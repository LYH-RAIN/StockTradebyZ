#!/usr/bin/env python3
"""
出坑战法测试脚本

用途：
1. 测试出坑战法的选股功能
2. 单只股票详细分析
3. 参数调优测试
"""

from pathlib import Path
import pandas as pd
from Selector import BreakoutPreviousHighSelector
from select_stock import load_data
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("test_breakout")


def test_single_stock(code: str, date: str = None):
    """
    测试单只股票
    
    Args:
        code: 股票代码（6位）
        date: 测试日期（YYYY-MM-DD），默认使用最新数据
    """
    print(f"\n{'='*60}")
    print(f"测试股票：{code}")
    print(f"{'='*60}\n")
    
    # 读取数据
    data_file = Path(f"./data/{code}.csv")
    if not data_file.exists():
        print(f"❌ 数据文件不存在：{data_file}")
        return
    
    df = pd.read_csv(data_file, parse_dates=['date'])
    df = df.sort_values('date')
    
    if date:
        hist = df[df['date'] <= date]
    else:
        hist = df
        date = hist.iloc[-1]['date']
    
    print(f"交易日期：{date}")
    print(f"数据量：{len(hist)}条")
    print(f"日期范围：{hist.iloc[0]['date']} 至 {hist.iloc[-1]['date']}\n")
    
    # 创建选择器
    selector = BreakoutPreviousHighSelector(
        lookback_days=60,
        consolidation_min_days=5,
        consolidation_max_days=30,
        approach_pct=0.15,
        vol_ratio_threshold=2.0,
        turnover_threshold=0.05,
        consolidation_shrink_ratio=0.6,
        pullback_pct_min=0.05,
        pullback_pct_max=0.25,
        ma_converge_threshold=0.05,
        max_window=120
    )
    
    # 逐步检查
    print("=" * 60)
    print("Step 1: 基础数据检查")
    print("=" * 60)
    last_row = hist.iloc[-1]
    prev_row = hist.iloc[-2]
    
    print(f"当日收盘：{last_row['close']:.2f}")
    print(f"当日成交量：{last_row['volume']:.0f}")
    print(f"当日涨跌幅：{(last_row['close']/prev_row['close']-1)*100:.2f}%")
    print(f"当日振幅：{(last_row['high']-last_row['low'])/last_row['low']*100:.2f}%\n")
    
    # 计算均线
    hist_copy = hist.copy()
    hist_copy['MA5'] = hist_copy['close'].rolling(5).mean()
    hist_copy['MA10'] = hist_copy['close'].rolling(10).mean()
    hist_copy['MA20'] = hist_copy['close'].rolling(20).mean()
    hist_copy['MA30'] = hist_copy['close'].rolling(30).mean()
    
    print("=" * 60)
    print("Step 2: 均线系统")
    print("=" * 60)
    print(f"MA5:  {hist_copy.iloc[-1]['MA5']:.2f}")
    print(f"MA10: {hist_copy.iloc[-1]['MA10']:.2f}")
    print(f"MA20: {hist_copy.iloc[-1]['MA20']:.2f}")
    print(f"MA30: {hist_copy.iloc[-1]['MA30']:.2f}")
    print(f"当前价格在MA5{'上方' if last_row['close'] > hist_copy.iloc[-1]['MA5'] else '下方'}")
    print(f"当前价格在MA20{'上方' if last_row['close'] > hist_copy.iloc[-1]['MA20'] else '下方'}\n")
    
    print("=" * 60)
    print("Step 3: 前高识别")
    print("=" * 60)
    prev_high_info = selector._find_previous_high(hist_copy)
    
    if prev_high_info:
        high_price = prev_high_info['high_price']
        high_date = prev_high_info['high_date']
        high_idx = prev_high_info['high_idx']
        
        print(f"✓ 找到前高")
        print(f"  前高价格：{high_price:.2f}")
        print(f"  前高日期：{high_date}")
        print(f"  距今天数：{len(hist)-high_idx-1}天")
        print(f"  当前距离前高：{abs(last_row['close']-high_price)/high_price*100:.2f}%")
        print(f"  {'在前高上方' if last_row['close'] > high_price else '在前高下方'}\n")
        
        print("=" * 60)
        print("Step 4: 回调幅度检查")
        print("=" * 60)
        after_high = hist_copy.iloc[high_idx + 1 : -1]
        if not after_high.empty:
            pullback_low = float(after_high['close'].min())
            pullback_pct = (high_price - pullback_low) / high_price
            
            print(f"回调最低价：{pullback_low:.2f}")
            print(f"回调幅度：{pullback_pct*100:.2f}%")
            print(f"要求范围：5%-25%")
            
            if 0.05 <= pullback_pct <= 0.25:
                print(f"✓ 回调幅度合理\n")
            else:
                print(f"✗ 回调幅度不符合要求\n")
        
        print("=" * 60)
        print("Step 5: 震荡整理检查")
        print("=" * 60)
        consol_info = selector._check_consolidation_phase(hist_copy, high_idx)
        
        if consol_info:
            consol_days = consol_info['consol_end_idx'] - consol_info['consol_start_idx'] + 1
            consol_avg_vol = consol_info['avg_volume']
            
            print(f"✓ 找到震荡区间")
            print(f"  震荡天数：{consol_days}天")
            print(f"  要求范围：5-30天")
            print(f"  震荡期平均成交量：{consol_avg_vol:.0f}\n")
            
            print("=" * 60)
            print("Step 6: 量能分析")
            print("=" * 60)
            
            # 拉升期成交量
            rally_vol_seg = hist_copy.iloc[max(0, high_idx-5):high_idx+1]['volume']
            rally_avg_vol = float(rally_vol_seg.mean())
            
            # 当日成交量
            vol_today = float(last_row['volume'])
            
            # 计算各种比率
            vol_ratio = vol_today / consol_avg_vol if consol_avg_vol > 0 else 0
            shrink_ratio = consol_avg_vol / rally_avg_vol if rally_avg_vol > 0 else 0
            
            print(f"拉升期平均量：{rally_avg_vol:.0f}")
            print(f"震荡期平均量：{consol_avg_vol:.0f}")
            print(f"当日成交量：{vol_today:.0f}")
            print(f"\n缩量比率：{shrink_ratio:.2f} (震荡期/拉升期，要求≤0.6)")
            print(f"当日量比：{vol_ratio:.2f} (当日/震荡期，要求≥2.0)")
            
            if shrink_ratio <= 0.6:
                print(f"✓ 震荡期缩量明显")
            else:
                print(f"✗ 震荡期缩量不足")
            
            if vol_ratio >= 2.0:
                print(f"✓ 当日放量突破")
            else:
                print(f"✗ 当日放量不足")
            print()
        else:
            print(f"✗ 未找到有效震荡区间\n")
    else:
        print(f"✗ 未找到有效前高\n")
    
    # 最终判断
    print("=" * 60)
    print("Step 7: 最终判断")
    print("=" * 60)
    
    result = selector._passes_filters(hist_copy)
    
    if result:
        print(f"✅ 符合出坑战法条件！")
        print(f"\n操作建议：")
        print(f"  买入价格：{last_row['close']:.2f}")
        if prev_high_info:
            print(f"  前高位置：{prev_high_info['high_price']:.2f}")
            print(f"  止损位：{hist_copy.iloc[-1]['MA10']:.2f} (MA10)")
            print(f"  止损幅度：{(hist_copy.iloc[-1]['MA10']/last_row['close']-1)*100:.2f}%")
            print(f"  目标位1：{prev_high_info['high_price']*1.05:.2f} (+5%)")
            print(f"  目标位2：{prev_high_info['high_price']*1.10:.2f} (+10%)")
            print(f"  目标位3：{prev_high_info['high_price']*1.15:.2f} (+15%)")
    else:
        print(f"❌ 不符合出坑战法条件")
    
    print(f"\n{'='*60}\n")
    return result


def test_batch_stocks(num_stocks: int = 50, date: str = None):
    """
    批量测试多只股票
    
    Args:
        num_stocks: 测试股票数量
        date: 测试日期
    """
    print(f"\n{'='*60}")
    print(f"批量测试 - 测试{num_stocks}只股票")
    print(f"{'='*60}\n")
    
    # 获取股票列表
    data_dir = Path("./data")
    csv_files = list(data_dir.glob("*.csv"))[:num_stocks]
    codes = [f.stem for f in csv_files]
    
    print(f"加载数据：{len(codes)}只股票...")
    data = load_data(data_dir, codes)
    print(f"成功加载：{len(data)}只股票\n")
    
    # 创建选择器
    selector = BreakoutPreviousHighSelector()
    
    # 确定交易日
    if date:
        trade_date = pd.Timestamp(date)
    else:
        trade_date = max(df['date'].max() for df in data.values())
    
    print(f"交易日期：{trade_date.date()}")
    print(f"开始选股...\n")
    
    # 执行选股
    picks = selector.select(trade_date, data)
    
    # 输出结果
    print(f"{'='*60}")
    print(f"选股结果")
    print(f"{'='*60}\n")
    print(f"符合条件股票数：{len(picks)}")
    print(f"选中率：{len(picks)/len(data)*100:.2f}%\n")
    
    if picks:
        print(f"股票代码：")
        for i, code in enumerate(picks, 1):
            print(f"  {i}. {code}")
    else:
        print(f"未选中任何股票")
    
    print(f"\n{'='*60}\n")
    return picks


def test_parameters():
    """
    测试不同参数组合
    """
    print(f"\n{'='*60}")
    print(f"参数测试")
    print(f"{'='*60}\n")
    
    # 加载数据
    data_dir = Path("./data")
    codes = [f.stem for f in data_dir.glob("*.csv")][:100]
    data = load_data(data_dir, codes)
    trade_date = max(df['date'].max() for df in data.values())
    
    # 测试不同参数组合
    param_sets = [
        {
            "name": "保守型",
            "vol_ratio_threshold": 2.5,
            "approach_pct": 0.10,
            "consolidation_min_days": 10
        },
        {
            "name": "标准型（默认）",
            "vol_ratio_threshold": 2.0,
            "approach_pct": 0.15,
            "consolidation_min_days": 5
        },
        {
            "name": "激进型",
            "vol_ratio_threshold": 1.8,
            "approach_pct": 0.20,
            "consolidation_min_days": 3
        },
    ]
    
    results = []
    
    for params in param_sets:
        name = params.pop("name")
        print(f"测试参数组：{name}")
        print(f"  参数：{params}")
        
        selector = BreakoutPreviousHighSelector(**params)
        picks = selector.select(trade_date, data)
        
        print(f"  选中：{len(picks)}只股票")
        print(f"  选中率：{len(picks)/len(data)*100:.2f}%\n")
        
        results.append({
            "name": name,
            "count": len(picks),
            "rate": len(picks)/len(data)*100
        })
    
    # 汇总
    print(f"{'='*60}")
    print(f"参数对比汇总")
    print(f"{'='*60}\n")
    print(f"{'参数组':<15} {'选中数量':<10} {'选中率':<10}")
    print(f"-" * 40)
    for r in results:
        print(f"{r['name']:<15} {r['count']:<10} {r['rate']:.2f}%")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "single":
            # 单只股票测试
            if len(sys.argv) > 2:
                code = sys.argv[2]
                date = sys.argv[3] if len(sys.argv) > 3 else None
                test_single_stock(code, date)
            else:
                print("用法：python test_breakout.py single <股票代码> [日期]")
                print("示例：python test_breakout.py single 000001 2025-09-10")
        
        elif command == "batch":
            # 批量测试
            num = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            date = sys.argv[3] if len(sys.argv) > 3 else None
            test_batch_stocks(num, date)
        
        elif command == "params":
            # 参数测试
            test_parameters()
        
        else:
            print(f"未知命令：{command}")
            print("可用命令：single, batch, params")
    
    else:
        # 默认执行批量测试
        print("使用方法：")
        print("  python test_breakout.py single <代码> [日期]  # 单只股票详细测试")
        print("  python test_breakout.py batch [数量] [日期]   # 批量测试")
        print("  python test_breakout.py params               # 参数对比测试")
        print()
        print("执行默认测试（50只股票）...")
        test_batch_stocks(50)


