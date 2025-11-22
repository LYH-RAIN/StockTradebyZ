#!/usr/bin/env python3
"""
出坑战法（突破前高战法）测试脚本

使用方法：
python test_breakout_strategy.py --data-dir ./data --date 2025-09-10
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict
import logging

from Selector import BreakoutPreviousHighSelector
from select_stock import load_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="测试出坑战法")
    parser.add_argument("--data-dir", default="./data", help="数据目录")
    parser.add_argument("--date", help="交易日 YYYY-MM-DD（可选）")
    parser.add_argument("--codes", help="指定股票代码，逗号分隔（可选）")
    
    # 策略参数（可调优）
    parser.add_argument("--lookback-days", type=int, default=60, help="回看窗口")
    parser.add_argument("--approach-pct", type=float, default=0.15, help="接近前高距离")
    parser.add_argument("--vol-ratio", type=float, default=2.0, help="放量倍数")
    
    args = parser.parse_args()
    
    # 加载数据
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",")]
    else:
        codes = [f.stem for f in data_dir.glob("*.csv")]
    
    logger.info(f"加载 {len(codes)} 只股票数据...")
    data = load_data(data_dir, codes)
    
    if not data:
        logger.error("未能加载任何数据")
        return
    
    # 确定交易日
    if args.date:
        trade_date = pd.to_datetime(args.date)
    else:
        trade_date = max(df["date"].max() for df in data.values())
    
    logger.info(f"测试日期: {trade_date.date()}")
    
    # 创建选择器
    selector = BreakoutPreviousHighSelector(
        lookback_days=args.lookback_days,
        consolidation_min_days=5,
        consolidation_max_days=30,
        approach_pct=args.approach_pct,
        vol_ratio_threshold=args.vol_ratio,
        turnover_threshold=0.05,
        consolidation_shrink_ratio=0.6,
        pullback_pct_min=0.05,
        pullback_pct_max=0.25,
        ma_converge_threshold=0.05,
        max_window=120
    )
    
    logger.info("执行选股...")
    picks = selector.select(trade_date, data)
    
    # 输出结果
    print("\n" + "="*60)
    print(f"【出坑战法】选股结果")
    print("="*60)
    print(f"测试日期: {trade_date.date()}")
    print(f"股票池数量: {len(data)}")
    print(f"符合条件股票数: {len(picks)}")
    print("-"*60)
    
    if picks:
        print("符合条件的股票代码:")
        # 按代码排序
        picks_sorted = sorted(picks)
        
        # 分行显示，每行10个
        for i in range(0, len(picks_sorted), 10):
            batch = picks_sorted[i:i+10]
            print("  " + ", ".join(batch))
        
        # 显示详细信息（前5只）
        print("\n" + "-"*60)
        print("前5只股票详细信息:")
        print("-"*60)
        
        for code in picks_sorted[:5]:
            df = data[code]
            hist = df[df["date"] <= trade_date].tail(120)
            
            if not hist.empty:
                last = hist.iloc[-1]
                print(f"\n股票代码: {code}")
                print(f"  当日收盘: {last['close']:.2f}")
                print(f"  当日成交量: {last['volume']:.0f}")
                
                # 计算前高
                window = hist.iloc[-(60 + 5):-5]
                if not window.empty:
                    prev_high = window["high"].max()
                    distance = (prev_high - last['close']) / prev_high * 100
                    print(f"  前期高点: {prev_high:.2f}")
                    print(f"  距离前高: {distance:.2f}%")
                
                # 计算均线
                ma5 = hist['close'].rolling(5).mean().iloc[-1]
                ma10 = hist['close'].rolling(10).mean().iloc[-1]
                ma20 = hist['close'].rolling(20).mean().iloc[-1]
                ma30 = hist['close'].rolling(30).mean().iloc[-1]
                
                print(f"  MA5: {ma5:.2f}, MA10: {ma10:.2f}")
                print(f"  MA20: {ma20:.2f}, MA30: {ma30:.2f}")
    else:
        print("未找到符合条件的股票")
    
    print("\n" + "="*60)
    
    # 参数调优建议
    if len(picks) == 0:
        print("\n💡 调优建议（结果为空时）:")
        print("  1. 放宽接近前高距离: --approach-pct 0.20")
        print("  2. 降低放量要求: --vol-ratio 1.5")
        print("  3. 扩大回看窗口: --lookback-days 90")
    elif len(picks) > 50:
        print("\n💡 调优建议（结果过多时）:")
        print("  1. 缩小接近前高距离: --approach-pct 0.10")
        print("  2. 提高放量要求: --vol-ratio 2.5")
        print("  3. 缩小回看窗口: --lookback-days 45")


if __name__ == "__main__":
    main()

