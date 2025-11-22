#!/usr/bin/env python3
"""
使用mplfinance生成专业K线图
"""

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def generate_kline_chart(code: str, df: pd.DataFrame, signals: list, output_dir: str = 'static/charts') -> str:
    """
    使用mplfinance生成专业K线图
    
    Args:
        code: 股票代码
        df: 股票数据（包含date, open, high, low, close, volume）
        signals: 交易信号列表
        output_dir: 输出目录
    
    Returns:
        图表文件路径
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    df_chart = df.copy()
    
    # 确保date列是索引
    if 'date' in df_chart.columns:
        df_chart.set_index('date', inplace=True)
    
    # 确保索引是DatetimeIndex
    if not isinstance(df_chart.index, pd.DatetimeIndex):
        df_chart.index = pd.to_datetime(df_chart.index)
    
    # 只保留mplfinance需要的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df_chart = df_chart[required_cols]
    
    # 准备附加绘制（均线、趋势线）
    addplot_list = []
    
    # MA5, MA10, MA20, MA60
    if 'MA5' in df.columns and df['MA5'].notna().any():
        addplot_list.append(
            mpf.make_addplot(df['MA5'], color='#FF6B6B', width=1, alpha=0.7, label='MA5')
        )
    if 'MA10' in df.columns and df['MA10'].notna().any():
        addplot_list.append(
            mpf.make_addplot(df['MA10'], color='#4ECDC4', width=1.5, alpha=0.8, label='MA10')
        )
    if 'MA20' in df.columns and df['MA20'].notna().any():
        addplot_list.append(
            mpf.make_addplot(df['MA20'], color='#FFE66D', width=1.5, alpha=0.8, label='MA20')
        )
    if 'MA60' in df.columns and df['MA60'].notna().any():
        addplot_list.append(
            mpf.make_addplot(df['MA60'], color='#95E1D3', width=1.5, alpha=0.8, label='MA60')
        )
    
    # 知行趋势线
    if 'trend_short' in df.columns and df['trend_short'].notna().any():
        addplot_list.append(
            mpf.make_addplot(df['trend_short'], color='#FF6B6B', width=2.5, label='短期趋势线')
        )
    if 'trend_long' in df.columns and df['trend_long'].notna().any():
        addplot_list.append(
            mpf.make_addplot(df['trend_long'], color='#1976D2', width=2.5, label='知行多空线')
        )
    
    # 准备信号标记
    buy_signals_dates = []
    buy_signals_prices = []
    sell_signals_dates = []
    sell_signals_prices = []
    
    for sig in signals:
        sig_date = pd.to_datetime(sig['date'])
        sig_price = sig['price']
        
        if sig['type'] in ['B', 'B1']:
            buy_signals_dates.append(sig_date)
            buy_signals_prices.append(sig_price)
        elif sig['type'] in ['S', 'S1']:
            sell_signals_dates.append(sig_date)
            sell_signals_prices.append(sig_price)
    
    # 创建信号序列（用于在K线图上标记）
    buy_markers = pd.Series(index=df_chart.index, data=float('nan'))
    sell_markers = pd.Series(index=df_chart.index, data=float('nan'))
    
    for date, price in zip(buy_signals_dates, buy_signals_prices):
        if date in df_chart.index:
            buy_markers.loc[date] = price * 0.97  # 标记在价格下方3%
    
    for date, price in zip(sell_signals_dates, sell_signals_prices):
        if date in df_chart.index:
            sell_markers.loc[date] = price * 1.03  # 标记在价格上方3%
    
    # 添加信号标记到addplot
    if buy_markers.notna().any():
        addplot_list.append(
            mpf.make_addplot(buy_markers, type='scatter', markersize=100, 
                           marker='^', color='green', label='买点')
        )
    if sell_markers.notna().any():
        addplot_list.append(
            mpf.make_addplot(sell_markers, type='scatter', markersize=100,
                           marker='v', color='red', label='卖点')
        )
    
    # 配置样式
    mc = mpf.make_marketcolors(
        up='#EF5350',      # 阳线红色
        down='#26A69A',    # 阴线绿色
        edge='inherit',
        wick='inherit',
        volume='in',
        alpha=0.9
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        gridcolor='#f0f0f0',
        facecolor='white',
        figcolor='white',
        y_on_right=False,
        rc={'font.size': 10, 'axes.labelsize': 11}
    )
    
    # 统计信号数量
    b_count = len(buy_signals_dates)
    s_count = len(sell_signals_dates)
    
    # 判断战法类型
    is_breakout = any(sig['type'] in ['B', 'S'] for sig in signals)
    signal_summary = f"出坑战法【B:{b_count} | S:{s_count}】" if is_breakout else f"少妇战法【B1:{b_count} | S1:{s_count}】"
    
    # 绘制图表
    output_file = Path(output_dir) / f"{code}.png"
    
    fig, axes = mpf.plot(
        df_chart,
        type='candle',
        style=s,
        volume=True,
        addplot=addplot_list if addplot_list else None,
        title=f'\n{code} K线图 【{signal_summary}】',
        ylabel='价格 (¥)',
        ylabel_lower='成交量',
        figsize=(14, 8),
        tight_layout=True,
        returnfig=True,
        show_nontrading=False,  # 自动去除非交易日
        savefig=dict(fname=str(output_file), dpi=150, bbox_inches='tight')
    )
    
    plt.close(fig)
    
    return f'/static/charts/{code}.png'


if __name__ == '__main__':
    # 测试
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    from web_app import StockSelector
    
    selector = StockSelector('data', 'configs.json')
    df = selector.get_stock_data('300350', 120, strategy='breakout')
    
    if df is not None:
        signals = []
        for idx, row in df.iterrows():
            if row.get('signal') and row['signal'] != '':
                signals.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'price': row.get('signal_price', row['close']),
                    'type': row['signal']
                })
        
        output_path = generate_kline_chart('300350', df, signals)
        print(f"✅ 图表已生成: {output_path}")
    else:
        print("❌ 获取数据失败")

