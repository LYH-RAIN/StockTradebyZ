#!/usr/bin/env python3
"""测试出坑战法B信号"""
import pandas as pd
import os

DATA_DIR = "data"

# 读取测试股票
code = "603027"
file_path = os.path.join(DATA_DIR, f"{code}.csv")

if not os.path.exists(file_path):
    print(f"❌ 文件不存在: {file_path}")
    exit(1)

df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

# 过滤非交易日
df = df[df['volume'] > 0].copy()

# 取最近220天
df = df.tail(220)

print(f"✅ 加载数据: {len(df)}个交易日")
print(f"   日期范围: {df.iloc[0]['date'].strftime('%Y-%m-%d')} ~ {df.iloc[-1]['date'].strftime('%Y-%m-%d')}")
print(f"   价格范围: ¥{df['close'].min():.2f} ~ ¥{df['close'].max():.2f}")

# 计算均线
df['MA5'] = df['close'].rolling(5).mean()
df['MA10'] = df['close'].rolling(10).mean()

# 测试B信号识别
b_signals = []

for i in range(20, len(df)):
    # 查找前期最高价
    lookback_start = max(0, i-60)
    lookback_end = max(lookback_start + 5, i-3)
    
    if lookback_end <= lookback_start:
        continue
    
    lookback_window = df.iloc[lookback_start:lookback_end]
    if len(lookback_window) < 5:
        continue
    
    prev_high = lookback_window['high'].max()
    current_close = df.iloc[i]['close']
    current_open = df.iloc[i]['open']
    current_high = df.iloc[i]['high']
    current_vol = df.iloc[i]['volume']
    current_date = df.iloc[i]['date']
    
    # 计算距离前高
    distance_to_high = (prev_high - current_high) / prev_high
    
    # 计算量比
    vol_window_start = max(0, i-5)
    vol_ma5 = df.iloc[vol_window_start:i]['volume'].mean() if i > vol_window_start else current_vol
    vol_ratio = current_vol / vol_ma5 if vol_ma5 > 0 else 0
    
    # 判断
    is_close_to_high = (distance_to_high <= 0.30 and distance_to_high >= -0.10)
    is_bullish = (current_close >= current_open * 0.97)
    is_vol_up = (current_vol > vol_ma5 * 1.3)
    
    if is_close_to_high and is_bullish and is_vol_up:
        b_signals.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'price': current_close,
            'prev_high': prev_high,
            'distance_pct': distance_to_high * 100,
            'vol_ratio': vol_ratio
        })
        
        if len(b_signals) <= 5:
            print(f"\n✅ B信号 #{len(b_signals)}")
            print(f"   日期: {current_date.strftime('%Y-%m-%d')}")
            print(f"   价格: ¥{current_close:.2f}")
            print(f"   前高: ¥{prev_high:.2f}")
            print(f"   距离: {distance_to_high*100:.1f}%")
            print(f"   量比: {vol_ratio:.2f}x")

print(f"\n{'='*50}")
print(f"总计: {len(b_signals)}个B信号")
if len(b_signals) == 0:
    print("\n⚠️ 没有发现B信号，可能原因:")
    print("   1. 该股票近期没有接近前高的走势")
    print("   2. 需要放宽条件或调整参数")
    print("   3. 测试其他股票")

