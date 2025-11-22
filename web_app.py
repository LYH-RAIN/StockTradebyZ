#!/usr/bin/env python3
"""
量化选股Web界面
提供可视化的选股结果展示和K线图分析
"""

from flask import Flask, render_template, jsonify, request
from pathlib import Path
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

from select_stock import load_data, load_config, instantiate_selector
from signal_identifier import SignalIdentifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("web_app")

# 创建Flask应用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 支持中文

# 全局配置
DATA_DIR = Path("./data")
CONFIG_FILE = Path("./configs.json")

class StockSelector:
    """选股器管理类"""
    
    def __init__(self, data_dir: Path, config_file: Path, max_stocks: int = None):
        self.data_dir = data_dir
        self.config_file = config_file
        self.max_stocks = max_stocks  # None表示加载全部
        self.data = None
        self.selectors = {}
        self.all_codes = []  # 所有股票代码
        self.load_data()
        self.load_selectors()
    
    def load_data(self):
        """加载股票数据"""
        logger.info(f"📊 开始加载数据：{self.data_dir}")
        
        # 获取所有股票代码
        all_files = list(self.data_dir.glob("*.csv"))
        self.all_codes = [f.stem for f in all_files]
        total_count = len(self.all_codes)
        logger.info(f"📋 发现 {total_count} 只股票")
        
        # 根据配置决定加载数量
        if self.max_stocks and self.max_stocks < total_count:
            codes_to_load = self.all_codes[:self.max_stocks]
            logger.info(f"⚡ 性能模式：仅加载前 {len(codes_to_load)} 只股票（约{len(codes_to_load)/total_count*100:.0f}%）")
        else:
            codes_to_load = self.all_codes
            logger.info(f"🔄 正在加载全部 {total_count} 只股票，请稍候...")
        
        self.data = load_data(self.data_dir, codes_to_load)
        logger.info(f"✅ 成功加载 {len(self.data)} 只股票数据！")
    
    def load_selectors(self):
        """加载所有选择器"""
        if not self.config_file.exists():
            logger.error(f"配置文件不存在: {self.config_file}")
            return
        
        selector_configs = load_config(self.config_file)
        for selector_cfg in selector_configs:
            if not selector_cfg.get("activate", False):
                continue
            
            try:
                # instantiate_selector 返回 (alias, instance) 元组
                alias, selector_instance = instantiate_selector(selector_cfg)
                self.selectors[alias] = {
                    "class": selector_cfg["class"],
                    "instance": selector_instance,
                    "params": selector_cfg.get("params", {})
                }
                logger.info(f"加载选择器: {alias}")
            except Exception as e:
                alias = selector_cfg.get("alias", selector_cfg["class"])
                logger.error(f"加载选择器失败 {alias}: {e}")
    
    def run_selector(self, selector_name: str, trade_date: str = None) -> List[str]:
        """运行指定选择器"""
        if selector_name not in self.selectors:
            return []
        
        if trade_date is None:
            # 使用最近的交易日
            all_dates = set()
            for df in self.data.values():
                all_dates.update(df['date'])
            trade_date = max(all_dates)
        else:
            trade_date = pd.Timestamp(trade_date)
        
        selector = self.selectors[selector_name]["instance"]
        picks = selector.select(trade_date, self.data)
        return picks
    
    def get_stock_data(self, code: str, days: int = 120, strategy: str = 'default') -> Optional[pd.DataFrame]:
        """获取股票数据（包含技术指标和交易信号）"""
        if code not in self.data:
            return None
        
        df = self.data[code].copy()
        
        # 调试date列状态
        print(f"DEBUG get_stock_data 开始: date列类型={df['date'].dtype if 'date' in df.columns else 'N/A'}, date是索引={df.index.name=='date'}")
        
        # 确保date列不是索引
        if 'date' not in df.columns and df.index.name == 'date':
            df.reset_index(inplace=True)
            print(f"DEBUG: reset_index后，date列类型={df['date'].dtype}")
        
        # 确保date列是datetime类型
        if 'date' in df.columns and df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'])
            print(f"DEBUG: to_datetime后，date列类型={df['date'].dtype}")
        
        # 过滤掉节假日（成交量为0的日期）
        df = df[df['volume'] > 0].copy()
        print(f"DEBUG: 过滤volume>0后，df形状={df.shape}, date列NaT数={df['date'].isna().sum() if 'date' in df.columns else 'N/A'}")
        
        # 取更多数据用于信号识别（需要回看60日，加上非交易日需要更多）
        df_full = df.tail(days + 150).copy()  # 使用copy避免SettingWithCopyWarning
        print(f"DEBUG: tail后，df_full形状={df_full.shape}, date列NaT数={df_full['date'].isna().sum()}")
        
        # 计算均线
        df_full['MA5'] = df_full['close'].rolling(5).mean()
        df_full['MA10'] = df_full['close'].rolling(10).mean()
        df_full['MA20'] = df_full['close'].rolling(20).mean()
        df_full['MA30'] = df_full['close'].rolling(30).mean()
        df_full['MA60'] = df_full['close'].rolling(60).mean()
        
        # 计算MACD
        df_full = self._calculate_macd(df_full)
        print(f"DEBUG: calculate_macd后，date列NaT数={df_full['date'].isna().sum()}")
        
        # 计算KDJ
        df_full = self._calculate_kdj(df_full)
        print(f"DEBUG: calculate_kdj后，date列NaT数={df_full['date'].isna().sum()}")
        
        # 计算趋势线
        df_full = self._calculate_trend_lines(df_full)
        print(f"DEBUG: calculate_trend_lines后，date列NaT数={df_full['date'].isna().sum()}")
        
        # 识别交易信号（在完整数据上）
        df_full = self._identify_trading_signals(df_full, strategy=strategy)
        print(f"DEBUG: identify_signals后，date列NaT数={df_full['date'].isna().sum()}, B信号行的date示例={df_full[df_full['signal']=='B']['date'].iloc[0] if len(df_full[df_full['signal']=='B'])>0 else 'N/A'}")
        
        # 最后截取需要的天数
        df_result = df_full.tail(days).copy()
        print(f"DEBUG: tail(days)后，df_result形状={df_result.shape}, date列NaT数={df_result['date'].isna().sum()}, B信号行数={len(df_result[df_result['signal']=='B'])}")
        
        # 重置索引（date列已经是数据列，不是索引）
        df_result.reset_index(drop=True, inplace=True)
        print(f"DEBUG: reset_index后，date列NaT数={df_result['date'].isna().sum()}, 前3个date={df_result['date'].head(3).tolist()}")
        
        # 确保date列不是NaT - 如果有NaT，从索引恢复
        if 'date' in df_result.columns:
            nat_count = df_result['date'].isna().sum()
            if nat_count > 0:
                print(f"WARNING: {nat_count}行date为NaT，尝试修复...")
                # 如果date列有NaT，尝试从原始数据恢复
                for idx in df_result.index:
                    if pd.isna(df_result.loc[idx, 'date']):
                        # 尝试从原始df恢复
                        if idx < len(df_full):
                            original_date = df_full.iloc[idx]['date']
                            if pd.notna(original_date):
                                df_result.loc[idx, 'date'] = original_date
        
        # 调试输出
        if strategy == 'breakout':
            total_b = len(df_full[df_full['signal'] == 'B'])
            result_b = len(df_result[df_result['signal'] == 'B'])
            print(f"DEBUG: 完整数据B信号: {total_b}个, 截取后B信号: {result_b}个")
        
        return df_result
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        # EMA12和EMA26
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # DIF = EMA12 - EMA26
        df['DIF'] = df['EMA12'] - df['EMA26']
        
        # DEA = DIF的9日EMA
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        
        # MACD柱 = (DIF - DEA) * 2
        df['MACD'] = (df['DIF'] - df['DEA']) * 2
        
        return df
    
    def _calculate_kdj(self, df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
        """计算KDJ指标"""
        # 计算RSV
        low_list = df['low'].rolling(n, min_periods=1).min()
        high_list = df['high'].rolling(n, min_periods=1).max()
        
        df['RSV'] = (df['close'] - low_list) / (high_list - low_list) * 100
        df['RSV'].fillna(0, inplace=True)
        
        # 计算K、D、J
        df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        return df
    
    def _calculate_trend_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势线（知行体系标准公式）"""
        # 短期趋势线 = EMA(EMA(CLOSE,10),10) - 双重指数移动平均
        ema1 = df['close'].ewm(span=10, adjust=False).mean()
        df['trend_short'] = ema1.ewm(span=10, adjust=False).mean()
        
        # 知行多空线 = (MA(14)+MA(28)+MA(57)+MA(114))/4 - 四条均线的平均值
        ma14 = df['close'].rolling(14, min_periods=1).mean()
        ma28 = df['close'].rolling(28, min_periods=1).mean()
        ma57 = df['close'].rolling(57, min_periods=1).mean()
        ma114 = df['close'].rolling(114, min_periods=1).mean()
        df['trend_long'] = (ma14 + ma28 + ma57 + ma114) / 4
        
        # 计算差值百分比
        df['trend_diff_pct'] = abs((df['close'] - df['trend_short']) / df['trend_short']) * 100
        
        # 判断短期趋势是否在知行多空线上方
        df['trend_short_above'] = df['trend_short'] > df['trend_long']
        
        return df
    
    def _identify_trading_signals(self, df: pd.DataFrame, strategy: str = 'default') -> pd.DataFrame:
        """
        识别交易信号
        
        strategy参数：
        - 'default': B1/B2/S1（少妇战法）
        - 'breakout': B/S（出坑战法）
        """
        print(f"DEBUG _identify_trading_signals 开始: df形状={df.shape}, date列NaT数={df['date'].isna().sum()}")
        
        df['signal'] = ''
        df['signal_price'] = 0.0
        
        print(f"DEBUG 添加signal列后: date列NaT数={df['date'].isna().sum()}")
        
        if strategy == 'breakout':
            result = self._identify_breakout_signals(df)
        else:
            result = self._identify_default_signals(df)
        
        print(f"DEBUG _identify_trading_signals 结束: date列NaT数={result['date'].isna().sum()}")
        return result
    
    def _identify_default_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别默认信号：B1（买入）、S1（卖出）、B2（加仓）- 少妇战法
        
        使用统一的信号识别模块
        """
        
        # 使用统一的信号识别模块
        signals = SignalIdentifier.identify_shaofv_signals(df)
        
        # 将信号添加到DataFrame
        df['signal'] = ''
        df['signal_price'] = 0.0
        
        for signal in signals:
            # 找到对应日期的行
            mask = df['date'] == signal['date']
            if mask.any():
                idx = df.index[mask][0]
                df.at[idx, 'signal'] = signal['type']
                df.at[idx, 'signal_price'] = signal['price']
        
        return df
        
    def _identify_default_signals_OLD(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别默认信号：B1（买入）、S1（卖出）、B2（加仓）- 少妇战法
        
        【已废弃】使用新的统一信号识别模块
        """
        
        # 识别B1信号
        for i in range(10, len(df)):
            # B1条件：
            # 1. J值曾到达大负值区域（J < -20）
            # 2. J值开始回升（从负值向上）
            # 3. 出现右侧转强信号：收出阳线或反包阳线
            # 4. 成交量放大
            
            j_current = df.iloc[i]['J']
            j_prev = df.iloc[i-1]['J']
            j_min_last10 = df.iloc[i-10:i]['J'].min()
            
            close_current = df.iloc[i]['close']
            close_prev = df.iloc[i-1]['close']
            open_current = df.iloc[i]['open']
            
            vol_current = df.iloc[i]['volume']
            vol_ma5 = df.iloc[i-5:i]['volume'].mean()
            
            # B1判断
            if (j_min_last10 < -20 and  # 近期曾到达超卖区
                j_current > j_prev and  # J值回升
                close_current > open_current and  # 收阳线
                close_current > close_prev and  # 价格上涨
                vol_current > vol_ma5 * 1.2):  # 成交量放大
                
                df.iloc[i, df.columns.get_loc('signal')] = 'B1'
                df.iloc[i, df.columns.get_loc('signal_price')] = close_current
        
        # 识别S1信号（放量大阴线）
        for i in range(5, len(df)):
            close_current = df.iloc[i]['close']
            open_current = df.iloc[i]['open']
            high_current = df.iloc[i]['high']
            low_current = df.iloc[i]['low']
            
            vol_current = df.iloc[i]['volume']
            vol_ma5 = df.iloc[i-5:i]['volume'].mean()
            
            # 计算阴线实体
            body_size = abs(close_current - open_current)
            candle_size = high_current - low_current
            
            # S1判断：放量大阴线
            if (close_current < open_current and  # 阴线
                body_size / candle_size > 0.6 and  # 实体占比>60%
                vol_current > vol_ma5 * 1.5 and  # 成交量明显放大
                (open_current - close_current) / open_current > 0.03):  # 跌幅>3%
                
                df.iloc[i, df.columns.get_loc('signal')] = 'S1'
                df.iloc[i, df.columns.get_loc('signal_price')] = close_current
        
        # 识别B2信号（B1后次日继续上涨）
        for i in range(1, len(df)):
            if df.iloc[i-1]['signal'] == 'B1':
                close_current = df.iloc[i]['close']
                close_prev = df.iloc[i-1]['close']
                open_current = df.iloc[i]['open']
                
                # B2判断：B1次日继续上涨
                if (close_current > open_current and  # 收阳线
                    close_current > close_prev):  # 价格高于B1日收盘价
                    
                    df.loc[i, 'signal'] = 'B2'
                    df.loc[i, 'signal_price'] = close_current
        
        return df
    
    def _identify_breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别出坑战法信号 - 使用统一的信号识别模块
        
        核心逻辑：
        - B信号：股价接近前期高点（75%-105%区间），放量，是突破前兆
        - S信号：高位放量大阴线或连续下跌累计-7%
        
        统一使用signal_identifier.SignalIdentifier，确保与选股逻辑一致
        """
        
        # 使用统一的信号识别模块
        signals = SignalIdentifier.identify_breakout_signals(df)
        
        # 将信号添加到DataFrame
        df['signal'] = ''
        df['signal_price'] = 0.0
        
        for signal in signals:
            # 找到对应日期的行
            mask = df['date'] == signal['date']
            if mask.any():
                idx = df.index[mask][0]
                df.at[idx, 'signal'] = signal['type']
                df.at[idx, 'signal_price'] = signal['price']
        
        return df
        
        # 识别B信号：接近前高买入（出坑时机）
        for i in range(20, len(df)):
            # 查找前期高点（30-60日前的最高价）
            lookback_start = max(0, i - 60)
            lookback_end = max(lookback_start + 5, i - 5)  # 至少排除最近5日
            
            if lookback_end <= lookback_start:
                continue
            
            prev_high = df.iloc[lookback_start:lookback_end]['high'].max()
            if pd.isna(prev_high) or prev_high == 0:
                continue
            
            current_close = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_vol = df.iloc[i]['volume']
            
            # 计算与前高的距离（用收盘价）
            distance_pct = (prev_high - current_close) / prev_high
            
            # 计算成交量比
            vol_ma5 = df.iloc[max(0, i-5):i]['volume'].mean() if i >= 5 else current_vol
            vol_ratio = current_vol / vol_ma5 if vol_ma5 > 0 else 0
            
            # MA支撑
            ma10 = df.iloc[i].get('MA10', 0)
            
            # B信号条件（更宽松）：
            # 1. 价格在前高的75%-105%区间（允许小幅超越）
            # 2. 成交量放大（1.2倍以上）
            # 3. 价格在MA10上方或附近（允许10%偏差）
            
            if (-0.05 <= distance_pct <= 0.25 and  # 前高的75%-105%区间
                vol_ratio >= 1.2 and  # 放量
                current_close >= ma10 * 0.9):  # 接近或站上MA10
                
                # 信号去重：检查最近5天内是否已有B信号
                recent_b_signal = False
                for j in range(max(0, i-5), i):
                    if df.iloc[j].get('signal') == 'B':
                        recent_b_signal = True
                        break
                
                # 只有没有近期B信号时才添加
                if not recent_b_signal:
                    df.iloc[i, df.columns.get_loc('signal')] = 'B'
                    df.iloc[i, df.columns.get_loc('signal_price')] = current_close
                    print(f"  B信号: 日期={df.iloc[i]['date']}, 价={current_close:.2f}, 前高={prev_high:.2f}, 距离={distance_pct*100:.1f}%, 量比={vol_ratio:.2f}")
        
        # 识别S信号：高位放量大阴线 或 连续阶梯阴量
        # 先找出最高点和次高点
        for i in range(10, len(df)):
            # 查找最近30日的最高价和次高价
            recent_window = df.iloc[max(0, i-30):i+1]
            if len(recent_window) < 10:
                continue
            
            # 最高价
            highest_price = recent_window['high'].max()
            if pd.isna(highest_price):
                continue
            
            highest_idx = recent_window['high'].idxmax()
            if pd.isna(highest_idx):
                continue
            
            # 次高价（排除最高价所在K线）
            temp_window = recent_window[recent_window.index != highest_idx]
            if temp_window.empty:
                second_highest_price = highest_price
            else:
                second_highest_price = temp_window['high'].max()
            
            current_close = df.iloc[i]['close']
            current_open = df.iloc[i]['open']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_vol = df.iloc[i]['volume']
            
            # 计算成交量平均值
            vol_ma5 = df.iloc[i-5:i]['volume'].mean() if i >= 5 else current_vol
            
            # S信号条件1：在最高点或次高点附近出现放量大阴线
            body_size = abs(current_open - current_close)
            candle_size = current_high - current_low
            is_big_red = (current_close < current_open and  # 阴线
                         body_size / candle_size > 0.6 if candle_size > 0 else False)  # 实体占比>60%
            
            is_at_peak = (current_high >= highest_price * 0.98 or  # 在最高点附近
                         current_high >= second_highest_price * 0.98)  # 或次高点附近
            
            if is_at_peak and is_big_red and current_vol > vol_ma5 * 1.5:
                df.iloc[i, df.columns.get_loc('signal')] = 'S'
                df.iloc[i, df.columns.get_loc('signal_price')] = current_close
                continue
            
            # S信号条件2：连续阶梯阴量累计-7
            # 阶梯阴量：连续阴线且每日跌幅累计
            if i >= 10:
                cumulative_drop = 0
                consecutive_red = 0
                
                for j in range(i, max(i-10, -1), -1):
                    if df.iloc[j]['close'] < df.iloc[j]['open']:  # 阴线
                        consecutive_red += 1
                        # 计算当日跌幅百分比
                        drop_pct = (df.iloc[j]['open'] - df.iloc[j]['close']) / df.iloc[j]['open'] * 100
                        cumulative_drop += drop_pct
                        
                        # 如果累计跌幅达到7%，标记S信号
                        if cumulative_drop >= 7.0 and consecutive_red >= 3:
                            df.iloc[i, df.columns.get_loc('signal')] = 'S'
                            df.iloc[i, df.columns.get_loc('signal_price')] = df.iloc[i]['close']
                            break
                    else:
                        break  # 遇到阳线中断
        
        return df
    
    def _backtest_signals(self, df: pd.DataFrame, signals: list, stop_loss_pct: float = 0.05) -> dict:
        """回测交易信号：B1/B买入 → S1/S卖出（支持止损）"""
        if not signals:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'win_count': 0,
                'loss_count': 0,
                'stop_loss_count': 0,
                'trades': []
            }
        
        # 创建日期到价格的映射
        date_to_data = {}
        for idx, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else None
            if date_str:
                date_to_data[date_str] = {
                    'low': row['low'],
                    'close': row['close'],
                    'high': row['high']
                }
        
        # 按日期排序信号
        signals_sorted = sorted(signals, key=lambda x: x['date'])
        
        trades = []
        current_position = None  # 当前持仓信息
        
        for signal in signals_sorted:
            signal_type = signal['type']
            signal_date = signal['date']
            signal_price = signal['price']
            
            # 买入信号
            if signal_type in ['B1', 'B'] and current_position is None:
                current_position = {
                    'buy_date': signal_date,
                    'buy_price': signal_price,
                    'buy_type': signal_type,
                    'stop_loss_price': signal_price * (1 - stop_loss_pct)  # 止损价
                }
            
            # 持仓期间检查止损
            elif current_position is not None:
                # 获取当前日期的数据
                if signal_date in date_to_data:
                    day_data = date_to_data[signal_date]
                    
                    # 检查是否触发止损（最低价跌破止损价）
                    if day_data['low'] <= current_position['stop_loss_price']:
                        # 止损卖出
                        buy_price = current_position['buy_price']
                        sell_price = current_position['stop_loss_price']
                        return_pct = (sell_price - buy_price) / buy_price * 100
                        
                        trades.append({
                            'buy_date': current_position['buy_date'],
                            'buy_price': buy_price,
                            'sell_date': signal_date,
                            'sell_price': sell_price,
                            'return_pct': round(return_pct, 2),
                            'days_held': (pd.to_datetime(signal_date) - pd.to_datetime(current_position['buy_date'])).days,
                            'sell_reason': '止损'
                        })
                        
                        current_position = None
                        continue
                
                # 卖出信号
                if signal_type in ['S1', 'S']:
                    buy_price = current_position['buy_price']
                    sell_price = signal_price
                    return_pct = (sell_price - buy_price) / buy_price * 100
                    
                    trades.append({
                        'buy_date': current_position['buy_date'],
                        'buy_price': buy_price,
                        'sell_date': signal_date,
                        'sell_price': sell_price,
                        'return_pct': round(return_pct, 2),
                        'days_held': (pd.to_datetime(signal_date) - pd.to_datetime(current_position['buy_date'])).days,
                        'sell_reason': signal_type
                    })
                    
                    current_position = None
        
        # 如果还有未平仓的持仓，用最后一天的收盘价计算
        if current_position is not None:
            last_row = df.iloc[-1]
            # 确保日期不是NaT
            if pd.notna(last_row['date']):
                last_date = last_row['date'].strftime('%Y-%m-%d')
            else:
                last_date = 'N/A'
            last_price = last_row['close']
            buy_price = current_position['buy_price']
            return_pct = (last_price - buy_price) / buy_price * 100
            
            trades.append({
                'buy_date': current_position['buy_date'],
                'buy_price': buy_price,
                'sell_date': last_date,
                'sell_price': last_price,
                'return_pct': round(return_pct, 2),
                'days_held': (pd.to_datetime(last_date) - pd.to_datetime(current_position['buy_date'])).days,
                'status': 'open'  # 标记为未平仓
            })
        
        # 计算统计指标
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'win_count': 0,
                'loss_count': 0,
                'trades': []
            }
        
        returns = [t['return_pct'] for t in trades]
        win_count = sum(1 for r in returns if r > 0)
        stop_loss_count = sum(1 for t in trades if t.get('sell_reason') == '止损')
        
        # 计算最大回撤
        cumulative_returns = []
        cum_return = 0
        for r in returns:
            cum_return = (1 + cum_return/100) * (1 + r/100) * 100 - 100
            cumulative_returns.append(cum_return)
        
        max_drawdown = 0
        peak = cumulative_returns[0]
        for cum_ret in cumulative_returns:
            if cum_ret > peak:
                peak = cum_ret
            drawdown = (peak - cum_ret)
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'total_trades': len(trades),
            'win_rate': round(win_count / len(trades) * 100, 2) if trades else 0,
            'avg_return': round(sum(returns) / len(returns), 2) if trades else 0,
            'total_return': round(cumulative_returns[-1], 2) if cumulative_returns else 0,
            'max_drawdown': round(max_drawdown, 2),
            'win_count': win_count,
            'loss_count': len(trades) - win_count,
            'stop_loss_count': stop_loss_count,
            'trades': trades
        }
    
    def get_stock_info(self, code: str, trade_date: str = None) -> Dict:
        """获取股票信息"""
        df = self.get_stock_data(code, days=120)
        if df is None or df.empty:
            return {}
        
        if trade_date:
            df = df[df['date'] <= pd.Timestamp(trade_date)]
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 计算涨跌幅，确保不是NaN
        change_value = (latest['close'] - prev['close']) / prev['close'] * 100
        if pd.isna(change_value):
            change_value = 0.0
        
        info = {
            'code': code,
            'date': latest['date'].strftime('%Y-%m-%d') if pd.notna(latest['date']) else 'N/A',
            'close': float(latest['close']),
            'change': float(change_value),
            'volume': float(latest['volume']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'open': float(latest['open']),
            'ma5': float(latest['MA5']) if pd.notna(latest['MA5']) else None,
            'ma10': float(latest['MA10']) if pd.notna(latest['MA10']) else None,
            'ma20': float(latest['MA20']) if pd.notna(latest['MA20']) else None,
            'ma60': float(latest['MA60']) if pd.notna(latest['MA60']) else None,
        }
        
        return info

# 初始化选股器（加载全部股票，启动会慢一些）
# 如需快速启动，可设置 max_stocks=1000
stock_selector = StockSelector(DATA_DIR, CONFIG_FILE, max_stocks=None)

# ==================== 路由 ==================== #

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/test')
def test():
    """测试页面"""
    return render_template('test.html')

@app.route('/backtest')
def backtest_page():
    """回测页面"""
    return render_template('backtest.html')

@app.route('/debug')
def debug():
    """调试页面 - 直接返回HTML"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>调试页面</title>
</head>
<body style="padding:30px; font-family:Arial; background:#f0f0f0;">
    <h1 style="color:#333;">🔧 系统调试信息</h1>
    
    <div style="background:white; padding:20px; margin:10px 0; border-radius:8px;">
        <h2>✅ Flask服务正常</h2>
        <p>如果你能看到这个页面，说明Flask服务工作正常。</p>
    </div>
    
    <div style="background:white; padding:20px; margin:10px 0; border-radius:8px;">
        <h2>🧪 JavaScript测试</h2>
        <p id="js-test" style="color:red;">❌ JavaScript未执行</p>
    </div>
    
    <div style="background:white; padding:20px; margin:10px 0; border-radius:8px;">
        <h2>🌐 API测试</h2>
        <button onclick="testAPI()" style="padding:10px 20px; font-size:16px;">点击测试API</button>
        <pre id="api-result" style="background:#f5f5f5; padding:10px; margin-top:10px;"></pre>
    </div>
    
    <script>
        // JavaScript测试
        document.getElementById('js-test').innerHTML = '✅ JavaScript正常工作';
        document.getElementById('js-test').style.color = 'green';
        
        // API测试函数
        function testAPI() {
            document.getElementById('api-result').textContent = '测试中...';
            
            fetch('/api/latest_date')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('api-result').textContent = 
                        'API响应正常:\\n' + JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    document.getElementById('api-result').textContent = 
                        'API错误:\\n' + error.message;
                });
        }
        
        // 自动测试API
        setTimeout(testAPI, 500);
    </script>
</body>
</html>
    '''

@app.route('/api/selectors')
def get_selectors():
    """获取所有选择器列表"""
    selectors = []
    for alias, info in stock_selector.selectors.items():
        selectors.append({
            'name': alias,
            'class': info['class'],
            'params': info['params']
        })
    return jsonify(selectors)

@app.route('/api/select/<selector_name>')
def run_selector(selector_name: str):
    """运行选择器"""
    trade_date = request.args.get('date', None)
    
    try:
        picks = stock_selector.run_selector(selector_name, trade_date)
        
        # 获取每只股票的详细信息
        stocks = []
        for code in picks:
            info = stock_selector.get_stock_info(code, trade_date)
            if info:
                stocks.append(info)
        
        # 清理NaN值，避免JSON序列化错误
        def clean_nan(obj):
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, float):
                try:
                    import math
                    if math.isnan(obj):
                        return None
                except:
                    pass
                if pd.isna(obj):
                    return None
            return obj
        
        cleaned_stocks = clean_nan(stocks)
        
        return jsonify({
            'success': True,
            'selector': selector_name,
            'date': trade_date if trade_date else 'latest',
            'count': len(cleaned_stocks),
            'stocks': cleaned_stocks
        })
    except Exception as e:
        logger.error(f"运行选择器失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/stock/<code>')
def get_stock(code: str):
    """获取股票详细数据"""
    days = int(request.args.get('days', 120))
    strategy = request.args.get('strategy', 'default')  # 默认少妇战法，可选breakout出坑战法
    
    df = stock_selector.get_stock_data(code, days, strategy=strategy)
    if df is None or df.empty:
        return jsonify({
            'success': False,
            'error': '股票数据不存在'
        })
    
    # 调试：检查signal列
    signal_counts = df['signal'].value_counts().to_dict()
    print(f"DEBUG API signal列统计: {signal_counts}")
    
    # 使用统一的SignalIdentifier重新识别信号（获取完整的signal信息包含reason）
    if strategy == 'breakout':
        signals = SignalIdentifier.identify_breakout_signals(df)
    else:
        signals = SignalIdentifier.identify_shaofv_signals(df)
    
    # 转换信号日期格式为字符串
    for signal in signals:
        if isinstance(signal['date'], pd.Timestamp):
            signal['date'] = signal['date'].strftime('%Y-%m-%d')
    
    print(f"DEBUG API: 识别到{len(signals)}个信号")
    
    # 过滤掉非交易日数据（volume为0的），但保留有信号的行
    df_filtered = df[(df['volume'] > 0) | (df['signal'].isin(['B1', 'B2', 'S1', 'B', 'S']))].copy()
    
    print(f"DEBUG API: 信号数={len(signals)}, 过滤前={len(df)}, 过滤后={len(df_filtered)}")
    
    # 清理NaN值的辅助函数
    def clean_nan_value(val):
        if isinstance(val, float):
            try:
                import math
                if math.isnan(val):
                    return 0.0
            except:
                pass
            if pd.isna(val):
                return 0.0
        return val
    
    def clean_list(lst):
        return [clean_nan_value(v) for v in lst]
    
    # 转换为前端需要的格式
    data = {
        'success': True,
        'code': code,
        'dates': df_filtered['date'].dt.strftime('%Y-%m-%d').tolist(),
        'open': clean_list(df_filtered['open'].tolist()),
        'close': clean_list(df_filtered['close'].tolist()),
        'high': clean_list(df_filtered['high'].tolist()),
        'low': clean_list(df_filtered['low'].tolist()),
        'volume': clean_list(df_filtered['volume'].tolist()),
        'ma5': clean_list(df_filtered['MA5'].fillna(0).tolist()),
        'ma10': clean_list(df_filtered['MA10'].fillna(0).tolist()),
        'ma20': clean_list(df_filtered['MA20'].fillna(0).tolist()),
        'ma30': clean_list(df_filtered['MA30'].fillna(0).tolist()),
        'ma60': clean_list(df_filtered['MA60'].fillna(0).tolist()),
        # 趋势线（知行体系标准公式）
        'trend_short': clean_list(df_filtered['trend_short'].fillna(0).tolist()),
        'trend_long': clean_list(df_filtered['trend_long'].fillna(0).tolist()),
        'trend_diff_pct': clean_list(df_filtered['trend_diff_pct'].fillna(0).tolist()),
        'trend_short_above': df_filtered['trend_short_above'].fillna(False).tolist(),
        # MACD
        'dif': clean_list(df_filtered['DIF'].fillna(0).tolist()),
        'dea': clean_list(df_filtered['DEA'].fillna(0).tolist()),
        'macd': clean_list(df_filtered['MACD'].fillna(0).tolist()),
        # KDJ
        'k': clean_list(df_filtered['K'].fillna(0).tolist()),
        'd': clean_list(df_filtered['D'].fillna(0).tolist()),
        'j': clean_list(df_filtered['J'].fillna(0).tolist()),
        # 交易信号
        'signals': signals,
        # 回测结果
        'backtest': stock_selector._backtest_signals(df_filtered, signals)
    }
    
    return jsonify(data)

@app.route('/api/compare')
def compare_selectors():
    """对比所有选择器结果"""
    trade_date = request.args.get('date', None)
    
    results = []
    for alias in stock_selector.selectors.keys():
        picks = stock_selector.run_selector(alias, trade_date)
        results.append({
            'name': alias,
            'count': len(picks),
            'codes': picks
        })
    
    return jsonify({
        'success': True,
        'date': trade_date if trade_date else 'latest',
        'results': results
    })

@app.route('/api/stock_info/<code>')
def get_stock_info(code: str):
    """获取股票基本信息"""
    trade_date = request.args.get('date', None)
    info = stock_selector.get_stock_info(code, trade_date)
    
    if not info:
        return jsonify({
            'success': False,
            'error': '股票不存在'
        })
    
    return jsonify({
        'success': True,
        'info': info
    })

@app.route('/api/backtest')
def api_backtest():
    """回测API"""
    code = request.args.get('code')
    days = int(request.args.get('days', 365))
    strategy = request.args.get('strategy', 'default')
    stop_loss = float(request.args.get('stop_loss', 0.05))
    
    if not code:
        return jsonify({
            'success': False,
            'error': '缺少股票代码参数'
        })
    
    df = stock_selector.get_stock_data(code, days, strategy=strategy)
    if df is None or df.empty:
        return jsonify({
            'success': False,
            'error': '股票数据不存在'
        })
    
    # 提取信号
    signals = []
    for idx, row in df.iterrows():
        if row['signal'] in ['B1', 'B2', 'S1', 'B', 'S']:
            if pd.notna(row['date']):
                signals.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'type': row['signal'],
                    'price': float(row['signal_price'])
                })
    
    # 回测
    backtest_result = stock_selector._backtest_signals(df, signals, stop_loss_pct=stop_loss)
    
    return jsonify({
        'success': True,
        'code': code,
        'strategy': strategy,
        'backtest': backtest_result
    })

@app.route('/api/latest_date')
def get_latest_date():
    """获取最新交易日"""
    all_dates = set()
    for df in stock_selector.data.values():
        all_dates.update(df['date'])
    latest = max(all_dates) if all_dates else None
    
    return jsonify({
        'success': True,
        'date': latest.strftime('%Y-%m-%d') if latest else None
    })

# ==================== 启动应用 ==================== #

def main():
    """启动Web服务"""
    import argparse
    
    parser = argparse.ArgumentParser(description="量化选股Web界面")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()
    
    logger.info(f"启动Web服务: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()

