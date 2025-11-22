"""
统一的交易信号识别模块

本模块提供统一的信号识别逻辑，供选股器和Web系统共同使用。
确保"选股结果 = 有买入信号的股票"

作者: AI Assistant
日期: 2025-11-22
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta


class SignalIdentifier:
    """统一的信号识别器"""
    
    @staticmethod
    def identify_breakout_signals(df: pd.DataFrame, max_days_lookback: int = 120) -> List[Dict[str, Any]]:
        """
        出坑战法信号识别
        
        B信号（买入）:
        - 价格接近前高（75%-105%区间）
        - 放量（≥5日均量的1.2倍）
        - 站上或接近MA10（≥MA10*0.9）
        - 收阳线或接近前高
        - 5天内无重复B信号（去重）
        
        S信号（卖出）:
        - 高位大阴柱（跌幅≥3%，放量≥1.5倍）
        - 或连续阶梯阴量累计-7%
        
        Args:
            df: K线数据DataFrame，必须包含date, open, close, high, low, volume, ma10列
            max_days_lookback: 计算前高的回看天数
            
        Returns:
            信号列表，每个信号包含: {type, date, price, volume, reason}
        """
        if df.empty or len(df) < 60:
            return []
        
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 添加必要的技术指标
        df = SignalIdentifier._add_indicators(df)
        
        signals = []
        
        # 计算60日前高
        lookback = min(max_days_lookback, len(df))
        
        for i in range(60, len(df)):
            # 计算前高（排除当前5天，避免自己和自己比）
            prev_data = df.iloc[max(0, i-60):max(0, i-5)]
            if prev_data.empty:
                continue
            
            prev_high = prev_data['high'].max()
            
            current = df.iloc[i]
            current_date = current['date']
            current_close = current['close']
            current_open = current['open']
            current_vol = current['volume']
            current_high = current['high']
            current_low = current['low']
            
            # 计算5日均量
            vol_ma5 = df.iloc[max(0, i-5):i]['volume'].mean() if i >= 5 else current_vol
            vol_ratio = current_vol / vol_ma5 if vol_ma5 > 0 else 0
            
            # 计算MA10
            ma10 = current.get('ma10', current_close)
            
            # ========== B信号识别 ==========
            # 1. 计算距离前高的百分比
            distance_to_high = (prev_high - current_close) / prev_high
            
            # 2. B信号条件（优化后）
            is_near_high = -0.05 <= distance_to_high <= 0.25  # 前高的75%-105%
            is_volume_up = vol_ratio >= 1.2  # 放量
            is_above_ma10 = current_close >= ma10 * 0.9  # 站上或接近MA10
            is_yang_or_near_high = (current_close > current_open) or (distance_to_high <= 0.05)
            
            # 3. 增强条件：整理期缩量确认（经典出坑特征）
            # 检查最近10-30天是否有缩量整理期
            consolidation_confirmed = False
            if i >= 30:
                # 计算前高之后的整理期平均成交量
                consolidation_period = df.iloc[max(0, i-30):max(0, i-5)]
                if len(consolidation_period) >= 10:
                    consol_avg_vol = consolidation_period['volume'].mean()
                    # 如果整理期平均成交量 < 当前放量的60%，说明有明显缩量整理
                    if consol_avg_vol < current_vol * 0.6:
                        consolidation_confirmed = True
            
            # 4. 均线系统确认（可选，不作为必要条件）
            ma20 = current.get('ma20', current_close)
            ma_system_good = current_close > ma20 * 0.95  # 站上或接近MA20
            
            # 5. MACD确认（可选，优先级加分项）
            macd_positive = False
            if 'MACD' in df.columns:
                macd_current = current.get('MACD', 0)
                macd_positive = macd_current > 0  # MACD红柱
            
            # 基础条件必须满足
            basic_conditions = is_near_high and is_volume_up and is_above_ma10 and is_yang_or_near_high
            
            # 增强条件：有缩量整理更好（但不强制）
            enhanced_score = 0
            if consolidation_confirmed:
                enhanced_score += 1
            if ma_system_good:
                enhanced_score += 1
            if macd_positive:
                enhanced_score += 1
            
            # 满足基础条件即可，增强条件用于reason说明
            if basic_conditions:
                # 检查最近5天内是否已有B信号（去重）
                recent_b = [s for s in signals 
                           if s['type'] == 'B' and 
                           (current_date - s['date']).days <= 5]
                
                if not recent_b:
                    # 构建详细的reason
                    reason_parts = [f'接近前高{prev_high:.2f}(距离{distance_to_high*100:.1f}%)']
                    reason_parts.append(f'放量{vol_ratio:.1f}倍')
                    
                    if consolidation_confirmed:
                        reason_parts.append('前期缩量整理')
                    if ma_system_good and current_close > ma20:
                        reason_parts.append('突破MA20')
                    if macd_positive:
                        reason_parts.append('MACD红柱')
                    
                    # 信号质量评分（0-3分）
                    quality = 'strong' if enhanced_score >= 2 else 'normal'
                    
                    signals.append({
                        'type': 'B',
                        'date': current_date,
                        'price': float(current_close),
                        'volume': float(current_vol),
                        'reason': ', '.join(reason_parts),
                        'prev_high': float(prev_high),
                        'vol_ratio': float(vol_ratio),
                        'quality': quality,  # 信号质量
                        'score': enhanced_score  # 增强特征评分
                    })
            
            # ========== S信号识别 ==========
            # 方式1：高位大阴柱
            drop_pct = (current_open - current_close) / current_open if current_open > 0 else 0
            is_big_bearish = drop_pct >= 0.03  # 跌幅≥3%
            is_high_volume = vol_ratio >= 1.5
            
            # 判断是否在高位（接近60日最高价）
            high_60 = df.iloc[max(0, i-60):i+1]['high'].max()
            is_near_peak = current_high >= high_60 * 0.95
            
            if is_big_bearish and is_high_volume and is_near_peak:
                signals.append({
                    'type': 'S',
                    'date': current_date,
                    'price': float(current_close),
                    'volume': float(current_vol),
                    'reason': f'高位大阴柱(跌{drop_pct*100:.1f}%, 放量{vol_ratio:.1f}倍)',
                    'drop_pct': float(drop_pct),
                    'vol_ratio': float(vol_ratio)
                })
            
            # 方式2：连续阶梯阴量累计-7%
            if i >= 5:
                recent_5 = df.iloc[i-4:i+1]
                cumulative_drop = 0
                consecutive_drops = 0
                
                for j in range(len(recent_5)):
                    day = recent_5.iloc[j]
                    day_drop = (day['open'] - day['close']) / day['open'] if day['open'] > 0 else 0
                    
                    if day_drop > 0:  # 阴线
                        cumulative_drop += day_drop
                        consecutive_drops += 1
                    else:
                        # 重置（必须连续）
                        cumulative_drop = 0
                        consecutive_drops = 0
                
                if consecutive_drops >= 3 and cumulative_drop >= 0.07:
                    # 检查是否已有相同日期的S信号
                    same_date_s = [s for s in signals 
                                  if s['type'] == 'S' and s['date'] == current_date]
                    
                    if not same_date_s:
                        signals.append({
                            'type': 'S',
                            'date': current_date,
                            'price': float(current_close),
                            'volume': float(current_vol),
                            'reason': f'连续{consecutive_drops}天阴线累计-{cumulative_drop*100:.1f}%',
                            'cumulative_drop': float(cumulative_drop),
                            'consecutive_days': int(consecutive_drops)
                        })
        
        return signals
    
    @staticmethod
    def identify_shaofv_signals(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        少妇战法信号识别
        
        B1信号（首次买入）:
        - KDJ超卖后首次转强（J<20且开始上升）
        - 收阳线
        - 放量（≥前日1.2倍）
        
        B2信号（加仓买入）:
        - B1后次日继续上涨
        - 持续放量
        
        S1信号（卖出）:
        - 放量大阴柱（跌幅≥3%，量比≥1.5倍）
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            信号列表
        """
        if df.empty or len(df) < 30:
            return []
        
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 添加KDJ指标
        df = SignalIdentifier._add_kdj(df)
        
        signals = []
        
        for i in range(14, len(df)):  # KDJ需要至少9天数据
            current = df.iloc[i]
            prev = df.iloc[i-1] if i > 0 else current
            
            current_date = current['date']
            current_close = current['close']
            current_open = current['open']
            current_vol = current['volume']
            
            # KDJ值
            j_current = current.get('J', 50)
            j_prev = prev.get('J', 50)
            k_current = current.get('K', 50)
            d_current = current.get('D', 50)
            
            # ========== B1信号识别 ==========
            # 1. KDJ超卖后转强
            is_oversold_recovery = (j_prev < 20) and (j_current > j_prev) and (k_current > d_current)
            
            # 2. 收阳线
            is_yang = current_close > current_open
            
            # 3. 放量
            vol_ratio = current_vol / prev['volume'] if prev['volume'] > 0 else 1
            is_volume_up = vol_ratio >= 1.2
            
            if is_oversold_recovery and is_yang and is_volume_up:
                signals.append({
                    'type': 'B1',
                    'date': current_date,
                    'price': float(current_close),
                    'volume': float(current_vol),
                    'reason': f'KDJ超卖转强(J:{j_current:.1f}↑), 收阳放量{vol_ratio:.1f}倍',
                    'kdj_j': float(j_current),
                    'vol_ratio': float(vol_ratio)
                })
            
            # ========== B2信号识别 ==========
            # B1后次日继续上涨
            recent_b1 = [s for s in signals 
                        if s['type'] == 'B1' and 
                        (current_date - s['date']).days == 1]
            
            if recent_b1:
                is_continue_rise = current_close > prev['close']
                is_continue_vol = vol_ratio >= 1.0
                
                if is_continue_rise and is_continue_vol:
                    signals.append({
                        'type': 'B2',
                        'date': current_date,
                        'price': float(current_close),
                        'volume': float(current_vol),
                        'reason': f'B1后继续上涨(涨{(current_close/prev["close"]-1)*100:.1f}%)',
                        'rise_pct': float((current_close / prev['close'] - 1) * 100)
                    })
            
            # ========== S1信号识别 ==========
            drop_pct = (current_open - current_close) / current_open if current_open > 0 else 0
            is_big_bearish = drop_pct >= 0.03
            
            # 计算5日均量
            vol_ma5 = df.iloc[max(0, i-5):i]['volume'].mean() if i >= 5 else current_vol
            vol_ratio_ma5 = current_vol / vol_ma5 if vol_ma5 > 0 else 0
            is_high_volume = vol_ratio_ma5 >= 1.5
            
            if is_big_bearish and is_high_volume:
                signals.append({
                    'type': 'S1',
                    'date': current_date,
                    'price': float(current_close),
                    'volume': float(current_vol),
                    'reason': f'放量大阴柱(跌{drop_pct*100:.1f}%, 放量{vol_ratio_ma5:.1f}倍)',
                    'drop_pct': float(drop_pct),
                    'vol_ratio': float(vol_ratio_ma5)
                })
        
        return signals
    
    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """添加基础技术指标"""
        df = df.copy()
        
        # MA均线
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        if 'ma20' not in df.columns:
            df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        if 'ma60' not in df.columns:
            df['ma60'] = df['close'].rolling(window=60, min_periods=1).mean()
        
        return df
    
    @staticmethod
    def _add_kdj(df: pd.DataFrame) -> pd.DataFrame:
        """添加KDJ指标"""
        df = df.copy()
        
        # 计算RSV
        low_9 = df['low'].rolling(window=9, min_periods=1).min()
        high_9 = df['high'].rolling(window=9, min_periods=1).max()
        
        df['RSV'] = ((df['close'] - low_9) / (high_9 - low_9) * 100).fillna(0)
        
        # 计算K, D, J
        df['K'] = 50.0
        df['D'] = 50.0
        
        for i in range(1, len(df)):
            df.loc[i, 'K'] = df.loc[i-1, 'K'] * 2/3 + df.loc[i, 'RSV'] * 1/3
            df.loc[i, 'D'] = df.loc[i-1, 'D'] * 2/3 + df.loc[i, 'K'] * 1/3
        
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        return df
    
    @staticmethod
    def get_latest_signal_date(signals: List[Dict[str, Any]], signal_type: str = 'B') -> Any:
        """
        获取最新信号日期
        
        Args:
            signals: 信号列表
            signal_type: 信号类型（B/S/B1/B2/S1）
            
        Returns:
            最新信号的日期，如果没有返回None
        """
        filtered = [s for s in signals if s['type'] == signal_type]
        if not filtered:
            return None
        
        return max(filtered, key=lambda x: x['date'])['date']
    
    @staticmethod
    def has_recent_signal(signals: List[Dict[str, Any]], 
                         current_date: Any, 
                         signal_type: str = 'B',
                         max_days: int = 2) -> bool:
        """
        检查是否有最近的信号
        
        Args:
            signals: 信号列表
            current_date: 当前日期
            signal_type: 信号类型
            max_days: 最多几天前（含当天）
            
        Returns:
            True if 有最近max_days天内的信号
        """
        latest_date = SignalIdentifier.get_latest_signal_date(signals, signal_type)
        
        if latest_date is None:
            return False
        
        # 计算天数差
        if isinstance(current_date, pd.Timestamp):
            current_date = current_date.to_pydatetime()
        if isinstance(latest_date, pd.Timestamp):
            latest_date = latest_date.to_pydatetime()
        
        if isinstance(current_date, datetime) and isinstance(latest_date, datetime):
            days_diff = (current_date - latest_date).days
        else:
            # 如果是字符串，转换为datetime
            if isinstance(current_date, str):
                current_date = datetime.strptime(current_date, '%Y-%m-%d')
            if isinstance(latest_date, str):
                latest_date = datetime.strptime(latest_date, '%Y-%m-%d')
            days_diff = (current_date - latest_date).days
        
        return days_diff <= max_days

