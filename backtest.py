#!/usr/bin/env python3
"""
量化策略回测框架

功能：
1. 支持所有选择器的回测
2. 多种止损止盈策略
3. 详细的收益统计分析
4. 可视化结果展示
5. 交易记录导出
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import logging

from select_stock import load_data, load_config, instantiate_selector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("backtest")


class Backtest:
    """
    回测框架
    
    使用方法：
    >>> bt = Backtest(
    >>>     data_dir="./data",
    >>>     start_date="2024-01-01",
    >>>     end_date="2024-12-31"
    >>> )
    >>> results = bt.run_strategy("BBIKDJSelector", params={...})
    >>> bt.print_stats(results)
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = 100000.0,  # 初始资金
        commission_rate: float = 0.0003,     # 佣金率（万3）
        slippage_rate: float = 0.001,        # 滑点率（0.1%）
    ):
        self.data_dir = Path(data_dir)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # 加载数据
        logger.info(f"加载数据：{self.data_dir}")
        codes = [f.stem for f in self.data_dir.glob("*.csv")]
        self.data = load_data(self.data_dir, codes)
        logger.info(f"成功加载 {len(self.data)} 只股票")
        
        # 获取所有交易日
        self.trade_dates = self._get_trade_dates()
        logger.info(f"回测期间：{self.start_date.date()} 至 {self.end_date.date()}")
        logger.info(f"交易日数量：{len(self.trade_dates)}\n")
    
    def _get_trade_dates(self) -> List[pd.Timestamp]:
        """获取回测期间的所有交易日"""
        all_dates = set()
        for df in self.data.values():
            dates = df[(df['date'] >= self.start_date) & 
                      (df['date'] <= self.end_date)]['date']
            all_dates.update(dates)
        return sorted(list(all_dates))
    
    def _get_price(
        self, 
        code: str, 
        date: pd.Timestamp, 
        price_type: str = 'close'
    ) -> Optional[float]:
        """获取指定日期的价格"""
        if code not in self.data:
            return None
        
        df = self.data[code]
        row = df[df['date'] == date]
        
        if row.empty:
            return None
        
        return float(row.iloc[0][price_type])
    
    def _calculate_commission(self, amount: float) -> float:
        """计算交易佣金"""
        commission = amount * self.commission_rate
        return max(commission, 5.0)  # 最低5元
    
    def run_strategy(
        self,
        selector_class: str,
        params: Dict = None,
        hold_days: int = 5,
        stop_loss_pct: float = -0.05,      # 止损-5%
        take_profit_pct: float = 0.15,      # 止盈+15%
        position_per_stock: float = 0.1,    # 单只股票仓位10%
        max_positions: int = 5,             # 最多持仓5只
        stop_loss_method: str = "price",    # 止损方式：price/ma/none
    ) -> pd.DataFrame:
        """
        运行策略回测
        
        Args:
            selector_class: 选择器类名（如 "BBIKDJSelector"）
            params: 选择器参数
            hold_days: 持有天数
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            position_per_stock: 单只股票仓位比例
            max_positions: 最大持仓数量
            stop_loss_method: 止损方式
                - "price": 固定比例止损
                - "ma": 跌破MA5止损
                - "none": 不止损
        
        Returns:
            DataFrame: 交易记录
        """
        logger.info(f"{'='*60}")
        logger.info(f"回测策略：{selector_class}")
        logger.info(f"{'='*60}\n")
        
        # 实例化选择器
        import importlib
        module = importlib.import_module("Selector")
        cls = getattr(module, selector_class)
        selector = cls(**params) if params else cls()
        
        # 初始化账户
        capital = self.initial_capital  # 当前资金
        positions = {}  # 持仓：{code: {'shares': 数量, 'buy_price': 买入价, 'buy_date': 买入日}}
        trades = []  # 交易记录
        
        # 遍历每个交易日
        for i, date in enumerate(self.trade_dates):
            # 显示进度
            if i % 50 == 0:
                logger.info(f"回测进度：{i}/{len(self.trade_dates)} ({i/len(self.trade_dates)*100:.1f}%)")
            
            # ============ 卖出逻辑 ============
            to_sell = []
            for code, pos in list(positions.items()):
                current_price = self._get_price(code, date, 'close')
                if current_price is None:
                    continue
                
                buy_price = pos['buy_price']
                buy_date = pos['buy_date']
                shares = pos['shares']
                hold_period = (date - buy_date).days
                
                # 计算当前收益率
                profit_pct = (current_price - buy_price) / buy_price
                
                sell_reason = None
                
                # 止损检查
                if stop_loss_method == "price" and profit_pct <= stop_loss_pct:
                    sell_reason = f"止损({profit_pct*100:.2f}%)"
                
                elif stop_loss_method == "ma":
                    # 跌破MA5止损
                    df = self.data[code]
                    hist = df[df['date'] <= date].tail(10)
                    if len(hist) >= 5:
                        ma5 = hist['close'].rolling(5).mean().iloc[-1]
                        if current_price < ma5:
                            sell_reason = f"跌破MA5({profit_pct*100:.2f}%)"
                
                # 止盈检查
                elif profit_pct >= take_profit_pct:
                    sell_reason = f"止盈({profit_pct*100:.2f}%)"
                
                # 时间止损（持有超过指定天数）
                elif hold_period >= hold_days:
                    sell_reason = f"到期({hold_period}天,{profit_pct*100:.2f}%)"
                
                # 执行卖出
                if sell_reason:
                    # 考虑滑点
                    sell_price = current_price * (1 - self.slippage_rate)
                    sell_amount = shares * sell_price
                    commission = self._calculate_commission(sell_amount)
                    net_amount = sell_amount - commission
                    
                    # 更新资金
                    capital += net_amount
                    
                    # 记录交易
                    trades.append({
                        'code': code,
                        'buy_date': buy_date,
                        'sell_date': date,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'shares': shares,
                        'hold_days': hold_period,
                        'profit_pct': profit_pct * 100,
                        'profit_amount': (sell_price - buy_price) * shares - commission - pos.get('buy_commission', 0),
                        'sell_reason': sell_reason,
                        'commission': commission + pos.get('buy_commission', 0)
                    })
                    
                    to_sell.append(code)
            
            # 清理已卖出的持仓
            for code in to_sell:
                del positions[code]
            
            # ============ 买入逻辑 ============
            # 如果持仓未满，执行选股
            if len(positions) < max_positions:
                picks = selector.select(date, self.data)
                
                # 排除已持仓的股票
                picks = [p for p in picks if p not in positions]
                
                # 计算可买入数量
                can_buy = max_positions - len(positions)
                picks = picks[:can_buy]
                
                # 执行买入
                for code in picks:
                    buy_price = self._get_price(code, date, 'close')
                    if buy_price is None or buy_price <= 0:
                        continue
                    
                    # 考虑滑点
                    actual_buy_price = buy_price * (1 + self.slippage_rate)
                    
                    # 计算买入金额（单只股票仓位）
                    buy_amount = capital * position_per_stock
                    commission = self._calculate_commission(buy_amount)
                    
                    # 检查资金是否充足
                    if capital < buy_amount + commission:
                        continue
                    
                    # 计算股数（取整100股）
                    shares = int((buy_amount - commission) / actual_buy_price / 100) * 100
                    if shares <= 0:
                        continue
                    
                    # 实际花费
                    actual_cost = shares * actual_buy_price + commission
                    
                    # 更新资金和持仓
                    capital -= actual_cost
                    positions[code] = {
                        'shares': shares,
                        'buy_price': actual_buy_price,
                        'buy_date': date,
                        'buy_commission': commission
                    }
        
        # 回测结束，清仓所有持仓
        logger.info(f"\n回测结束，清仓所有持仓...")
        last_date = self.trade_dates[-1]
        for code, pos in positions.items():
            sell_price = self._get_price(code, last_date, 'close')
            if sell_price is None:
                continue
            
            sell_price = sell_price * (1 - self.slippage_rate)
            shares = pos['shares']
            buy_price = pos['buy_price']
            buy_date = pos['buy_date']
            
            sell_amount = shares * sell_price
            commission = self._calculate_commission(sell_amount)
            net_amount = sell_amount - commission
            capital += net_amount
            
            profit_pct = (sell_price - buy_price) / buy_price
            
            trades.append({
                'code': code,
                'buy_date': buy_date,
                'sell_date': last_date,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'shares': shares,
                'hold_days': (last_date - buy_date).days,
                'profit_pct': profit_pct * 100,
                'profit_amount': (sell_price - buy_price) * shares - commission - pos.get('buy_commission', 0),
                'sell_reason': '回测结束',
                'commission': commission + pos.get('buy_commission', 0)
            })
        
        # 转换为DataFrame
        df_trades = pd.DataFrame(trades)
        
        # 添加最终资金
        logger.info(f"\n最终资金：{capital:.2f}")
        logger.info(f"总收益：{capital - self.initial_capital:.2f}")
        logger.info(f"总收益率：{(capital / self.initial_capital - 1) * 100:.2f}%\n")
        
        return df_trades
    
    def print_stats(self, trades: pd.DataFrame):
        """打印统计信息"""
        if trades.empty:
            logger.warning("没有交易记录")
            return
        
        print(f"\n{'='*60}")
        print(f"回测统计报告")
        print(f"{'='*60}\n")
        
        # 基础统计
        print(f"【基础信息】")
        print(f"交易次数：{len(trades)}")
        print(f"盈利次数：{(trades['profit_pct'] > 0).sum()}")
        print(f"亏损次数：{(trades['profit_pct'] <= 0).sum()}")
        print(f"胜率：{(trades['profit_pct'] > 0).sum() / len(trades) * 100:.2f}%\n")
        
        # 收益统计
        print(f"【收益统计】")
        print(f"平均收益率：{trades['profit_pct'].mean():.2f}%")
        print(f"中位数收益率：{trades['profit_pct'].median():.2f}%")
        print(f"最大单笔收益：{trades['profit_pct'].max():.2f}%")
        print(f"最大单笔亏损：{trades['profit_pct'].min():.2f}%")
        print(f"总盈利金额：{trades[trades['profit_amount'] > 0]['profit_amount'].sum():.2f}")
        print(f"总亏损金额：{trades[trades['profit_amount'] <= 0]['profit_amount'].sum():.2f}")
        print(f"净利润：{trades['profit_amount'].sum():.2f}\n")
        
        # 持仓统计
        print(f"【持仓统计】")
        print(f"平均持仓天数：{trades['hold_days'].mean():.1f}天")
        print(f"最长持仓：{trades['hold_days'].max()}天")
        print(f"最短持仓：{trades['hold_days'].min()}天\n")
        
        # 费用统计
        print(f"【费用统计】")
        print(f"总交易佣金：{trades['commission'].sum():.2f}\n")
        
        # 卖出原因统计
        print(f"【卖出原因】")
        sell_reasons = trades['sell_reason'].value_counts()
        for reason, count in sell_reasons.items():
            print(f"  {reason}: {count}次 ({count/len(trades)*100:.1f}%)")
        
        print(f"\n{'='*60}\n")
    
    def plot_equity_curve(self, trades: pd.DataFrame, save_path: str = None):
        """
        绘制资金曲线
        
        需要安装：pip install matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib import rcParams
            
            # 设置中文字体
            rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            rcParams['axes.unicode_minus'] = False
        except ImportError:
            logger.error("请先安装 matplotlib: pip install matplotlib")
            return
        
        if trades.empty:
            logger.warning("没有交易数据，无法绘图")
            return
        
        # 计算累计收益
        trades = trades.sort_values('sell_date')
        trades['cumulative_profit'] = trades['profit_amount'].cumsum()
        trades['cumulative_return'] = (trades['cumulative_profit'] + self.initial_capital) / self.initial_capital - 1
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 图1：资金曲线
        ax1.plot(trades['sell_date'], 
                trades['cumulative_profit'] + self.initial_capital,
                label='账户资金', linewidth=2, color='#2E86AB')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                   label='初始资金', linewidth=1)
        ax1.set_title('资金曲线', fontsize=14, fontweight='bold')
        ax1.set_xlabel('日期', fontsize=12)
        ax1.set_ylabel('资金（元）', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 图2：累计收益率
        ax2.plot(trades['sell_date'], 
                trades['cumulative_return'] * 100,
                label='累计收益率', linewidth=2, color='#A23B72')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax2.fill_between(trades['sell_date'], 
                         trades['cumulative_return'] * 100, 
                         0, alpha=0.3)
        ax2.set_title('累计收益率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('收益率 (%)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存：{save_path}")
        else:
            plt.show()
    
    def export_trades(self, trades: pd.DataFrame, filename: str = "backtest_trades.csv"):
        """导出交易记录"""
        trades.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"交易记录已导出：{filename}")
    
    def compare_strategies(
        self,
        strategies: List[Dict],
        hold_days: int = 5
    ) -> pd.DataFrame:
        """
        对比多个策略
        
        Args:
            strategies: 策略列表，格式：
                [
                    {"name": "少妇战法", "class": "BBIKDJSelector", "params": {...}},
                    {"name": "出坑战法", "class": "BreakoutPreviousHighSelector", "params": {...}}
                ]
            hold_days: 持有天数
        
        Returns:
            DataFrame: 对比结果
        """
        results = []
        
        for strategy in strategies:
            name = strategy.get("name", strategy["class"])
            logger.info(f"\n开始回测：{name}")
            
            trades = self.run_strategy(
                selector_class=strategy["class"],
                params=strategy.get("params", {}),
                hold_days=hold_days
            )
            
            if not trades.empty:
                win_rate = (trades['profit_pct'] > 0).sum() / len(trades) * 100
                avg_return = trades['profit_pct'].mean()
                total_profit = trades['profit_amount'].sum()
                total_return = total_profit / self.initial_capital * 100
                
                results.append({
                    '策略': name,
                    '交易次数': len(trades),
                    '胜率(%)': f"{win_rate:.2f}",
                    '平均收益率(%)': f"{avg_return:.2f}",
                    '总收益率(%)': f"{total_return:.2f}",
                    '净利润': f"{total_profit:.2f}"
                })
            else:
                results.append({
                    '策略': name,
                    '交易次数': 0,
                    '胜率(%)': 0,
                    '平均收益率(%)': 0,
                    '总收益率(%)': 0,
                    '净利润': 0
                })
        
        df_results = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print(f"策略对比结果")
        print(f"{'='*80}\n")
        print(df_results.to_string(index=False))
        print(f"\n{'='*80}\n")
        
        return df_results


def main():
    """示例：回测出坑战法"""
    
    # 创建回测实例
    bt = Backtest(
        data_dir="./data",
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
        commission_rate=0.0003,
        slippage_rate=0.001
    )
    
    # 回测出坑战法
    trades = bt.run_strategy(
        selector_class="BreakoutPreviousHighSelector",
        params={
            "lookback_days": 60,
            "consolidation_min_days": 5,
            "consolidation_max_days": 30,
            "approach_pct": 0.15,
            "vol_ratio_threshold": 2.0,
            "turnover_threshold": 0.05,
            "consolidation_shrink_ratio": 0.6,
            "pullback_pct_min": 0.05,
            "pullback_pct_max": 0.25,
            "ma_converge_threshold": 0.05,
            "max_window": 120
        },
        hold_days=5,
        stop_loss_pct=-0.05,
        take_profit_pct=0.15,
        position_per_stock=0.1,
        max_positions=5,
        stop_loss_method="price"
    )
    
    # 打印统计信息
    bt.print_stats(trades)
    
    # 导出交易记录
    bt.export_trades(trades, "出坑战法_回测记录.csv")
    
    # 绘制资金曲线
    # bt.plot_equity_curve(trades, "出坑战法_资金曲线.png")


def compare_all_strategies():
    """对比所有策略"""
    
    bt = Backtest(
        data_dir="./data",
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0
    )
    
    strategies = [
        {
            "name": "少妇战法",
            "class": "BBIKDJSelector",
            "params": {
                "j_threshold": 15,
                "bbi_min_window": 20,
                "max_window": 120,
                "price_range_pct": 1,
                "bbi_q_threshold": 0.2,
                "j_q_threshold": 0.10
            }
        },
        {
            "name": "出坑战法",
            "class": "BreakoutPreviousHighSelector",
            "params": {
                "lookback_days": 60,
                "consolidation_min_days": 5,
                "consolidation_max_days": 30,
                "approach_pct": 0.15,
                "vol_ratio_threshold": 2.0,
                "turnover_threshold": 0.05,
                "consolidation_shrink_ratio": 0.6,
                "pullback_pct_min": 0.05,
                "pullback_pct_max": 0.25,
                "ma_converge_threshold": 0.05,
                "max_window": 120
            }
        },
        # 可以添加更多策略
    ]
    
    results = bt.compare_strategies(strategies, hold_days=5)
    results.to_csv("策略对比结果.csv", index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_all_strategies()
    else:
        main()


