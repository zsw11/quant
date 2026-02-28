"""
回测结果 - 存储交易记录并计算绩效指标

指标：
  total_return   : 总收益率
  annual_return  : 年化收益率
  max_drawdown   : 最大回撤
  sharpe_ratio   : 夏普比率（年化，无风险利率3%）
  calmar_ratio   : 卡玛比率 = 年化收益 / 最大回撤
  win_rate       : 胜率（盈利卖出次数 / 总卖出次数）
  total_trades   : 总交易次数
  total_commission: 累计手续费
"""
from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np


@dataclass
class TradeRecord:
    """单笔交易记录"""
    date:       pd.Timestamp
    symbol:     str
    action:     str    # "BUY" | "SELL"
    quantity:   float
    price:      float
    amount:     float  # 成交金额
    commission: float  # 手续费+印花税
    pnl:        float = 0.0   # 本笔盈亏（仅卖出有效）
    reason:     str   = ""


@dataclass
class BacktestResult:
    """完整的回测结果"""
    strategy_name:   str
    start_date:      str
    end_date:        str
    initial_capital: float

    # 每日净值曲线（index=日期，值=总资产）
    equity_curve: pd.Series = field(default_factory=pd.Series)

    # 所有成交记录
    trades: List[TradeRecord] = field(default_factory=list)

    # ----------------------------------------------------------------
    # 绩效指标（属性计算，不存储）
    # ----------------------------------------------------------------

    @property
    def final_value(self) -> float:
        if len(self.equity_curve) == 0:
            return self.initial_capital
        return float(self.equity_curve.iloc[-1])

    @property
    def total_return(self) -> float:
        """总收益率"""
        return (self.final_value - self.initial_capital) / self.initial_capital

    @property
    def annual_return(self) -> float:
        """年化收益率"""
        if len(self.equity_curve) < 2:
            return 0.0
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days == 0:
            return 0.0
        return (1 + self.total_return) ** (365 / days) - 1

    @property
    def max_drawdown(self) -> float:
        """最大回撤（负数）"""
        if len(self.equity_curve) == 0:
            return 0.0
        roll_max = self.equity_curve.cummax()
        dd = (self.equity_curve - roll_max) / roll_max
        return float(dd.min())

    @property
    def sharpe_ratio(self) -> float:
        """年化夏普比率（无风险利率 3%）"""
        if len(self.equity_curve) < 2:
            return 0.0
        ret = self.equity_curve.pct_change().dropna()
        if ret.std() == 0:
            return 0.0
        rf_daily = 0.03 / 252
        excess   = ret - rf_daily
        return float(excess.mean() / excess.std() * np.sqrt(252))

    @property
    def calmar_ratio(self) -> float:
        """卡玛比率"""
        if self.max_drawdown == 0:
            return 0.0
        return self.annual_return / abs(self.max_drawdown)

    @property
    def volatility(self) -> float:
        """年化波动率"""
        if len(self.equity_curve) < 2:
            return 0.0
        return float(self.equity_curve.pct_change().dropna().std() * np.sqrt(252))

    @property
    def win_rate(self) -> float:
        """胜率（卖出盈利次数 / 总卖出次数）"""
        sells = [t for t in self.trades if t.action == "SELL"]
        if not sells:
            return 0.0
        wins = sum(1 for t in sells if t.pnl > 0)
        return wins / len(sells)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def total_commission(self) -> float:
        return sum(t.commission for t in self.trades)

    @property
    def total_pnl(self) -> float:
        """所有卖出交易的累计盈亏"""
        return sum(t.pnl for t in self.trades if t.action == "SELL")

    # ----------------------------------------------------------------
    # 打印
    # ----------------------------------------------------------------

    def print_summary(self):
        """在控制台打印格式化的回测结果表格"""
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  回测结果报告 ── {self.strategy_name}")
        print(sep)
        rows = [
            ("回测区间",    f"{self.start_date} ~ {self.end_date}"),
            ("初始资金",    f"{self.initial_capital:>14,.0f} 元"),
            ("最终资产",    f"{self.final_value:>14,.2f} 元"),
            ("总收益率",    f"{self.total_return*100:>13.2f} %"),
            ("年化收益率",  f"{self.annual_return*100:>13.2f} %"),
            ("最大回撤",    f"{self.max_drawdown*100:>13.2f} %"),
            ("夏普比率",    f"{self.sharpe_ratio:>14.4f}"),
            ("卡玛比率",    f"{self.calmar_ratio:>14.4f}"),
            ("年化波动率",  f"{self.volatility*100:>13.2f} %"),
            ("总交易次数",  f"{self.total_trades:>14}"),
            ("胜率",        f"{self.win_rate*100:>13.1f} %"),
            ("累计手续费",  f"{self.total_commission:>14,.2f} 元"),
        ]
        for label, value in rows:
            print(f"  {label:<10}: {value}")
        print(sep)
