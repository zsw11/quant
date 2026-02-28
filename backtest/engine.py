"""
回测引擎 - 事件驱动，逐日模拟历史交易

特性：
  - 真实 A 股交易成本：佣金 + 印花税（仅卖出）+ 滑点
  - T+1 保护：当日买入的股票当日不能卖出
  - 最小交易单位：买入100股整数倍，卖出无限制
  - 每日更新持仓市价，计算净值曲线
  - 风控模块（RiskManager）可选接入
"""
import pandas as pd
from typing import Dict, List, Optional
import logging

import config
from strategy.base import BaseStrategy, Signal
from risk.manager import RiskManager
from .result import BacktestResult, TradeRecord

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    回测引擎

    使用方式：
        engine = BacktestEngine()
        result = engine.run(strategy, data, "20220101", "20241231")
        result.print_summary()
    """

    def __init__(self):
        self.commission  = config.COMMISSION
        self.stamp_duty  = config.STAMP_DUTY
        self.slippage    = config.SLIPPAGE
        self.min_comm    = config.MIN_COMMISSION
        self.init_cash   = config.INITIAL_CAPITAL

    def run(
        self,
        strategy: BaseStrategy,
        data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date:   str,
        risk_manager: Optional[RiskManager] = None,
    ) -> BacktestResult:
        """
        执行回测

        Args:
            strategy    : 策略实例（MACrossStrategy 等）
            data        : {symbol: DataFrame}，含技术指标
            start_date  : 回测开始日期 "YYYYMMDD"
            end_date    : 回测结束日期 "YYYYMMDD"
            risk_manager: 风控模块（可选）

        Returns:
            BacktestResult（含净值曲线和交易记录）
        """
        # 初始化策略资金
        strategy.on_start(self.init_cash)

        # 获取回测期间所有交易日（取各股票日期的并集）
        dates = self._get_dates(data, start_date, end_date)
        if len(dates) == 0:
            raise ValueError("回测区间内没有数据，请检查日期范围")

        logger.info(
            f"开始回测 [{strategy.name}] "
            f"{start_date}~{end_date}，共 {len(dates)} 个交易日"
        )

        equity_records = []
        all_trades: List[TradeRecord] = []

        # 记录每日买入的股票（用于 T+1 判断）
        bought_today: set = set()

        for i, date in enumerate(dates):
            # 当日收盘价
            prices = {}
            for sym, df in data.items():
                if date in df.index:
                    prices[sym] = float(df.loc[date, "close"])

            # 更新策略持仓的当前市价
            strategy.update_prices(prices)

            # 新的一天，清空"今日买入"记录
            if i == 0 or dates[i].date() != dates[i - 1].date():
                bought_today = set()

            # 生成信号
            try:
                signals = strategy.generate_signals(data, date)
            except Exception as e:
                logger.error(f"{date.date()} 策略异常: {e}")
                signals = []

            # 风控过滤（仓位 + 止损）
            if risk_manager and signals:
                signals = risk_manager.filter_signals(signals, strategy, prices)

            # 移动止盈检查（每日收盘后，基于当日收盘价）
            if risk_manager:
                trail_signals = risk_manager.update_trailing(strategy, prices)
                # 移动止盈信号优先，过滤已有卖出信号的股票
                selling_now = {s.symbol for s in signals if s.action == "SELL"}
                for ts in trail_signals:
                    if ts.symbol not in selling_now and ts.symbol not in bought_today:
                        signals.append(ts)

            # 执行信号
            for sig in signals:
                # T+1 检查：今日买入的不能今日卖出
                if sig.action == "SELL" and sig.symbol in bought_today:
                    logger.debug(f"T+1限制，跳过卖出 {sig.symbol}")
                    continue

                record = self._execute(sig, prices, strategy, date)
                if record:
                    all_trades.append(record)
                    if sig.action == "BUY":
                        bought_today.add(sig.symbol)
                        # 通知风控：新买入，初始化移动止盈基准
                        if risk_manager:
                            risk_manager.on_buy(sig.symbol, prices.get(sig.symbol, 0))
                    elif sig.action == "SELL":
                        # 通知风控：已卖出，清除移动止盈记录
                        if risk_manager:
                            risk_manager.on_sell(sig.symbol)

            # 记录当日净值
            equity_records.append({
                "date":   date,
                "equity": strategy.total_assets,
            })

        strategy.on_end()

        # 构造结果
        eq_df = pd.DataFrame(equity_records).set_index("date")["equity"]
        result = BacktestResult(
            strategy_name   = strategy.name,
            start_date      = start_date,
            end_date        = end_date,
            initial_capital = self.init_cash,
            equity_curve    = eq_df,
            trades          = all_trades,
        )

        logger.info(
            f"回测完成 | 总收益={result.total_return*100:.2f}% "
            f"最大回撤={result.max_drawdown*100:.2f}% "
            f"夏普={result.sharpe_ratio:.3f}"
        )
        return result

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _execute(
        self,
        sig:      Signal,
        prices:   Dict[str, float],
        strategy: BaseStrategy,
        date:     pd.Timestamp,
    ) -> Optional[TradeRecord]:
        """执行单个信号，返回成交记录"""

        if sig.symbol not in prices or prices[sig.symbol] <= 0:
            logger.warning(f"{date.date()} {sig.symbol} 无行情，跳过")
            return None

        base_price = prices[sig.symbol]

        # 滑点：买入价格略高，卖出略低
        if sig.action == "BUY":
            exec_price = base_price * (1 + self.slippage)
        else:
            exec_price = base_price * (1 - self.slippage)

        qty    = sig.quantity
        amount = exec_price * qty

        # 手续费
        comm = max(amount * self.commission, self.min_comm)
        if sig.action == "SELL":
            comm += amount * self.stamp_duty   # 印花税仅卖出收

        # 买入：检查资金
        if sig.action == "BUY":
            if amount + comm > strategy.cash:
                # 按可用资金缩减数量
                budget = strategy.cash * 0.99
                qty    = int(budget / exec_price / 100) * 100
                if qty <= 0:
                    logger.debug(f"{sig.symbol} 资金不足，跳过买入")
                    return None
                amount = exec_price * qty
                comm   = max(amount * self.commission, self.min_comm)

        # 卖出：检查持仓
        elif sig.action == "SELL":
            pos = strategy.get_position(sig.symbol)
            if not pos or pos.quantity <= 0:
                return None
            qty    = min(qty, pos.quantity)
            amount = exec_price * qty
            comm   = max(amount * self.commission, self.min_comm)
            comm  += amount * self.stamp_duty

        # 计算本笔盈亏（仅卖出）
        pnl = 0.0
        if sig.action == "SELL":
            pos = strategy.get_position(sig.symbol)
            if pos:
                pnl = (exec_price - pos.avg_cost) * qty - comm

        # 更新策略状态
        sig.quantity = qty
        strategy.on_trade(sig, exec_price, qty, comm)

        logger.debug(
            f"{date.date()} {sig.action} {sig.symbol} "
            f"x{qty} @{exec_price:.2f} 手续费={comm:.2f} | {sig.reason}"
        )

        return TradeRecord(
            date       = date,
            symbol     = sig.symbol,
            action     = sig.action,
            quantity   = qty,
            price      = exec_price,
            amount     = amount,
            commission = comm,
            pnl        = pnl,
            reason     = sig.reason,
        )

    @staticmethod
    def _get_dates(
        data:       Dict[str, pd.DataFrame],
        start_date: str,
        end_date:   str,
    ) -> pd.DatetimeIndex:
        """取所有股票在回测区间内的交易日并集"""
        start = pd.to_datetime(start_date)
        end   = pd.to_datetime(end_date)
        all_dates: set = set()
        for df in data.values():
            mask = (df.index >= start) & (df.index <= end)
            all_dates.update(df[mask].index.tolist())
        return pd.DatetimeIndex(sorted(all_dates))
