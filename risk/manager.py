"""
风控管理器 - 在策略信号执行前做合规过滤

规则（来自 config.py）：
  1. 单票最大仓位 MAX_POSITION_PCT（默认30%）
  2. 全仓最大仓位 MAX_TOTAL_POS_PCT（默认95%）
  3. 最大回撤超限 MAX_DRAWDOWN_LIMIT（默认15%）→ 触发全部清仓
  4. 单票止损 STOP_LOSS_PCT（默认5%）→ 产生强制卖出信号
  5. 最低现金保留 MIN_CASH_RESERVE（默认5%）

使用方式：
    rm = RiskManager(initial_capital=1_000_000)
    signals = rm.filter_signals(signals, strategy, prices)
    stop_signals = rm.check_stop_loss(strategy, prices)
"""
from typing import Dict, List
import logging

import config
from strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class RiskManager:
    """
    风控管理器

    Args:
        initial_capital: 初始资金（用于最大回撤基准计算）
    """

    def __init__(self, initial_capital: float = None):
        self.initial_capital    = initial_capital or config.INITIAL_CAPITAL
        self.max_pos_pct        = config.MAX_POSITION_PCT
        self.max_total_pct      = config.MAX_TOTAL_POS_PCT
        self.max_drawdown_limit = config.MAX_DRAWDOWN_LIMIT
        self.stop_loss_pct      = config.STOP_LOSS_PCT
        self.min_cash_reserve   = config.MIN_CASH_RESERVE
        self.trailing_stop_pct  = getattr(config, "TRAILING_STOP_PCT", 0.0)

        # 追踪历史峰值净值（用于计算当前回撤）
        self._peak_equity: float = initial_capital or config.INITIAL_CAPITAL
        # {symbol: highest_price_since_buy} 追踪每只持仓的历史最高价
        self._highest_price: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signals:  List[Signal],
        strategy: BaseStrategy,
        prices:   Dict[str, float],
    ) -> List[Signal]:
        """
        对策略生成的信号列表做风控过滤

        流程：
          1. 更新峰值净值
          2. 如果触发最大回撤 → 返回全部清仓信号，丢弃原信号
          3. 对剩余信号逐个检查仓位限制
          4. 追加止损信号

        Returns:
            过滤并补充后的合规信号列表
        """
        # 更新峰值
        total = strategy.total_assets
        if total > self._peak_equity:
            self._peak_equity = total

        # 检查最大回撤
        if self._is_max_drawdown_breached(total):
            logger.warning(
                f"[风控] 最大回撤触发！当前净值={total:,.0f} "
                f"峰值={self._peak_equity:,.0f} "
                f"回撤={(total - self._peak_equity)/self._peak_equity*100:.1f}%，"
                f"执行全仓清仓"
            )
            return self._build_liquidate_all(strategy, prices)

        # 过滤买入信号（检查仓位限制，超限时削减数量而非直接拒绝）
        approved: List[Signal] = []
        for sig in signals:
            if sig.action == "BUY":
                adjusted = self._adjust_buy_qty(sig, strategy, prices)
                if adjusted is not None:
                    approved.append(adjusted)
            else:
                # 卖出信号直接通过
                approved.append(sig)

        # 追加止损 + 移动止盈信号（对现有持仓）
        stop_signals = self.check_stop_loss(strategy, prices)
        # 避免重复卖出同一股票
        selling_symbols = {s.symbol for s in approved if s.action == "SELL"}
        for ss in stop_signals:
            if ss.symbol not in selling_symbols:
                approved.append(ss)
                selling_symbols.add(ss.symbol)

        return approved

    def update_trailing(self, strategy: BaseStrategy, prices: Dict[str, float]):
        """
        每日收盘后调用：更新每只持仓的最高价，检查移动止盈。
        返回需要执行的移动止盈卖出信号列表。
        """
        if self.trailing_stop_pct <= 0:
            return []
        signals = []
        for sym, pos in strategy.positions.items():
            if pos.quantity <= 0:
                continue
            price = prices.get(sym, 0)
            if price <= 0:
                continue
            # 更新历史最高价
            if sym not in self._highest_price or price > self._highest_price[sym]:
                self._highest_price[sym] = price
            highest = self._highest_price[sym]
            trail_stop = highest * (1 - self.trailing_stop_pct)
            if price <= trail_stop:
                drop_pct = (price - highest) / highest * 100
                logger.warning(
                    f"[风控] 移动止盈触发 {sym}："
                    f"最高={highest:.2f} 当前={price:.2f} 回落={drop_pct:.1f}%"
                )
                signals.append(Signal(
                    symbol   = sym,
                    action   = "SELL",
                    quantity = pos.quantity,
                    price    = 0,
                    reason   = f"移动止盈 从最高{highest:.2f}回落{abs(drop_pct):.1f}%",
                ))
        return signals

    def on_buy(self, symbol: str, price: float):
        """买入时初始化该股票的最高价记录"""
        self._highest_price[symbol] = price

    def on_sell(self, symbol: str):
        """卖出时清除该股票的最高价记录"""
        self._highest_price.pop(symbol, None)

    def check_stop_loss(
        self,
        strategy: BaseStrategy,
        prices:   Dict[str, float],
    ) -> List[Signal]:
        """
        扫描所有持仓，对亏损超过止损线的仓位生成卖出信号

        Returns:
            止损卖出信号列表
        """
        signals = []
        for sym, pos in strategy.positions.items():
            if pos.quantity <= 0:
                continue
            price = prices.get(sym, 0)
            if price <= 0:
                continue
            loss_pct = (price - pos.avg_cost) / pos.avg_cost
            if loss_pct <= -self.stop_loss_pct:
                logger.warning(
                    f"[风控] 止损触发 {sym}："
                    f"成本={pos.avg_cost:.2f} 当前={price:.2f} "
                    f"亏损={loss_pct*100:.1f}%"
                )
                signals.append(Signal(
                    symbol   = sym,
                    action   = "SELL",
                    quantity = pos.quantity,
                    price    = 0,
                    reason   = f"止损 亏损{loss_pct*100:.1f}%",
                ))
        return signals

    def update_peak(self, equity: float):
        """手动更新峰值净值（实盘每根K线调用一次）"""
        if equity > self._peak_equity:
            self._peak_equity = equity

    def current_drawdown(self, equity: float) -> float:
        """
        计算当前回撤（负数）
        例如 -0.12 表示从峰值回撤了12%
        """
        if self._peak_equity <= 0:
            return 0.0
        return (equity - self._peak_equity) / self._peak_equity

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _is_max_drawdown_breached(self, equity: float) -> bool:
        dd = self.current_drawdown(equity)
        return dd <= -self.max_drawdown_limit

    def _adjust_buy_qty(
        self,
        sig:      Signal,
        strategy: BaseStrategy,
        prices:   Dict[str, float],
    ):
        """
        检查买入信号是否符合风控限制，超限时削减数量。

        返回调整后的 Signal（quantity 可能变小），
        如果完全无法买入则返回 None。
        """
        import copy

        total = strategy.total_assets
        if total <= 0:
            logger.info(f"[风控] 拒绝买入 {sig.symbol}：总资产为0")
            return None

        price = prices.get(sig.symbol, 0)
        if price <= 0:
            logger.info(f"[风控] 拒绝买入 {sig.symbol}：无价格数据")
            return None

        # 当前该票已有仓位市值
        pos = strategy.get_position(sig.symbol)
        current_sym_value = pos.market_value if pos else 0.0

        # 当前所有持仓市值（含该票）
        total_pos_value = sum(p.market_value for p in strategy.positions.values())

        # 计算三个维度的最大可买金额，取最小值
        # 1. 单票仓位上限：该票买入后市值 ≤ total * MAX_POSITION_PCT
        max_by_sym  = total * self.max_pos_pct - current_sym_value

        # 2. 总仓位上限：所有仓位买入后 ≤ total * MAX_TOTAL_POS_PCT
        max_by_total = total * self.max_total_pct - total_pos_value

        # 3. 现金保留：买入后现金 ≥ total * MIN_CASH_RESERVE
        max_by_cash = strategy.cash - total * self.min_cash_reserve

        # 取最严格的约束
        max_buy_value = min(max_by_sym, max_by_total, max_by_cash)

        if max_buy_value <= 0:
            logger.info(
                f"[风控] 拒绝买入 {sig.symbol}："
                f"单票上限剩余={max_by_sym:,.0f} "
                f"总仓上限剩余={max_by_total:,.0f} "
                f"可用现金={max_by_cash:,.0f}"
            )
            return None

        # 原始期望买入金额
        original_value = sig.quantity * price

        if original_value <= max_buy_value:
            # 在限额内，直接放行
            return sig

        # 超限：按风控上限削减股数（取整到100的整数倍）
        allowed_qty = int(max_buy_value / price / 100) * 100
        if allowed_qty <= 0:
            logger.info(
                f"[风控] 拒绝买入 {sig.symbol}："
                f"削减后不足1手（上限={max_buy_value:,.0f} 元，价格={price:.2f}）"
            )
            return None

        adjusted = copy.copy(sig)
        adjusted.quantity = allowed_qty
        logger.info(
            f"[风控] 削减买入 {sig.symbol}："
            f"原始={int(sig.quantity)}股 → 调整={allowed_qty}股"
            f"（仓位上限约束）"
        )
        return adjusted

    def _build_liquidate_all(
        self,
        strategy: BaseStrategy,
        prices:   Dict[str, float],
    ) -> List[Signal]:
        """生成清仓所有持仓的卖出信号列表"""
        signals = []
        for sym, pos in strategy.positions.items():
            if pos.quantity > 0:
                signals.append(Signal(
                    symbol   = sym,
                    action   = "SELL",
                    quantity = pos.quantity,
                    price    = 0,
                    reason   = "最大回撤触发全仓清仓",
                ))
        return signals
