"""
模拟盘 Broker - 内存撮合，假钱，不联网

特性：
  - 所有委托立即以"当前价格 ± 滑点"成交
  - 真实计算 A 股手续费：佣金（万3）+ 印花税（千1，仅卖出）
  - 最低佣金 5 元
  - 内置 T+1 检查（记录每日买入，卖出时拦截）
  - 无需任何外部依赖，随时可单独测试
"""
import uuid
import logging
from typing import Dict
from datetime import date

import config
from .base import BaseBroker, OrderResult

logger = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """
    模拟盘经纪商

    Args:
        initial_cash: 初始资金（默认读取 config.INITIAL_CAPITAL）

    实盘同步模式：
        如果从真实账户同步了持仓后切换为纸盘测试，
        可以在构造后调用 set_state(cash, positions) 注入真实状态。
    """

    def __init__(self, initial_cash: float = None):
        self.cash: float = initial_cash or config.INITIAL_CAPITAL
        # {symbol: {"quantity": float, "avg_cost": float, "current_price": float}}
        self._positions: Dict[str, Dict] = {}
        # {symbol: date} 记录每只股票最近一次买入日期（T+1）
        self._buy_dates: Dict[str, date] = {}
        self._today: date = date.today()

    # ------------------------------------------------------------------
    # BaseBroker 接口实现
    # ------------------------------------------------------------------

    def buy(
        self,
        symbol:   str,
        quantity: float,
        price:    float = 0.0,
    ) -> OrderResult:
        """
        模拟买入：立即以市价 + 滑点成交

        Args:
            symbol  : 纯6位代码，如 "600519"
            quantity: 买入股数（需是100整数倍）
            price   : 市价单传 0；限价单传目标价（模拟盘均按市价处理）
        """
        if quantity <= 0:
            return OrderResult(
                success=False,
                symbol=symbol,
                action="BUY",
                message="买入数量必须大于0",
            )

        # 按100股取整
        quantity = int(quantity / 100) * 100
        if quantity <= 0:
            return OrderResult(
                success=False,
                symbol=symbol,
                action="BUY",
                message="有效买入数量为0（不足1手）",
            )

        exec_price = (price if price > 0 else self._get_last_price(symbol))
        exec_price *= (1 + config.SLIPPAGE)

        amount = exec_price * quantity
        commission = max(amount * config.COMMISSION, config.MIN_COMMISSION)
        total_cost = amount + commission

        if total_cost > self.cash:
            # 按可用资金自动缩减
            budget = self.cash * 0.99
            quantity = int(budget / exec_price / 100) * 100
            if quantity <= 0:
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    action="BUY",
                    message=f"资金不足 (可用={self.cash:,.0f}，需要={total_cost:,.0f})",
                )
            amount = exec_price * quantity
            commission = max(amount * config.COMMISSION, config.MIN_COMMISSION)
            total_cost = amount + commission

        # 更新现金
        self.cash -= total_cost

        # 更新持仓
        if symbol in self._positions:
            pos = self._positions[symbol]
            new_qty  = pos["quantity"] + quantity
            new_cost = (pos["avg_cost"] * pos["quantity"] + exec_price * quantity) / new_qty
            pos["quantity"]      = new_qty
            pos["avg_cost"]      = new_cost
            pos["current_price"] = exec_price
        else:
            self._positions[symbol] = {
                "quantity":      quantity,
                "avg_cost":      exec_price,
                "current_price": exec_price,
            }

        # 记录买入日期（T+1）
        self._buy_dates[symbol] = self._today

        order_id = str(uuid.uuid4())[:8]
        logger.info(
            f"[模拟盘] BUY {symbol} x{quantity} @{exec_price:.2f} "
            f"手续费={commission:.2f} 剩余现金={self.cash:,.0f}"
        )

        return OrderResult(
            success      = True,
            order_id     = order_id,
            symbol       = symbol,
            action       = "BUY",
            quantity     = quantity,
            price        = price,
            filled_price = exec_price,
            filled_qty   = quantity,
            commission   = commission,
            message      = "成交",
        )

    def sell(
        self,
        symbol:   str,
        quantity: float,
        price:    float = 0.0,
    ) -> OrderResult:
        """
        模拟卖出：立即以市价 - 滑点成交

        T+1 检查：当天买入的股票不能当天卖出
        """
        if quantity <= 0:
            return OrderResult(
                success=False,
                symbol=symbol,
                action="SELL",
                message="卖出数量必须大于0",
            )

        # T+1 检查
        buy_date = self._buy_dates.get(symbol)
        if buy_date and buy_date == self._today:
            logger.warning(f"[模拟盘] T+1限制，{symbol} 今日买入不能今日卖出")
            return OrderResult(
                success=False,
                symbol=symbol,
                action="SELL",
                message="T+1限制：今日买入不能今日卖出",
            )

        pos = self._positions.get(symbol)
        if not pos or pos["quantity"] <= 0:
            return OrderResult(
                success=False,
                symbol=symbol,
                action="SELL",
                message=f"无持仓 {symbol}",
            )

        quantity = min(quantity, pos["quantity"])

        exec_price = (price if price > 0 else pos["current_price"])
        exec_price *= (1 - config.SLIPPAGE)

        amount = exec_price * quantity
        commission = max(amount * config.COMMISSION, config.MIN_COMMISSION)
        commission += amount * config.STAMP_DUTY   # 印花税（仅卖出）

        # 计算本笔盈亏
        pnl = (exec_price - pos["avg_cost"]) * quantity - commission

        # 更新现金
        self.cash += (amount - commission)

        # 更新持仓
        pos["quantity"] -= quantity
        if pos["quantity"] <= 0:
            del self._positions[symbol]
            if symbol in self._buy_dates:
                del self._buy_dates[symbol]

        order_id = str(uuid.uuid4())[:8]
        logger.info(
            f"[模拟盘] SELL {symbol} x{quantity} @{exec_price:.2f} "
            f"手续费={commission:.2f} 盈亏={pnl:+.2f} 现金={self.cash:,.0f}"
        )

        return OrderResult(
            success      = True,
            order_id     = order_id,
            symbol       = symbol,
            action       = "SELL",
            quantity     = quantity,
            price        = price,
            filled_price = exec_price,
            filled_qty   = quantity,
            commission   = commission,
            message      = f"成交 盈亏={pnl:+.2f}",
        )

    def get_cash(self) -> float:
        return self.cash

    def get_positions(self) -> Dict[str, Dict]:
        return {k: v.copy() for k, v in self._positions.items()}

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float):
        """更新股票当前市价（每根K线后调用）"""
        if symbol in self._positions and price > 0:
            self._positions[symbol]["current_price"] = price

    def update_prices(self, prices: Dict[str, float]):
        """批量更新市价"""
        for sym, price in prices.items():
            self.update_price(sym, price)

    def advance_day(self):
        """推进到下一个交易日（实盘模式在每日开盘时调用）"""
        self._today = date.today()

    def set_state(self, cash: float, positions: Dict[str, Dict]):
        """
        注入外部状态（用于从真实账户同步后切换模拟盘）

        Args:
            cash     : 账户可用现金
            positions: {symbol: {"quantity": ..., "avg_cost": ..., "current_price": ...}}
        """
        self.cash = cash
        self._positions = {k: v.copy() for k, v in positions.items()}
        logger.info(f"[模拟盘] 同步状态：现金={cash:,.0f} 持仓={list(positions.keys())}")

    def total_assets(self) -> float:
        """总资产 = 现金 + 持仓市值"""
        pos_value = sum(
            p["quantity"] * p["current_price"]
            for p in self._positions.values()
        )
        return self.cash + pos_value

    def _get_last_price(self, symbol: str) -> float:
        """获取最近已知价格（用于市价单估算）"""
        pos = self._positions.get(symbol)
        if pos:
            return pos.get("current_price", pos["avg_cost"])
        return 0.0
