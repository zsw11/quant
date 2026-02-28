"""
策略基类 - 定义所有策略共用的数据结构和接口

Signal   : 交易信号（买/卖指令）
Position : 持仓信息
BaseStrategy : 所有策略必须继承的抽象类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import logging


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------

@dataclass
class Signal:
    """
    交易信号

    属性：
        symbol    : 股票代码，如 "600519"
        action    : "BUY"买入 | "SELL"卖出 | "HOLD"持有
        quantity  : 股数（必须是100的整数倍，A股最小1手=100股）
        price     : 期望价格，0 表示市价单
        reason    : 信号原因（用于日志）
        confidence: 置信度 0-1（观察策略用）
    """
    symbol:     str
    action:     str
    quantity:   float
    price:      float = 0.0
    timestamp:  Optional[pd.Timestamp] = None
    reason:     str   = ""
    confidence: float = 1.0
    meta:       Dict  = field(default_factory=dict)


@dataclass
class Position:
    """
    单只股票的持仓信息

    属性：
        symbol       : 股票代码
        quantity     : 持仓股数
        avg_cost     : 平均持仓成本（元/股）
        current_price: 当前市价（实时更新）
        buy_date     : 买入日期（用于 T+1 判断）
    """
    symbol:        str
    quantity:      float
    avg_cost:      float
    current_price: float = 0.0
    buy_date:      Optional[pd.Timestamp] = None

    @property
    def market_value(self) -> float:
        """持仓市值"""
        return self.quantity * self.current_price

    @property
    def cost_value(self) -> float:
        """持仓成本"""
        return self.quantity * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        """浮动盈亏（元）"""
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """浮动盈亏（%）"""
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


# ------------------------------------------------------------------
# 策略抽象基类
# ------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    所有策略的抽象基类

    子类必须实现：
        generate_signals(data, current_time) -> List[Signal]

    内置功能：
        - 持仓管理（positions）
        - 现金管理（cash）
        - 成交回调（on_trade）
        - 总资产计算（total_assets）
    """

    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name   = name
        self.params = params or {}
        self.cash:   float = 0.0
        self.positions: Dict[str, Position] = {}
        self.logger = logging.getLogger(f"strategy.{name}")

    # ---- 子类必须实现 ----

    @abstractmethod
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        current_time: pd.Timestamp,
    ) -> List[Signal]:
        """
        根据当前数据生成交易信号列表

        Args:
            data        : {symbol: DataFrame}，含技术指标
            current_time: 当前时间（回测时为日期，实盘时为K线结束时间）
        Returns:
            Signal 列表（空列表表示无操作）
        """
        pass

    # ---- 生命周期钩子 ----

    def on_start(self, initial_cash: float):
        """策略启动"""
        self.cash = initial_cash
        self.logger.info(f"策略[{self.name}]启动，初始资金: {initial_cash:,.0f}")

    def on_end(self):
        """策略结束"""
        self.logger.info(f"策略[{self.name}]结束，总资产: {self.total_assets:,.2f}")

    def on_trade(
        self,
        signal: Signal,
        filled_price: float,
        filled_qty: float,
        commission: float = 0.0,
    ):
        """
        成交回调 —— 更新持仓和现金

        由回测引擎或实盘 broker 在成交后调用
        """
        cost = filled_price * filled_qty

        if signal.action == "BUY":
            self.cash -= (cost + commission)
            sym = signal.symbol
            if sym in self.positions:
                pos = self.positions[sym]
                new_qty  = pos.quantity + filled_qty
                new_cost = (pos.avg_cost * pos.quantity + cost) / new_qty
                pos.quantity      = new_qty
                pos.avg_cost      = new_cost
                pos.current_price = filled_price
            else:
                self.positions[sym] = Position(
                    symbol        = sym,
                    quantity      = filled_qty,
                    avg_cost      = filled_price,
                    current_price = filled_price,
                    buy_date      = signal.timestamp,
                )

        elif signal.action == "SELL":
            self.cash += (cost - commission)
            sym = signal.symbol
            if sym in self.positions:
                self.positions[sym].quantity -= filled_qty
                if self.positions[sym].quantity <= 0:
                    del self.positions[sym]

    # ---- 工具方法 ----

    def update_prices(self, prices: Dict[str, float]):
        """更新持仓的当前市价（每根K线收盘后调用）"""
        for sym, price in prices.items():
            if sym in self.positions and price > 0:
                self.positions[sym].current_price = price

    def sync_from_broker(
        self,
        cash: float,
        positions: Dict[str, Dict],
    ):
        """
        从掘金实盘账户同步真实持仓（仅实盘模式启动时调用）

        Args:
            cash     : 账户可用现金
            positions: {symbol: {"quantity": int, "avg_cost": float}}
        """
        self.cash = cash
        self.positions.clear()
        for sym, info in positions.items():
            self.positions[sym] = Position(
                symbol        = sym,
                quantity      = info["quantity"],
                avg_cost      = info["avg_cost"],
                current_price = info.get("current_price", info["avg_cost"]),
            )
        self.logger.info(
            f"[实盘同步] 现金={cash:,.0f} "
            f"持仓={list(positions.keys())}"
        )

    @property
    def total_assets(self) -> float:
        """总资产 = 现金 + 持仓市值"""
        pos_value = sum(p.market_value for p in self.positions.values())
        return self.cash + pos_value

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定股票的持仓，无持仓返回 None"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """是否持有该股票"""
        pos = self.positions.get(symbol)
        return pos is not None and pos.quantity > 0

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)
