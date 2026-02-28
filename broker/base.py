"""
Broker 抽象基类 - 定义下单接口

子类：
  PaperBroker : 模拟盘（内存撮合，假钱）
  GmBroker    : 掘金实盘（调用 gm SDK 真实下单）

所有下单方法返回 OrderResult，统一错误处理。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class OrderResult:
    """
    下单结果

    Attributes:
        success      : 是否成功提交委托
        order_id     : 委托编号（失败时为空）
        symbol       : 股票代码
        action       : "BUY" | "SELL"
        quantity     : 委托数量
        price        : 委托价格（0表示市价）
        filled_price : 实际成交价（模拟盘立即成交，实盘异步回报）
        filled_qty   : 实际成交数量
        commission   : 手续费 + 印花税
        message      : 说明信息（错误原因等）
    """
    success:      bool
    order_id:     str   = ""
    symbol:       str   = ""
    action:       str   = ""
    quantity:     float = 0.0
    price:        float = 0.0
    filled_price: float = 0.0
    filled_qty:   float = 0.0
    commission:   float = 0.0
    message:      str   = ""


class BaseBroker(ABC):
    """
    Broker 抽象基类

    子类必须实现：
        buy(symbol, quantity, price)  -> OrderResult
        sell(symbol, quantity, price) -> OrderResult
        get_cash()                    -> float
        get_positions()               -> Dict[str, Dict]
    """

    # ---- 子类必须实现 ----

    @abstractmethod
    def buy(
        self,
        symbol:   str,
        quantity: float,
        price:    float = 0.0,
    ) -> OrderResult:
        """
        发出买入委托

        Args:
            symbol  : 股票代码（内部使用纯6位，如 "600519"）
            quantity: 买入股数（必须是100的整数倍）
            price   : 委托价格，0 表示市价单

        Returns:
            OrderResult
        """
        pass

    @abstractmethod
    def sell(
        self,
        symbol:   str,
        quantity: float,
        price:    float = 0.0,
    ) -> OrderResult:
        """
        发出卖出委托

        Args:
            symbol  : 股票代码
            quantity: 卖出股数
            price   : 委托价格，0 表示市价单

        Returns:
            OrderResult
        """
        pass

    @abstractmethod
    def get_cash(self) -> float:
        """返回账户当前可用现金"""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Dict]:
        """
        返回当前持仓字典

        Returns:
            {
                "600519": {
                    "quantity":      1000,
                    "avg_cost":      1800.0,
                    "current_price": 1850.0,
                },
                ...
            }
        """
        pass

    # ---- 通用工具方法 ----

    @staticmethod
    def to_gm_symbol(symbol: str) -> str:
        """
        将纯6位代码转换为掘金格式

        规则：
          6 开头 → 上交所 SHSE.xxxxxx
          其余   → 深交所 SZSE.xxxxxx
        """
        if symbol.startswith("6"):
            return f"SHSE.{symbol}"
        return f"SZSE.{symbol}"

    @staticmethod
    def from_gm_symbol(gm_symbol: str) -> str:
        """将掘金格式转换回纯6位代码"""
        return gm_symbol.split(".")[-1]
