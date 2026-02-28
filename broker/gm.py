"""
掘金实盘 Broker - 调用 gm SDK 真实下单

依赖：
  pip install gm  （掘金量化 Python SDK）

设计：
  - 所有下单使用掘金 order_volume() 函数
  - 股票代码自动转换：600519 → SHSE.600519（6开头=上交所）
  - 委托结果异步回报（on_order_status 回调），此处仅返回提交结果
  - get_positions() 从掘金账户实时拉取真实持仓
  - T+1 由掘金风控系统保障，本地不重复检查

掘金 SDK 常用函数说明：
  order_volume(symbol, volume, side, order_type, position_effect, price)
    side          : OrderSide_Buy=1 买入 / OrderSide_Sell=2 卖出
    order_type    : OrderType_Market=1 市价 / OrderType_Limit=2 限价
    position_effect: PositionEffect_Open=1 开仓 / PositionEffect_Close=2 平仓

  get_unfinished_orders() → 未成交委托列表
  get_position(account_id, symbol) → 单只股票持仓
"""
import logging
from typing import Dict, Optional

import config
from .base import BaseBroker, OrderResult

logger = logging.getLogger(__name__)

# 尝试导入掘金 SDK（不安装时不影响回测和模拟盘）
try:
    from gm.api import (
        order_volume,
        get_cash,
        get_position,
        OrderSide_Buy,
        OrderSide_Sell,
        OrderType_Market,
        OrderType_Limit,
        PositionEffect_Open,
        PositionEffect_Close,
    )
    GM_AVAILABLE = True
except ImportError:
    GM_AVAILABLE = False
    logger.warning(
        "掘金 SDK 未安装，GmBroker 不可用。"
        "安装命令：pip install gm"
    )


class GmBroker(BaseBroker):
    """
    掘金实盘经纪商

    Args:
        account_id: 掘金账户ID（来自 config.ACCOUNT_ID）

    使用：
        broker = GmBroker(account_id=config.ACCOUNT_ID)
        result = broker.buy("600519", 100)
    """

    def __init__(self, account_id: str = None):
        if not GM_AVAILABLE:
            raise RuntimeError(
                "掘金 SDK 未安装，无法使用实盘。"
                "请运行：pip install gm"
            )
        self.account_id = account_id or config.ACCOUNT_ID

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
        发出买入委托（市价单）

        Args:
            symbol  : 纯6位代码，如 "600519"（内部自动转为 SHSE.600519）
            quantity: 买入股数（100整数倍）
            price   : 0=市价单；非0=限价单
        """
        gm_sym = self.to_gm_symbol(symbol)
        qty    = int(quantity / 100) * 100
        if qty <= 0:
            return OrderResult(
                success=False, symbol=symbol, action="BUY",
                message="有效买入数量为0（不足1手）",
            )

        order_type = OrderType_Market if price == 0 else OrderType_Limit
        limit_price = price if price > 0 else 0

        try:
            orders = order_volume(
                symbol          = gm_sym,
                volume          = qty,
                side            = OrderSide_Buy,
                order_type      = order_type,
                position_effect = PositionEffect_Open,
                price           = limit_price,
                account         = self.account_id,
            )
            if orders:
                order = orders[0]
                order_id = getattr(order, "cl_ord_id", "")
                logger.info(
                    f"[实盘] BUY {gm_sym} x{qty} "
                    f"{'市价' if price==0 else f'限价@{price:.2f}'} "
                    f"委托ID={order_id}"
                )
                return OrderResult(
                    success  = True,
                    order_id = order_id,
                    symbol   = symbol,
                    action   = "BUY",
                    quantity = qty,
                    price    = price,
                    message  = "委托已提交",
                )
            else:
                return OrderResult(
                    success=False, symbol=symbol, action="BUY",
                    message="掘金 order_volume 返回空列表",
                )
        except Exception as e:
            logger.error(f"[实盘] 买入 {gm_sym} 异常: {e}")
            return OrderResult(
                success=False, symbol=symbol, action="BUY",
                message=str(e),
            )

    def sell(
        self,
        symbol:   str,
        quantity: float,
        price:    float = 0.0,
    ) -> OrderResult:
        """
        发出卖出委托（市价单）

        注意：T+1 由掘金风控系统保障，委托会被拒绝或等到下一交易日。
        """
        gm_sym = self.to_gm_symbol(symbol)

        try:
            orders = order_volume(
                symbol          = gm_sym,
                volume          = int(quantity),
                side            = OrderSide_Sell,
                order_type      = OrderType_Market if price == 0 else OrderType_Limit,
                position_effect = PositionEffect_Close,
                price           = price if price > 0 else 0,
                account         = self.account_id,
            )
            if orders:
                order    = orders[0]
                order_id = getattr(order, "cl_ord_id", "")
                logger.info(
                    f"[实盘] SELL {gm_sym} x{quantity} "
                    f"{'市价' if price==0 else f'限价@{price:.2f}'} "
                    f"委托ID={order_id}"
                )
                return OrderResult(
                    success  = True,
                    order_id = order_id,
                    symbol   = symbol,
                    action   = "SELL",
                    quantity = quantity,
                    price    = price,
                    message  = "委托已提交",
                )
            else:
                return OrderResult(
                    success=False, symbol=symbol, action="SELL",
                    message="掘金 order_volume 返回空列表",
                )
        except Exception as e:
            logger.error(f"[实盘] 卖出 {gm_sym} 异常: {e}")
            return OrderResult(
                success=False, symbol=symbol, action="SELL",
                message=str(e),
            )

    def get_cash(self) -> float:
        """从掘金账户实时获取可用现金"""
        try:
            cash_info = get_cash(account=self.account_id)
            if cash_info:
                return float(cash_info.available)
            return 0.0
        except Exception as e:
            logger.error(f"[实盘] 获取现金异常: {e}")
            return 0.0

    def get_positions(self) -> Dict[str, Dict]:
        """
        从掘金账户实时获取持仓

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
        result = {}
        try:
            for gm_sym in config.SYMBOLS_GM:
                pos = get_position(account=self.account_id, symbol=gm_sym)
                if pos and pos.volume > 0:
                    sym = self.from_gm_symbol(gm_sym)
                    result[sym] = {
                        "quantity":      float(pos.volume),
                        "avg_cost":      float(pos.cost_price),
                        "current_price": float(pos.price),
                    }
        except Exception as e:
            logger.error(f"[实盘] 获取持仓异常: {e}")
        return result

    def get_all_positions(self) -> Dict[str, Dict]:
        """别名，同 get_positions()"""
        return self.get_positions()
