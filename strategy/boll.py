"""
布林带策略（观察策略）- 只记录日志，不实际下单

逻辑：
  价格 ≤ 布林带下轨 → 超卖，记录"买入观察"日志
  价格 ≥ 布林带中轨 且已"观察持仓" → 记录"卖出观察"日志

用途：
  和主策略（双均线）的信号做对比
  布林带适合震荡市，双均线适合趋势市，两者互为补充
"""
from typing import Dict, List, Set
import pandas as pd

from .base import BaseStrategy, Signal


class BollStrategy(BaseStrategy):
    """布林带观察策略（不实际下单）"""

    def __init__(self, symbols: List[str]):
        super().__init__("Boll_Observer", {})
        self.symbols = symbols
        # 记录哪些股票已触及下轨（用于跟踪"观察持仓"状态）
        self._watching_buy: Set[str] = set()

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        current_time: pd.Timestamp,
    ) -> List[Signal]:

        signals = []

        for symbol in self.symbols:
            if symbol not in data:
                continue

            df = data[symbol]
            if current_time not in df.index:
                continue

            idx = df.index.get_loc(current_time)
            if idx < 1:
                continue

            required = ["boll_upper", "boll_mid", "boll_lower"]
            if not all(c in df.columns for c in required):
                continue

            curr = df.iloc[idx]
            price      = curr["close"]
            boll_lower = curr["boll_lower"]
            boll_mid   = curr["boll_mid"]
            boll_upper = curr["boll_upper"]

            if pd.isna(boll_lower):
                continue

            # 触及下轨 → 超卖，观察买入
            if price <= boll_lower and symbol not in self._watching_buy:
                self._watching_buy.add(symbol)
                signals.append(Signal(
                    symbol    = symbol,
                    action    = "BUY",
                    quantity  = 0,
                    price     = 0,
                    timestamp = current_time,
                    reason    = (
                        f"[布林观察-买入] 价格={price:.2f} "
                        f"触及下轨={boll_lower:.2f}，超卖"
                    ),
                    confidence = 0.7,
                ))

            # 价格回归中轨以上 → 观察卖出
            elif price >= boll_mid and symbol in self._watching_buy:
                self._watching_buy.discard(symbol)
                signals.append(Signal(
                    symbol    = symbol,
                    action    = "SELL",
                    quantity  = 0,
                    price     = 0,
                    timestamp = current_time,
                    reason    = (
                        f"[布林观察-卖出] 价格={price:.2f} "
                        f"回归中轨={boll_mid:.2f}"
                    ),
                    confidence = 0.7,
                ))

        return signals
