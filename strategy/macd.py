"""
MACD 策略（观察策略）- 只记录日志，不实际下单

逻辑：
  MACD 柱由负转正（DIF 上穿 DEA）→ 记录"买入观察"日志
  MACD 柱由正转负（DIF 下穿 DEA）→ 记录"卖出观察"日志

用途：
  和主策略（双均线）的信号做对比，判断两个策略是否同向
  信号一致时说明趋势较强，信号背离时需注意风险
"""
from typing import Dict, List
import pandas as pd

from .base import BaseStrategy, Signal


class MACDStrategy(BaseStrategy):
    """MACD 观察策略（generate_signals 返回信号但调用方不会下单）"""

    def __init__(self, symbols: List[str]):
        super().__init__("MACD_Observer", {})
        self.symbols = symbols

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
            if idx < 2:
                continue

            if "macd_bar" not in df.columns:
                continue

            curr_bar = df.iloc[idx]["macd_bar"]
            prev_bar = df.iloc[idx - 1]["macd_bar"]
            curr_dif = df.iloc[idx]["macd_dif"]
            curr_dea = df.iloc[idx]["macd_dea"]

            if pd.isna(curr_bar) or pd.isna(prev_bar):
                continue

            # MACD 金叉
            if prev_bar < 0 and curr_bar > 0:
                signals.append(Signal(
                    symbol    = symbol,
                    action    = "BUY",
                    quantity  = 0,          # 0 表示仅观察，不下单
                    price     = 0,
                    timestamp = current_time,
                    reason    = (
                        f"[MACD观察-买入] "
                        f"DIF={curr_dif:.4f} DEA={curr_dea:.4f} "
                        f"MACD柱由负转正"
                    ),
                    confidence = 0.8,
                ))

            # MACD 死叉
            elif prev_bar > 0 and curr_bar < 0:
                signals.append(Signal(
                    symbol    = symbol,
                    action    = "SELL",
                    quantity  = 0,
                    price     = 0,
                    timestamp = current_time,
                    reason    = (
                        f"[MACD观察-卖出] "
                        f"DIF={curr_dif:.4f} DEA={curr_dea:.4f} "
                        f"MACD柱由正转负"
                    ),
                    confidence = 0.8,
                ))

        return signals
