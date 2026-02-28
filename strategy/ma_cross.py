"""
双均线策略（主策略）- 唯一实际下单的策略

逻辑：
  金叉：MA_FAST 上穿 MA_SLOW → 买入
  死叉：MA_FAST 下穿 MA_SLOW → 卖出

特点：
  - 未持仓时才买入，已持仓时才卖出（避免重复信号）
  - 买入数量按照可用资金 × position_pct 计算，按手取整（100股整数倍）
  - 实盘中同时检查止损（由 RiskManager 处理）
"""
from typing import Dict, List
import pandas as pd

import config
from .base import BaseStrategy, Signal


class MACrossStrategy(BaseStrategy):
    """
    双均线金叉/死叉策略

    参数（来自 config.py）：
        MA_FAST    : 短期均线周期，默认 10
        MA_SLOW    : 长期均线周期，默认 30
        MA_TREND   : 趋势过滤均线周期，默认 60（0 表示关闭）
        POSITION_PCT: 每次建仓使用可用资金比例，默认 0.9
    """

    def __init__(self, symbols: List[str]):
        params = {
            "fast":             config.MA_FAST,
            "slow":             config.MA_SLOW,
            "trend":            config.MA_TREND,
            "position_pct":     config.POSITION_PCT,
            "rsi_buy_max":      config.RSI_BUY_MAX,
            "rsi_vol_exempt":   config.RSI_VOL_EXEMPT,
            "vol_confirm_ratio":config.VOL_CONFIRM_RATIO,
            "sell_cooldown":    config.SELL_COOLDOWN_BARS,
            "surge_lookback":   config.SURGE_LOOKBACK,
            "surge_max_pct":    config.SURGE_MAX_PCT,
            "day_surge_pct":    config.DAY_SURGE_PCT,
            "day_surge_rsi":    config.DAY_SURGE_RSI,
        }
        super().__init__("MACross", params)
        self.symbols             = symbols
        self.fast                = config.MA_FAST
        self.slow                = config.MA_SLOW
        self.trend               = config.MA_TREND          # 0 = 关闭趋势过滤
        self.position_pct        = config.POSITION_PCT
        self.rsi_buy_max         = config.RSI_BUY_MAX       # RSI 超买上限，100=关闭
        self.rsi_vol_exempt      = config.RSI_VOL_EXEMPT    # 放量豁免RSI的倍数
        self.vol_confirm_ratio   = config.VOL_CONFIRM_RATIO # 量能确认倍数，0=关闭
        self.sell_cooldown_bars  = config.SELL_COOLDOWN_BARS  # 卖出后冷却期（交易日数）
        self.surge_lookback      = config.SURGE_LOOKBACK    # 短期涨幅回看天数
        self.surge_max_pct       = config.SURGE_MAX_PCT     # 短期涨幅上限，0=关闭
        self.day_surge_pct       = config.DAY_SURGE_PCT     # 单日跳涨上限，0=关闭
        self.day_surge_rsi       = config.DAY_SURGE_RSI     # 单日跳涨时同时检查RSI阈值
        # {symbol: DataFrame行索引(idx)} 记录上次卖出时的 bar 位置
        self._last_sell_idx: Dict[str, int] = {}

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        current_time: pd.Timestamp,
    ) -> List[Signal]:
        """
        遍历所有监控股票，判断是否出现金叉/死叉

        Returns:
            Signal 列表（通常每次最多1-2个信号）
        """
        signals = []

        for symbol in self.symbols:
            if symbol not in data:
                continue

            df = data[symbol]

            # 找到当前时间在 df 中的位置
            if current_time not in df.index:
                continue
            idx = df.index.get_loc(current_time)
            if idx < self.slow:          # 数据不足，跳过
                continue

            fast_col = f"ma{self.fast}"
            slow_col = f"ma{self.slow}"
            if fast_col not in df.columns or slow_col not in df.columns:
                self.logger.warning(f"{symbol} 缺少均线列，跳过")
                continue

            curr = df.iloc[idx]
            prev = df.iloc[idx - 1]

            curr_fast = curr[fast_col]
            curr_slow = curr[slow_col]
            prev_fast = prev[fast_col]
            prev_slow = prev[slow_col]

            if pd.isna(curr_fast) or pd.isna(curr_slow):
                continue

            price = curr["close"]

            # ---- 金叉：买入 ----
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                if not self.has_position(symbol):
                    # 卖出冷却期：避免死叉后隔天立刻再入场（震荡市假信号）
                    if self.sell_cooldown_bars > 0 and symbol in self._last_sell_idx:
                        bars_since_sell = idx - self._last_sell_idx[symbol]
                        if bars_since_sell < self.sell_cooldown_bars:
                            self.logger.debug(
                                f"{symbol} 距上次卖出仅 {bars_since_sell} 根K线，"
                                f"冷却期 {self.sell_cooldown_bars} 根，跳过买入"
                            )
                            continue

                    # MA_TREND 趋势过滤：价格须在趋势均线上方
                    if self.trend > 0:
                        trend_col = f"ma{self.trend}"
                        if trend_col not in df.columns or pd.isna(curr.get(trend_col)):
                            self.logger.debug(
                                f"{symbol} MA{self.trend} 数据不足，跳过买入"
                            )
                            continue
                        ma_trend_val = curr[trend_col]
                        if price <= ma_trend_val:
                            self.logger.debug(
                                f"{symbol} 价格{price:.2f} <= MA{self.trend}"
                                f"={ma_trend_val:.2f}，趋势过滤，跳过买入"
                            )
                            continue

                    # RSI 超买过滤：RSI 过高时，若非放量突破则跳过（追顶）
                    if self.rsi_buy_max < 100 and "rsi" in df.columns:
                        rsi_val = curr.get("rsi")
                        if rsi_val is not None and not pd.isna(rsi_val) and rsi_val > self.rsi_buy_max:
                            # 放量豁免：量能强劲时即使 RSI 高也是真突破
                            vol_now  = curr.get("volume", 0)
                            vol_ma20 = curr.get("vol_ma20", 0)
                            vol_r    = (vol_now / vol_ma20) if vol_ma20 > 0 else 0
                            if vol_r < self.rsi_vol_exempt:
                                self.logger.debug(
                                    f"{symbol} RSI={rsi_val:.1f} > {self.rsi_buy_max} "
                                    f"且量能={vol_r:.2f}x < {self.rsi_vol_exempt}x，超买过滤，跳过买入"
                                )
                                continue
                            else:
                                self.logger.debug(
                                    f"{symbol} RSI={rsi_val:.1f} > {self.rsi_buy_max} "
                                    f"但放量={vol_r:.2f}x >= {self.rsi_vol_exempt}x，豁免过滤，允许买入"
                                )

                    # 成交量确认：量能不足说明突破信号不可靠
                    if self.vol_confirm_ratio > 0 and "vol_ma20" in df.columns:
                        vol_now   = curr.get("volume")
                        vol_ma20  = curr.get("vol_ma20")
                        if (vol_now is not None and vol_ma20 is not None
                                and not pd.isna(vol_now) and not pd.isna(vol_ma20)
                                and vol_ma20 > 0):
                            vol_ratio = vol_now / vol_ma20
                            if vol_ratio < self.vol_confirm_ratio:
                                self.logger.debug(
                                    f"{symbol} 成交量比={vol_ratio:.2f}x < "
                                    f"{self.vol_confirm_ratio}x，量能不足，跳过买入"
                                )
                                continue

                    # 短期涨幅过滤：N日内急涨超过阈值，追高风险大，跳过
                    if self.surge_max_pct > 0 and self.surge_lookback > 0:
                        lookback_idx = idx - self.surge_lookback
                        if lookback_idx >= 0:
                            past_close = df.iloc[lookback_idx]["close"]
                            surge_pct  = (price - past_close) / past_close
                            if surge_pct > self.surge_max_pct:
                                self.logger.debug(
                                    f"{symbol} {self.surge_lookback}日涨幅={surge_pct*100:.1f}% "
                                    f"> {self.surge_max_pct*100:.0f}%，急涨过滤，跳过买入"
                                )
                                continue

                    # 单日跳涨过滤：当日涨幅过大且RSI高 → 节假日跳空/追高，跳过
                    if self.day_surge_pct > 0 and "pct_change" in df.columns:
                        day_pct = curr.get("pct_change")
                        rsi_val = curr.get("rsi", 0)
                        if (day_pct is not None and not pd.isna(day_pct)
                                and day_pct > self.day_surge_pct * 100
                                and rsi_val > self.day_surge_rsi):
                            self.logger.debug(
                                f"{symbol} 当日涨幅={day_pct:.2f}% > {self.day_surge_pct*100:.1f}% "
                                f"且 RSI={rsi_val:.1f} > {self.day_surge_rsi}，跳涨过滤，跳过买入"
                            )
                            continue

                    qty = self._calc_buy_qty(price)
                    if qty > 0:
                        trend_info = ""
                        if self.trend > 0:
                            trend_info = f" | MA{self.trend}={curr[f'ma{self.trend}']:.2f}↑"
                        signals.append(Signal(
                            symbol    = symbol,
                            action    = "BUY",
                            quantity  = qty,
                            price     = 0,        # 市价单
                            timestamp = current_time,
                            reason    = (
                                f"金叉 MA{self.fast}={curr_fast:.2f} "
                                f"上穿 MA{self.slow}={curr_slow:.2f}"
                                f"{trend_info}"
                            ),
                        ))

            # ---- 死叉：卖出 ----
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                pos = self.get_position(symbol)
                if pos and pos.quantity > 0:
                    # 记录卖出时的 bar 位置，供冷却期判断使用
                    self._last_sell_idx[symbol] = idx
                    signals.append(Signal(
                        symbol    = symbol,
                        action    = "SELL",
                        quantity  = pos.quantity,
                        price     = 0,
                        timestamp = current_time,
                        reason    = (
                            f"死叉 MA{self.fast}={curr_fast:.2f} "
                            f"下穿 MA{self.slow}={curr_slow:.2f}"
                        ),
                    ))

        return signals

    def _calc_buy_qty(self, price: float) -> int:
        """
        根据可用资金计算买入股数

        A股规则：最小单位100股（1手），向下取整到100的整数倍
        """
        budget = self.cash * self.position_pct
        qty    = int(budget / price / 100) * 100
        return qty
