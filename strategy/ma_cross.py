"""
增强型双均线策略（主策略）- 唯一实际下单的策略

在原版双均线基础上增加：
  1. MACD 金叉作为辅助买入信号
  2. 分级止盈：达到目标利润时部分平仓锁定收益
  3. MACD 死叉 + RSI 高位作为辅助卖出信号
  4. RSI 超买减仓
  5. 趋势过滤（MA_TREND）防止在下跌趋势中接刀
  6. 动量过滤：跳过近N日下跌的弱势股（避免隆基/茅台拖累）
  7. 死叉确认：强趋势中延迟卖出，等待收盘确认

买入条件（满足任一）：
  A) MA 金叉（MA_FAST 上穿 MA_SLOW）
  B) MACD 柱由负转正（可选开关）
  + 趋势过滤、RSI过滤、量能确认、急涨过滤、冷却期、动量过滤

卖出条件：
  A) MA 死叉（全仓，可配置确认天数）
  B) 分级止盈（部分卖出）
  C) MACD 死叉 + RSI > 阈值（全仓）
  D) RSI 极端超买 + 盈利（减仓50%）
"""
from typing import Dict, List, Optional
import pandas as pd

import config
from .base import BaseStrategy, Signal


class MACrossStrategy(BaseStrategy):
    """增强型双均线 + MACD + 分级止盈策略"""

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
        self.trend               = config.MA_TREND
        self.position_pct        = config.POSITION_PCT
        self.rsi_buy_max         = config.RSI_BUY_MAX
        self.rsi_vol_exempt      = config.RSI_VOL_EXEMPT
        self.vol_confirm_ratio   = config.VOL_CONFIRM_RATIO
        self.sell_cooldown_bars  = config.SELL_COOLDOWN_BARS
        self.surge_lookback      = config.SURGE_LOOKBACK
        self.surge_max_pct       = config.SURGE_MAX_PCT
        self.day_surge_pct       = config.DAY_SURGE_PCT
        self.day_surge_rsi       = config.DAY_SURGE_RSI

        # 新增参数（带默认值）
        self.macd_buy_enabled    = getattr(config, 'MACD_BUY_ENABLED', True)
        self.macd_sell_rsi_min   = getattr(config, 'MACD_SELL_RSI_MIN', 70)
        self.rsi_overbought      = getattr(config, 'RSI_OVERBOUGHT', 88)
        self.partial_profit      = getattr(config, 'PARTIAL_PROFIT_ENABLED', True)
        self.profit_target_1     = getattr(config, 'PROFIT_TARGET_1', 0.06)
        self.profit_target_2     = getattr(config, 'PROFIT_TARGET_2', 0.12)
        self.profit_target_3     = getattr(config, 'PROFIT_TARGET_3', 0.20)
        self.profit_sell_pct     = getattr(config, 'PROFIT_SELL_PCT_1', 0.30)

        # R5 新增：动量过滤 —— 跳过近 N 日下跌的弱势股
        self.momentum_filter_enabled = getattr(config, 'MOMENTUM_FILTER_ENABLED', False)
        self.momentum_lookback       = getattr(config, 'MOMENTUM_LOOKBACK', 20)
        self.momentum_min_pct        = getattr(config, 'MOMENTUM_MIN_PCT', 0.0)

        # R5 新增：死叉确认天数 —— 要求连续N天MA_FAST < MA_SLOW才确认死叉
        self.death_cross_confirm     = getattr(config, 'DEATH_CROSS_CONFIRM', 1)

        # R5 新增：盈利保护 —— 持仓盈利超过阈值时，不因死叉卖出（让利润跑）
        self.profit_protect_enabled  = getattr(config, 'PROFIT_PROTECT_ENABLED', False)
        self.profit_protect_pct      = getattr(config, 'PROFIT_PROTECT_PCT', 0.15)

        # 内部状态
        self._last_sell_idx: Dict[str, int] = {}
        self._entry_price: Dict[str, float] = {}
        self._profit_stage: Dict[str, int] = {}
        self._death_cross_count: Dict[str, int] = {}  # 死叉确认计数

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
            if idx < self.slow:
                continue

            fast_col = f"ma{self.fast}"
            slow_col = f"ma{self.slow}"
            if fast_col not in df.columns or slow_col not in df.columns:
                continue

            curr = df.iloc[idx]
            prev = df.iloc[idx - 1]

            cf = curr[fast_col]
            cs = curr[slow_col]
            pf = prev[fast_col]
            ps = prev[slow_col]

            if pd.isna(cf) or pd.isna(cs):
                continue

            price = curr["close"]
            has_pos = self.has_position(symbol)

            if has_pos:
                sell_sigs = self._check_sell(
                    symbol, df, idx, price, cf, cs, pf, ps, curr, prev, current_time
                )
                signals.extend(sell_sigs)
            else:
                buy_sig = self._check_buy(
                    symbol, df, idx, price, cf, cs, pf, ps, curr, prev, current_time
                )
                if buy_sig:
                    signals.append(buy_sig)

        return signals

    # ----------------------------------------------------------------
    # 买入逻辑
    # ----------------------------------------------------------------

    def _check_buy(self, symbol, df, idx, price, cf, cs, pf, ps, curr, prev,
                    current_time) -> Optional[Signal]:
        """检查买入条件"""

        # 冷却期
        if self.sell_cooldown_bars > 0 and symbol in self._last_sell_idx:
            if idx - self._last_sell_idx[symbol] < self.sell_cooldown_bars:
                return None

        # === 核心信号 ===
        reason = ""

        # 信号A: MA金叉
        ma_cross = (pf <= ps and cf > cs)
        if ma_cross:
            reason = f"金叉 MA{self.fast}={cf:.2f} > MA{self.slow}={cs:.2f}"

        # 信号B: MACD金叉（可选）
        macd_cross = False
        if self.macd_buy_enabled and not ma_cross:
            macd_bar = curr.get('macd_bar', 0)
            prev_bar = prev.get('macd_bar', 0)
            if not pd.isna(macd_bar) and not pd.isna(prev_bar):
                if prev_bar <= 0 and macd_bar > 0:
                    macd_cross = True
                    reason = f"MACD金叉 DIF={curr.get('macd_dif', 0):.4f}"

        # 无信号则跳过
        if not ma_cross and not macd_cross:
            return None

        # === 过滤条件 ===

        # 趋势过滤
        if self.trend > 0:
            trend_col = f"ma{self.trend}"
            if trend_col in df.columns:
                ma_t = curr.get(trend_col)
                if ma_t is not None and not pd.isna(ma_t) and price <= ma_t:
                    self.logger.debug(f"{symbol} 趋势过滤 price={price:.2f} <= MA{self.trend}={ma_t:.2f}")
                    return None

        # RSI 超买过滤
        rsi = curr.get('rsi', 50)
        if self.rsi_buy_max < 100 and not pd.isna(rsi) and rsi > self.rsi_buy_max:
            vol = curr.get('volume', 0)
            vol_ma20 = curr.get('vol_ma20', 1)
            vol_r = vol / vol_ma20 if vol_ma20 and vol_ma20 > 0 else 0
            if vol_r < self.rsi_vol_exempt:
                return None

        # 成交量确认
        if self.vol_confirm_ratio > 0:
            vol = curr.get('volume', 0)
            vol_ma20 = curr.get('vol_ma20', 0)
            if vol_ma20 and vol_ma20 > 0:
                if vol / vol_ma20 < self.vol_confirm_ratio:
                    return None

        # 急涨过滤
        if self.surge_max_pct > 0 and self.surge_lookback > 0:
            lb = idx - self.surge_lookback
            if lb >= 0:
                past = df.iloc[lb]["close"]
                if (price - past) / past > self.surge_max_pct:
                    return None

        # 单日跳涨过滤
        if self.day_surge_pct > 0 and "pct_change" in df.columns:
            pct = curr.get("pct_change")
            if pct is not None and not pd.isna(pct):
                if pct > self.day_surge_pct * 100 and rsi > self.day_surge_rsi:
                    return None

        # R5 新增：动量过滤 —— 跳过近 N 日涨幅不足 momentum_min_pct 的弱势股
        if self.momentum_filter_enabled and self.momentum_lookback > 0:
            lb_idx = idx - self.momentum_lookback
            if lb_idx >= 0:
                past_price = df.iloc[lb_idx]["close"]
                momentum = (price - past_price) / past_price
                if momentum < self.momentum_min_pct:
                    self.logger.debug(
                        f"{symbol} 动量过滤 {self.momentum_lookback}日涨幅"
                        f"={momentum*100:.1f}% < {self.momentum_min_pct*100:.1f}%"
                    )
                    return None

        # === 下单（支持动量加权）===
        weight = 1.0
        if getattr(config, 'MOMENTUM_WEIGHT_ENABLED', False):
            mw_lookback = getattr(config, 'MOMENTUM_WEIGHT_LOOKBACK', 20)
            lb_idx = idx - mw_lookback
            if lb_idx >= 0:
                past_price = df.iloc[lb_idx]["close"]
                mom = (price - past_price) / past_price
                # 强势股 (涨幅>0) 加权 1.0~2.0；弱势股 减权 0.5~1.0
                weight = max(0.5, min(2.0, 1.0 + mom * 3))

        qty = self._calc_buy_qty(price, weight)
        if qty <= 0:
            return None

        self._entry_price[symbol] = price
        self._profit_stage[symbol] = 0

        trend_info = ""
        if self.trend > 0:
            trend_col = f"ma{self.trend}"
            if trend_col in df.columns:
                mt = curr.get(trend_col)
                if mt and not pd.isna(mt):
                    trend_info = f" | MA{self.trend}={mt:.2f}"

        return Signal(
            symbol    = symbol,
            action    = "BUY",
            quantity  = qty,
            price     = 0,
            timestamp = current_time,
            reason    = f"{reason}{trend_info}",
        )

    # ----------------------------------------------------------------
    # 卖出逻辑
    # ----------------------------------------------------------------

    def _check_sell(self, symbol, df, idx, price, cf, cs, pf, ps, curr, prev,
                     current_time) -> List[Signal]:
        """检查卖出条件"""
        signals = []
        pos = self.get_position(symbol)
        if not pos or pos.quantity <= 0:
            return signals

        # === 分级止盈 ===
        if self.partial_profit:
            entry = self._entry_price.get(symbol, pos.avg_cost)
            pnl_pct = (price - entry) / entry if entry > 0 else 0
            stage = self._profit_stage.get(symbol, 0)
            targets = [self.profit_target_1, self.profit_target_2, self.profit_target_3]

            if stage < len(targets) and pnl_pct >= targets[stage]:
                sell_qty = int(pos.quantity * self.profit_sell_pct / 100) * 100
                if sell_qty >= 100:
                    self._profit_stage[symbol] = stage + 1
                    signals.append(Signal(
                        symbol    = symbol,
                        action    = "SELL",
                        quantity  = sell_qty,
                        price     = 0,
                        timestamp = current_time,
                        reason    = f"止盈L{stage+1} +{pnl_pct*100:.1f}%",
                    ))
                    return signals

        # === MA死叉 → 全仓（支持确认天数和盈利保护）===
        is_dead_cross_today = (pf >= ps and cf < cs)
        is_below = (cf < cs)

        if is_dead_cross_today or (is_below and symbol in self._death_cross_count):
            # 更新死叉确认计数
            if is_dead_cross_today:
                self._death_cross_count[symbol] = 1
            elif is_below and symbol in self._death_cross_count:
                self._death_cross_count[symbol] += 1

            confirmed = self._death_cross_count.get(symbol, 0) >= self.death_cross_confirm

            if confirmed:
                # R5 盈利保护：持仓盈利超过阈值时，不因死叉卖出
                if self.profit_protect_enabled:
                    entry = self._entry_price.get(symbol, pos.avg_cost)
                    pnl_pct = (price - entry) / entry if entry > 0 else 0
                    if pnl_pct >= self.profit_protect_pct:
                        self.logger.debug(
                            f"{symbol} 盈利保护 +{pnl_pct*100:.1f}% >= "
                            f"{self.profit_protect_pct*100:.0f}%，跳过死叉卖出"
                        )
                        self._death_cross_count.pop(symbol, None)
                        # 不卖出，继续持有
                    else:
                        self._last_sell_idx[symbol] = idx
                        self._clean(symbol)
                        self._death_cross_count.pop(symbol, None)
                        signals.append(Signal(
                            symbol    = symbol,
                            action    = "SELL",
                            quantity  = pos.quantity,
                            price     = 0,
                            timestamp = current_time,
                            reason    = f"死叉 MA{self.fast}={cf:.2f} < MA{self.slow}={cs:.2f}",
                        ))
                        return signals
                else:
                    self._last_sell_idx[symbol] = idx
                    self._clean(symbol)
                    self._death_cross_count.pop(symbol, None)
                    signals.append(Signal(
                        symbol    = symbol,
                        action    = "SELL",
                        quantity  = pos.quantity,
                        price     = 0,
                        timestamp = current_time,
                        reason    = f"死叉 MA{self.fast}={cf:.2f} < MA{self.slow}={cs:.2f}",
                    ))
                    return signals
        else:
            # MA_FAST >= MA_SLOW，重置死叉确认计数
            self._death_cross_count.pop(symbol, None)

        # === MACD死叉 + RSI较高 → 全仓 ===
        macd_bar = curr.get('macd_bar', 0)
        prev_bar = prev.get('macd_bar', 0)
        rsi = curr.get('rsi', 50)
        if not pd.isna(macd_bar) and not pd.isna(prev_bar):
            if prev_bar > 0 and macd_bar <= 0 and not pd.isna(rsi) and rsi > self.macd_sell_rsi_min:
                self._last_sell_idx[symbol] = idx
                self._clean(symbol)
                signals.append(Signal(
                    symbol    = symbol,
                    action    = "SELL",
                    quantity  = pos.quantity,
                    price     = 0,
                    timestamp = current_time,
                    reason    = f"MACD死叉+RSI高({rsi:.0f})",
                ))
                return signals

        # === RSI 极端超买 + 盈利 → 减仓 ===
        if not pd.isna(rsi) and rsi > self.rsi_overbought:
            entry = self._entry_price.get(symbol, pos.avg_cost)
            pnl_pct = (price - entry) / entry if entry > 0 else 0
            if pnl_pct > 0.03:
                sell_qty = int(pos.quantity * 0.5 / 100) * 100
                if sell_qty >= 100:
                    signals.append(Signal(
                        symbol    = symbol,
                        action    = "SELL",
                        quantity  = sell_qty,
                        price     = 0,
                        timestamp = current_time,
                        reason    = f"RSI超买减仓({rsi:.0f}) +{pnl_pct*100:.1f}%",
                    ))

        return signals

    def _clean(self, symbol: str):
        self._profit_stage.pop(symbol, None)
        self._entry_price.pop(symbol, None)
        self._death_cross_count.pop(symbol, None)

    def _calc_buy_qty(self, price: float, weight: float = 1.0) -> int:
        budget = self.cash * self.position_pct * weight
        qty = int(budget / price / 100) * 100
        return qty
