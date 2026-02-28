"""
数据管理器 - 缓存管理 + 技术指标计算

职责：
  1. 调用 AKShareProvider 下载历史数据
  2. 本地磁盘缓存，避免重复下载
  3. 自动计算 MA / MACD / 布林带 / RSI / ATR 等技术指标
  4. 提供给策略和回测引擎使用
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .provider import AKShareProvider

logger = logging.getLogger(__name__)


class DataManager:
    """
    数据管理器

    使用方式：
        dm = DataManager()
        df = dm.get("600519", "20220101", "20241231")
        data = dm.get_multi(["600519","300750"], "20220101", "20241231")
    """

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = AKShareProvider()
        self._mem: Dict[str, pd.DataFrame] = {}   # 内存缓存

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取单只股票带技术指标的完整数据

        Args:
            symbol    : 股票代码，如 "600519"
            start_date: 开始日期 "YYYYMMDD"
            end_date  : 结束日期 "YYYYMMDD"
            adjust    : 复权 "qfq"前复权（推荐）
            use_cache : 是否使用本地缓存

        Returns:
            DataFrame（index=日期），含 open/high/low/close/volume
            以及 ma5/ma20/macd_bar/boll_upper 等技术指标列
        """
        key = f"{symbol}_{start_date}_{end_date}_{adjust}"

        # 1. 内存缓存精确命中
        if key in self._mem:
            return self._mem[key]

        # 2. 磁盘缓存精确命中
        cache_file = self.cache_dir / f"{key}.pkl"
        if use_cache and cache_file.exists():
            try:
                df = pd.read_pickle(cache_file)
                self._mem[key] = df
                logger.info(f"[缓存] 加载 {symbol}")
                return df
            except Exception:
                logger.warning(f"[缓存] 读取失败，重新下载 {symbol}")

        # 3. 查找已有缓存文件，判断是否能覆盖请求范围
        if use_cache:
            cached_df, cached_end = self._find_best_cache(symbol, start_date, end_date, adjust)

            if cached_df is not None:
                # 3a. 缓存完全覆盖请求范围 → 裁剪直接返回
                if cached_end >= end_date:
                    mask = (cached_df.index >= pd.to_datetime(start_date)) & \
                           (cached_df.index <= pd.to_datetime(end_date))
                    result = self._add_indicators(
                        cached_df[mask][["open","high","low","close","volume"]].copy()
                    )
                    self._mem[key] = result
                    return result

                # 3b. 缓存有头部数据但尾部缺口 → 尝试联网补增量
                logger.info(
                    f"[增量更新] {symbol} 本地数据到 {cached_end}，"
                    f"请求到 {end_date}，尝试联网补充..."
                )
                # 从缓存末日的下一天开始下载
                next_day = (pd.to_datetime(cached_end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
                try:
                    new_df = self.provider.get_stock_history(symbol, next_day, end_date, adjust)
                    # 拼接：旧缓存 OHLCV + 新增量，去重保留最新
                    combined = pd.concat([
                        cached_df[["open","high","low","close","volume"]],
                        new_df[["open","high","low","close","volume"]],
                    ]).sort_index()
                    combined = combined[~combined.index.duplicated(keep="last")]
                    df = self._add_indicators(combined.copy())
                    # 保存扩展后的缓存（文件名更新为新的结束日期）
                    new_end        = combined.index[-1].strftime("%Y%m%d")
                    new_cache_file = self.cache_dir / f"{symbol}_{start_date}_{new_end}_{adjust}.pkl"
                    df.to_pickle(new_cache_file)
                    logger.info(f"[增量更新] {symbol} 已拼接至 {new_end}，缓存已保存")
                    # 裁剪到请求范围后返回
                    mask = (df.index >= pd.to_datetime(start_date)) & \
                           (df.index <= pd.to_datetime(end_date))
                    result = df[mask].copy()
                    self._mem[key] = result
                    return result
                except Exception as e:
                    logger.warning(
                        f"[增量更新] {symbol} 联网补充失败（{e}），使用已有缓存部分数据"
                    )
                    # 联网失败：用已有缓存数据（尽管不完整）作为兜底
                    actual_end = min(end_date, cached_end)
                    mask = (cached_df.index >= pd.to_datetime(start_date)) & \
                           (cached_df.index <= pd.to_datetime(actual_end))
                    sliced = cached_df[mask]
                    if len(sliced) > 0:
                        result = self._add_indicators(
                            sliced[["open","high","low","close","volume"]].copy()
                        )
                        self._mem[key] = result
                        return result

        # 4. 无任何缓存 → 全量联网下载
        df = self.provider.get_stock_history(symbol, start_date, end_date, adjust)
        df = self._add_indicators(df)

        # 5. 保存磁盘缓存
        if use_cache:
            df.to_pickle(cache_file)

        self._mem[key] = df
        return df

    def _find_best_cache(
        self,
        symbol:     str,
        start_date: str,
        end_date:   str,
        adjust:     str,
    ) -> tuple:
        """
        在本地缓存目录中寻找头部覆盖 start_date 的最优缓存文件。

        Returns:
            (DataFrame, cached_end_str) 若找到有交集的缓存
            (None, None)               若无任何可用缓存
        """
        candidates = list(self.cache_dir.glob(f"{symbol}_*_{adjust}.pkl"))
        best_df  = None
        best_end = None   # 字符串 "YYYYMMDD"

        for f in candidates:
            # 文件名格式: symbol_startDate_endDate_adjust.pkl
            parts = f.stem.split("_")
            if len(parts) < 4:
                continue
            cached_start = parts[1]
            cached_end   = parts[2]

            # 必须与请求范围有交集，且缓存起点不晚于 start_date
            if cached_start > start_date or cached_end < start_date:
                continue

            try:
                df = pd.read_pickle(f)
                # 以缓存文件实际最后一条记录为准（比文件名更可靠）
                actual_end = df.index[-1].strftime("%Y%m%d")
                # 选覆盖最多的缓存（实际结束日最大）
                if best_end is None or actual_end > best_end:
                    best_df  = df
                    best_end = actual_end
            except Exception:
                continue

        return best_df, best_end

    def get_multi(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据

        Returns:
            dict {symbol: DataFrame}
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get(symbol, start_date, end_date, adjust)
            except Exception as e:
                logger.error(f"获取 {symbol} 失败，已跳过: {e}")
        return result

    def build_from_bars(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        将掘金推送的 Bar 数据（含 open/high/low/close/volume）
        计算技术指标后返回，供实盘策略使用

        Args:
            bars_df: 已按时间排好序的 DataFrame
        Returns:
            带技术指标的 DataFrame
        """
        return self._add_indicators(bars_df.copy())

    def clear_cache(self, symbol: Optional[str] = None):
        """清除缓存（symbol=None 清全部）"""
        if symbol:
            for f in self.cache_dir.glob(f"{symbol}_*.pkl"):
                f.unlink()
            self._mem = {k: v for k, v in self._mem.items()
                         if not k.startswith(symbol)}
        else:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
            self._mem.clear()
        logger.info(f"缓存已清除: {symbol or '全部'}")

    # ------------------------------------------------------------------
    # 技术指标计算（内部方法）
    # ------------------------------------------------------------------

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """在原始 OHLCV 数据上追加技术指标列"""

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # ---- 日涨跌幅（百分比，与AKShare原始列同口径）----
        df["pct_change"] = close.pct_change() * 100   # 单位：%，如 3.45 表示涨3.45%

        # ---- 移动平均线 ----
        # 基础周期 + 从 config 动态读取策略所需周期，避免遗漏
        import config as _cfg
        ma_periods = sorted(set([5, 10, 20, 60,
                                 getattr(_cfg, "MA_FAST",  10),
                                 getattr(_cfg, "MA_SLOW",  30),
                                 getattr(_cfg, "MA_TREND", 60)]) - {0})
        for p in ma_periods:
            df[f"ma{p}"] = close.rolling(p).mean()

        # ---- EMA ----
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["ema12"] = ema12
        df["ema26"] = ema26

        # ---- MACD ----
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        df["macd_dif"] = dif
        df["macd_dea"] = dea
        df["macd_bar"] = (dif - dea) * 2   # MACD柱

        # ---- RSI (14) ----
        df["rsi"] = self._rsi(close, 14)

        # ---- 布林带 (20, 2σ) ----
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        df["boll_mid"]   = mid
        df["boll_upper"] = mid + 2 * std
        df["boll_lower"] = mid - 2 * std

        # ---- ATR (14) ----
        df["atr"] = self._atr(high, low, close, 14)

        # ---- 成交量均线 ----
        df["vol_ma5"]  = volume.rolling(5).mean()
        df["vol_ma20"] = volume.rolling(20).mean()

        return df

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series,
             close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
