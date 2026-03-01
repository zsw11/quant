"""
数据提供器 - 通过 AKShare 获取A股历史行情数据

数据源优先级：
  历史数据：东方财富（快但限流严重）→ 腾讯财经（不限流，兜底）
  实时价格：东方财富 → 腾讯财经
"""
import time
import logging
from typing import Optional

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


def _sym_to_tencent(symbol: str) -> str:
    """纯数字代码 → 腾讯格式：'600519' → 'sh600519', '300750' → 'sz300750'"""
    if symbol.startswith(("6", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"


class AKShareProvider:
    """
    A股历史数据提供器

    数据来源（自动切换）：
      1. 东方财富 stock_zh_a_hist  - 速度快，但限流严重
      2. 腾讯财经 stock_zh_a_daily - 不限流，作为备用
    """

    MAX_RETRIES  = 2   # 东方财富最多重试次数（减少等待时间）
    RETRY_DELAY  = 3   # 重试间隔
    TIMEOUT      = 15  # 超时秒数

    def get_stock_history(
        self,
        symbol:     str,
        start_date: str,
        end_date:   str,
        adjust:     str = "qfq",
    ) -> pd.DataFrame:
        """
        获取股票历史日线K线。

        先尝试东方财富（快），失败后自动切换到腾讯财经（稳定不限流）。
        """
        # ---- 方案 1：东方财富 ----
        df = self._try_eastmoney(symbol, start_date, end_date, adjust)
        if df is not None:
            return df

        # ---- 方案 2：腾讯财经（兜底，不限流）----
        logger.info(f"东方财富不可用，切换到腾讯数据源...")
        df = self._try_tencent(symbol, start_date, end_date, adjust)
        if df is not None:
            return df

        raise ConnectionError(
            f"无法下载 {symbol} 数据：东方财富和腾讯数据源均失败，请检查网络"
        )

    # ------------------------------------------------------------------
    # 东方财富（stock_zh_a_hist）
    # ------------------------------------------------------------------
    def _try_eastmoney(self, symbol, start_date, end_date, adjust) -> Optional[pd.DataFrame]:
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                if attempt > 1:
                    time.sleep(self.RETRY_DELAY * attempt)

                logger.info(
                    f"[东方财富] 下载 {symbol} {start_date}~{end_date}"
                    + (f"（第{attempt}次）" if attempt > 1 else "")
                )

                df = ak.stock_zh_a_hist(
                    symbol     = symbol,
                    period     = "daily",
                    start_date = start_date,
                    end_date   = end_date,
                    adjust     = adjust,
                    timeout    = self.TIMEOUT,
                )

                if df is None or df.empty:
                    raise ValueError("返回数据为空")

                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low",  "收盘": "close", "成交量": "volume",
                    "成交额": "amount", "涨跌幅": "pct_change",
                    "涨跌额": "change", "振幅": "amplitude", "换手率": "turnover",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()

                keep = [c for c in ["open","high","low","close","volume",
                                    "amount","pct_change","turnover"]
                        if c in df.columns]
                df = df[keep]

                logger.info(
                    f"[东方财富] 下载成功 {symbol}：{len(df)} 条 "
                    f"({df.index[0].date()} ~ {df.index[-1].date()})"
                )
                return df

            except Exception as e:
                logger.warning(f"[东方财富] {symbol} 第{attempt}次失败: {e}")

        return None

    # ------------------------------------------------------------------
    # 腾讯财经（stock_zh_a_daily）—— 不限流
    # ------------------------------------------------------------------
    def _try_tencent(self, symbol, start_date, end_date, adjust) -> Optional[pd.DataFrame]:
        try:
            tencent_sym = _sym_to_tencent(symbol)
            tencent_adjust = "" if adjust == "" else adjust

            logger.info(f"[腾讯] 下载 {symbol}（{tencent_sym}）...")

            df = ak.stock_zh_a_daily(
                symbol=tencent_sym,
                adjust=tencent_adjust,
            )

            if df is None or df.empty:
                raise ValueError("返回数据为空")

            # 腾讯返回的列名已经是英文，但 date 是普通列不是 index
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()

            keep = [c for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]
                    if c in df.columns]
            df = df[keep]

            # 按日期范围裁剪
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)
            df = df[(df.index >= start_ts) & (df.index <= end_ts)]

            if df.empty:
                raise ValueError(f"日期范围 {start_date}~{end_date} 内无数据")

            logger.info(
                f"[腾讯] 下载成功 {symbol}：{len(df)} 条 "
                f"({df.index[0].date()} ~ {df.index[-1].date()})"
            )
            return df

        except Exception as e:
            logger.warning(f"[腾讯] {symbol} 下载失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 实时价格
    # ------------------------------------------------------------------
    def get_realtime_price(self, symbol: str) -> float:
        """获取单只股票最新实时价格，失败返回 0.0"""
        prices = self.get_realtime_prices([symbol])
        return prices.get(symbol, 0.0)

    def get_realtime_prices(self, symbols: list) -> dict:
        """
        批量获取实时价格。
        优先东方财富（一次请求全部），失败后用腾讯逐只获取。
        """
        result = {s: 0.0 for s in symbols}

        # 方案1：东方财富实时快照（一次拿全市场）
        try:
            df = ak.stock_zh_a_spot_em()
            for symbol in symbols:
                row = df[df["代码"] == symbol]
                if not row.empty:
                    result[symbol] = float(row.iloc[0]["最新价"])
            # 检查是否全部获取到
            if all(result[s] > 0 for s in symbols):
                return result
        except Exception as e:
            logger.debug(f"东方财富实时接口失败: {e}")

        # 方案2：腾讯逐只获取（用最近一条日线收盘价作为近似实时价）
        for symbol in symbols:
            if result[symbol] > 0:
                continue
            try:
                tencent_sym = _sym_to_tencent(symbol)
                df = ak.stock_zh_a_daily(symbol=tencent_sym, adjust="qfq")
                if df is not None and not df.empty:
                    result[symbol] = float(df.iloc[-1]["close"])
                    logger.debug(f"[腾讯] {symbol} 实时价: {result[symbol]}")
            except Exception as e:
                logger.debug(f"[腾讯] {symbol} 实时价获取失败: {e}")

        return result
