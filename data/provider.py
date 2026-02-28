"""
数据提供器 - 通过 AKShare 获取A股历史行情数据

底层数据源：东方财富 push2his.eastmoney.com
限流策略：同一 IP 短时间频繁请求会被断开，因此加入重试+冷却间隔
"""
import time
import logging
from typing import Optional

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

_NETWORK_HELP = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [网络错误] 无法从东方财富下载数据（已重试3次）

  数据源：push2his.eastmoney.com（东方财富行情服务器）
  东方财富会对频繁访问的 IP 限流，通常几分钟后恢复。

  排查步骤：
  1. 等 2～3 分钟后重新运行
  2. 如果一直失败，尝试关闭 VPN/代理 后重新运行
  3. 或切换到手机热点
  4. 升级 AKShare：pip install --upgrade akshare

  如果只想用已有缓存数据（不联网），
  把 config.py 的日期改为缓存已有的范围即可，
  缓存文件在 data_cache/ 目录，文件名包含日期范围。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class AKShareProvider:
    """
    A股历史数据提供器

    数据来源：AKShare → 东方财富（每天更新一次）
    支持：日线K线、前复权/后复权/不复权
    """

    MAX_RETRIES  = 3   # 最多重试次数
    RETRY_DELAY  = 5   # 每次重试前等待秒数（递增）
    TIMEOUT      = 30  # 单次请求超时秒数

    def get_stock_history(
        self,
        symbol:     str,
        start_date: str,
        end_date:   str,
        adjust:     str = "qfq",
    ) -> pd.DataFrame:
        """
        获取股票历史日线K线，失败自动重试3次，每次间隔递增。

        Args:
            symbol    : 股票代码，如 "600519"
            start_date: 开始日期 "YYYYMMDD"
            end_date  : 结束日期 "YYYYMMDD"
            adjust    : "qfq" 前复权 | "hfq" 后复权 | "" 不复权
        """
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                if attempt > 1:
                    wait = self.RETRY_DELAY * attempt
                    logger.warning(f"  {wait}秒后进行第 {attempt} 次重试...")
                    time.sleep(wait)

                logger.info(
                    f"下载 {symbol} {start_date}~{end_date}"
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
                    raise ValueError("返回数据为空，可能该日期范围内无交易数据")

                # 统一列名
                df = df.rename(columns={
                    "日期":   "date",
                    "开盘":   "open",
                    "最高":   "high",
                    "最低":   "low",
                    "收盘":   "close",
                    "成交量": "volume",
                    "成交额": "amount",
                    "涨跌幅": "pct_change",
                    "涨跌额": "change",
                    "振幅":   "amplitude",
                    "换手率": "turnover",
                })

                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()

                keep = [c for c in ["open","high","low","close","volume",
                                    "amount","pct_change","turnover"]
                        if c in df.columns]
                df = df[keep]

                logger.info(
                    f"下载成功 {symbol}：{len(df)} 条 "
                    f"({df.index[0].date()} ~ {df.index[-1].date()})"
                )
                return df

            except Exception as e:
                last_error = e
                logger.warning(f"下载 {symbol} 第{attempt}次失败: {e}")

        # 全部重试失败
        print(_NETWORK_HELP)
        logger.error(f"下载 {symbol} 失败，已重试 {self.MAX_RETRIES} 次")
        raise ConnectionError(
            f"无法下载 {symbol} 数据，请检查网络。原始错误: {last_error}"
        )

    def get_realtime_price(self, symbol: str) -> float:
        """获取单只股票最新实时价格，失败返回 0.0"""
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == symbol]
            if not row.empty:
                return float(row.iloc[0]["最新价"])
        except Exception as e:
            logger.warning(f"获取实时价格失败 {symbol}: {e}")
        return 0.0

    def get_realtime_prices(self, symbols: list) -> dict:
        """批量获取实时价格，返回 {symbol: price}，失败的返回 0.0"""
        result = {s: 0.0 for s in symbols}
        try:
            df = ak.stock_zh_a_spot_em()
            for symbol in symbols:
                row = df[df["代码"] == symbol]
                if not row.empty:
                    result[symbol] = float(row.iloc[0]["最新价"])
        except Exception as e:
            logger.warning(f"批量获取实时价格失败: {e}")
        return result
