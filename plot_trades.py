"""
交易买卖点 K 线图生成器

读取回测生成的交易记录 CSV，为每只股票绘制完整的日 K 线图，
并在图上标注每一个买入点（绿色向上三角）和卖出点（红色向下三角）。

生成结果保存到 reports/ 目录：
  reports/kline_600519_贵州茅台.png
  reports/kline_300750_宁德时代.png

用法：
  python plot_trades.py
  python plot_trades.py --csv reports/trades_MACross_20260228.csv
"""
import os
import sys
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.collections import PatchCollection
import matplotlib.font_manager as fm
import platform
import logging

# ---------- 路径 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------- 中文字体 ----------
def _set_chinese_font():
    if platform.system() == "Windows":
        for name in ["Microsoft YaHei", "SimHei", "SimSun"]:
            if any(f.name == name for f in fm.fontManager.ttflist):
                plt.rcParams["font.family"] = name
                break
    elif platform.system() == "Darwin":
        plt.rcParams["font.family"] = "PingFang SC"
    plt.rcParams["axes.unicode_minus"] = False


# ---------- 找最新 CSV ----------
def find_latest_csv() -> str:
    pattern = os.path.join(ROOT, "reports", "trades_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"找不到交易记录 CSV，请先运行 main_backtest.py")
    return files[-1]


# ---------- K 线绘图核心 ----------
def draw_kline_with_trades(
    symbol:     str,
    name:       str,
    ohlcv:      pd.DataFrame,
    trades:     pd.DataFrame,
    save_dir:   str,
):
    """
    为单只股票绘制日 K 线图并标注买卖点

    Args:
        symbol : 股票代码，如 "600519"
        name   : 股票名称，如 "贵州茅台"
        ohlcv  : 日线数据 DataFrame（index=日期，含 open/high/low/close/volume）
        trades : 该股票的所有交易记录（含 日期/操作/价格/原因）
        save_dir: 图片保存目录
    """
    buys  = trades[trades["操作"] == "买入"].copy()
    sells = trades[trades["操作"] == "卖出"].copy()

    # 只取有交易的前后各 20 个交易日，但最少显示全部区间
    fig, (ax_k, ax_vol) = plt.subplots(
        2, 1,
        figsize=(20, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)

    dates = ohlcv.index
    x     = range(len(dates))          # 用整数序号做 x 轴，避免节假日空白
    date2x = {d: i for i, d in enumerate(dates)}

    # ---- 绘制 K 线 ----
    for i, (dt, row) in enumerate(ohlcv.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#E8231A" if c >= o else "#1DB954"   # 涨红跌绿

        # 实体
        rect = plt.Rectangle(
            (i - 0.35, min(o, c)),
            0.7,
            abs(c - o) if abs(c - o) > 0 else 0.01,
            color=color,
        )
        ax_k.add_patch(rect)

        # 上下影线
        ax_k.plot([i, i], [l, min(o, c)], color=color, linewidth=0.8)
        ax_k.plot([i, i], [max(o, c), h], color=color, linewidth=0.8)

    # ---- 均线 ----
    for period, color, lw in [(5, "#FF9800", 1.2), (20, "#2196F3", 1.2)]:
        ma = ohlcv["close"].rolling(period).mean()
        ax_k.plot(x, ma.values, color=color, linewidth=lw,
                  label=f"MA{period}", alpha=0.9)

    # ---- 买入点 ----
    for _, row in buys.iterrows():
        dt = row["日期"]
        if dt not in date2x:
            continue
        xi    = date2x[dt]
        price = row["价格"]
        reason = row.get("原因", "")
        ax_k.scatter(xi, price * 0.975, marker="^", color="#00C853",
                     s=180, zorder=5)
        ax_k.annotate(
            f"买\n{price:.1f}",
            xy=(xi, price * 0.975),
            xytext=(xi, price * 0.935),
            fontsize=7, color="#00C853", ha="center",
            arrowprops=dict(arrowstyle="-", color="#00C853", lw=0.6),
        )

    # ---- 卖出点 ----
    for _, row in sells.iterrows():
        dt = row["日期"]
        if dt not in date2x:
            continue
        xi    = date2x[dt]
        price = row["价格"]
        pnl   = row.get("盈亏", 0)
        pnl   = float(pnl) if str(pnl).strip() not in ("", "nan") else 0.0
        pnl_str = f"+{pnl:,.0f}" if pnl >= 0 else f"{pnl:,.0f}"
        ax_k.scatter(xi, price * 1.025, marker="v", color="#FF3D00",
                     s=180, zorder=5)
        ax_k.annotate(
            f"卖\n{price:.1f}\n({pnl_str})",
            xy=(xi, price * 1.025),
            xytext=(xi, price * 1.06),
            fontsize=7, color="#FF3D00", ha="center",
            arrowprops=dict(arrowstyle="-", color="#FF3D00", lw=0.6),
        )

    # ---- 连线：同一笔交易的买卖点 ----
    # 简单匹配：按时间顺序，买入等待对应的卖出
    trade_pairs = []
    buy_stack = []
    for _, row in trades.sort_values("日期").iterrows():
        if row["操作"] == "买入":
            buy_stack.append(row)
        elif row["操作"] == "卖出" and buy_stack:
            b = buy_stack.pop(0)
            trade_pairs.append((b, row))

    for b_row, s_row in trade_pairs:
        bx = date2x.get(b_row["日期"])
        sx = date2x.get(s_row["日期"])
        if bx is None or sx is None:
            continue
        bp = b_row["价格"]
        sp = s_row["价格"]
        ax_k.plot([bx, sx], [bp, sp],
                  linestyle="--", color="#9E9E9E", linewidth=0.6, alpha=0.6)

    # ---- K 线图装饰 ----
    ax_k.set_title(
        f"{name}（{symbol}）  日 K 线图  {dates[0].date()} ~ {dates[-1].date()}",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax_k.set_ylabel("价格（元）", fontsize=10)
    ax_k.legend(loc="upper left", fontsize=9)
    ax_k.grid(True, alpha=0.25, linestyle="--")

    # x 轴刻度：每隔约 20 个交易日显示一次日期
    step = max(1, len(dates) // 25)
    tick_positions = list(range(0, len(dates), step))
    tick_labels    = [dates[i].strftime("%Y-%m") for i in tick_positions]
    ax_k.set_xticks(tick_positions)
    ax_k.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax_k.set_xlim(-1, len(dates))

    # ---- 成交量柱 ----
    for i, (dt, row) in enumerate(ohlcv.iterrows()):
        o, c = row["open"], row["close"]
        color = "#E8231A" if c >= o else "#1DB954"
        ax_vol.bar(i, row["volume"], color=color, alpha=0.7, width=0.8)

    ax_vol.set_ylabel("成交量", fontsize=9)
    ax_vol.grid(True, alpha=0.2, linestyle="--")

    # ---- 图例 ----
    buy_patch  = mpatches.Patch(color="#00C853", label="买入点 ▲")
    sell_patch = mpatches.Patch(color="#FF3D00", label="卖出点 ▼")
    ax_k.legend(
        handles=[
            plt.Line2D([0], [0], color="#FF9800", lw=1.5, label="MA5"),
            plt.Line2D([0], [0], color="#2196F3", lw=1.5, label="MA20"),
            buy_patch, sell_patch,
        ],
        loc="upper left", fontsize=9,
    )

    # ---- 统计信息文本框 ----
    total_pnl   = sells["盈亏"].apply(
        lambda v: float(v) if str(v).strip() not in ("", "nan") else 0.0
    ).sum()
    win_count   = sells["盈亏"].apply(
        lambda v: 1 if str(v).strip() not in ("", "nan") and float(v) > 0 else 0
    ).sum()
    total_count = len(sells)
    win_rate    = win_count / total_count * 100 if total_count else 0

    info = (
        f"交易次数：买入 {len(buys)} 次 / 卖出 {len(sells)} 次\n"
        f"累计盈亏：{'+'if total_pnl>=0 else ''}{total_pnl:,.0f} 元\n"
        f"胜率：{win_rate:.1f}%（{int(win_count)}/{total_count}）"
    )
    ax_k.text(
        0.99, 0.98, info,
        transform=ax_k.transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#BDBDBD", alpha=0.85),
    )

    # ---- 保存 ----
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"kline_{symbol}_{name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存: {save_path}")
    return save_path


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="绘制交易买卖点 K 线图")
    parser.add_argument(
        "--csv", default=None,
        help="交易记录 CSV 路径（默认自动找 reports/ 下最新的）",
    )
    args = parser.parse_args()

    _set_chinese_font()

    # 1. 读取交易记录
    csv_path = args.csv or find_latest_csv()
    logger.info(f"读取交易记录: {csv_path}")
    trades_all = pd.read_csv(csv_path, encoding="utf-8-sig")
    trades_all["日期"] = pd.to_datetime(trades_all["日期"])

    # 2. 读取历史 K 线数据
    from data.manager import DataManager
    cache_dir = os.path.join(ROOT, config.DATA_CACHE_DIR.lstrip("./\\"))
    dm = DataManager(cache_dir=cache_dir)

    # 使用 config 里的回测日期（确保命中本地缓存）
    start_ext = config.BACKTEST_START
    end_ext   = config.BACKTEST_END

    logger.info(f"K 线范围: {start_ext} ~ {end_ext}")

    saved_files = []

    # 3. 按股票分组绘图
    for symbol in trades_all["代码"].unique():
        sym_str = str(symbol).zfill(6)
        name    = trades_all[trades_all["代码"] == symbol]["股票"].iloc[0]
        trades_sym = trades_all[trades_all["代码"] == symbol].copy()

        logger.info(f"\n处理 {name}（{sym_str}）：{len(trades_sym)} 笔交易")

        try:
            ohlcv = dm.get(
                symbol     = sym_str,
                start_date = start_ext,
                end_date   = end_ext,
            )
        except Exception as e:
            logger.error(f"获取 {sym_str} K 线数据失败: {e}")
            continue

        save_path = draw_kline_with_trades(
            symbol   = sym_str,
            name     = name,
            ohlcv    = ohlcv,
            trades   = trades_sym,
            save_dir = os.path.join(ROOT, "reports"),
        )
        saved_files.append(save_path)

    print("\n" + "=" * 50)
    print(f"共生成 {len(saved_files)} 张图表：")
    for p in saved_files:
        print(f"  {p}")
    print("=" * 50)


if __name__ == "__main__":
    main()
