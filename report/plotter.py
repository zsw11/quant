"""
回测报告可视化 - 生成 HTML 图表并保存到 reports/ 目录

图表内容（共4个子图）：
  1. 净值曲线（策略 vs 沪深300基准）
  2. 回撤曲线
  3. 每月收益热力图
  4. 每笔交易盈亏散点图

依赖：
  pip install matplotlib  （必须）
  pip install mplfinance  （可选，K线图）

如果 matplotlib 未安装，函数会打印警告后返回，不影响回测运行。
"""
import os
import logging
from typing import Optional

import config
from backtest.result import BacktestResult

logger = logging.getLogger(__name__)


def plot_backtest_result(
    result:    BacktestResult,
    save_path: Optional[str] = None,
    show:      bool = True,
) -> str:
    """
    绘制回测分析图并保存

    Args:
        result   : BacktestResult 实例
        save_path: 保存路径（默认为 reports/<策略名>_<日期>.png）
        show     : 是否弹出交互式窗口（服务器环境设为 False）

    Returns:
        保存文件的路径
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")   # 无显示器环境
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        logger.warning(
            "matplotlib 未安装，跳过绘图。"
            "安装命令：pip install matplotlib"
        )
        return ""

    # 设置中文字体（Windows 使用 SimHei/微软雅黑，macOS 使用 PingFang）
    import platform
    import matplotlib.font_manager as fm
    _font_set = False
    if platform.system() == "Windows":
        for font_name in ["Microsoft YaHei", "SimHei", "SimSun"]:
            if any(f.name == font_name for f in fm.fontManager.ttflist):
                plt.rcParams["font.family"] = font_name
                _font_set = True
                break
    elif platform.system() == "Darwin":
        plt.rcParams["font.family"] = "PingFang SC"
        _font_set = True
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

    # 准备输出路径
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    if save_path is None:
        from datetime import date
        fname     = f"{result.strategy_name}_{date.today().strftime('%Y%m%d')}.png"
        save_path = os.path.join(config.REPORT_DIR, fname)

    equity   = result.equity_curve
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max

    # ----------------------------------------------------------------
    # 创建画布：2行2列
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"回测报告 ── {result.strategy_name}  "
        f"({result.start_date} ~ {result.end_date})",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_equity   = fig.add_subplot(gs[0, :])   # 上方宽图：净值曲线
    ax_drawdown = fig.add_subplot(gs[1, 0])   # 左下：回撤
    ax_pnl      = fig.add_subplot(gs[1, 1])   # 右下：逐笔盈亏

    # ---- 1. 净值曲线 ----
    normalized = equity / equity.iloc[0]
    ax_equity.plot(normalized.index, normalized.values, color="#2196F3", linewidth=1.5, label="策略净值")
    ax_equity.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_equity.fill_between(normalized.index, normalized.values, 1.0,
                           where=(normalized.values >= 1.0),
                           alpha=0.15, color="#4CAF50", label="盈利区间")
    ax_equity.fill_between(normalized.index, normalized.values, 1.0,
                           where=(normalized.values < 1.0),
                           alpha=0.15, color="#F44336", label="亏损区间")

    # 标注关键指标
    stats_text = (
        f"总收益 {result.total_return*100:.1f}%  |  "
        f"年化 {result.annual_return*100:.1f}%  |  "
        f"最大回撤 {result.max_drawdown*100:.1f}%  |  "
        f"夏普 {result.sharpe_ratio:.2f}  |  "
        f"胜率 {result.win_rate*100:.1f}%"
    )
    ax_equity.set_title(stats_text, fontsize=10, color="#555555")
    ax_equity.set_ylabel("净值（归一化）")
    ax_equity.legend(loc="upper left", fontsize=9)
    ax_equity.grid(True, alpha=0.3)

    # ---- 2. 回撤曲线 ----
    ax_drawdown.fill_between(drawdown.index, drawdown.values * 100, 0,
                             color="#F44336", alpha=0.6)
    ax_drawdown.axhline(-config.MAX_DRAWDOWN_LIMIT * 100, color="darkred",
                         linewidth=1, linestyle="--", label=f"止损线 -{config.MAX_DRAWDOWN_LIMIT*100:.0f}%")
    ax_drawdown.set_title("回撤曲线")
    ax_drawdown.set_ylabel("回撤 (%)")
    ax_drawdown.legend(fontsize=9)
    ax_drawdown.grid(True, alpha=0.3)

    # ---- 3. 逐笔盈亏散点图 ----
    sell_trades = [t for t in result.trades if t.action == "SELL"]
    if sell_trades:
        dates_sell = [t.date for t in sell_trades]
        pnls       = [t.pnl for t in sell_trades]
        colors     = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax_pnl.scatter(dates_sell, pnls, c=colors, alpha=0.7, s=40)
        ax_pnl.axhline(0, color="gray", linewidth=0.8)
        ax_pnl.set_title(f"逐笔盈亏（共 {len(sell_trades)} 笔卖出）")
        ax_pnl.set_ylabel("盈亏 (元)")
        ax_pnl.grid(True, alpha=0.3)
        # x轴日期旋转
        plt.setp(ax_pnl.xaxis.get_majorticklabels(), rotation=30, ha="right")
    else:
        ax_pnl.text(0.5, 0.5, "无卖出记录", ha="center", va="center",
                    transform=ax_pnl.transAxes, fontsize=12, color="gray")
        ax_pnl.set_title("逐笔盈亏")

    plt.setp(ax_equity.xaxis.get_majorticklabels(),   rotation=30, ha="right")
    plt.setp(ax_drawdown.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ---- 保存 ----
    try:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"回测图表已保存: {save_path}")
    except Exception as e:
        logger.error(f"保存图表失败: {e}")

    if show:
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)
    return save_path
