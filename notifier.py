"""
消息提醒模块 - 交易信号到达时弹窗+声音+红色控制台通知

功能：
  1. 红色醒目控制台输出（ANSI 颜色，兼容 Windows Terminal / CMD）
  2. Windows 系统弹窗通知（win10toast / plyer / ctypes MessageBox）
  3. 系统提示音（winsound）
  4. 日志记录（所有信号无论模式都会写日志）

使用方式：
    from notifier import notify_signal
    notify_signal(signal, prices)              # 弹窗+声音+红色提醒
    notify_signal(signal, prices, silent=True)  # 仅日志，不弹窗不声音

配置：
    config.SIGNAL_MODE = "notify"  -> 信号只提醒不下单（默认）
    config.SIGNAL_MODE = "auto"    -> 信号自动下单
"""
import logging
import os
import platform
import sys
from datetime import datetime
from typing import Dict, Optional

import config

logger = logging.getLogger("notifier")

# ---- ANSI 颜色码 ----
_RED     = "\033[91m"
_GREEN   = "\033[92m"
_YELLOW  = "\033[93m"
_BOLD    = "\033[1m"
_RESET   = "\033[0m"


def _enable_ansi_colors():
    """Windows 10+ 需要启用虚拟终端序列才能显示 ANSI 颜色"""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11
        handle = kernel32.GetStdHandle(-11)
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass

_enable_ansi_colors()


def notify_signal(
    signal,
    prices: Dict[str, float],
    silent: bool = False,
):
    """
    发送交易信号提醒

    Args:
        signal : Signal 对象（含 symbol, action, quantity, reason）
        prices : {symbol: price} 当前价格
        silent : True=仅日志不弹窗（用于回测模式）
    """
    sym   = signal.symbol
    name  = config.SYMBOL_NAMES.get(sym, sym)
    price = prices.get(sym, 0)
    action_cn = "买入" if signal.action == "BUY" else "卖出"

    # 构建提醒文本
    title = f"[交易信号] {action_cn} {name}"
    body  = (
        f"股票: {name} ({sym})\n"
        f"操作: {action_cn}\n"
        f"数量: {int(signal.quantity)} 股\n"
        f"当前价: {price:.2f}\n"
        f"金额: {price * signal.quantity:,.0f} 元\n"
        f"原因: {signal.reason}\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # 1. 始终写日志
    log_line = (
        f"{'='*50}\n"
        f"  {title}\n"
        f"  {body.replace(chr(10), ' | ')}\n"
        f"{'='*50}"
    )
    logger.info(log_line)

    if silent:
        return

    # 2. 播放提示音
    _play_sound(signal.action)

    # 3. 弹窗通知
    _show_notification(title, body)

    # 4. 控制台醒目输出
    _console_alert(title, body, action=signal.action)


def _play_sound(action: str):
    """播放系统提示音"""
    if platform.system() != "Windows":
        return
    try:
        import winsound
        if action == "BUY":
            # 买入：两声短促提示
            winsound.Beep(800, 300)
            winsound.Beep(1000, 300)
        else:
            # 卖出：三声急促提示
            winsound.Beep(1000, 200)
            winsound.Beep(800, 200)
            winsound.Beep(600, 300)
    except Exception:
        pass  # 非 Windows 或 winsound 不可用，静默忽略


def _show_notification(title: str, body: str):
    """
    弹出系统通知（Windows 优先用 win10toast，降级为 ctypes MessageBox）
    """
    if platform.system() != "Windows":
        return

    # 方案1: 尝试 win10toast（现代 Windows 10/11 通知中心）
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(
            title,
            body.replace("\n", " | "),
            duration=10,        # 显示10秒
            threaded=True,      # 不阻塞主线程
        )
        return
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"win10toast 通知失败: {e}")

    # 方案2: 尝试 plyer（跨平台通知）
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=body.replace("\n", " | "),
            timeout=10,
        )
        return
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"plyer 通知失败: {e}")

    # 方案3: ctypes MessageBox（阻塞式弹窗，最可靠的兜底）
    try:
        import ctypes
        # MB_OK | MB_ICONINFORMATION | MB_SYSTEMMODAL（置顶）
        MB_OK = 0x00000000
        MB_ICONINFO = 0x00000040
        MB_SYSTEMMODAL = 0x00001000
        ctypes.windll.user32.MessageBoxW(
            0,
            body.replace("\n", "\r\n"),
            title,
            MB_OK | MB_ICONINFO | MB_SYSTEMMODAL,
        )
    except Exception as e:
        logger.warning(f"弹窗通知全部失败: {e}，请查看日志和控制台输出")


def _console_alert(title: str, body: str, action: str = "BUY"):
    """
    在控制台输出醒目的红色/绿色提醒框

    买入信号 → 红底白字
    卖出信号 → 绿底白字
    """
    if action == "BUY":
        color = _RED
        action_icon = ">>> BUY  <<<  买入"
    else:
        color = _GREEN
        action_icon = ">>> SELL <<<  卖出"

    width = 64
    border = "=" * width

    print()
    print(f"{color}{_BOLD}{border}{_RESET}")
    print(f"{color}{_BOLD}||  {action_icon:^{width - 6}}  ||{_RESET}")
    print(f"{color}{_BOLD}{border}{_RESET}")
    print(f"{color}{_BOLD}  {title}{_RESET}")
    print(f"{color}{'-' * width}{_RESET}")
    for line in body.split("\n"):
        print(f"{color}{_BOLD}  {line}{_RESET}")
    print(f"{color}{'-' * width}{_RESET}")
    if getattr(config, 'SIGNAL_MODE', 'auto') == 'notify':
        print(f"{_YELLOW}{_BOLD}  --> 仅提醒模式，请在交易终端手动操作 <--{_RESET}")
    else:
        print(f"{_YELLOW}{_BOLD}  --> 已自动下单 <--{_RESET}")
    print(f"{color}{_BOLD}{border}{_RESET}")
    print()


# ---- ANSI 背景色 ----
_BG_RED   = "\033[41m"
_BG_GREEN = "\033[42m"
_BG_YELLOW = "\033[43m"
_CYAN     = "\033[96m"
_WHITE    = "\033[97m"


def notify_zone_analysis(analysis_results: list):
    """
    启动时的股票区间分析提醒 - 控制台彩色输出 + 弹窗 + 声音

    Args:
        analysis_results: list of dict, 每个元素:
            {
                "symbol": str,
                "name": str,
                "zone": "BUY" | "SELL" | "NEUTRAL",
                "zone_cn": str,  # 中文描述
                "price": float,
                "details": list[str],  # 各维度分析
            }
    """
    if not analysis_results:
        return

    width = 68
    border = "=" * width

    # 统计
    buy_count = sum(1 for r in analysis_results if r["zone"] == "BUY")
    sell_count = sum(1 for r in analysis_results if r["zone"] == "SELL")
    neutral_count = sum(1 for r in analysis_results if r["zone"] == "NEUTRAL")

    # ---- 控制台输出 ----
    print()
    print(f"{_CYAN}{_BOLD}{border}{_RESET}")
    print(f"{_CYAN}{_BOLD}||  {'盘前分析 - 各股当前所处区间':^{width - 16}}  ||{_RESET}")
    print(f"{_CYAN}{_BOLD}{border}{_RESET}")
    print(
        f"{_CYAN}  "
        f"{_RED}买入区间: {buy_count} 只{_RESET}  "
        f"{_GREEN}卖出区间: {sell_count} 只{_RESET}  "
        f"{_YELLOW}中性观望: {neutral_count} 只{_RESET}"
    )
    print(f"{_CYAN}{'-' * width}{_RESET}")

    for r in analysis_results:
        zone = r["zone"]
        if zone == "BUY":
            color = _RED
            icon = " BUY "
        elif zone == "SELL":
            color = _GREEN
            icon = " SELL"
        else:
            color = _YELLOW
            icon = " WAIT"

        # 股票名 + 价格行
        header = f"[{icon}] {r['name']}({r['symbol']})  {r['price']:.2f} 元"
        print(f"{color}{_BOLD}  {header}{_RESET}")
        print(f"{color}        {r['zone_cn']}{_RESET}")

        # 各维度详情
        for detail in r["details"]:
            print(f"        {detail}")

        print(f"{_CYAN}{'-' * width}{_RESET}")

    print(f"{_CYAN}{_BOLD}{border}{_RESET}")
    print()

    # ---- 声音提示 ----
    _play_sound("BUY")  # 启动提示音

    # ---- 弹窗通知 ----
    title = f"盘前分析: {buy_count}只买入区 {sell_count}只卖出区 {neutral_count}只观望"
    body_lines = []
    for r in analysis_results:
        zone_icon = {"BUY": "[买]", "SELL": "[卖]", "NEUTRAL": "[观望]"}[r["zone"]]
        body_lines.append(
            f"{zone_icon} {r['name']} {r['price']:.2f} - {r['zone_cn']}"
        )
    body = "\n".join(body_lines)
    _show_notification(title, body)
