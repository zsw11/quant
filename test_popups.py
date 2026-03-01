"""
Test all 3 types of popup notifications
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
from notifier import notify_signal, notify_zone_analysis, _show_notification, _play_sound

# ======================================
# Test 1: Basic popup notification
# ======================================
print("=" * 60)
print("  TEST 1: Basic popup notification")
print("=" * 60)
print("  Sending a test popup now...")
_show_notification(
    "Test Popup",
    "This is a test notification from the quant system."
)
_play_sound("BUY")
print("  Test 1 done. Did you see a popup?")
print()
time.sleep(2)

# ======================================
# Test 2: Zone analysis popup
# ======================================
print("=" * 60)
print("  TEST 2: Zone analysis popup (startup pre-market)")
print("=" * 60)
print("  Sending zone analysis popup now...")

fake_zone_results = [
    {
        "symbol": "600519",
        "name": "贵州茅台",
        "zone": "SELL",
        "zone_cn": "处于卖出区间 - 空头趋势",
        "price": 1500.00,
        "score": -5,
        "max_score": 10,
        "details": ["  均线: MA5 < MA30 空头排列", "  RSI: 45.2 中性"],
    },
    {
        "symbol": "300750",
        "name": "宁德时代",
        "zone": "BUY",
        "zone_cn": "处于买入区间 - 多头趋势",
        "price": 280.00,
        "score": 7,
        "max_score": 10,
        "details": ["  均线: MA5 > MA30 多头排列", "  RSI: 55.3 中性"],
    },
    {
        "symbol": "601899",
        "name": "紫金矿业",
        "zone": "NEUTRAL",
        "zone_cn": "中性观望 - 趋势不明朗",
        "price": 22.50,
        "score": 1,
        "max_score": 10,
        "details": ["  均线: MA5 略高于 MA30", "  MACD: 柱缩小"],
    },
]
notify_zone_analysis(fake_zone_results)
print("  Test 2 done. Did you see zone analysis popup + console output?")
print()
time.sleep(2)

# ======================================
# Test 3: Trade signal popup
# ======================================
print("=" * 60)
print("  TEST 3: Trade signal popup (buy/sell alert)")
print("=" * 60)
print("  Sending BUY signal popup now...")

class FakeSignal:
    def __init__(self, symbol, action, quantity, reason):
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.reason = reason

buy_signal = FakeSignal("300750", "BUY", 200, "MA5上穿MA30 金叉")
notify_signal(buy_signal, {"300750": 280.00})
print()

time.sleep(2)
print("  Sending SELL signal popup now...")
sell_signal = FakeSignal("600519", "SELL", 100, "MA5下穿MA30 死叉")
notify_signal(sell_signal, {"600519": 1500.00})
print()

print("=" * 60)
print("  ALL TESTS COMPLETE")
print("=" * 60)
print("  If you saw 4 popups (1 basic + 1 zone + 1 buy + 1 sell),")
print("  the notification system is working correctly.")
