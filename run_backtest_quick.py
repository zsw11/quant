"""Quick backtest runner - outputs only key metrics"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Only suppress non-critical logs
logging.basicConfig(level=logging.WARNING, format="%(message)s")

import config
from data.manager import DataManager
from strategy.ma_cross import MACrossStrategy
from backtest.engine import BacktestEngine
from risk.manager import RiskManager

import pandas as pd
dm = DataManager()
warmup_days = max(getattr(config, "MA_TREND", 0), 60) * 2
load_start = (pd.to_datetime(config.BACKTEST_START) - pd.Timedelta(days=warmup_days)).strftime("%Y%m%d")
data = dm.get_multi(config.SYMBOLS_RAW, load_start, config.BACKTEST_END)
print(f"Loaded {len(data)} stocks, checking data sizes...")
for sym, df in data.items():
    print(f"  {sym}: {len(df)} bars")

strategy = MACrossStrategy(symbols=config.SYMBOLS_RAW)
risk_mgr = RiskManager(initial_capital=config.INITIAL_CAPITAL)
engine = BacktestEngine()
result = engine.run(strategy, data, config.BACKTEST_START, config.BACKTEST_END, risk_mgr)

print(f"\n=== BACKTEST RESULTS ({len(config.SYMBOLS_RAW)} stocks) ===")
print(f"Period:         {config.BACKTEST_START} ~ {config.BACKTEST_END}")
print(f"Total Return:   {result.total_return*100:.2f}%")
print(f"Annual Return:  {result.annual_return*100:.2f}%")
print(f"Max Drawdown:   {result.max_drawdown*100:.2f}%")
print(f"Sharpe Ratio:   {result.sharpe_ratio:.3f}")
print(f"Calmar Ratio:   {result.calmar_ratio:.3f}")
print(f"Win Rate:       {result.win_rate*100:.1f}%")
print(f"Total Trades:   {result.total_trades}")
print(f"Total Comm:     {result.total_commission:,.0f}")
print(f"Final Value:    {result.final_value:,.0f}")
