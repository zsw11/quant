"""
14-stock pool optimizer - R6
Optimize key parameters for the expanded 14-stock pool
"""
import sys, os, logging, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.CRITICAL)

import pandas as pd
import config
from data.manager import DataManager
from strategy.ma_cross import MACrossStrategy
from backtest.engine import BacktestEngine
from risk.manager import RiskManager

# Load data once
dm = DataManager()
warmup_days = max(getattr(config, "MA_TREND", 0), 60) * 2
load_start = (pd.to_datetime(config.BACKTEST_START) - pd.Timedelta(days=warmup_days)).strftime("%Y%m%d")
DATA = dm.get_multi(config.SYMBOLS_RAW, load_start, config.BACKTEST_END)
print(f"Loaded {len(DATA)} stocks")

# Parameter grid - focus on position sizing and risk for 14 stocks
GRID = {
    "MA_FAST":          [5, 10],
    "MA_SLOW":          [15, 20, 30],
    "MA_TREND":         [0],
    "POSITION_PCT":     [0.70, 0.80, 0.90],
    "MAX_POSITION_PCT": [0.20, 0.25, 0.30, 0.35],
    "SELL_COOLDOWN_BARS": [2, 3],
    "STOP_LOSS_PCT":    [0.12, 0.15, 0.20],
    "PARTIAL_PROFIT_ENABLED": [False],
    "MACD_BUY_ENABLED": [True],
}

keys = list(GRID.keys())
combos = list(itertools.product(*[GRID[k] for k in keys]))
print(f"Total combos: {len(combos)}")

results = []
best_return = -999
best_params = None
t0 = time.time()

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    
    # Apply params temporarily
    for k, v in params.items():
        setattr(config, k, v)
    
    try:
        strategy = MACrossStrategy(symbols=config.SYMBOLS_RAW)
        risk_mgr = RiskManager(initial_capital=config.INITIAL_CAPITAL)
        engine = BacktestEngine()
        result = engine.run(strategy, DATA, config.BACKTEST_START, config.BACKTEST_END, risk_mgr)
        
        ret = result.total_return * 100
        dd = result.max_drawdown * 100
        sharpe = result.sharpe_ratio
        trades = result.total_trades
        
        results.append({
            **params,
            "total_return": ret,
            "max_drawdown": dd,
            "sharpe": sharpe,
            "trades": trades,
        })
        
        if ret > best_return:
            best_return = ret
            best_params = params.copy()
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(combos)}] NEW BEST: {ret:.2f}% (DD={dd:.1f}%, Sharpe={sharpe:.3f}, "
                  f"Trades={trades}) | MA{params['MA_FAST']}/{params['MA_SLOW']} "
                  f"POS={params['POSITION_PCT']} MAX={params['MAX_POSITION_PCT']} "
                  f"SL={params['STOP_LOSS_PCT']} | {elapsed:.0f}s")
        
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(combos) - i - 1)
            print(f"  Progress: {i+1}/{len(combos)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    
    except Exception as e:
        if (i+1) % 100 == 0:
            print(f"  [{i+1}] Error: {e}")

# Save results
df = pd.DataFrame(results).sort_values("total_return", ascending=False)
os.makedirs("reports", exist_ok=True)
df.to_csv("reports/optimize_14stocks_r6.csv", index=False)

print(f"\n{'='*70}")
print(f"OPTIMIZATION COMPLETE: {len(results)} combos tested in {time.time()-t0:.0f}s")
print(f"{'='*70}")
print(f"\nBest return: {best_return:.2f}%")
print(f"Best params: {best_params}")
print(f"\nTop 10:")
print(df.head(10).to_string(index=False))

# Restore config
config.MA_FAST = 10
config.MA_SLOW = 20
config.POSITION_PCT = 0.90
config.MAX_POSITION_PCT = 0.35
config.SELL_COOLDOWN_BARS = 3
config.STOP_LOSS_PCT = 0.15
