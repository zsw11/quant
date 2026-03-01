"""
Focused optimizer v3 - Combine best findings from v2
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)

import config
from data.manager import DataManager
from backtest.engine import BacktestEngine
from risk.manager import RiskManager
from strategy.ma_cross import MACrossStrategy


def run_one(params, raw_data, symbols, start, end):
    for k, v in params.items():
        setattr(config, k, v)

    dm = DataManager(cache_dir=config.DATA_CACHE_DIR)
    warmup_days = max(params.get('MA_TREND', 0), 60) * 2
    load_start_ts = pd.to_datetime(start) - pd.Timedelta(days=warmup_days)

    data = {}
    for sym in symbols:
        raw = raw_data[sym]
        mask = (raw.index >= load_start_ts) & (raw.index <= pd.to_datetime(end))
        sliced = raw[mask].copy()
        if len(sliced) < 60:
            continue
        data[sym] = dm._add_indicators(sliced)

    if not data:
        return None

    strategy = MACrossStrategy(symbols=symbols)
    risk_manager = RiskManager(initial_capital=config.INITIAL_CAPITAL)
    engine = BacktestEngine()

    try:
        result = engine.run(strategy=strategy, data=data,
                            start_date=start, end_date=end,
                            risk_manager=risk_manager)
        return {
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'max_drawdown': result.max_drawdown,
            'sharpe': result.sharpe_ratio,
            'win_rate': result.win_rate,
            'trades': result.total_trades,
        }
    except:
        return None


def evaluate(params, raw_data, symbols, periods):
    period_results = {}
    for start, end, label in periods:
        r = run_one(params, raw_data, symbols, start, end)
        if r:
            period_results[label] = r

    if len(period_results) < 2:
        return None

    annuals = [r['annual_return'] for r in period_results.values()]
    returns = [r['total_return'] for r in period_results.values()]
    sharpes = [r['sharpe'] for r in period_results.values()]
    drawdowns = [r['max_drawdown'] for r in period_results.values()]

    avg_annual = np.mean(annuals)
    min_annual = min(annuals)
    worst_dd = min(drawdowns)
    avg_sharpe = np.mean(sharpes)
    score = avg_annual * 100 + avg_sharpe * 3 - abs(min(min(returns), 0)) * 150

    return {
        'avg_annual': avg_annual,
        'min_annual': min_annual,
        'worst_dd': worst_dd,
        'avg_sharpe': avg_sharpe,
        'score': score,
        'details': period_results,
    }


def main():
    symbols = ['600519', '300750']
    periods = [
        ('20220101', '20221231', '2022'),
        ('20230101', '20231231', '2023'),
        ('20240101', '20241231', '2024'),
    ]

    raw_data = {}
    print("Loading data...")
    for sym in symbols:
        df = pd.read_pickle(f'F:/quant/data_cache/{sym}_20230903_20260227_qfq.pkl')
        raw_data[sym] = df[["open", "high", "low", "close", "volume"]].copy()
        print(f"  {sym}: {len(df)} rows")

    # Base params
    base = {
        'MA_FAST': 7, 'MA_SLOW': 15, 'MA_TREND': 40,
        'TRAILING_STOP_PCT': 0.08, 'STOP_LOSS_PCT': 0.06,
        'SELL_COOLDOWN_BARS': 3,
        'MAX_TOTAL_POS_PCT': 0.95,
        'MAX_DRAWDOWN_LIMIT': 0.25, 'MIN_CASH_RESERVE': 0.05,
        'RSI_BUY_MAX': 80, 'VOL_CONFIRM_RATIO': 0.5,
        'SURGE_MAX_PCT': 0.12, 'SURGE_LOOKBACK': 3,
        'DAY_SURGE_PCT': 0, 'RSI_VOL_EXEMPT': 1.5,
        'MACD_BUY_ENABLED': True,
        'RSI_BOUNCE_ENABLED': False,
        'RSI_OVERSOLD': 28, 'RSI_BUY_THRESHOLD': 33,
        'PULLBACK_BUY_ENABLED': False,
        'PARTIAL_PROFIT_ENABLED': True,
        'PROFIT_TARGET_1': 0.06, 'PROFIT_TARGET_2': 0.12, 'PROFIT_TARGET_3': 0.20,
        'PROFIT_SELL_PCT_1': 0.30, 'PROFIT_SELL_PCT_2': 0.30,
        'MACD_SELL_RSI_MIN': 55,
    }

    experiments = []

    # Combine: Position size × RSI overbought × profit targets × MACD sell threshold
    for pos in [0.60, 0.70, 0.80, 0.90]:
        for rsi_ob in [73, 78, 83, 88]:
            for macd_rsi in [55, 65, 75]:
                for pt1 in [0.05, 0.06, 0.08]:
                    for trail in [0.06, 0.08, 0.10]:
                        p = base.copy()
                        p['POSITION_PCT'] = pos
                        p['MAX_POSITION_PCT'] = min(pos + 0.10, 0.95)
                        p['RSI_OVERBOUGHT'] = rsi_ob
                        p['MACD_SELL_RSI_MIN'] = macd_rsi
                        p['PROFIT_TARGET_1'] = pt1
                        p['PROFIT_TARGET_2'] = pt1 * 2
                        p['PROFIT_TARGET_3'] = pt1 * 3.5
                        p['TRAILING_STOP_PCT'] = trail
                        experiments.append((f'P{pos}/OB{rsi_ob}/MR{macd_rsi}/PT{pt1}/T{trail}', p))

    print(f"\nRunning {len(experiments)} experiments...")
    results = []

    for i, (name, params) in enumerate(experiments):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(experiments)}...")
        ev = evaluate(params, raw_data, symbols, periods)
        if ev:
            results.append((name, ev))

    results.sort(key=lambda x: x[1]['score'], reverse=True)

    print(f"\n{'='*130}")
    print(f"Top 30 results:")
    print(f"{'='*130}")
    print(f"{'Rank':>4} {'Config':<45} {'AvgAnn':>8} {'MinAnn':>8} {'WorstDD':>8} {'AvgShp':>7} {'Score':>7}")
    print("-" * 130)

    for i, (name, ev) in enumerate(results[:30]):
        print(f"{i+1:4d} {name:<45} {ev['avg_annual']*100:7.2f}% {ev['min_annual']*100:7.2f}% {ev['worst_dd']*100:7.2f}% {ev['avg_sharpe']:6.3f} {ev['score']:6.2f}")
        if i < 10:
            for label, d in ev['details'].items():
                print(f"      {label}: Ret={d['total_return']*100:6.2f}% DD={d['max_drawdown']*100:6.2f}% T={d['trades']:3d} WR={d['win_rate']*100:.0f}%")

    # Also run the top config across the full period
    print(f"\n{'='*80}")
    print("Running top 5 configs across FULL period (2022-2024)...")
    full_periods = [('20220101', '20241231', 'Full')]
    for i, (name, ev) in enumerate(results[:5]):
        # Extract params from name
        parts = name.split('/')
        pos = float(parts[0][1:])
        rsi_ob = int(parts[1][2:])
        macd_rsi = int(parts[2][2:])
        pt1 = float(parts[3][2:])
        trail = float(parts[4][1:])

        p = base.copy()
        p['POSITION_PCT'] = pos
        p['MAX_POSITION_PCT'] = min(pos + 0.10, 0.95)
        p['RSI_OVERBOUGHT'] = rsi_ob
        p['MACD_SELL_RSI_MIN'] = macd_rsi
        p['PROFIT_TARGET_1'] = pt1
        p['PROFIT_TARGET_2'] = pt1 * 2
        p['PROFIT_TARGET_3'] = pt1 * 3.5
        p['TRAILING_STOP_PCT'] = trail

        full_ev = evaluate(p, raw_data, symbols, full_periods)
        if full_ev:
            d = full_ev['details']['Full']
            print(f"  #{i+1} {name}")
            print(f"      Full 2022-2024: Ret={d['total_return']*100:6.2f}% Ann={d['annual_return']*100:6.2f}% DD={d['max_drawdown']*100:6.2f}% T={d['trades']:3d} WR={d['win_rate']*100:.0f}% Sharpe={d['sharpe']:.3f}")


if __name__ == "__main__":
    main()
