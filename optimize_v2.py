"""
Focused parameter optimizer v2 - Tests key hypotheses to maximize returns
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
    """Run single backtest with given params, return result dict or None"""
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
    except Exception as e:
        return None


def evaluate(params, raw_data, symbols, periods):
    """Run across all periods and compute score"""
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

    # Score: reward average return, penalize losses
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

    # Load data
    raw_data = {}
    print("Loading data...")
    for sym in symbols:
        df = pd.read_pickle(f'F:/quant/data_cache/{sym}_20230903_20260227_qfq.pkl')
        raw_data[sym] = df[["open", "high", "low", "close", "volume"]].copy()
        print(f"  {sym}: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")

    # Base params (best known)
    base = {
        'MA_FAST': 7, 'MA_SLOW': 15, 'MA_TREND': 40,
        'TRAILING_STOP_PCT': 0.08, 'STOP_LOSS_PCT': 0.06,
        'SELL_COOLDOWN_BARS': 3, 'POSITION_PCT': 0.45,
        'MAX_POSITION_PCT': 0.55, 'MAX_TOTAL_POS_PCT': 0.95,
        'MAX_DRAWDOWN_LIMIT': 0.25, 'MIN_CASH_RESERVE': 0.05,
        'RSI_BUY_MAX': 80, 'VOL_CONFIRM_RATIO': 0.5,
        'SURGE_MAX_PCT': 0.12, 'SURGE_LOOKBACK': 3,
        'DAY_SURGE_PCT': 0, 'RSI_VOL_EXEMPT': 1.5,
        'MACD_BUY_ENABLED': True,
        'RSI_BOUNCE_ENABLED': False,
        'RSI_OVERSOLD': 28, 'RSI_BUY_THRESHOLD': 33,
        'RSI_OVERBOUGHT': 73,
        'PULLBACK_BUY_ENABLED': False,
        'PARTIAL_PROFIT_ENABLED': True,
        'PROFIT_TARGET_1': 0.06, 'PROFIT_TARGET_2': 0.12, 'PROFIT_TARGET_3': 0.20,
        'PROFIT_SELL_PCT_1': 0.30, 'PROFIT_SELL_PCT_2': 0.30,
        'MACD_SELL_RSI_MIN': 55,
    }

    # Generate targeted experiments
    experiments = []

    # Experiment 1: MACD sell RSI threshold
    for macd_rsi in [55, 60, 65, 70, 75, 80, 100]:
        p = base.copy()
        p['MACD_SELL_RSI_MIN'] = macd_rsi
        experiments.append(('MACD_sell_RSI', macd_rsi, p))

    # Experiment 2: RSI overbought reduction threshold
    for rsi_ob in [70, 73, 78, 83, 88, 95]:
        p = base.copy()
        p['RSI_OVERBOUGHT'] = rsi_ob
        experiments.append(('RSI_OB', rsi_ob, p))

    # Experiment 3: Profit targets
    for pt1, pt2, pt3 in [(0.04, 0.08, 0.15), (0.06, 0.12, 0.20), (0.08, 0.15, 0.25),
                           (0.10, 0.20, 0.35), (0.15, 0.30, 0.50)]:
        p = base.copy()
        p['PROFIT_TARGET_1'] = pt1
        p['PROFIT_TARGET_2'] = pt2
        p['PROFIT_TARGET_3'] = pt3
        experiments.append(('ProfitT', f'{pt1}/{pt2}/{pt3}', p))

    # Experiment 4: Profit sell percentages
    for ps in [0.20, 0.25, 0.30, 0.40, 0.50]:
        p = base.copy()
        p['PROFIT_SELL_PCT_1'] = ps
        p['PROFIT_SELL_PCT_2'] = ps
        experiments.append(('ProfSell%', ps, p))

    # Experiment 5: Trailing stop
    for trail in [0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.0]:
        p = base.copy()
        p['TRAILING_STOP_PCT'] = trail
        experiments.append(('Trail', trail, p))

    # Experiment 6: Stop loss
    for stop in [0.04, 0.05, 0.06, 0.08, 0.10, 0.0]:
        p = base.copy()
        p['STOP_LOSS_PCT'] = stop
        experiments.append(('StopL', stop, p))

    # Experiment 7: Position size
    for pos in [0.30, 0.45, 0.60, 0.80, 0.90]:
        p = base.copy()
        p['POSITION_PCT'] = pos
        p['MAX_POSITION_PCT'] = min(pos + 0.10, 0.95)
        experiments.append(('PosPct', pos, p))

    # Experiment 8: Disable partial profit
    p = base.copy()
    p['PARTIAL_PROFIT_ENABLED'] = False
    experiments.append(('NoProfit', '-', p))

    # Experiment 9: Disable MACD buy
    p = base.copy()
    p['MACD_BUY_ENABLED'] = False
    experiments.append(('NoMACD', '-', p))

    # Experiment 10: No trend filter
    p = base.copy()
    p['MA_TREND'] = 0
    experiments.append(('NoTrend', '-', p))

    # Experiment 11: Different MA combos
    for fast, slow, trend in [(3, 8, 20), (3, 10, 30), (5, 10, 30), (5, 12, 30),
                               (5, 15, 40), (7, 20, 40), (10, 20, 50), (10, 30, 60)]:
        p = base.copy()
        p['MA_FAST'] = fast
        p['MA_SLOW'] = slow
        p['MA_TREND'] = trend
        experiments.append(('MA', f'{fast}/{slow}/{trend}', p))

    # Experiment 12: Cooldown
    for cd in [0, 1, 2, 3, 5]:
        p = base.copy()
        p['SELL_COOLDOWN_BARS'] = cd
        experiments.append(('Cooldown', cd, p))

    # Experiment 13: Combined best candidates
    # Hypothesis: MACD sell RSI=100 (disable) + higher RSI OB + wider trail
    for macd_rsi in [80, 100]:
        for rsi_ob in [80, 88]:
            for trail in [0.08, 0.10, 0.12]:
                for pt1 in [0.06, 0.08]:
                    p = base.copy()
                    p['MACD_SELL_RSI_MIN'] = macd_rsi
                    p['RSI_OVERBOUGHT'] = rsi_ob
                    p['TRAILING_STOP_PCT'] = trail
                    p['PROFIT_TARGET_1'] = pt1
                    p['PROFIT_TARGET_2'] = pt1 * 2
                    p['PROFIT_TARGET_3'] = pt1 * 3
                    experiments.append(('Combo', f'MR{macd_rsi}/OB{rsi_ob}/T{trail}/P{pt1}', p))

    print(f"\nRunning {len(experiments)} experiments...")
    results = []

    for i, (name, val, params) in enumerate(experiments):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(experiments)}...")
        ev = evaluate(params, raw_data, symbols, periods)
        if ev:
            results.append((name, val, ev))

    results.sort(key=lambda x: x[2]['score'], reverse=True)

    print(f"\n{'='*120}")
    print(f"Results (sorted by score):")
    print(f"{'='*120}")
    print(f"{'Rank':>4} {'Experiment':<25} {'Value':<25} {'AvgAnn':>8} {'MinAnn':>8} {'WorstDD':>8} {'AvgShp':>7} {'Score':>7}")
    print("-" * 120)

    for i, (name, val, ev) in enumerate(results[:50]):
        print(f"{i+1:4d} {name:<25} {str(val):<25} {ev['avg_annual']*100:7.2f}% {ev['min_annual']*100:7.2f}% {ev['worst_dd']*100:7.2f}% {ev['avg_sharpe']:6.3f} {ev['score']:6.2f}")
        if i < 15:
            for label, d in ev['details'].items():
                print(f"      {label}: Ret={d['total_return']*100:6.2f}% DD={d['max_drawdown']*100:6.2f}% T={d['trades']:3d} WR={d['win_rate']*100:.0f}%")

    # Print bottom 10 for comparison
    print(f"\n--- Bottom 10 ---")
    for i, (name, val, ev) in enumerate(results[-10:]):
        idx = len(results) - 10 + i + 1
        print(f"{idx:4d} {name:<25} {str(val):<25} {ev['avg_annual']*100:7.2f}% {ev['min_annual']*100:7.2f}% {ev['worst_dd']*100:7.2f}% {ev['avg_sharpe']:6.3f} {ev['score']:6.2f}")


if __name__ == "__main__":
    main()
