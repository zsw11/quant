"""
参数优化器 - 网格搜索最佳策略参数组合
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import itertools
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)

import config
from data.manager import DataManager
from backtest.engine import BacktestEngine
from risk.manager import RiskManager
from strategy.ma_cross import MACrossStrategy


def run_backtest(params, raw_data, symbols, start, end):
    """运行单次回测"""
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


def grid_search():
    symbols = ['600519', '300750']

    periods = [
        ('20220101', '20221231', '2022'),
        ('20230101', '20231231', '2023'),
        ('20240101', '20241231', '2024'),
        ('20250101', '20260227', '2025'),
    ]

    # 预加载数据
    raw_data = {}
    print("Loading data...")
    for sym in symbols:
        df = pd.read_pickle(f'F:/quant/data_cache/{sym}_20230903_20260227_qfq.pkl')
        raw_data[sym] = df[["open", "high", "low", "close", "volume"]].copy()
        print(f"  {sym}: {len(df)} rows")

    # 参数组合
    configs = []

    # 策略配置组合
    for fast, slow, trend in [(3, 10, 0), (3, 10, 20), (3, 12, 0), (3, 12, 20), (3, 15, 0), (3, 15, 20), (3, 15, 40),
                               (5, 12, 0), (5, 12, 20), (5, 15, 0), (5, 15, 20), (5, 15, 40), (5, 20, 0), (5, 20, 40),
                               (7, 15, 0), (7, 15, 40), (7, 20, 0), (7, 20, 40)]:
        for trail in [0.04, 0.05, 0.06, 0.08, 0.10]:
            for stop in [0.04, 0.05, 0.06, 0.08]:
                for cooldown in [0, 1, 2, 3]:
                    for pos_pct in [0.45, 0.7, 0.9]:
                        for macd_en, rsi_en, profit_en in [(True, True, True), (False, False, False), (True, False, True), (True, True, False)]:
                            for pt1, pt2 in [(0.05, 0.12), (0.06, 0.15), (0.08, 0.18)]:
                                if not profit_en and pt1 != 0.05:
                                    continue  # skip unnecessary combos
                                configs.append({
                                    'MA_FAST': fast, 'MA_SLOW': slow, 'MA_TREND': trend,
                                    'TRAILING_STOP_PCT': trail, 'STOP_LOSS_PCT': stop,
                                    'SELL_COOLDOWN_BARS': cooldown, 'POSITION_PCT': pos_pct,
                                    'MAX_POSITION_PCT': 0.60, 'MAX_TOTAL_POS_PCT': 0.95,
                                    'MAX_DRAWDOWN_LIMIT': 0.25, 'MIN_CASH_RESERVE': 0.05,
                                    'RSI_BUY_MAX': 80, 'VOL_CONFIRM_RATIO': 0.5,
                                    'SURGE_MAX_PCT': 0.12, 'SURGE_LOOKBACK': 3,
                                    'DAY_SURGE_PCT': 0, 'RSI_VOL_EXEMPT': 1.5,
                                    'MACD_BUY_ENABLED': macd_en,
                                    'RSI_BOUNCE_ENABLED': rsi_en,
                                    'RSI_OVERSOLD': 28, 'RSI_BUY_THRESHOLD': 33,
                                    'RSI_OVERBOUGHT': 73,
                                    'PULLBACK_BUY_ENABLED': False,
                                    'PARTIAL_PROFIT_ENABLED': profit_en,
                                    'PROFIT_TARGET_1': pt1, 'PROFIT_TARGET_2': pt2,
                                    'PROFIT_SELL_PCT_1': 0.3, 'PROFIT_SELL_PCT_2': 0.3,
                                })

    # Too many combos, sample randomly
    import random
    random.seed(42)
    if len(configs) > 2000:
        configs = random.sample(configs, 2000)
    
    print(f"Testing {len(configs)} parameter combinations...")

    results = []
    for i, params in enumerate(configs):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(configs)}...")

        period_results = []
        for start, end, label in periods:
            r = run_backtest(params, raw_data, symbols, start, end)
            if r:
                period_results.append((label, r))

        if len(period_results) < len(periods):
            continue

        avg_annual = np.mean([r['annual_return'] for _, r in period_results])
        worst_dd = min(r['max_drawdown'] for _, r in period_results)
        avg_sharpe = np.mean([r['sharpe'] for _, r in period_results])
        min_return = min(r['total_return'] for _, r in period_results)
        max_annual = max(r['annual_return'] for _, r in period_results)
        min_annual = min(r['annual_return'] for _, r in period_results)

        # Score: maximize average returns, penalize losses heavily
        score = avg_annual * 100 + avg_sharpe * 5 - abs(min(min_return, 0)) * 200

        results.append({
            'params': params,
            'avg_annual': avg_annual,
            'max_annual': max_annual,
            'min_annual': min_annual,
            'worst_dd': worst_dd,
            'avg_sharpe': avg_sharpe,
            'min_return': min_return,
            'score': score,
            'details': {label: r for label, r in period_results},
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n{'='*140}")
    print(f"Grid search complete. Tested {len(results)} valid combinations.")
    print(f"{'='*140}")
    
    header = f"{'Rank':>4} {'F':>2} {'S':>2} {'T':>2} {'Trail':>5} {'Stop':>4} {'CD':>2} {'Pos':>4} {'MACD':>4} {'RSI':>3} {'Prof':>4} {'PT1':>4} | {'AvgAnn':>8} {'MinAnn':>8} {'MaxAnn':>8} {'WorstDD':>8} {'AvgShp':>7} | {'Score':>7}"
    print(header)
    print("-" * 140)
    
    for i, r in enumerate(results[:40]):
        p = r['params']
        print(f"{i+1:4d} {p['MA_FAST']:2d} {p['MA_SLOW']:2d} {p['MA_TREND']:2d} {p['TRAILING_STOP_PCT']:5.2f} {p['STOP_LOSS_PCT']:4.2f} {p['SELL_COOLDOWN_BARS']:2d} {p['POSITION_PCT']:4.2f} {'Y' if p['MACD_BUY_ENABLED'] else 'N':>4} {'Y' if p['RSI_BOUNCE_ENABLED'] else 'N':>3} {'Y' if p['PARTIAL_PROFIT_ENABLED'] else 'N':>4} {p.get('PROFIT_TARGET_1',0):4.2f} | "
              f"{r['avg_annual']*100:7.2f}% {r['min_annual']*100:7.2f}% {r['max_annual']*100:7.2f}% {r['worst_dd']*100:7.2f}% {r['avg_sharpe']:6.3f} | {r['score']:6.2f}")
        if i < 10:
            for label, d in r['details'].items():
                print(f"      {label}: Ret={d['total_return']*100:7.2f}% Ann={d['annual_return']*100:7.2f}% DD={d['max_drawdown']*100:7.2f}% Trades={d['trades']:3d} WR={d['win_rate']*100:.0f}%")

    return results


if __name__ == "__main__":
    grid_search()
