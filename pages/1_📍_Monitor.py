"""
Pairs Position Monitor v5.3
v5.3: Hurst warning fix (threshold 0.48), full open positions CSV
v5.2: Full Open Pos CSV, adaptive stop, MTF sync

Запуск: streamlit run pairs_position_monitor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
import json
import os
# v27: Unified config
try:
    from config_loader import CFG
except ImportError:
    def CFG(section, key=None, default=None):
        _d = {'strategy': {'entry_z': 1.8, 'exit_z': 0.8, 'stop_z_offset': 2.0,
              'min_stop_z': 4.0, 'max_hold_hours': 72, 'commission_pct': 0.10},
              'monitor': {'refresh_interval_sec': 120, 'exit_z_target': 0.5,
              'pnl_stop_pct': -5.0, 'hurst_critical': 0.50, 'hurst_warning': 0.48,
              'hurst_border': 0.45, 'pvalue_warning': 0.10, 'correlation_warning': 0.20,
              'trailing_z_bounce': 0.8, 'time_warning_ratio': 1.0,
              'time_exit_ratio': 1.5, 'time_critical_ratio': 2.0,
              'overshoot_deep_z': 1.0, 'pnl_trailing_threshold': 0.5,
              'pnl_trailing_fraction': 0.4}}
        if key is None:
            return _d.get(section, {})
        return _d.get(section, {}).get(key, default)
from datetime import datetime, timedelta, timezone

MSK = timezone(timedelta(hours=3))
def now_msk():
    return datetime.now(MSK)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint

# ═══════════════════════════════════════════════════════
# DRY: Import shared utilities from analysis module
# ═══════════════════════════════════════════════════════
try:
    from mean_reversion_analysis import (
        calculate_hurst_exponent,
        calculate_hurst_ema,
        calculate_adaptive_robust_zscore,
        calculate_garch_zscore,
        calc_halflife_from_spread,
        assess_entry_readiness,
        check_pnl_z_disagreement,
        smart_exit_analysis,
        z_velocity_analysis,
    )
    _USE_MRA = True
except ImportError:
    _USE_MRA = False

# v5.3: assess_entry_readiness — imported from analysis module when available
# Local fallback always defined (used when analysis module unavailable)

def assess_entry_readiness(p):
    """
    v8.0: Единая оценка с HARD HURST GATE.
    Hurst ≥ 0.45 → max УСЛОВНО. Hurst=0.500 fallback → max СЛАБЫЙ.
    """
    mandatory = [
        ('Статус ≥ READY', p.get('signal', 'NEUTRAL') in ('SIGNAL', 'READY'), p.get('signal', 'NEUTRAL')),
        ('|Z| ≥ Thr', abs(p.get('zscore', 0)) >= p.get('threshold', 2.0),
         f"|{p.get('zscore',0):.2f}| vs {p.get('threshold',2.0)}"),
        ('Q ≥ 50', p.get('quality_score', 0) >= 50, f"Q={p.get('quality_score', 0)}"),
        ('Dir ≠ NONE', p.get('direction', 'NONE') != 'NONE', p.get('direction', 'NONE')),
    ]
    all_mandatory = all(m[1] for m in mandatory)
    
    fdr_ok = p.get('fdr_passed', False)
    stab_ok = p.get('stability_passed', 0) >= 3
    hurst_val = p.get('hurst', 0.5)
    hurst_ok = hurst_val < 0.35
    hurst_is_fallback = hurst_val == 0.5
    
    optional = [
        ('FDR ✅', fdr_ok, '✅' if fdr_ok else '❌'),
        ('Conf=HIGH', p.get('confidence', 'LOW') == 'HIGH', p.get('confidence', 'LOW')),
        ('S ≥ 60', p.get('signal_score', 0) >= 60, f"S={p.get('signal_score', 0)}"),
        ('ρ ≥ 0.5', p.get('correlation', 0) >= 0.5, f"ρ={p.get('correlation', 0):.2f}"),
        ('Stab ≥ 3/4', stab_ok, f"{p.get('stability_passed',0)}/{p.get('stability_total',4)}"),
        ('Hurst < 0.35', hurst_ok, f"H={hurst_val:.3f}"),
    ]
    opt_count = sum(1 for _, met, _ in optional if met)
    fdr_bypass = (not fdr_ok and p.get('quality_score', 0) >= 70 and
                  stab_ok and p.get('adf_passed', False) and hurst_ok)
    
    if all_mandatory:
        if hurst_is_fallback:
            level, label = 'CONDITIONAL', '🟡 СЛАБЫЙ ⚠️H=0.5'
        elif hurst_val >= 0.45:
            level, label = 'CONDITIONAL', '🟡 УСЛОВНО ⚠️H≥0.45'
        elif opt_count >= 4:
            level, label = 'ENTRY', '🟢 ВХОД'
        elif opt_count >= 2 or fdr_bypass:
            level, label = 'CONDITIONAL', '🟡 УСЛОВНО'
        else:
            level, label = 'CONDITIONAL', '🟡 СЛАБЫЙ'
    else:
        level, label = 'WAIT', '⚪ ЖДАТЬ'
    
    return {'level': level, 'label': label, 'all_mandatory': all_mandatory,
            'mandatory': mandatory, 'optional': optional,
            'fdr_bypass': fdr_bypass, 'opt_count': opt_count}

# ═══════════════════════════════════════════════════════
# CORE MATH (standalone — не зависит от analysis module)
# ═══════════════════════════════════════════════════════

def kalman_hr(s1, s2, delta=1e-4, ve=1e-3):
    s1, s2 = np.array(s1, float), np.array(s2, float)
    n = min(len(s1), len(s2))
    if n < 10: return None
    s1, s2 = s1[:n], s2[:n]
    init_n = min(30, n // 3)
    try:
        X = np.column_stack([np.ones(init_n), s2[:init_n]])
        beta = np.linalg.lstsq(X, s1[:init_n], rcond=None)[0]
    except: beta = np.array([0.0, 1.0])
    P = np.eye(2); Q = np.eye(2) * delta; R = ve
    hrs, ints, spread = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        x = np.array([1.0, s2[t]]); P += Q
        e = s1[t] - x @ beta; S = x @ P @ x + R
        K = P @ x / S; beta += K * e
        P -= np.outer(K, x) @ P; P = (P + P.T) / 2
        np.fill_diagonal(P, np.maximum(np.diag(P), 1e-10))
        hrs[t], ints[t] = beta[1], beta[0]
        spread[t] = s1[t] - beta[1] * s2[t] - beta[0]
    return {'hrs': hrs, 'intercepts': ints, 'spread': spread,
            'hr': float(hrs[-1]), 'intercept': float(ints[-1])}


def calc_zscore(spread, halflife_bars=None, min_w=10, max_w=60):
    spread = np.array(spread, float); n = len(spread)
    if halflife_bars and not np.isinf(halflife_bars) and halflife_bars > 0:
        w = int(np.clip(2.5 * halflife_bars, min_w, max_w))
    else: w = 30
    w = min(w, max(10, n // 2))
    zs = np.full(n, np.nan)
    for i in range(w, n):
        lb = spread[i - w:i]; med = np.median(lb)
        mad = np.median(np.abs(lb - med)) * 1.4826
        if mad < 1e-10:
            s = np.std(lb)
            zs[i] = (spread[i] - np.mean(lb)) / s if s > 1e-10 else 0
        else: zs[i] = (spread[i] - med) / mad
    return zs, w


def calc_halflife(spread, dt=None):
    """OU halflife через регрессию. dt=1/24 для 1h, 1/6 для 4h, 1 для 1d."""
    s = np.array(spread, float)
    if len(s) < 20: return 999
    sl, sd = s[:-1], np.diff(s)
    n = len(sl)
    sx, sy = np.sum(sl), np.sum(sd)
    sxy, sx2 = np.sum(sl * sd), np.sum(sl**2)
    denom = n * sx2 - sx**2
    if abs(denom) < 1e-10: return 999
    b = (n * sxy - sx * sy) / denom
    if dt is None: dt = 1.0
    theta = max(0.001, min(10.0, -b / dt))
    hl = np.log(2) / theta  # в единицах dt
    return float(hl) if hl < 999 else 999


def calc_hurst(series, min_window=8):
    """DFA Hurst exponent (упрощённый, совместимый с сканером)."""
    x = np.array(series, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 50: return 0.5
    
    y = np.cumsum(x - np.mean(x))
    
    scales = []
    flucts = []
    min_seg = max(min_window, 4)
    max_seg = n // 4
    
    for seg_len in range(min_seg, max_seg + 1, max(1, (max_seg - min_seg) // 20)):
        n_segs = n // seg_len
        if n_segs < 2: continue
        f2_list = []
        for i in range(n_segs):
            seg = y[i * seg_len:(i + 1) * seg_len]
            t = np.arange(len(seg))
            if len(seg) < 2: continue
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            f2_list.append(np.mean((seg - trend) ** 2))
        if f2_list:
            scales.append(seg_len)
            flucts.append(np.sqrt(np.mean(f2_list)))
    
    if len(scales) < 4: return 0.5
    
    log_s = np.log(scales)
    log_f = np.log(np.array(flucts) + 1e-10)
    coeffs = np.polyfit(log_s, log_f, 1)
    
    # R² check
    pred = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_f - pred) ** 2)
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    if r_sq < 0.8: return 0.5  # fallback
    return float(np.clip(coeffs[0], 0.01, 0.99))


def calc_correlation(p1, p2, window=60):
    """Rolling корреляция."""
    n = min(len(p1), len(p2))
    if n < window: return 0.0
    r1 = np.diff(np.log(p1[-n:] + 1e-10))
    r2 = np.diff(np.log(p2[-n:] + 1e-10))
    if len(r1) < 10: return 0.0
    return float(np.corrcoef(r1[-window:], r2[-window:])[0, 1])


def calc_cointegration_pvalue(p1, p2):
    """P-value коинтеграции."""
    try:
        _, pval, _ = coint(p1, p2)
        return float(pval)
    except:
        return 1.0


# ═══════════════════════════════════════════════════════
# POSITIONS FILE (JSON persistence)
# ═══════════════════════════════════════════════════════
POSITIONS_FILE = "positions.json"

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return []

def save_positions(positions):
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2, default=str)


def add_position(coin1, coin2, direction, entry_z, entry_hr, 
                 entry_price1, entry_price2, timeframe, notes="",
                 max_hold_hours=None, pnl_stop_pct=None):
    positions = load_positions()
    # v27: defaults from unified config
    if max_hold_hours is None:
        max_hold_hours = CFG('strategy', 'max_hold_hours', 72)
    if pnl_stop_pct is None:
        pnl_stop_pct = CFG('monitor', 'pnl_stop_pct', -5.0)
    # v5.0: Adaptive stop_z — at least offset Z-units beyond entry
    _stop_offset = CFG('strategy', 'stop_z_offset', 2.0)
    _min_stop = CFG('strategy', 'min_stop_z', 4.0)
    adaptive_stop = max(abs(entry_z) + _stop_offset, _min_stop)
    pos = {
        'id': len(positions) + 1,
        'coin1': coin1, 'coin2': coin2,
        'direction': direction,
        'entry_z': entry_z,
        'entry_hr': entry_hr,
        'entry_price1': entry_price1,
        'entry_price2': entry_price2,
        'entry_time': now_msk().isoformat(),
        'timeframe': timeframe,
        'status': 'OPEN',
        'notes': notes,
        'exit_z_target': CFG('monitor', 'exit_z_target', 0.5),
        'stop_z': adaptive_stop,
        'max_hold_hours': max_hold_hours,
        'pnl_stop_pct': pnl_stop_pct,
    }
    positions.append(pos)
    save_positions(positions)
    return pos


def close_position(pos_id, exit_price1, exit_price2, exit_z, reason):
    positions = load_positions()
    closed_pos = None
    for p in positions:
        if p['id'] == pos_id and p['status'] == 'OPEN':
            p['status'] = 'CLOSED'
            p['exit_price1'] = exit_price1
            p['exit_price2'] = exit_price2
            p['exit_z'] = exit_z
            p['exit_time'] = now_msk().isoformat()
            p['exit_reason'] = reason
            # P&L
            r1 = (exit_price1 - p['entry_price1']) / p['entry_price1']
            r2 = (exit_price2 - p['entry_price2']) / p['entry_price2']
            hr = p['entry_hr']
            if p['direction'] == 'LONG':
                raw = r1 - hr * r2
            else:
                raw = -r1 + hr * r2
            p['pnl_pct'] = round(raw / (1 + abs(hr)) * 100, 3)
            closed_pos = p.copy()
            break
    save_positions(positions)
    
    # v25: R8 Performance Tracker — save to persistent history
    if closed_pos:
        try:
            save_trade_to_history(closed_pos)
        except Exception:
            pass
        # v27: Update pair memory
        try:
            from config_loader import pair_memory_update
            _pair = f"{closed_pos['coin1']}/{closed_pos['coin2']}"
            _entry_dt = closed_pos.get('entry_time', '')
            _exit_dt = closed_pos.get('exit_time', '')
            try:
                from datetime import datetime
                _et = datetime.fromisoformat(str(_entry_dt))
                _xt = datetime.fromisoformat(str(_exit_dt))
                _hold_h = (_xt - _et).total_seconds() / 3600
            except Exception:
                _hold_h = 0
            pair_memory_update(
                _pair, closed_pos.get('pnl_pct', 0), _hold_h,
                closed_pos.get('direction', ''), 
                closed_pos.get('entry_z', 0),
                closed_pos.get('exit_z', 0)
            )
        except Exception:
            pass


def save_trade_to_history(trade):
    """R8: Save closed trade to persistent CSV history."""
    import csv
    history_file = "trade_history.csv"
    fields = [
        'id', 'pair', 'coin1', 'coin2', 'direction', 'timeframe',
        'entry_z', 'exit_z', 'entry_hr', 'pnl_pct',
        'entry_time', 'exit_time', 'exit_reason',
        'entry_price1', 'entry_price2', 'exit_price1', 'exit_price2',
        'notes', 'best_pnl',
    ]
    
    row = {
        'id': trade.get('id', 0),
        'pair': f"{trade.get('coin1', '')}/{trade.get('coin2', '')}",
        'coin1': trade.get('coin1', ''),
        'coin2': trade.get('coin2', ''),
        'direction': trade.get('direction', ''),
        'timeframe': trade.get('timeframe', '4h'),
        'entry_z': trade.get('entry_z', 0),
        'exit_z': trade.get('exit_z', 0),
        'entry_hr': trade.get('entry_hr', 0),
        'pnl_pct': trade.get('pnl_pct', 0),
        'entry_time': trade.get('entry_time', ''),
        'exit_time': trade.get('exit_time', ''),
        'exit_reason': trade.get('exit_reason', ''),
        'entry_price1': trade.get('entry_price1', 0),
        'entry_price2': trade.get('entry_price2', 0),
        'exit_price1': trade.get('exit_price1', 0),
        'exit_price2': trade.get('exit_price2', 0),
        'notes': trade.get('notes', ''),
        'best_pnl': trade.get('best_pnl', 0),
    }
    
    file_exists = os.path.exists(history_file)
    with open(history_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_trade_history():
    """R8: Load all trade history."""
    import csv
    history_file = "trade_history.csv"
    if not os.path.exists(history_file):
        return []
    
    with open(history_file, 'r') as f:
        reader = csv.DictReader(f)
        trades = []
        for row in reader:
            # Convert numeric fields
            for k in ['entry_z', 'exit_z', 'entry_hr', 'pnl_pct', 
                       'entry_price1', 'entry_price2', 'exit_price1', 'exit_price2', 'best_pnl']:
                try:
                    row[k] = float(row.get(k, 0) or 0)
                except (ValueError, TypeError):
                    row[k] = 0
            try:
                row['id'] = int(row.get('id', 0) or 0)
            except:
                row['id'] = 0
            trades.append(row)
    return trades


# ═══════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════

# v4.0: Exchange fallback chain (Binance/Bybit block cloud servers)
EXCHANGE_FALLBACK = ['okx', 'kucoin', 'bybit', 'binance']

def _get_exchange(exchange_name):
    """Получить рабочую биржу с fallback."""
    tried = set()
    chain = [exchange_name] + [e for e in EXCHANGE_FALLBACK if e != exchange_name]
    for exch in chain:
        if exch in tried: continue
        tried.add(exch)
        try:
            ex = getattr(ccxt, exch)({'enableRateLimit': True})
            ex.load_markets()
            return ex, exch
        except:
            continue
    return None, None


@st.cache_data(ttl=120)
def fetch_prices(exchange_name, coin, timeframe, lookback_bars=300):
    """v27: Fetch with retry + futures first."""
    import ccxt as _ccxt
    # Try futures first, then spot
    symbols = [f"{coin}/USDT:USDT", f"{coin}/USDT"]
    for symbol in symbols:
        for _attempt in range(3):
            try:
                ex, actual = _get_exchange(exchange_name)
                if ex is None: return None
                ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=lookback_bars)
                df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                return df
            except (_ccxt.NetworkError, _ccxt.RequestTimeout, _ccxt.ExchangeNotAvailable):
                time.sleep([2, 5, 15][_attempt])
            except:
                break  # try next symbol
    return None


def get_current_price(exchange_name, coin):
    """v27: Get price with retry + futures."""
    import ccxt as _ccxt
    symbols = [f"{coin}/USDT:USDT", f"{coin}/USDT"]
    for symbol in symbols:
        for _attempt in range(3):
            try:
                ex, actual = _get_exchange(exchange_name)
                if ex is None: return None
                ticker = ex.fetch_ticker(symbol)
                return ticker['last']
            except (_ccxt.NetworkError, _ccxt.RequestTimeout, _ccxt.ExchangeNotAvailable):
                time.sleep([2, 5, 15][_attempt])
            except:
                break
    return None


# ═══════════════════════════════════════════════════════
# MONITOR LOGIC
# ═══════════════════════════════════════════════════════

def monitor_position(pos, exchange_name):
    """Полный мониторинг одной позиции v3.0 — с quality metrics."""
    c1, c2 = pos['coin1'], pos['coin2']
    tf = pos['timeframe']
    
    bars_map = {'1h': 300, '4h': 300, '1d': 120}
    n_bars = bars_map.get(tf, 300)
    
    df1 = fetch_prices(exchange_name, c1, tf, n_bars)
    df2 = fetch_prices(exchange_name, c2, tf, n_bars)
    
    if df1 is None or df2 is None:
        return None
    
    # Align timestamps
    merged = pd.merge(df1[['ts', 'c']], df2[['ts', 'c']], on='ts', suffixes=('_1', '_2'))
    if len(merged) < 50:
        return None
    
    p1 = merged['c_1'].values
    p2 = merged['c_2'].values
    ts = merged['ts'].tolist()
    
    # Kalman
    kf = kalman_hr(p1, p2)
    if kf is None:
        return None
    
    spread = kf['spread']
    hr_current = kf['hr']
    
    # v3.0: OU Half-life (dt-correct, как в сканере)
    dt_ou = {'1h': 1/24, '4h': 1/6, '1d': 1.0}.get(tf, 1/6)
    hpb = {'1h': 1, '4h': 4, '1d': 24}.get(tf, 4)
    
    # v18: Use SAME halflife function as scanner (critical for Z-window sync)
    if _USE_MRA:
        hl_days = calc_halflife_from_spread(spread, dt=dt_ou)
    else:
        hl_days = calc_halflife(spread, dt=dt_ou)
    hl_hours = hl_days * 24 if hl_days < 999 else 999
    hl_bars = (hl_hours / hpb) if hl_hours < 999 else None
    
    # v15: Use SAME Z-score function as scanner for consistency
    if _USE_MRA:
        z_now, zs, zw = calculate_adaptive_robust_zscore(spread, halflife_bars=hl_bars)
        # v18: GARCH Z for false convergence detection
        garch_info = calculate_garch_zscore(spread, halflife_bars=hl_bars)
        z_garch = garch_info.get('z_garch', z_now)
        garch_vol_ratio = garch_info.get('vol_ratio', 1.0)
        garch_var_expanding = garch_info.get('variance_expanding', False)
    else:
        zs, zw = calc_zscore(spread, halflife_bars=hl_bars)
        z_now = float(zs[~np.isnan(zs)][-1]) if any(~np.isnan(zs)) else 0
        z_garch = z_now
        garch_vol_ratio = 1.0
        garch_var_expanding = False
    
    # v3.0: Quality metrics (как в сканере)
    # v14: CRITICAL FIX — use SAME Hurst as scanner (DFA on increments)
    # v16: Hurst EMA smoothing
    if _USE_MRA:
        hurst_ema_info = calculate_hurst_ema(spread)
        hurst = hurst_ema_info.get('hurst_ema', 0.5)  # Use EMA, not raw
        hurst_raw = hurst_ema_info.get('hurst_raw', hurst)
        hurst_std = hurst_ema_info.get('hurst_std', 0)
    else:
        hurst = calc_hurst(spread)  # fallback
        hurst_raw = hurst
        hurst_std = 0
    corr = calc_correlation(p1, p2, window=min(60, len(p1) // 3))
    pvalue = calc_cointegration_pvalue(p1, p2)
    
    # v3.0: Entry readiness data
    quality_data = {
        'signal': 'SIGNAL' if abs(z_now) >= 2.0 else ('READY' if abs(z_now) >= 1.5 else 'NEUTRAL'),
        'zscore': z_now,
        'threshold': 2.0,
        'quality_score': max(0, int(100 - pvalue * 200 - max(0, hurst - 0.35) * 200)),
        'direction': pos['direction'],
        'fdr_passed': pvalue < 0.01,
        'confidence': 'HIGH' if (hurst < 0.4 and pvalue < 0.03) else ('MEDIUM' if pvalue < 0.05 else 'LOW'),
        'signal_score': max(0, int(abs(z_now) / 2.0 * 50 + (0.5 - hurst) * 100)),
        'correlation': corr,
        'stability_passed': 3 if pvalue < 0.05 else 1,
        'stability_total': 4,
        'hurst': hurst,
        'adf_passed': pvalue < 0.05,
    }
    
    # P&L (v4.0: price-based + spread-based + disagreement warning)
    r1 = (p1[-1] - pos['entry_price1']) / pos['entry_price1']
    r2 = (p2[-1] - pos['entry_price2']) / pos['entry_price2']
    hr = pos['entry_hr']
    if pos['direction'] == 'LONG':
        raw_pnl = r1 - hr * r2
    else:
        raw_pnl = -r1 + hr * r2
    pnl_pct = raw_pnl / (1 + abs(hr)) * 100
    
    # v4.0: Spread-based P&L (фиксированный HR от входа)
    entry_spread_val = pos['entry_price1'] - hr * pos['entry_price2']
    current_spread_val = p1[-1] - hr * p2[-1]
    spread_change = current_spread_val - entry_spread_val
    if pos['direction'] == 'LONG':
        spread_direction = 'profit' if spread_change > 0 else 'loss'
    else:
        spread_direction = 'profit' if spread_change < 0 else 'loss'
    
    # v4.0: Z-direction check
    z_entry = pos['entry_z']
    # v22: Directional Z check (fixes SOL/OKSOL false disagree on overshoot)
    # OLD: z_towards_zero = abs(z_now) < abs(z_entry) — WRONG for overshoot!
    # NEW: Check if Z moved in the CORRECT direction for our trade
    if pos['direction'] == 'LONG':
        # LONG entered at Z<0, wants Z to go UP (toward 0 and beyond)
        z_towards_zero = z_now > z_entry
    else:
        # SHORT entered at Z>0, wants Z to go DOWN (toward 0 and beyond)
        z_towards_zero = z_now < z_entry
    
    # v4.0: Предупреждение при расхождении P&L и Z-направления
    # v14: Enhanced with variance collapse detection (рассуждение #1)
    pnl_z_disagree = False
    pnl_z_warning = ""
    
    # Use shared function if available
    if _USE_MRA:
        disagree_info = check_pnl_z_disagreement(z_entry, z_now, pnl_pct, pos['direction'])
        if disagree_info.get('disagreement'):
            pnl_z_disagree = True
            pnl_z_warning = disagree_info.get('warning', '')
    
    # Legacy checks (still useful as fallback)
    if not pnl_z_disagree:
        if pnl_pct > 0 and not z_towards_zero:
            pnl_z_disagree = True
            pnl_z_warning = (
                f"⚠️ P&L положительный (+{pnl_pct:.2f}%), но Z ушёл дальше от нуля "
                f"({z_entry:+.2f} → {z_now:+.2f}). "
                f"HR изменился ({pos['entry_hr']:.4f} → {hr_current:.4f})."
            )
        elif pnl_pct < -0.5 and z_towards_zero:
            pnl_z_disagree = True
            pnl_z_warning = (
                f"⚠️ Z → 0 ({z_entry:+.2f} → {z_now:+.2f}), но P&L={pnl_pct:+.2f}%. "
                f"Возможно ложное схождение (σ спреда выросла)."
            )
    
    # Time in trade (вычисляем ДО использования)
    entry_dt = datetime.fromisoformat(pos['entry_time'])
    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.replace(tzinfo=MSK)  # assume MSK if no tz
    hours_in = (now_msk() - entry_dt).total_seconds() / 3600
    
    # Exit signals
    exit_signal = None
    exit_urgency = 0
    ez = pos.get('exit_z_target', 0.5)
    # v5.0: Adaptive stop — at least 2.0 Z-units beyond entry, minimum 4.0
    default_stop = max(abs(pos['entry_z']) + 2.0, 4.0)
    sz = pos.get('stop_z', default_stop)
    max_hours = pos.get('max_hold_hours', 72)
    pnl_stop = pos.get('pnl_stop_pct', -5.0)
    
    if pos['direction'] == 'LONG':
        if z_now >= -ez and z_now <= ez:
            # v16: Check PnL before declaring convergence (рассуждение #1)
            # v18: Also check GARCH Z — if GARCH still far, it's variance collapse
            garch_still_far = abs(z_garch) > 1.5
            if pnl_pct > -0.3 and not garch_still_far:
                exit_signal = '✅ MEAN REVERT — закрывать!'
                exit_urgency = 2
            elif garch_still_far:
                exit_signal = (f'⚠️ ЛОЖНОЕ СХОЖДЕНИЕ: Z_std→0 но Z_GARCH={z_garch:+.1f}. '
                               f'σ выросла в {garch_vol_ratio:.1f}x. Реального возврата нет.')
                exit_urgency = 1
            else:
                exit_signal = (f'⚠️ ЛОЖНОЕ СХОЖДЕНИЕ: Z→0 но P&L={pnl_pct:+.2f}%. '
                               f'σ спреда выросла. Ждите реального возврата цен.')
                exit_urgency = 1
        elif z_now > 1.0:
            exit_signal = '✅ OVERSHOOT — фиксировать прибыль!'
            exit_urgency = 2
        elif z_now < -sz:
            exit_signal = '🛑 STOP LOSS (Z) — экстренный выход!'
            exit_urgency = 2
    else:
        if z_now <= ez and z_now >= -ez:
            garch_still_far = abs(z_garch) > 1.5
            if pnl_pct > -0.3 and not garch_still_far:
                exit_signal = '✅ MEAN REVERT — закрывать!'
                exit_urgency = 2
            elif garch_still_far:
                exit_signal = (f'⚠️ ЛОЖНОЕ СХОЖДЕНИЕ: Z_std→0 но Z_GARCH={z_garch:+.1f}. '
                               f'σ выросла в {garch_vol_ratio:.1f}x. Реального возврата нет.')
                exit_urgency = 1
            else:
                exit_signal = (f'⚠️ ЛОЖНОЕ СХОЖДЕНИЕ: Z→0 но P&L={pnl_pct:+.2f}%. '
                               f'σ спреда выросла. Ждите реального возврата цен.')
                exit_urgency = 1
        elif z_now < -1.0:
            exit_signal = '✅ OVERSHOOT — фиксировать прибыль!'
            exit_urgency = 2
        elif z_now > sz:
            exit_signal = '🛑 STOP LOSS (Z) — экстренный выход!'
            exit_urgency = 2
    
    # P&L stop
    if pnl_pct <= pnl_stop and exit_urgency < 2:
        exit_signal = f'🛑 STOP LOSS (P&L {pnl_pct:.1f}% < {pnl_stop:.0f}%) — выход!'
        exit_urgency = 2
    
    # Time-based
    if hours_in > max_hours and exit_urgency < 2:
        if exit_signal is None:
            exit_signal = f'⏰ TIMEOUT ({hours_in:.0f}ч > {max_hours:.0f}ч) — рассмотрите выход'
            exit_urgency = 1
    elif hours_in > max_hours * 0.75 and exit_urgency == 0:
        exit_signal = f'⚠️ Позиция открыта {hours_in:.0f}ч (лимит {max_hours:.0f}ч)'
        exit_urgency = 1
    
    # v27: Quality warnings — thresholds from unified config
    quality_warnings = []
    _h_crit = CFG('monitor', 'hurst_critical', 0.50)
    _h_warn = CFG('monitor', 'hurst_warning', 0.48)
    _h_border = CFG('monitor', 'hurst_border', 0.45)
    _pv_warn = CFG('monitor', 'pvalue_warning', 0.10)
    _corr_warn = CFG('monitor', 'correlation_warning', 0.20)
    
    if hurst >= _h_crit:
        quality_warnings.append(
            f"🚨 Hurst(EMA)={hurst:.3f} ≥ {_h_crit} — нет mean reversion!"
            + (f" (raw={hurst_raw:.3f}, σ={hurst_std:.3f})" if hurst_std > 0 else ""))
    elif hurst >= _h_warn:
        quality_warnings.append(f"⚠️ Hurst(EMA)={hurst:.3f} ≥ {_h_warn} — ослабевает")
    elif hurst >= _h_border:
        quality_warnings.append(f"💡 Hurst(EMA)={hurst:.3f} — пограничное")
    if pvalue >= _pv_warn:
        quality_warnings.append(f"⚠️ P-value={pvalue:.3f} — коинтеграция ослабла!")
    if corr < _corr_warn:
        quality_warnings.append(f"⚠️ Корреляция ρ={corr:.2f} < {_corr_warn} — хедж не работает!")
    
    # v18: Direction sanity check — warn if direction contradicts entry Z
    entry_z = pos.get('entry_z', 0)
    direction = pos.get('direction', '')
    if entry_z < -0.5 and direction == 'SHORT':
        quality_warnings.append(
            f"🚨 НАПРАВЛЕНИЕ ИНВЕРТИРОВАНО: Entry_Z={entry_z:+.2f} (отрицательный) "
            f"но Dir=SHORT. Для Z<0 должен быть LONG! Проверьте ввод.")
    elif entry_z > 0.5 and direction == 'LONG':
        quality_warnings.append(
            f"🚨 НАПРАВЛЕНИЕ ИНВЕРТИРОВАНО: Entry_Z={entry_z:+.2f} (положительный) "
            f"но Dir=LONG. Для Z>0 должен быть SHORT! Проверьте ввод.")
    
    # Build base result dict
    base_result = {
        'z_now': z_now,
        'z_entry': pos['entry_z'],
        'pnl_pct': pnl_pct,
        'spread_direction': spread_direction,
        'z_towards_zero': z_towards_zero,
        'pnl_z_disagree': pnl_z_disagree,
        'pnl_z_warning': pnl_z_warning,
        'price1_now': p1[-1],
        'price2_now': p2[-1],
        'hr_now': hr_current,
        'hr_entry': pos['entry_hr'],
        'exit_signal': exit_signal,
        'exit_urgency': exit_urgency,
        'hours_in': hours_in,
        'spread': spread,
        'zscore_series': zs,
        'timestamps': ts,
        'hr_series': kf['hrs'],
        'halflife_hours': hl_hours,
        'z_window': zw,
        'hurst': hurst,
        'correlation': corr,
        'pvalue': pvalue,
        'quality_data': quality_data,
        'quality_warnings': quality_warnings,
        'z_garch': z_garch,
        'garch_vol_ratio': garch_vol_ratio,
        'garch_var_expanding': garch_var_expanding,
    }
    
    # v27: R6 Correlation Monitor — track quality degradation
    _pair_key = f"{pos['coin1']}/{pos['coin2']}"
    _qh_key = f"_quality_history_{pos['id']}"
    if _qh_key not in st.session_state:
        st.session_state[_qh_key] = []
    _qh = st.session_state[_qh_key]
    _qh.append({'ts': time.time(), 'corr': corr, 'hurst': hurst, 'pval': pvalue})
    if len(_qh) > 30:
        st.session_state[_qh_key] = _qh[-30:]
    
    # R6: Quality degradation alerts
    if len(_qh) >= 3:
        _recent_corr = [q['corr'] for q in _qh[-5:]]
        _recent_hurst = [q['hurst'] for q in _qh[-5:]]
        _corr_trend = _recent_corr[-1] - _recent_corr[0] if len(_recent_corr) > 1 else 0
        _hurst_trend = _recent_hurst[-1] - _recent_hurst[0] if len(_recent_hurst) > 1 else 0
        
        if _corr_trend < -0.1:
            quality_warnings.append(f"📉 R6: ρ падает ({_recent_corr[0]:.2f}→{_recent_corr[-1]:.2f}). Хедж деградирует!")
        if _hurst_trend > 0.05:
            quality_warnings.append(f"📈 R6: Hurst растёт ({_recent_hurst[0]:.3f}→{_recent_hurst[-1]:.3f}). MR ослабевает!")
    
    base_result['quality_warnings'] = quality_warnings
    
    # v24: R5 Smart Exit Analysis (was dead code — FIXED in v27)
    base_result['smart_exit'] = None
    base_result['smart_signals'] = []
    base_result['smart_recommendation'] = ''
    base_result['smart_urgency'] = 0
    
    if _USE_MRA:
        try:
            smart_exit = smart_exit_analysis(
                z_entry=pos['entry_z'],
                z_now=z_now,
                z_history=zs[~np.isnan(zs)] if len(zs) > 0 else np.array([z_now]),
                pnl_pct=pnl_pct,
                hours_in=hours_in,
                halflife_hours=hl_hours,
                direction=pos['direction'],
                best_pnl=pos.get('best_pnl', max(pnl_pct, 0)),
            )
            base_result['smart_exit'] = smart_exit
            base_result['smart_signals'] = smart_exit.get('signals', [])
            base_result['smart_recommendation'] = smart_exit.get('recommendation', '')
            base_result['smart_urgency'] = smart_exit.get('urgency', 0)
            
            # Override exit_signal if smart exit has higher urgency
            if smart_exit.get('urgency', 0) > exit_urgency:
                base_result['exit_urgency'] = smart_exit['urgency']
                smart_msgs = [s['message'] for s in smart_exit.get('signals', [])]
                if smart_msgs:
                    base_result['exit_signal'] = ' | '.join(smart_msgs[:2])
        except Exception:
            pass
    
    return base_result


# ═══════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════

st.set_page_config(page_title="Position Monitor", page_icon="📍", layout="wide")

st.markdown("""
<style>
    .exit-signal { padding: 15px; border-radius: 10px; font-size: 1.2em; 
                   font-weight: bold; text-align: center; margin: 10px 0; }
    .signal-exit { background: #1b5e20; color: #a5d6a7; }
    .signal-stop { background: #b71c1c; color: #ef9a9a; }
</style>
""", unsafe_allow_html=True)

st.title("📍 Pairs Position Monitor")
st.caption("v19.0 | 24.02.2026 | Unified Config + Auto-refresh FIX + R8/R5 Smart Exit")

# Sidebar
with st.sidebar:
    st.header("⚙️ Настройки")
    exchange = st.selectbox("Биржа", ['okx', 'kucoin', 'bybit', 'binance'], index=0,
                           help="⚠️ Binance/Bybit заблокированы на облачных серверах. Используйте OKX/KuCoin.")
    auto_refresh = st.checkbox("Авто-обновление (2 мин)", value=False)
    
    st.divider()
    st.header("➕ Новая позиция")
    
    # v22: R3 — Auto-Import from Scanner
    import glob, json as _json
    pending_files = sorted(glob.glob("monitor_import/pending_*.json"))
    
    # v27: Cleanup — remove pending files if pair already open
    if pending_files:
        _open_pairs = set()
        for _op in load_positions():
            if _op.get('status') == 'OPEN':
                _open_pairs.add(f"{_op['coin1']}/{_op['coin2']}")
        
        _remaining = []
        for pf in pending_files:
            try:
                with open(pf, 'r') as f:
                    imp = _json.load(f)
                _pname = f"{imp['coin1']}/{imp['coin2']}"
                if _pname in _open_pairs:
                    import os; os.remove(pf)  # Already imported
                else:
                    _remaining.append(pf)
            except Exception:
                _remaining.append(pf)
        pending_files = _remaining
    
    if pending_files:
        st.markdown("#### 📥 Импорт из сканера")
        for pf in pending_files:
            try:
                with open(pf, 'r') as f:
                    imp = _json.load(f)
                pair_name = f"{imp['coin1']}/{imp['coin2']}"
                st.info(
                    f"📤 **{pair_name}** {imp['direction']} | "
                    f"Z={imp['entry_z']:.2f} HR={imp['entry_hr']:.4f} "
                    f"| {imp.get('notes', '')}"
                )
                if st.button(f"✅ Импортировать {pair_name}", key=f"imp_{pair_name}"):
                    with st.spinner(f"Загружаю цены {pair_name}..."):
                        p1 = imp.get('entry_price1', 0)
                        p2 = imp.get('entry_price2', 0)
                        if p1 == 0 or p2 == 0:
                            p1 = get_current_price(exchange, imp['coin1']) or 0
                            p2 = get_current_price(exchange, imp['coin2']) or 0
                        if p1 > 0 and p2 > 0:
                            pos = add_position(
                                imp['coin1'], imp['coin2'], imp['direction'],
                                imp['entry_z'], imp['entry_hr'],
                                p1, p2, imp.get('timeframe', '4h'),
                                imp.get('notes', ''))
                            st.success(f"✅ #{pos['id']} {pair_name} добавлена!")
                            import os; os.remove(pf)
                            st.rerun()
                        else:
                            st.error("Не удалось получить цены")
            except Exception as ex:
                st.warning(f"⚠️ {pf}: {ex}")
        st.divider()
    
    # Upload JSON manually
    uploaded_json = st.file_uploader("📤 Или загрузи JSON из сканера", type=['json'], key='json_import')
    if uploaded_json:
        try:
            imp = _json.load(uploaded_json)
            pair_name = f"{imp['coin1']}/{imp['coin2']}"
            st.info(f"📤 **{pair_name}** {imp['direction']} Z={imp['entry_z']:.2f} HR={imp['entry_hr']:.4f}")
            if st.button(f"✅ Импортировать {pair_name}", key="imp_upload"):
                with st.spinner("Загружаю цены..."):
                    p1 = imp.get('entry_price1', 0) or get_current_price(exchange, imp['coin1']) or 0
                    p2 = imp.get('entry_price2', 0) or get_current_price(exchange, imp['coin2']) or 0
                    if p1 > 0 and p2 > 0:
                        pos = add_position(imp['coin1'], imp['coin2'], imp['direction'],
                                         imp['entry_z'], imp['entry_hr'], p1, p2,
                                         imp.get('timeframe', '4h'), imp.get('notes', ''))
                        st.success(f"✅ #{pos['id']} импортирована!")
                        st.rerun()
        except Exception as ex:
            st.error(f"Ошибка JSON: {ex}")
    
    st.divider()
    
    with st.form("add_position"):
        col1, col2 = st.columns(2)
        with col1:
            new_c1 = st.text_input("Coin 1", "ETH").upper().strip()
        with col2:
            new_c2 = st.text_input("Coin 2", "STETH").upper().strip()
        
        new_dir = st.selectbox("Направление", ["LONG", "SHORT"])
        new_tf = st.selectbox("Таймфрейм", ['1h', '4h', '1d'], index=1)
        
        col3, col4 = st.columns(2)
        with col3:
            new_z = st.number_input("Entry Z", value=2.0, step=0.1)
        with col4:
            new_hr = st.number_input("Hedge Ratio", value=1.0, step=0.01, format="%.4f")
        
        col5, col6 = st.columns(2)
        with col5:
            new_p1 = st.number_input("Цена Coin1", value=0.0, step=0.01, format="%.4f")
        with col6:
            new_p2 = st.number_input("Цена Coin2", value=0.0, step=0.01, format="%.4f")
        
        new_notes = st.text_input("Заметки", "")
        
        # v2.0: Risk management
        st.markdown("**⚠️ Риск-менеджмент**")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            new_max_hours = st.number_input("Max часов в позиции", value=72, step=12)
        with col_r2:
            new_pnl_stop = st.number_input("P&L Stop (%)", value=-5.0, step=0.5)
        
        # Автозагрузка цен
        fetch_prices_btn = st.form_submit_button("📥 Загрузить цены + Добавить")
    
    if fetch_prices_btn and new_c1 and new_c2:
        if new_p1 == 0 or new_p2 == 0:
            with st.spinner("Загружаю текущие цены..."):
                p1_live = get_current_price(exchange, new_c1)
                p2_live = get_current_price(exchange, new_c2)
                if p1_live and p2_live:
                    new_p1 = p1_live
                    new_p2 = p2_live
                    st.info(f"💰 {new_c1}: ${p1_live:.4f} | {new_c2}: ${p2_live:.4f}")
                else:
                    st.error("Не удалось загрузить цены")
        
        if new_p1 > 0 and new_p2 > 0:
            # v22: HR sanity check — warn if HR doesn't match price ratio
            expected_hr_approx = new_p1 / new_p2 if new_p2 > 0 else 0
            if new_hr > 0 and expected_hr_approx > 0:
                ratio = new_hr / expected_hr_approx if expected_hr_approx > 0 else 999
                if ratio > 10 or ratio < 0.1:
                    st.warning(
                        f"⚠️ **HR подозрительный!** HR={new_hr:.4f}, "
                        f"P1/P2={expected_hr_approx:.4f} (отношение {ratio:.1f}x). "
                        f"Проверьте правильность HR. Возможно опечатка.")
            
            pos = add_position(new_c1, new_c2, new_dir, new_z, new_hr,
                             new_p1, new_p2, new_tf, new_notes,
                             max_hold_hours=new_max_hours,
                             pnl_stop_pct=new_pnl_stop)
            st.success(f"✅ Позиция #{pos['id']} добавлена: {new_dir} {new_c1}/{new_c2}")
            st.rerun()

# ═══════ MAIN AREA ═══════
positions = load_positions()
open_positions = [p for p in positions if p['status'] == 'OPEN']
closed_positions = [p for p in positions if p['status'] == 'CLOSED']

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([f"📍 Открытые ({len(open_positions)})", 
                       f"📋 История ({len(closed_positions)})",
                       f"📊 Портфель",
                       f"📈 Performance (R8)"])

with tab1:
    if not open_positions:
        st.info("📭 Нет открытых позиций. Добавьте через боковую панель.")
    else:
        # Dashboard metrics
        total_pnl = 0
        
        for pos in open_positions:
            with st.container():
                st.markdown("---")
                
                # Header
                dir_emoji = '🟢' if pos['direction'] == 'LONG' else '🔴'
                pair_name = f"{pos['coin1']}/{pos['coin2']}"
                
                # Monitor
                with st.spinner(f"Обновляю {pair_name}..."):
                    mon = monitor_position(pos, exchange)
                
                if mon is None:
                    st.error(f"❌ Не удалось получить данные для {pair_name}")
                    continue
                
                total_pnl += mon['pnl_pct']
                
                # v24: Track best P&L for trailing stop
                current_best = pos.get('best_pnl', 0)
                if mon['pnl_pct'] > current_best:
                    pos['best_pnl'] = mon['pnl_pct']
                    # Save updated best_pnl
                    try:
                        all_pos = load_positions()
                        for p in all_pos:
                            if p['id'] == pos['id']:
                                p['best_pnl'] = mon['pnl_pct']
                        save_positions(all_pos)
                    except Exception:
                        pass
                
                # Exit signal banner
                if mon['exit_signal']:
                    if 'STOP' in mon['exit_signal'] or 'СРОЧН' in str(mon['exit_signal']):
                        st.error(mon['exit_signal'])
                    elif 'MEAN REVERT' in mon['exit_signal'] or 'OVERSHOOT' in mon['exit_signal']:
                        st.success(mon['exit_signal'])
                    else:
                        st.warning(mon['exit_signal'])
                
                # v24: R5 Smart Exit Signals panel
                smart_sigs = mon.get('smart_signals', [])
                smart_rec = mon.get('smart_recommendation', '')
                if smart_sigs:
                    with st.expander(f"🧠 Smart Exit: {smart_rec} ({len(smart_sigs)} сигнал{'ов' if len(smart_sigs) > 1 else ''})", expanded=mon.get('smart_urgency', 0) >= 2):
                        for sig in smart_sigs:
                            sig_type = sig.get('type', '')
                            sig_urg = sig.get('urgency', 0)
                            if sig_urg >= 3:
                                st.error(sig['message'])
                            elif sig_urg >= 2:
                                st.warning(sig['message'])
                            else:
                                st.info(sig['message'])
                
                # Header row
                dir_emoji_c1 = '🟢 LONG' if pos['direction'] == 'LONG' else '🔴 SHORT'
                dir_emoji_c2 = '🔴 SHORT' if pos['direction'] == 'LONG' else '🟢 LONG'
                st.subheader(f"{dir_emoji} {pos['direction']} | {pair_name} | #{pos['id']}")
                st.caption(f"{pos['coin1']}: {dir_emoji_c1} | {pos['coin2']}: {dir_emoji_c2}")
                
                # v4.0: P&L / Z disagreement warning
                if mon.get('pnl_z_disagree'):
                    st.warning(mon['pnl_z_warning'])
                
                # KPI row
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                
                # v23: Fix color display — Streamlit colors delta green=up, red=down
                # v24: P&L with CORRECT coloring
                # Streamlit st.metric: numeric delta → green if positive, red if negative
                # String delta "loss" was showing as green text — WRONG!
                # Fix: pass NUMERIC delta so Streamlit applies correct color
                pnl_val = mon['pnl_pct']
                pnl_emoji = "🟢" if pnl_val > 0.01 else "🔴" if pnl_val < -0.01 else "⚪"
                c1.metric(
                    f"P&L {pnl_emoji}", 
                    f"{pnl_val:+.2f}%", 
                    delta=f"{pnl_val:+.2f}%",  # String starting with - → red, + → green
                    delta_color="normal"  # positive=green, negative=red
                )
                
                # v22: Directional Z explanation
                z_dir_ok = mon.get('z_towards_zero', False)
                z_crossed_zero = (pos['direction'] == 'LONG' and mon['z_now'] > 0 and pos['entry_z'] < 0) or \
                                 (pos['direction'] == 'SHORT' and mon['z_now'] < 0 and pos['entry_z'] > 0)
                if z_crossed_zero:
                    z_delta_text = f"✅ OVERSHOOT (вход: {mon['z_entry']:+.2f})"
                elif z_dir_ok:
                    z_delta_text = f"✅ → 0 (вход: {mon['z_entry']:+.2f})"
                else:
                    z_delta_text = f"❌ ← от 0 (вход: {mon['z_entry']:+.2f})"
                c2.metric("Z сейчас", f"{mon['z_now']:+.2f}",
                         delta=z_delta_text)
                c3.metric("HR", f"{mon['hr_now']:.4f}",
                         delta=f"вход: {mon['hr_entry']:.4f}")
                
                # v23: Price display with directional coloring
                # Coin1: LONG=want price UP, SHORT=want price DOWN
                p1_now = mon['price1_now']
                p1_entry = pos['entry_price1']
                p1_change = (p1_now - p1_entry) / p1_entry * 100 if p1_entry > 0 else 0
                # For LONG coin1: price up = good (green), For SHORT coin1: price down = good
                p1_good = (pos['direction'] == 'LONG' and p1_change >= 0) or \
                          (pos['direction'] == 'SHORT' and p1_change <= 0)
                c4.metric(
                    f"{pos['coin1']} {'🟢' if pos['direction']=='LONG' else '🔴'}", 
                    f"${p1_now:.4f}",
                    delta=f"{p1_change:+.2f}% (вход: ${p1_entry:.4f})",
                    delta_color="normal" if p1_good else "inverse")
                
                # Coin2: opposite direction to coin1
                p2_now = mon['price2_now']
                p2_entry = pos['entry_price2']
                p2_change = (p2_now - p2_entry) / p2_entry * 100 if p2_entry > 0 else 0
                # For LONG: coin2 is SHORT (want price down), For SHORT: coin2 is LONG (want price up)
                p2_good = (pos['direction'] == 'LONG' and p2_change <= 0) or \
                          (pos['direction'] == 'SHORT' and p2_change >= 0)
                c5.metric(
                    f"{pos['coin2']} {'🔴' if pos['direction']=='LONG' else '🟢'}", 
                    f"${p2_now:.4f}",
                    delta=f"{p2_change:+.2f}% (вход: ${p2_entry:.4f})",
                    delta_color="normal" if p2_good else "inverse")
                
                c6.metric("В позиции", f"{mon['hours_in']:.0f}ч",
                         delta=f"HL: {mon['halflife_hours']:.0f}ч")
                
                # v3.0: Quality metrics row
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Hurst", f"{mon.get('hurst', 0.5):.3f}",
                         delta="🟢 MR" if mon.get('hurst', 0.5) < 0.45 else "🔴 No MR")
                q2.metric("P-value", f"{mon.get('pvalue', 1.0):.4f}",
                         delta="✅ Coint" if mon.get('pvalue', 1.0) < 0.05 else "⚠️ Weak")
                q3.metric("Корреляция ρ", f"{mon.get('correlation', 0):.3f}",
                         delta="🟢" if mon.get('correlation', 0) >= 0.5 else "⚠️")
                q4.metric("Z-window", f"{mon.get('z_window', 30)} баров")
                
                # v18: GARCH Z row
                if mon.get('z_garch') is not None:
                    gq1, gq2, gq3, gq4 = st.columns(4)
                    gq1.metric("Z GARCH", f"{mon.get('z_garch', 0):+.2f}",
                               f"vs std={mon.get('z_now',0):+.2f}")
                    vr = mon.get('garch_vol_ratio', 1.0)
                    gq2.metric("σ ratio", f"{vr:.2f}x",
                               "🔴 растёт" if mon.get('garch_var_expanding') else "✅ стабильна")
                    gq3.metric("HL часов", f"{mon.get('halflife_hours', 0):.1f}")
                    gq4.metric("Z-window", f"{mon.get('z_window', 30)} бар")
                
                # v20: Dynamic HR Drift Monitoring (P4 Roadmap)
                hr_entry = pos.get('entry_hr', 0)
                hr_now = mon.get('hr_now', hr_entry)
                if hr_entry > 0 and hr_now > 0:
                    hr_drift_pct = abs(hr_now - hr_entry) / hr_entry * 100
                    
                    if hr_drift_pct > 5:  # Only show if drift is significant
                        st.markdown("#### 📐 HR Drift Monitor")
                        hd1, hd2, hd3 = st.columns(3)
                        with hd1:
                            dr_emoji = '✅' if hr_drift_pct < 10 else '🟡' if hr_drift_pct < 20 else '🔴'
                            st.metric("HR дрейф", f"{dr_emoji} {hr_drift_pct:.1f}%",
                                     f"Entry: {hr_entry:.4f} → Now: {hr_now:.4f}")
                        with hd2:
                            # Calculate impact: how much spread changed due to HR drift alone
                            p2_now = mon.get('price2_now', pos.get('entry_price2', 1))
                            hr_impact = abs(hr_now - hr_entry) * p2_now
                            st.metric("Влияние на спред", f"{hr_impact:.4f}",
                                     "USD сдвиг от дрейфа HR")
                        with hd3:
                            # Rebalance suggestion
                            if hr_drift_pct > 15:
                                st.metric("Ребаланс", "🔴 НУЖЕН",
                                         f"HR изменился на {hr_drift_pct:.0f}%")
                            elif hr_drift_pct > 10:
                                st.metric("Ребаланс", "🟡 Рассмотрите",
                                         f"HR дрейфует")
                            else:
                                st.metric("Ребаланс", "✅ Не нужен", "Дрейф в норме")
                        
                        if hr_drift_pct > 100:
                            st.error(
                                f"🚨 **ОШИБКА ВВОДА HR?** Дрейф {hr_drift_pct:.0f}% — "
                                f"Entry={hr_entry:.4f}, Now={hr_now:.4f}. "
                                f"Вероятно HR был введён неверно при открытии позиции. "
                                f"**Проверьте и пересоздайте позицию с правильным HR!**")
                        elif hr_drift_pct > 20:
                            st.error(
                                f"🚨 **HR ДРЕЙФ КРИТИЧЕСКИЙ: {hr_drift_pct:.1f}%**. "
                                f"Entry HR={hr_entry:.4f}, текущий={hr_now:.4f}. "
                                f"Коинтеграция могла разрушиться. Рассмотрите закрытие.")
                        elif hr_drift_pct > 15:
                            st.warning(
                                f"⚠️ **HR дрейф {hr_drift_pct:.1f}%**: Entry={hr_entry:.4f}, "
                                f"Now={hr_now:.4f}. Ребалансируйте позицию или закройте.")
                
                # v3.0: Quality warnings
                for qw in mon.get('quality_warnings', []):
                    st.warning(qw)
                
                # v3.0: Entry readiness assessment
                qd = mon.get('quality_data', {})
                if qd:
                    ea = assess_entry_readiness(qd)
                    with st.expander("📋 Критерии входа (как в сканере)", expanded=False):
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            st.markdown("**🟢 Обязательные:**")
                            for name, met, val in ea['mandatory']:
                                st.markdown(f"  {'✅' if met else '❌'} **{name}** → `{val}`")
                        with ec2:
                            st.markdown("**🔵 Желательные:**")
                            for name, met, val in ea['optional']:
                                st.markdown(f"  {'✅' if met else '⬜'} {name} → `{val}`")
                
                # Chart
                with st.expander("📈 Графики", expanded=False):
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.08,
                                       subplot_titles=['Z-Score', 'Спред'],
                                       row_heights=[0.6, 0.4])
                    
                    ts = mon['timestamps']
                    
                    # Z-score
                    fig.add_trace(go.Scatter(
                        x=ts, y=mon['zscore_series'],
                        name='Z-Score', line=dict(color='#4fc3f7', width=2)
                    ), row=1, col=1)
                    
                    fig.add_hline(y=0, line_dash='dash', line_color='gray', 
                                 opacity=0.5, row=1, col=1)
                    fig.add_hline(y=pos.get('exit_z_target', 0.5), 
                                 line_dash='dot', line_color='#4caf50',
                                 opacity=0.5, row=1, col=1)
                    fig.add_hline(y=-pos.get('exit_z_target', 0.5), 
                                 line_dash='dot', line_color='#4caf50',
                                 opacity=0.5, row=1, col=1)
                    
                    # Entry Z marker
                    entry_dt = datetime.fromisoformat(pos['entry_time'])
                    fig.add_trace(go.Scatter(
                        x=[entry_dt], y=[pos['entry_z']],
                        mode='markers', marker=dict(size=14, color='yellow',
                                                     symbol='star'),
                        name='Entry', showlegend=True
                    ), row=1, col=1)
                    
                    # Spread
                    fig.add_trace(go.Scatter(
                        x=ts, y=mon['spread'],
                        name='Spread', line=dict(color='#ffa726', width=1.5)
                    ), row=2, col=1)
                    
                    fig.update_layout(height=400, template='plotly_dark',
                                     showlegend=False,
                                     margin=dict(l=50, r=30, t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Close button
                col_close1, col_close2, col_close3 = st.columns([2, 2, 1])
                with col_close3:
                    if st.button(f"❌ Закрыть #{pos['id']}", key=f"close_{pos['id']}"):
                        close_position(
                            pos['id'], 
                            mon['price1_now'], mon['price2_now'],
                            mon['z_now'], 'MANUAL'
                        )
                        st.success(f"Позиция #{pos['id']} закрыта | P&L: {mon['pnl_pct']:+.2f}%")
                        st.rerun()
        
        # Total P&L
        st.markdown("---")
        st.metric("📊 Суммарный P&L (открытые)", f"{total_pnl:+.2f}%")
        
        # v5.2: FULL open positions CSV with live monitoring data
        open_rows = []
        for pos in open_positions:
            row = {
                '#': pos['id'],
                'Пара': f"{pos['coin1']}/{pos['coin2']}",
                'Dir': pos['direction'],
                'TF': pos['timeframe'],
                'Entry_Z': pos['entry_z'],
                'Entry_HR': pos.get('entry_hr', 0),
                'Stop_Z': pos.get('stop_z', 4.0),
                'Entry_Time': pos['entry_time'][:16],
                'Entry_Price1': pos.get('entry_price1', 0),
                'Entry_Price2': pos.get('entry_price2', 0),
            }
            # Add live data if available
            try:
                mon = monitor_position(pos, exchange)
                if mon:
                    row.update({
                        'Current_Z': round(mon['z_now'], 4),
                        'Current_HR': round(mon['hr_now'], 4),
                        'P&L_%': round(mon['pnl_pct'], 4),
                        'Hours_In': round(mon['hours_in'], 1),
                        'HL_hours': round(mon['halflife_hours'], 1),
                        'Price1_Now': round(mon['price1_now'], 6),
                        'Price2_Now': round(mon['price2_now'], 6),
                        'Hurst': round(mon.get('hurst', 0.5), 4),
                        'Correlation': round(mon.get('correlation', 0), 4),
                        'P-value': round(mon.get('pvalue', 1.0), 6),
                        'Z_Window': mon.get('z_window', 30),
                        'Exit_Signal': mon.get('exit_signal', ''),
                        'Exit_Urgency': mon.get('exit_urgency', ''),
                        'Z_Toward_Zero': mon.get('z_towards_zero', False),
                        'PnL_Z_Disagree': mon.get('pnl_z_disagree', False),
                        'Quality_Warnings': '; '.join(mon.get('quality_warnings', [])),
                    })
            except Exception:
                pass
            open_rows.append(row)
        
        if open_rows:
            csv_open = pd.DataFrame(open_rows).to_csv(index=False)
            st.download_button("📥 Скачать открытые позиции (CSV)", csv_open,
                f"positions_open_{now_msk().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
                key="open_pos_csv")
            
            # v20.1: Auto-save positions to disk every 10 minutes
            try:
                import os
                os.makedirs("position_exports", exist_ok=True)
                last_auto_save = st.session_state.get('_last_pos_save', 0)
                now_ts = time.time()
                if now_ts - last_auto_save > 600:  # 10 minutes
                    save_path = f"position_exports/positions_open_{now_msk().strftime('%Y%m%d_%H%M')}.csv"
                    pd.DataFrame(open_rows).to_csv(save_path, index=False)
                    st.session_state['_last_pos_save'] = now_ts
                    st.toast(f"💾 Позиции сохранены: {save_path}")
            except Exception:
                pass

with tab2:
    if not closed_positions:
        st.info("📭 Нет закрытых позиций")
    else:
        # Summary
        pnls = [p.get('pnl_pct', 0) for p in closed_positions]
        wins = [p for p in pnls if p > 0]
        
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Сделок", len(closed_positions))
        sc2.metric("Win Rate", f"{len(wins)/len(closed_positions)*100:.0f}%" if closed_positions else "0%")
        sc3.metric("Total P&L", f"{sum(pnls):+.2f}%")
        sc4.metric("Avg P&L", f"{np.mean(pnls):+.2f}%" if pnls else "0%")
        
        # Table
        rows = []
        for p in reversed(closed_positions):
            rows.append({
                '#': p['id'],
                'Пара': f"{p['coin1']}/{p['coin2']}",
                'Dir': p['direction'],
                'TF': p['timeframe'],
                'Entry Z': f"{p['entry_z']:+.2f}",
                'Exit Z': f"{p.get('exit_z', 0):+.2f}",
                'P&L %': f"{p.get('pnl_pct', 0):+.2f}",
                'Причина': p.get('exit_reason', ''),
                'Вход': p['entry_time'][:16],
                'Выход': p.get('exit_time', '')[:16] if p.get('exit_time') else '',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
        # v5.1: CSV export with date in filename
        csv_history = pd.DataFrame(rows).to_csv(index=False)
        # Date range from trades
        dates = [p.get('exit_time', '')[:10] for p in closed_positions if p.get('exit_time')]
        date_suffix = dates[-1] if dates else now_msk().strftime('%Y-%m-%d')
        st.download_button("📥 Скачать историю сделок (CSV)", csv_history,
                          f"trades_history_{date_suffix}_{now_msk().strftime('%H%M')}.csv", "text/csv")

# ═══════════════════════════════════════════════════════
# TAB 3: PORTFOLIO RISK MANAGER (v19.0)
# ═══════════════════════════════════════════════════════
with tab3:
    if not open_positions:
        st.info("📭 Нет открытых позиций для анализа портфеля.")
    else:
        st.markdown("### 📊 Portfolio Risk Manager v2.0")
        
        # === 1. Collect all monitoring data upfront ===
        mon_cache = {}
        for pos in open_positions:
            pair = f"{pos['coin1']}/{pos['coin2']}"
            try:
                mon = monitor_position(pos, exchange)
                if mon:
                    mon_cache[pos['id']] = mon
            except Exception:
                pass
        
        # === 2. Portfolio summary metrics ===
        total_pnl_port = sum(m['pnl_pct'] for m in mon_cache.values())
        n_pos = len(open_positions)
        n_profit = sum(1 for m in mon_cache.values() if m['pnl_pct'] > 0)
        n_loss = sum(1 for m in mon_cache.values() if m['pnl_pct'] < 0)
        
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Позиций", n_pos)
        pc2.metric("Совокупный P&L", f"{total_pnl_port:+.3f}%")
        pc3.metric("Прибыльных", f"{n_profit}/{n_pos}",
                  f"{n_profit/n_pos*100:.0f}%" if n_pos > 0 else "—")
        avg_hours = sum(pos.get('hours_in', 0) for pos in open_positions) / n_pos if n_pos > 0 else 0
        pc4.metric("Ср. время в позиции", f"{avg_hours:.1f}ч")
        
        # === 3. Coin exposure map ===
        st.markdown("#### 🪙 Экспозиция по монетам")
        coin_exposure = {}
        for pos in open_positions:
            c1, c2 = pos['coin1'], pos['coin2']
            d = pos['direction']
            for coin, coin_dir in [(c1, d), (c2, 'SHORT' if d == 'LONG' else 'LONG')]:
                if coin not in coin_exposure:
                    coin_exposure[coin] = {'long': 0, 'short': 0, 'pairs': [], 'pnl': 0.0}
                if coin_dir == 'LONG':
                    coin_exposure[coin]['long'] += 1
                else:
                    coin_exposure[coin]['short'] += 1
                coin_exposure[coin]['pairs'].append(f"{c1}/{c2}")
                mon = mon_cache.get(pos['id'])
                if mon:
                    coin_exposure[coin]['pnl'] += mon['pnl_pct'] / 2  # Split P&L between legs
        
        for coin, data in coin_exposure.items():
            data['net'] = data['long'] - data['short']
            data['total'] = data['long'] + data['short']
        
        sorted_coins = sorted(coin_exposure.items(), key=lambda x: x[1]['total'], reverse=True)
        
        # Concentration metric
        max_coin = sorted_coins[0] if sorted_coins else ('—', {'total': 0})
        max_exposure_pct = max_coin[1]['total'] / (n_pos * 2) * 100 if n_pos > 0 else 0
        
        # Exposure table
        coin_rows = []
        for coin, data in sorted_coins:
            conflict = '🚨 КОНФЛИКТ' if data['long'] > 0 and data['short'] > 0 else ''
            pct_of_port = data['total'] / (n_pos * 2) * 100 if n_pos > 0 else 0
            bar = '█' * int(pct_of_port / 5) + '░' * (20 - int(pct_of_port / 5))
            coin_rows.append({
                'Монета': coin,
                'LONG': data['long'],
                'SHORT': data['short'],
                'Всего': data['total'],
                'Net': f"+{data['net']}" if data['net'] > 0 else str(data['net']),
                '% порт.': f"{pct_of_port:.0f}%",
                'P&L': f"{data['pnl']:+.3f}%",
                'Конфликт': conflict,
                'Пары': ', '.join(set(data['pairs'])),
            })
        if coin_rows:
            st.dataframe(pd.DataFrame(coin_rows), use_container_width=True, hide_index=True)
        
        # === 4. RISK LIMITS CHECK ===
        st.markdown("#### ⚠️ Лимиты риска")
        
        MAX_POSITIONS = 6
        MAX_COIN_EXPOSURE = 3  # max positions per coin
        MAX_CONCENTRATION_PCT = 40  # max % of portfolio in one coin
        
        lc1, lc2, lc3 = st.columns(3)
        
        with lc1:
            pos_ok = n_pos <= MAX_POSITIONS
            st.metric(
                "Позиций", f"{n_pos}/{MAX_POSITIONS}",
                "✅ OK" if pos_ok else "🔴 ПРЕВЫШЕН",
                delta_color="normal" if pos_ok else "inverse"
            )
        
        with lc2:
            max_c = max_coin[1]['total'] if sorted_coins else 0
            coin_ok = max_c <= MAX_COIN_EXPOSURE
            st.metric(
                f"Макс на монету ({max_coin[0]})", f"{max_c}/{MAX_COIN_EXPOSURE}",
                "✅ OK" if coin_ok else "🔴 ПРЕВЫШЕН",
                delta_color="normal" if coin_ok else "inverse"
            )
        
        with lc3:
            conc_ok = max_exposure_pct <= MAX_CONCENTRATION_PCT
            st.metric(
                "Концентрация", f"{max_exposure_pct:.0f}%/{MAX_CONCENTRATION_PCT}%",
                "✅ OK" if conc_ok else "🔴 ПРЕВЫШЕНА",
                delta_color="normal" if conc_ok else "inverse"
            )
        
        # Warnings
        warnings_found = False
        for coin, data in sorted_coins:
            if data['total'] >= MAX_COIN_EXPOSURE:
                st.error(
                    f"🚨 **{coin}** в {data['total']} позициях (лимит: {MAX_COIN_EXPOSURE}). "
                    f"При обвале {coin} на 10% ВСЕ {data['total']} позиции пострадают! "
                    f"**Закройте {data['total'] - MAX_COIN_EXPOSURE + 1} наименее прибыльную.**")
                warnings_found = True
            elif data['total'] >= 2:
                st.warning(f"⚠️ **{coin}** в {data['total']} позициях ({data['long']}L/{data['short']}S)")
                warnings_found = True
            
            if data['long'] > 0 and data['short'] > 0:
                st.error(
                    f"🚨 **{coin}** КОНФЛИКТ: LONG×{data['long']} + SHORT×{data['short']} "
                    f"одновременно → хеджирование самого себя!")
                warnings_found = True
        
        if not warnings_found:
            st.success("✅ Портфель в пределах лимитов.")
        
        # === 5. Position P&L table ===
        st.markdown("#### 📈 P&L по позициям")
        pnl_data = []
        for pos in open_positions:
            pair = f"{pos['coin1']}/{pos['coin2']}"
            mon = mon_cache.get(pos['id'])
            if mon:
                hours_in = pos.get('hours_in', 0)
                pnl_data.append({
                    '#': pos['id'],
                    'Пара': pair,
                    'Dir': pos['direction'],
                    'Entry Z': f"{mon['z_entry']:+.2f}",
                    'Now Z': f"{mon['z_now']:+.2f}",
                    'P&L': f"{mon['pnl_pct']:+.3f}%",
                    'Z→0': '✅' if mon['z_towards_zero'] else '❌',
                    'Часов': f"{hours_in:.1f}",
                    'Сигнал': (mon.get('exit_signal') or '—')[:35],
                })
        if pnl_data:
            st.dataframe(pd.DataFrame(pnl_data), use_container_width=True, hide_index=True)
        
        # === 6. Quick recommendations ===
        st.markdown("#### 💡 Рекомендации")
        recs = []
        
        # Find worst position
        worst_pos = None
        worst_pnl = 0
        for pos in open_positions:
            mon = mon_cache.get(pos['id'])
            if mon and mon['pnl_pct'] < worst_pnl:
                worst_pnl = mon['pnl_pct']
                worst_pos = pos
        
        if worst_pos and worst_pnl < -0.5:
            recs.append(f"🔴 Худшая позиция: **{worst_pos['coin1']}/{worst_pos['coin2']}** "
                       f"(P&L={worst_pnl:+.3f}%). Рассмотрите закрытие.")
        
        # Exit signals
        exits = []
        for pos in open_positions:
            mon = mon_cache.get(pos['id'])
            if mon and mon.get('exit_signal'):
                exits.append(f"**{pos['coin1']}/{pos['coin2']}**: {mon['exit_signal'][:40]}")
        if exits:
            recs.append(f"📍 Сигналы выхода: " + "; ".join(exits))
        
        # Concentration
        for coin, data in sorted_coins:
            if data['total'] >= 3:
                # Find least profitable pair with this coin
                least_profit = None
                least_pnl = 999
                for pos in open_positions:
                    if pos['coin1'] == coin or pos['coin2'] == coin:
                        mon = mon_cache.get(pos['id'])
                        if mon and mon['pnl_pct'] < least_pnl:
                            least_pnl = mon['pnl_pct']
                            least_profit = pos
                if least_profit:
                    recs.append(
                        f"⚠️ Для снижения экспозиции на **{coin}** закройте "
                        f"**{least_profit['coin1']}/{least_profit['coin2']}** "
                        f"(наименее прибыльная: {least_pnl:+.3f}%)")
        
        if recs:
            for r in recs:
                st.markdown(r)
        else:
            st.success("✅ Нет критических рекомендаций. Портфель выглядит здоровым.")
        
        # === 7. Portfolio Download ===
        st.markdown("#### 📥 Экспорт портфеля")
        portfolio_rows = []
        for pos in open_positions:
            mon = mon_cache.get(pos['id'])
            portfolio_rows.append({
                '#': pos['id'],
                'Пара': f"{pos['coin1']}/{pos['coin2']}",
                'Dir': pos['direction'],
                'TF': pos.get('timeframe', '4h'),
                'Entry_Z': pos.get('entry_z', 0),
                'Current_Z': mon['z_now'] if mon else '',
                'Entry_HR': pos.get('entry_hr', 0),
                'Current_HR': mon['hr_now'] if mon else '',
                'HR_Drift_%': round(abs(mon['hr_now'] - pos.get('entry_hr', 0)) / max(0.0001, pos.get('entry_hr', 0)) * 100, 1) if mon else '',
                'P&L_%': round(mon['pnl_pct'], 4) if mon else '',
                'Hours_In': round(mon['hours_in'], 1) if mon else '',
                'HL_hours': round(mon.get('halflife_hours', 0), 1) if mon else '',
                'Hurst': round(mon.get('hurst', 0.5), 3) if mon else '',
                'P-value': round(mon.get('pvalue', 1.0), 4) if mon else '',
                'Z_Toward_Zero': mon.get('z_towards_zero', '') if mon else '',
                'Exit_Signal': (mon.get('exit_signal', '') or '')[:40] if mon else '',
                'Entry_Time': pos.get('entry_time', ''),
                'Entry_P1': pos.get('entry_price1', ''),
                'Entry_P2': pos.get('entry_price2', ''),
                'Now_P1': mon.get('price1_now', '') if mon else '',
                'Now_P2': mon.get('price2_now', '') if mon else '',
            })
        if portfolio_rows:
            portfolio_df = pd.DataFrame(portfolio_rows)
            csv_portfolio = portfolio_df.to_csv(index=False)
            
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("📥 Портфель (CSV)", csv_portfolio,
                    f"portfolio_{now_msk().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
                    key="portfolio_csv_btn")
            with dl2:
                # Also auto-save to disk
                try:
                    import os
                    os.makedirs("position_exports", exist_ok=True)
                    pf_path = f"position_exports/portfolio_{now_msk().strftime('%Y%m%d_%H%M')}.csv"
                    portfolio_df.to_csv(pf_path, index=False)
                    st.caption(f"💾 Сохранено: {pf_path}")
                except Exception:
                    pass

# Auto refresh
# v27: Non-blocking auto-refresh — rerun OUTSIDE try/except
_monitor_needs_rerun = False
if auto_refresh:
    _mon_wait = CFG('monitor', 'refresh_interval_sec', 120)
    st.info(f"⏱️ Авто-обновление через {_mon_wait}с...")
    time.sleep(_mon_wait)
    _monitor_needs_rerun = True

if _monitor_needs_rerun:
    st.rerun()

# ═══════════════════════════════════════════════════════
# TAB 4: R8 Performance Tracker
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 Performance Tracker (R8)")
    st.caption("Накопительная статистика по всем закрытым сделкам")
    
    # Load history from persistent file + current session closed
    history = load_trade_history()
    
    # Also include closed positions from current session that might not be in history yet
    history_ids = {t.get('id', 0) for t in history}
    for cp in closed_positions:
        if cp.get('id', 0) not in history_ids:
            history.append({
                'id': cp.get('id', 0),
                'pair': f"{cp.get('coin1', '')}/{cp.get('coin2', '')}",
                'coin1': cp.get('coin1', ''), 'coin2': cp.get('coin2', ''),
                'direction': cp.get('direction', ''),
                'timeframe': cp.get('timeframe', '4h'),
                'entry_z': cp.get('entry_z', 0), 'exit_z': cp.get('exit_z', 0),
                'entry_hr': cp.get('entry_hr', 0), 'pnl_pct': cp.get('pnl_pct', 0),
                'entry_time': cp.get('entry_time', ''),
                'exit_time': cp.get('exit_time', ''),
                'exit_reason': cp.get('exit_reason', ''),
                'entry_price1': cp.get('entry_price1', 0),
                'entry_price2': cp.get('entry_price2', 0),
                'exit_price1': cp.get('exit_price1', 0),
                'exit_price2': cp.get('exit_price2', 0),
                'notes': cp.get('notes', ''),
                'best_pnl': cp.get('best_pnl', 0),
            })
    
    if not history:
        st.info("📭 Нет закрытых сделок в истории. Закройте позицию чтобы начать накапливать статистику.")
        st.markdown("💡 **Ручной импорт:** Загрузите CSV с прошлыми сделками.")
        
        uploaded_hist = st.file_uploader("📤 Импорт истории (CSV)", type=['csv'], key='hist_import')
        if uploaded_hist:
            try:
                import io
                hist_df = pd.read_csv(io.StringIO(uploaded_hist.getvalue().decode()))
                st.dataframe(hist_df)
                
                if st.button("✅ Импортировать эти сделки"):
                    for _, row in hist_df.iterrows():
                        trade = {
                            'id': int(row.get('#', row.get('id', 0))),
                            'coin1': str(row.get('Пара', '')).split('/')[0] if '/' in str(row.get('Пара', '')) else '',
                            'coin2': str(row.get('Пара', '')).split('/')[1] if '/' in str(row.get('Пара', '')) else '',
                            'direction': row.get('Dir', row.get('direction', '')),
                            'timeframe': row.get('TF', row.get('timeframe', '4h')),
                            'entry_z': float(str(row.get('Entry Z', row.get('entry_z', 0))).replace('+', '')),
                            'exit_z': float(str(row.get('Exit Z', row.get('exit_z', 0))).replace('+', '')),
                            'entry_hr': float(row.get('entry_hr', 1.0)),
                            'pnl_pct': float(str(row.get('P&L %', row.get('pnl_pct', 0))).replace('+', '').replace('%', '')),
                            'entry_time': str(row.get('Вход', row.get('entry_time', ''))),
                            'exit_time': str(row.get('Выход', row.get('exit_time', ''))),
                            'exit_reason': str(row.get('Причина', row.get('exit_reason', 'MANUAL'))),
                            'notes': '',
                            'best_pnl': 0,
                            'entry_price1': 0, 'entry_price2': 0,
                            'exit_price1': 0, 'exit_price2': 0,
                        }
                        save_trade_to_history(trade)
                    st.success(f"✅ Импортировано {len(hist_df)} сделок!")
                    st.rerun()
            except Exception as ex:
                st.error(f"❌ Ошибка импорта: {ex}")
    else:
        # === DASHBOARD ===
        pnls = [float(t.get('pnl_pct', 0)) for t in history]
        n_trades = len(history)
        total_pnl = sum(pnls)
        winners = sum(1 for p in pnls if p > 0)
        losers = sum(1 for p in pnls if p < 0)
        win_rate = winners / n_trades * 100 if n_trades > 0 else 0
        avg_pnl = total_pnl / n_trades if n_trades > 0 else 0
        avg_win = np.mean([p for p in pnls if p > 0]) if winners > 0 else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if losers > 0 else 0
        pf = abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p < 0)) if losers > 0 and sum(p for p in pnls if p < 0) != 0 else float('inf')
        
        # Max drawdown
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        max_dd = min(drawdown) if len(drawdown) > 0 else 0
        
        # Metrics row 1
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Всего сделок", n_trades)
        m2.metric("Суммарный P&L", f"{total_pnl:+.2f}%",
                 delta=f"{total_pnl:+.2f}%", delta_color="normal")
        m3.metric("Win Rate", f"{win_rate:.0f}%",
                 delta=f"{winners}W / {losers}L")
        m4.metric("Avg P&L", f"{avg_pnl:+.3f}%")
        m5.metric("Profit Factor", f"{pf:.2f}" if pf < 100 else "∞")
        
        # Metrics row 2
        m6, m7, m8, m9 = st.columns(4)
        m6.metric("Avg Win", f"{avg_win:+.3f}%")
        m7.metric("Avg Loss", f"{avg_loss:+.3f}%")
        m8.metric("Max Drawdown", f"{max_dd:+.2f}%")
        
        # Best streak
        streaks = []
        current_streak = 0
        for p in pnls:
            if p > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)
        best_streak = max(streaks) if streaks else 0
        m9.metric("Best Win Streak", f"{best_streak}")
        
        # === EQUITY CURVE ===
        st.markdown("#### 📈 Equity Curve")
        import plotly.graph_objects as go
        
        cum_pnl = list(np.cumsum(pnls))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[0] + cum_pnl,
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='#00c853', width=2),
            marker=dict(size=5, color=['green' if p > 0 else 'red' for p in [0] + list(pnls)])
        ))
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Cumulative P&L %",
            xaxis_title="Trade #",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # === BY PAIR ANALYSIS ===
        st.markdown("#### 🪙 Статистика по парам")
        pair_stats = {}
        for t in history:
            pair = t.get('pair', f"{t.get('coin1','')}/{t.get('coin2','')}")
            if pair not in pair_stats:
                pair_stats[pair] = {'pnls': [], 'count': 0}
            pair_stats[pair]['pnls'].append(float(t.get('pnl_pct', 0)))
            pair_stats[pair]['count'] += 1
        
        pair_rows = []
        for pair, stats in sorted(pair_stats.items(), key=lambda x: sum(x[1]['pnls']), reverse=True):
            ppnls = stats['pnls']
            pair_rows.append({
                'Пара': pair,
                'Сделок': stats['count'],
                'Total P&L': f"{sum(ppnls):+.2f}%",
                'Avg P&L': f"{np.mean(ppnls):+.3f}%",
                'WR': f"{sum(1 for p in ppnls if p > 0)/len(ppnls)*100:.0f}%",
                'Best': f"{max(ppnls):+.2f}%",
                'Worst': f"{min(ppnls):+.2f}%",
            })
        if pair_rows:
            st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)
        
        # === BY DAY ANALYSIS ===
        st.markdown("#### 📅 Статистика по дням")
        day_stats = {}
        for t in history:
            day = str(t.get('exit_time', t.get('entry_time', '')))[:10]
            if day and day != 'None':
                if day not in day_stats:
                    day_stats[day] = {'pnls': [], 'count': 0}
                day_stats[day]['pnls'].append(float(t.get('pnl_pct', 0)))
                day_stats[day]['count'] += 1
        
        day_rows = []
        for day, stats in sorted(day_stats.items()):
            dpnls = stats['pnls']
            day_rows.append({
                'Дата': day,
                'Сделок': stats['count'],
                'Total P&L': f"{sum(dpnls):+.2f}%",
                'WR': f"{sum(1 for p in dpnls if p > 0)/len(dpnls)*100:.0f}%",
                'Avg P&L': f"{np.mean(dpnls):+.3f}%",
            })
        if day_rows:
            st.dataframe(pd.DataFrame(day_rows), use_container_width=True, hide_index=True)
        
        # === TRADES TABLE ===
        st.markdown("#### 📋 Все сделки")
        trade_rows = []
        for t in reversed(history):
            trade_rows.append({
                '#': t.get('id', ''),
                'Пара': t.get('pair', ''),
                'Dir': t.get('direction', ''),
                'Entry Z': f"{float(t.get('entry_z', 0)):+.2f}",
                'Exit Z': f"{float(t.get('exit_z', 0)):+.2f}",
                'P&L': f"{float(t.get('pnl_pct', 0)):+.2f}%",
                'Причина': t.get('exit_reason', ''),
                'Вход': str(t.get('entry_time', ''))[-5:],
                'Выход': str(t.get('exit_time', ''))[-5:],
            })
        if trade_rows:
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
        
        # === EXPORT ===
        st.markdown("#### 📥 Экспорт")
        hist_df = pd.DataFrame(history)
        csv_hist = hist_df.to_csv(index=False)
        st.download_button("📥 Скачать полную историю (CSV)", csv_hist,
                          f"trade_history_{now_msk().strftime('%Y%m%d_%H%M')}.csv",
                          "text/csv", key="hist_export_btn")

st.divider()
st.caption("""
Как добавить позицию:
1. Найди 🟢 ВХОД в скринере
2. Скопируй данные: Coin1, Coin2, Direction, Z, HR, цены
3. Введи в форму слева → "Загрузить цены + Добавить"
4. Монитор покажет когда закрывать + предупредит если пара потеряла качество
""")
