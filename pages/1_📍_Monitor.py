"""
Pairs Position Monitor v5.0
═══════════════════════════════════════════════════════
Изменения v5.0:
  [CRITICAL] Dual Z-score: z_static (entry_hr) для выходов + z_dynamic (Kalman) для здоровья
  [NEW] Phantom Tracking — отслеживание пары после закрытия (24ч) → «не режу ли прибыль?»
  [NEW] Position Sizing — рекомендация объёма из сканера ($100 по умолчанию)
  [NEW] Performance Tracker с основанием сделки (signal/ready + вход/слабый/условно)
  [NEW] Время входа/выхода в HH:MM МСК
  [NEW] Кнопки «Закрыть сделку» + «Закрыть все»
  [NEW] Pattern Analysis с основанием сделки
  [NEW] Rally Filter Variant A — блокировка + alert + cooldown, Оптимум (1.0/1.2/-0.5)
  [FIX] Exit signals теперь по STATIC Z-score → совпадение с реальным P&L

Запуск: streamlit run pairs_position_monitor_v5.py
═══════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
import json
import os
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint

# ═══════════════════════════════════════════════════════
# TIMEZONE & HELPERS
# ═══════════════════════════════════════════════════════
MSK = timezone(timedelta(hours=3))

def now_msk():
    return datetime.now(MSK)

def to_msk(dt_str_or_obj):
    if not dt_str_or_obj: return ''
    try:
        dt = datetime.fromisoformat(str(dt_str_or_obj)) if isinstance(dt_str_or_obj, str) else dt_str_or_obj
        if dt.tzinfo is None: dt = dt.replace(tzinfo=MSK)
        else: dt = dt.astimezone(MSK)
        return dt.strftime("%H:%M")
    except: return str(dt_str_or_obj)[:5]

def to_msk_full(dt_str_or_obj):
    if not dt_str_or_obj: return ''
    try:
        dt = datetime.fromisoformat(str(dt_str_or_obj)) if isinstance(dt_str_or_obj, str) else dt_str_or_obj
        if dt.tzinfo is None: dt = dt.replace(tzinfo=MSK)
        else: dt = dt.astimezone(MSK)
        return dt.strftime("%d.%m %H:%M")
    except: return str(dt_str_or_obj)[:16]


# ═══════════════════════════════════════════════════════
# ENTRY READINESS (единая логика с сканером)
# ═══════════════════════════════════════════════════════

def assess_entry_readiness(p):
    mandatory = [
        ('Статус ≥ READY', p.get('signal', 'NEUTRAL') in ('SIGNAL', 'READY'), p.get('signal', 'NEUTRAL')),
        ('|Z| ≥ Thr', abs(p.get('zscore', 0)) >= p.get('threshold', 2.0),
         f"|{p.get('zscore',0):.2f}| vs {p.get('threshold',2.0):.1f}"),
        ('Q ≥ 50', p.get('quality_score', 0) >= 50, f"Q={p.get('quality_score', 0)}"),
        ('Dir ≠ NONE', p.get('direction', 'NONE') != 'NONE', p.get('direction', 'NONE')),
    ]
    all_mandatory = all(m[1] for m in mandatory)
    fdr_ok = p.get('fdr_passed', False)
    stab_ok = p.get('stability_passed', 0) >= 3
    hurst_ok = p.get('hurst', 0.5) < 0.35
    optional = [
        ('FDR ✅', fdr_ok, '✅' if fdr_ok else '❌'),
        ('Conf=HIGH', p.get('confidence', 'LOW') == 'HIGH', p.get('confidence', 'LOW')),
        ('S ≥ 60', p.get('signal_score', 0) >= 60, f"S={p.get('signal_score', 0)}"),
        ('ρ ≥ 0.5', p.get('correlation', 0) >= 0.5, f"ρ={p.get('correlation', 0):.2f}"),
        ('Stab ≥ 3/4', stab_ok, f"{p.get('stability_passed',0)}/{p.get('stability_total',4)}"),
        ('Hurst < 0.35', hurst_ok, f"H={p.get('hurst', 0.5):.3f}"),
    ]
    opt_count = sum(1 for _, met, _ in optional if met)
    fdr_bypass = (not fdr_ok and p.get('quality_score', 0) >= 70 and
                  stab_ok and p.get('adf_passed', False) and hurst_ok)
    if all_mandatory:
        if opt_count >= 4:
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
# v5.0: POSITION SIZING
# Рекомендация объёма на сделку ($100 базовый размер)
# ═══════════════════════════════════════════════════════

DEFAULT_TRADE_SIZE = 100.0  # $100

# ═══════════════════════════════════════════════════════
# КОМИССИЯ (round-trip)
# OKX: 4 ноги × 0.1% = 0.40%  (вход 2 ноги + выход 2 ноги)
# Можно менять в sidebar
# ═══════════════════════════════════════════════════════
COMMISSION_ROUND_TRIP_PCT = 0.40  # % на полный цикл сделки

def recommend_position_size(quality_score=0, confidence='LOW',
                            entry_readiness='', hurst=0.5, correlation=0.0):
    """
    Рекомендация объёма позиции на основе параметров сканера.
    Базовый размер = $100.
    
    Множители:
      Quality ≥ 80 + HIGH conf → 1.0× ($100)
      Quality ≥ 60 + MEDIUM    → 0.75× ($75)
      Quality < 60 или LOW     → 0.50× ($50)
    
    Модификаторы:
      🟢 ВХОД    → +0%   (стандарт)
      🟡 УСЛОВНО → −10%
      🟡 СЛАБЫЙ  → −25%
      Hurst > 0.45 → −20%
      ρ < 0.3      → −15%
    """
    base = DEFAULT_TRADE_SIZE
    
    # Quality + Confidence множитель
    if quality_score >= 80 and confidence == 'HIGH':
        multiplier = 1.0
    elif quality_score >= 60 and confidence in ('HIGH', 'MEDIUM'):
        multiplier = 0.75
    else:
        multiplier = 0.50
    
    # Модификаторы
    mod = 1.0
    if 'СЛАБЫЙ' in str(entry_readiness):
        mod -= 0.25
    elif 'УСЛОВНО' in str(entry_readiness):
        mod -= 0.10
    
    if hurst > 0.45:
        mod -= 0.20
    if correlation < 0.3:
        mod -= 0.15
    
    mod = max(0.25, mod)  # минимум 25% от базы
    
    recommended = round(base * multiplier * mod, 0)
    return max(25.0, recommended)  # минимум $25


# ═══════════════════════════════════════════════════════
# CORE MATH
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
    except:
        beta = np.array([0.0, 1.0])
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
    else:
        w = 30
    w = min(w, max(10, n // 2))
    zs = np.full(n, np.nan)
    for i in range(w, n):
        lb = spread[i - w:i]; med = np.median(lb)
        mad = np.median(np.abs(lb - med)) * 1.4826
        if mad < 1e-10:
            s = np.std(lb)
            zs[i] = (spread[i] - np.mean(lb)) / s if s > 1e-10 else 0
        else:
            zs[i] = (spread[i] - med) / mad
    return zs, w


def calc_halflife(spread, dt=None):
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
    hl = np.log(2) / theta
    return float(hl) if hl < 999 else 999


def calc_hurst(series, min_window=8):
    x = np.array(series, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 50: return 0.5
    y = np.cumsum(x - np.mean(x))
    scales, flucts = [], []
    min_seg = max(min_window, 4); max_seg = n // 4
    for seg_len in range(min_seg, max_seg + 1, max(1, (max_seg - min_seg) // 20)):
        n_segs = n // seg_len
        if n_segs < 2: continue
        f2_list = []
        for i in range(n_segs):
            seg = y[i * seg_len:(i + 1) * seg_len]
            t = np.arange(len(seg))
            if len(seg) < 2: continue
            coeffs = np.polyfit(t, seg, 1); trend = np.polyval(coeffs, t)
            f2_list.append(np.mean((seg - trend) ** 2))
        if f2_list: scales.append(seg_len); flucts.append(np.sqrt(np.mean(f2_list)))
    if len(scales) < 4: return 0.5
    log_s = np.log(scales); log_f = np.log(np.array(flucts) + 1e-10)
    coeffs = np.polyfit(log_s, log_f, 1)
    pred = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_f - pred)**2); ss_tot = np.sum((log_f - np.mean(log_f))**2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    if r_sq < 0.8: return 0.5
    return float(np.clip(coeffs[0], 0.01, 0.99))


def calc_correlation(p1, p2, window=60):
    n = min(len(p1), len(p2))
    if n < window: return 0.0
    r1 = np.diff(np.log(p1[-n:] + 1e-10)); r2 = np.diff(np.log(p2[-n:] + 1e-10))
    if len(r1) < 10: return 0.0
    return float(np.corrcoef(r1[-window:], r2[-window:])[0, 1])


def calc_cointegration_pvalue(p1, p2):
    try: _, pval, _ = coint(p1, p2); return float(pval)
    except: return 1.0


# ═══════════════════════════════════════════════════════
# v5.0: STATIC SPREAD — честный Z-score позиции
# ═══════════════════════════════════════════════════════

def calc_static_spread(p1_array, p2_array, entry_hr, entry_intercept=0.0):
    """
    Фактический спред позиции = P1 - entry_HR × P2 - entry_intercept.
    
    Это спред, который ТОЧНО отражает P&L твоей реальной позиции,
    в отличие от Калман-спреда, который пересчитывает HR на каждом баре.
    """
    return np.array(p1_array, float) - entry_hr * np.array(p2_array, float) - entry_intercept


def calc_static_zscore(static_spread, halflife_bars=None, min_w=10, max_w=60):
    """Z-score от статичного спреда (по фиксированному HR)."""
    return calc_zscore(static_spread, halflife_bars, min_w, max_w)


# ═══════════════════════════════════════════════════════
# RALLY FILTER — Variant A, Оптимум (1.0/1.2/-0.5)
# ═══════════════════════════════════════════════════════

RALLY_CONFIG = {
    'alert_threshold': 1.0,
    'block_threshold': 1.2,
    'resume_threshold': -0.5,
    'lookback_bars': 6,
    'cooldown_bars': 3,
}
RALLY_STATE_FILE = "rally_state.json"

def load_rally_state():
    if os.path.exists(RALLY_STATE_FILE):
        with open(RALLY_STATE_FILE) as f: return json.load(f)
    return {'status': 'NORMAL', 'btc_move_pct': 0.0, 'block_since': None,
            'cooldown_until': None, 'last_check': None}

def save_rally_state(state):
    with open(RALLY_STATE_FILE, 'w') as f: json.dump(state, f, indent=2, default=str)

def check_rally_filter(exchange_name, timeframe='4h'):
    state = load_rally_state(); cfg = RALLY_CONFIG
    try:
        ex = getattr(ccxt, exchange_name)({'enableRateLimit': True}); ex.load_markets()
        ohlcv = ex.fetch_ohlcv('BTC/USDT', timeframe, limit=cfg['lookback_bars'] + 1)
        if len(ohlcv) < 2:
            state['status'] = 'NORMAL'; save_rally_state(state); return state
        closes = [c[4] for c in ohlcv]
        btc_move = (closes[-1] - closes[0]) / closes[0] * 100
        state['btc_move_pct'] = round(btc_move, 2)
        state['last_check'] = now_msk().isoformat()
        abs_move = abs(btc_move); prev = state['status']
        if abs_move >= cfg['block_threshold']:
            state['status'] = 'BLOCKED'
            if prev != 'BLOCKED': state['block_since'] = now_msk().isoformat()
        elif abs_move >= cfg['alert_threshold']:
            if prev == 'BLOCKED':
                state['status'] = 'COOLDOWN'
                hpb = {'1h': 1, '4h': 4, '1d': 24}.get(timeframe, 4)
                state['cooldown_until'] = (now_msk() + timedelta(hours=cfg['cooldown_bars']*hpb)).isoformat()
            elif prev != 'COOLDOWN':
                state['status'] = 'ALERT'
        elif btc_move <= cfg['resume_threshold'] and prev in ('BLOCKED', 'COOLDOWN'):
            state['status'] = 'COOLDOWN'
            hpb = {'1h': 1, '4h': 4, '1d': 24}.get(timeframe, 4)
            state['cooldown_until'] = (now_msk() + timedelta(hours=cfg['cooldown_bars']*hpb)).isoformat()
        else:
            if prev == 'COOLDOWN' and state.get('cooldown_until'):
                cd_end = datetime.fromisoformat(state['cooldown_until'])
                if cd_end.tzinfo is None: cd_end = cd_end.replace(tzinfo=MSK)
                if now_msk() >= cd_end:
                    state['status'] = 'NORMAL'; state['cooldown_until'] = None
            elif prev not in ('COOLDOWN', 'BLOCKED'):
                state['status'] = 'NORMAL'
    except Exception as e:
        state['_error'] = str(e)
    save_rally_state(state); return state

def render_rally_banner(state):
    status = state.get('status', 'NORMAL'); move = state.get('btc_move_pct', 0)
    cfg = RALLY_CONFIG
    if status == 'BLOCKED':
        st.error(f"🚫 **RALLY BLOCKED** | BTC {move:+.2f}% (≥±{cfg['block_threshold']}%) — Новые позиции ЗАПРЕЩЕНЫ"); return False
    elif status == 'COOLDOWN':
        cd = to_msk_full(state.get('cooldown_until', ''))
        st.warning(f"⏳ **RALLY COOLDOWN** | BTC {move:+.2f}% | До {cd} МСК"); return False
    elif status == 'ALERT':
        st.warning(f"⚠️ **RALLY ALERT** | BTC {move:+.2f}% — С осторожностью"); return True
    else:
        st.caption(f"🟢 Rally Filter: NORMAL | BTC {move:+.2f}%"); return True


# ═══════════════════════════════════════════════════════
# POSITIONS FILE (JSON persistence)
# ═══════════════════════════════════════════════════════
POSITIONS_FILE = "positions.json"

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f: return json.load(f)
    return []

def save_positions(positions):
    with open(POSITIONS_FILE, 'w') as f: json.dump(positions, f, indent=2, default=str)

def add_position(coin1, coin2, direction, entry_z, entry_hr,
                 entry_price1, entry_price2, timeframe, notes="",
                 max_hold_hours=72, pnl_stop_pct=-5.0,
                 signal_basis="", entry_readiness="",
                 entry_intercept=0.0, recommended_size=100.0):
    """v5.0: добавлены entry_intercept (для static spread) и recommended_size."""
    positions = load_positions()
    pos = {
        'id': len(positions) + 1,
        'coin1': coin1, 'coin2': coin2, 'direction': direction,
        'entry_z': entry_z, 'entry_hr': entry_hr,
        'entry_intercept': entry_intercept,  # v5.0: для static spread
        'entry_price1': entry_price1, 'entry_price2': entry_price2,
        'entry_time': now_msk().isoformat(),
        'timeframe': timeframe, 'status': 'OPEN', 'notes': notes,
        'exit_z_target': 0.5, 'stop_z': 4.5,
        'max_hold_hours': max_hold_hours, 'pnl_stop_pct': pnl_stop_pct,
        'signal_basis': signal_basis,
        'entry_readiness': entry_readiness,
        'recommended_size': recommended_size,  # v5.0: $
        # v5.0: Phantom tracking fields (заполняются при закрытии)
        'phantom_track_until': None,
        'phantom_max_pnl': None,
        'phantom_min_pnl': None,
        'phantom_last_pnl': None,
        'phantom_last_check': None,
        'best_pnl_during_trade': 0.0,  # v5.0: лучший P&L во время жизни сделки
    }
    positions.append(pos); save_positions(positions)
    return pos

def close_position(pos_id, exit_price1, exit_price2, exit_z, reason,
                   z_static=None, best_pnl=None):
    """v5.0: сохраняет z_static и запускает phantom tracking на 24ч."""
    positions = load_positions()
    for p in positions:
        if p['id'] == pos_id and p['status'] == 'OPEN':
            p['status'] = 'CLOSED'
            p['exit_price1'] = exit_price1; p['exit_price2'] = exit_price2
            p['exit_z'] = exit_z
            p['exit_z_static'] = z_static  # v5.0
            p['exit_time'] = now_msk().isoformat()
            p['exit_reason'] = reason
            r1 = (exit_price1 - p['entry_price1']) / p['entry_price1'] if p['entry_price1'] > 0 else 0
            r2 = (exit_price2 - p['entry_price2']) / p['entry_price2'] if p['entry_price2'] > 0 else 0
            hr = p['entry_hr']
            raw = (r1 - hr * r2) if p['direction'] == 'LONG' else (-r1 + hr * r2)
            pnl_gross = raw / (1 + abs(hr)) * 100
            # v5.0: вычитаем комиссию на круг (4 ноги × 0.1%)
            p['pnl_pct'] = round(pnl_gross - COMMISSION_ROUND_TRIP_PCT, 3)
            p['pnl_gross_pct'] = round(pnl_gross, 3)  # P&L до комиссий (для анализа)
            if best_pnl is not None:
                p['best_pnl_during_trade'] = best_pnl
            # v5.0: Phantom tracking — 24 часа после закрытия
            p['phantom_track_until'] = (now_msk() + timedelta(hours=24)).isoformat()
            p['phantom_max_pnl'] = p['pnl_pct']
            p['phantom_min_pnl'] = p['pnl_pct']
            p['phantom_last_pnl'] = p['pnl_pct']
            break
    save_positions(positions)

def close_all_positions(exchange_name):
    positions = load_positions(); closed = 0
    for p in positions:
        if p['status'] == 'OPEN':
            p1 = get_current_price(exchange_name, p['coin1'])
            p2 = get_current_price(exchange_name, p['coin2'])
            close_position(p['id'], p1 or p.get('entry_price1', 0),
                           p2 or p.get('entry_price2', 0), 0, 'CLOSE_ALL')
            closed += 1
    return closed

def update_phantom_tracking(pos, exchange_name):
    """
    v5.0: Phantom Tracking — продолжает отслеживать пару после закрытия.
    Обновляет phantom_max_pnl / phantom_min_pnl / phantom_last_pnl.
    Отвечает на вопрос: «Не режу ли я прибыль?»
    """
    if pos['status'] != 'CLOSED': return None
    if not pos.get('phantom_track_until'): return None
    
    track_end = datetime.fromisoformat(pos['phantom_track_until'])
    if track_end.tzinfo is None: track_end = track_end.replace(tzinfo=MSK)
    if now_msk() > track_end: return None  # tracking expired
    
    p1 = get_current_price(exchange_name, pos['coin1'])
    p2 = get_current_price(exchange_name, pos['coin2'])
    if not p1 or not p2: return None
    
    r1 = (p1 - pos['entry_price1']) / pos['entry_price1'] if pos['entry_price1'] > 0 else 0
    r2 = (p2 - pos['entry_price2']) / pos['entry_price2'] if pos['entry_price2'] > 0 else 0
    hr = pos['entry_hr']
    raw = (r1 - hr * r2) if pos['direction'] == 'LONG' else (-r1 + hr * r2)
    phantom_pnl = round(raw / (1 + abs(hr)) * 100 - COMMISSION_ROUND_TRIP_PCT, 3)  # с комиссией
    
    # Update
    positions = load_positions()
    for p in positions:
        if p['id'] == pos['id']:
            p['phantom_last_pnl'] = phantom_pnl
            p['phantom_last_check'] = now_msk().isoformat()
            if phantom_pnl > (p.get('phantom_max_pnl') or -999):
                p['phantom_max_pnl'] = phantom_pnl
            if phantom_pnl < (p.get('phantom_min_pnl') or 999):
                p['phantom_min_pnl'] = phantom_pnl
            break
    save_positions(positions)
    
    return {
        'phantom_pnl': phantom_pnl,
        'phantom_max': pos.get('phantom_max_pnl', phantom_pnl),
        'phantom_min': pos.get('phantom_min_pnl', phantom_pnl),
        'exit_pnl': pos.get('pnl_pct', 0),
    }


# ═══════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=120)
def fetch_prices(exchange_name, coin, timeframe, lookback_bars=300):
    try:
        ex = getattr(ccxt, exchange_name)({'enableRateLimit': True}); ex.load_markets()
        ohlcv = ex.fetch_ohlcv(f"{coin}/USDT", timeframe, limit=lookback_bars)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return None

def get_current_price(exchange_name, coin):
    try:
        ex = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        ticker = ex.fetch_ticker(f"{coin}/USDT")
        return ticker['last']
    except: return None


# ═══════════════════════════════════════════════════════
# v5.0: MONITOR LOGIC — DUAL Z-SCORE
# ═══════════════════════════════════════════════════════

def monitor_position(pos, exchange_name):
    """
    v5.0: Двойной Z-score мониторинг.
    
    z_dynamic (Kalman) — показывает здоровье коинтеграции, HR drift
    z_static  (Entry HR) — РЕАЛЬНЫЙ Z-score позиции, используется для EXIT сигналов
    
    Это решает проблему рассинхрона Z/PnL: когда Kalman-Z = 0, 
    но P&L в минусе, потому что Kalman "убежал" вслед за ценой.
    """
    c1, c2 = pos['coin1'], pos['coin2']
    tf = pos['timeframe']
    n_bars = {'1h': 300, '4h': 300, '1d': 120}.get(tf, 300)
    
    df1 = fetch_prices(exchange_name, c1, tf, n_bars)
    df2 = fetch_prices(exchange_name, c2, tf, n_bars)
    if df1 is None or df2 is None: return None
    
    merged = pd.merge(df1[['ts', 'c']], df2[['ts', 'c']], on='ts', suffixes=('_1', '_2'))
    if len(merged) < 50: return None
    
    p1 = merged['c_1'].values; p2 = merged['c_2'].values
    ts = merged['ts'].tolist()
    
    # ── DYNAMIC (Kalman) ──
    kf = kalman_hr(p1, p2)
    if kf is None: return None
    spread_dynamic = kf['spread']; hr_current = kf['hr']
    
    # ── STATIC (Entry HR) ── v5.0: КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ
    entry_hr = pos['entry_hr']
    entry_intercept = pos.get('entry_intercept', 0.0)
    spread_static = calc_static_spread(p1, p2, entry_hr, entry_intercept)
    
    # Half-life и Z-score
    dt_ou = {'1h': 1/24, '4h': 1/6, '1d': 1.0}.get(tf, 1/6)
    hpb = {'1h': 1, '4h': 4, '1d': 24}.get(tf, 4)
    hl_days = calc_halflife(spread_dynamic, dt=dt_ou)
    hl_hours = hl_days * 24 if hl_days < 999 else 999
    hl_bars = (hl_hours / hpb) if hl_hours < 999 else None
    
    # Dynamic Z-score (для здоровья коинтеграции)
    zs_dynamic, zw = calc_zscore(spread_dynamic, halflife_bars=hl_bars)
    z_dynamic = float(zs_dynamic[~np.isnan(zs_dynamic)][-1]) if any(~np.isnan(zs_dynamic)) else 0
    
    # Static Z-score (для EXIT сигналов) — v5.0
    zs_static, _ = calc_static_zscore(spread_static, halflife_bars=hl_bars)
    z_static = float(zs_static[~np.isnan(zs_static)][-1]) if any(~np.isnan(zs_static)) else 0
    
    # Quality metrics
    hurst = calc_hurst(spread_dynamic)
    corr = calc_correlation(p1, p2, window=min(60, len(p1) // 3))
    pvalue = calc_cointegration_pvalue(p1, p2)
    
    quality_data = {
        'signal': 'SIGNAL' if abs(z_dynamic) >= 2.0 else ('READY' if abs(z_dynamic) >= 1.5 else 'NEUTRAL'),
        'zscore': z_dynamic, 'threshold': 2.0,
        'quality_score': max(0, int(100 - pvalue * 200 - max(0, hurst - 0.35) * 200)),
        'direction': pos['direction'],
        'fdr_passed': pvalue < 0.01,
        'confidence': 'HIGH' if (hurst < 0.4 and pvalue < 0.03) else ('MEDIUM' if pvalue < 0.05 else 'LOW'),
        'signal_score': max(0, int(abs(z_dynamic) / 2.0 * 50 + (0.5 - hurst) * 100)),
        'correlation': corr,
        'stability_passed': 3 if pvalue < 0.05 else 1, 'stability_total': 4,
        'hurst': hurst, 'adf_passed': pvalue < 0.05,
    }
    
    # P&L (с комиссией на круг)
    r1 = (p1[-1] - pos['entry_price1']) / pos['entry_price1'] if pos['entry_price1'] > 0 else 0
    r2 = (p2[-1] - pos['entry_price2']) / pos['entry_price2'] if pos['entry_price2'] > 0 else 0
    hr = pos['entry_hr']
    raw_pnl = (r1 - hr * r2) if pos['direction'] == 'LONG' else (-r1 + hr * r2)
    pnl_gross = raw_pnl / (1 + abs(hr)) * 100
    pnl_pct = pnl_gross - COMMISSION_ROUND_TRIP_PCT  # v5.0: после комиссий
    
    # Track best P&L during trade
    best_pnl = max(pos.get('best_pnl_during_trade', 0), pnl_pct)
    positions = load_positions()
    for pp in positions:
        if pp['id'] == pos['id'] and pp['status'] == 'OPEN':
            pp['best_pnl_during_trade'] = best_pnl
    save_positions(positions)
    
    # Time in trade
    entry_dt = datetime.fromisoformat(pos['entry_time'])
    if entry_dt.tzinfo is None: entry_dt = entry_dt.replace(tzinfo=MSK)
    hours_in = (now_msk() - entry_dt).total_seconds() / 3600
    
    # ── EXIT SIGNALS — теперь по z_static! ── v5.0
    exit_signal = None; exit_urgency = 0
    ez = pos.get('exit_z_target', 0.5); sz = pos.get('stop_z', 4.5)
    max_hours = pos.get('max_hold_hours', 72); pnl_stop = pos.get('pnl_stop_pct', -5.0)
    
    # Используем z_static для exit signals (честный Z реальной позиции)
    z_exit = z_static
    
    if pos['direction'] == 'LONG':
        if -ez <= z_exit <= ez:
            exit_signal = '✅ MEAN REVERT (static Z) — закрывать!'; exit_urgency = 2
        elif z_exit > 1.0:
            exit_signal = '✅ OVERSHOOT (static Z) — фиксировать!'; exit_urgency = 2
        elif z_exit < -sz:
            exit_signal = '🛑 STOP LOSS (static Z)!'; exit_urgency = 2
    else:
        if -ez <= z_exit <= ez:
            exit_signal = '✅ MEAN REVERT (static Z) — закрывать!'; exit_urgency = 2
        elif z_exit < -1.0:
            exit_signal = '✅ OVERSHOOT (static Z) — фиксировать!'; exit_urgency = 2
        elif z_exit > sz:
            exit_signal = '🛑 STOP LOSS (static Z)!'; exit_urgency = 2
    
    if pnl_pct <= pnl_stop and exit_urgency < 2:
        exit_signal = f'🛑 STOP LOSS (P&L {pnl_pct:.1f}%)!'; exit_urgency = 2
    if hours_in > max_hours and exit_urgency < 2:
        exit_signal = f'⏰ TIMEOUT ({hours_in:.0f}ч > {max_hours:.0f}ч)'; exit_urgency = 1
    elif hours_in > max_hours * 0.75 and exit_urgency == 0:
        exit_signal = f'⚠️ {hours_in:.0f}ч (лимит {max_hours:.0f}ч)'; exit_urgency = 1
    
    # Z drift warning (dynamic vs static divergence)
    z_drift = abs(z_dynamic - z_static)
    
    # Quality warnings
    qw = []
    if hurst >= 0.45: qw.append(f"⚠️ Hurst={hurst:.3f} ≥ 0.45")
    if pvalue >= 0.10: qw.append(f"⚠️ P-value={pvalue:.3f}")
    if corr < 0.2: qw.append(f"⚠️ ρ={corr:.2f} < 0.2")
    if z_drift > 1.5:
        qw.append(f"⚠️ Z-drift={z_drift:.2f} — HR Калмана сильно уплыл от entry HR!")
    
    return {
        'z_static': z_static,           # v5.0: РЕАЛЬНЫЙ Z позиции
        'z_dynamic': z_dynamic,          # v5.0: теоретический Kalman Z
        'z_now': z_static,               # для обратной совместимости → static
        'z_drift': z_drift,              # v5.0: расхождение static vs dynamic
        'z_entry': pos['entry_z'],
        'pnl_pct': pnl_pct,
        'best_pnl': best_pnl,
        'price1_now': p1[-1], 'price2_now': p2[-1],
        'hr_now': hr_current, 'hr_entry': pos['entry_hr'],
        'exit_signal': exit_signal, 'exit_urgency': exit_urgency,
        'hours_in': hours_in,
        'spread_static': spread_static,  # v5.0
        'spread_dynamic': spread_dynamic,
        'zscore_series_static': zs_static,    # v5.0
        'zscore_series_dynamic': zs_dynamic,
        'timestamps': ts, 'hr_series': kf['hrs'],
        'halflife_hours': hl_hours, 'z_window': zw,
        'hurst': hurst, 'correlation': corr, 'pvalue': pvalue,
        'quality_data': quality_data, 'quality_warnings': qw,
    }


# ═══════════════════════════════════════════════════════
# PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════

def analyze_patterns(closed_positions):
    if not closed_positions: return {}
    results = {'by_basis': {}, 'by_readiness': {}, 'by_direction': {},
               'by_timeframe': {}, 'by_exit_reason': {}, 'by_basis_readiness': {}}
    for p in closed_positions:
        pnl = p.get('pnl_pct', 0)
        basis = p.get('signal_basis', 'N/A') or 'N/A'
        readiness = p.get('entry_readiness', 'N/A') or 'N/A'
        br = f"{basis} | {readiness}"
        for gv, gd in [(basis, results['by_basis']), (readiness, results['by_readiness']),
                        (p.get('direction','N/A'), results['by_direction']),
                        (p.get('timeframe','N/A'), results['by_timeframe']),
                        (p.get('exit_reason','N/A'), results['by_exit_reason']),
                        (br, results['by_basis_readiness'])]:
            if gv not in gd: gd[gv] = {'pnls': [], 'count': 0}
            gd[gv]['pnls'].append(pnl); gd[gv]['count'] += 1
    for gn, gd in results.items():
        for k in gd:
            pnls = gd[k]['pnls']; wins = [p for p in pnls if p > 0]
            gd[k]['total_pnl'] = round(sum(pnls), 2)
            gd[k]['avg_pnl'] = round(np.mean(pnls), 3) if pnls else 0
            gd[k]['win_rate'] = round(len(wins)/len(pnls)*100, 1) if pnls else 0
            gd[k]['best'] = round(max(pnls), 2) if pnls else 0
            gd[k]['worst'] = round(min(pnls), 2) if pnls else 0
    return results


# ═══════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════

st.set_page_config(page_title="Position Monitor v5", page_icon="📍", layout="wide")

st.markdown("""<style>
    .exit-signal { padding: 15px; border-radius: 10px; font-size: 1.2em; font-weight: bold; text-align: center; margin: 10px 0; }
    .signal-exit { background: #1b5e20; color: #a5d6a7; }
    .signal-stop { background: #b71c1c; color: #ef9a9a; }
    .dual-z { display: flex; gap: 10px; margin: 8px 0; }
    .z-static { background: #1b5e20; color: white; padding: 8px 16px; border-radius: 6px; font-weight: bold; }
    .z-dynamic { background: #37474f; color: #b0bec5; padding: 8px 16px; border-radius: 6px; }
</style>""", unsafe_allow_html=True)

st.title("📍 Pairs Position Monitor")
st.caption("v5.0 | Static Z-score + Phantom Tracking + Position Sizing + Rally Filter")

# ═══════ SIDEBAR ═══════
with st.sidebar:
    st.header("⚙️ Настройки")
    exchange = st.selectbox("Биржа", ['okx', 'bybit', 'binance'], index=0)
    auto_refresh = st.checkbox("Авто-обновление (2 мин)", value=False)
    
    st.divider(); st.header("🛡️ Rally Filter")
    rally_enabled = st.checkbox("Rally Filter", value=True)
    if rally_enabled:
        st.caption(f"Alert ±{RALLY_CONFIG['alert_threshold']}% | Block ±{RALLY_CONFIG['block_threshold']}%")
    
    st.divider(); st.header("💰 Position Sizing")
    trade_size = st.number_input("Базовый размер ($)", value=100.0, step=25.0, min_value=25.0)
    DEFAULT_TRADE_SIZE = trade_size
    
    st.divider(); st.header("➕ Новая позиция")
    with st.form("add_pos"):
        fc1, fc2 = st.columns(2)
        with fc1: new_c1 = st.text_input("Coin 1", "").upper().strip()
        with fc2: new_c2 = st.text_input("Coin 2", "").upper().strip()
        new_dir = st.selectbox("Направление", ["LONG", "SHORT"])
        new_tf = st.selectbox("Таймфрейм", ['1h', '4h', '1d'], index=1)
        fc3, fc4 = st.columns(2)
        with fc3: new_z = st.number_input("Entry Z", value=2.0, step=0.1)
        with fc4: new_hr = st.number_input("Hedge Ratio", value=1.0, step=0.01, format="%.4f")
        fc5, fc6 = st.columns(2)
        with fc5: new_p1 = st.number_input("Цена Coin1", value=0.0, step=0.01, format="%.6f")
        with fc6: new_p2 = st.number_input("Цена Coin2", value=0.0, step=0.01, format="%.6f")
        new_intercept = st.number_input("Intercept (из сканера)", value=0.0, step=0.001, format="%.6f",
                                        help="Kalman intercept на момент входа. 0 = авто")
        new_notes = st.text_input("Заметки", "")
        st.markdown("**📋 Основание сделки**")
        fb1, fb2 = st.columns(2)
        with fb1: new_basis = st.selectbox("Статус сканера", ["SIGNAL", "READY"])
        with fb2: new_readiness = st.selectbox("Готовность", ["🟢 ВХОД", "🟡 СЛАБЫЙ", "🟡 УСЛОВНО"])
        st.markdown("**⚠️ Риск-менеджмент**")
        fr1, fr2 = st.columns(2)
        with fr1: new_max_h = st.number_input("Max часов", value=72, step=12)
        with fr2: new_pnl_stop = st.number_input("P&L Stop %", value=-5.0, step=0.5)
        fetch_btn = st.form_submit_button("📥 Загрузить цены + Добавить")
    
    if fetch_btn and new_c1 and new_c2:
        can_add = True
        if rally_enabled:
            rs = check_rally_filter(exchange, new_tf)
            if rs.get('status') == 'BLOCKED':
                st.error(f"🚫 Rally BLOCKED"); can_add = False
            elif rs.get('status') == 'COOLDOWN':
                st.warning("⏳ Rally COOLDOWN"); can_add = False
        if can_add:
            if new_p1 == 0 or new_p2 == 0:
                with st.spinner("Загружаю цены..."):
                    p1l = get_current_price(exchange, new_c1); p2l = get_current_price(exchange, new_c2)
                    if p1l and p2l: new_p1 = p1l; new_p2 = p2l
                    else: st.error("Не удалось загрузить цены")
            if new_p1 > 0 and new_p2 > 0:
                rec_size = recommend_position_size(
                    quality_score=70, confidence='MEDIUM',
                    entry_readiness=new_readiness, hurst=0.4, correlation=0.5)
                pos = add_position(new_c1, new_c2, new_dir, new_z, new_hr,
                                   new_p1, new_p2, new_tf, new_notes,
                                   new_max_h, new_pnl_stop, new_basis, new_readiness,
                                   entry_intercept=new_intercept, recommended_size=rec_size)
                st.success(f"✅ #{pos['id']}: {new_dir} {new_c1}/{new_c2} | ${rec_size:.0f}")
                st.rerun()

# ═══════ RALLY BANNER ═══════
if rally_enabled:
    rally_state = check_rally_filter(exchange, '4h')
    trades_allowed = render_rally_banner(rally_state)
else:
    trades_allowed = True

# ═══════ MAIN ═══════
positions = load_positions()
open_pos = [p for p in positions if p['status'] == 'OPEN']
closed_pos = [p for p in positions if p['status'] == 'CLOSED']

# Phantom tracking: закрытые позиции с активным отслеживанием
phantom_pos = [p for p in closed_pos 
               if p.get('phantom_track_until') and 
               datetime.fromisoformat(p['phantom_track_until']).replace(tzinfo=MSK) > now_msk()]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"📍 Портфель ({len(open_pos)})",
    f"👻 Phantom ({len(phantom_pos)})",
    f"📋 Performance ({len(closed_pos)})",
    "📊 Patterns",
    "🛡️ Rally",
])

# ═══════ TAB 1: ПОРТФЕЛЬ ═══════
with tab1:
    if not open_pos:
        st.info("📭 Нет открытых позиций.")
    else:
        h1, h2 = st.columns([3, 1])
        with h1: st.subheader(f"P&L по позициям ({len(open_pos)})")
        with h2:
            if st.button("🛑 Закрыть ВСЕ", type="secondary", use_container_width=True):
                n = close_all_positions(exchange); st.success(f"✅ Закрыто {n}"); st.rerun()
        
        total_pnl = 0; monitor_results = {}
        for pos in open_pos:
            with st.spinner(f"Обновляю {pos['coin1']}/{pos['coin2']}..."):
                mon = monitor_position(pos, exchange)
            if mon: monitor_results[pos['id']] = mon; total_pnl += mon['pnl_pct']
        
        st.metric("📊 Суммарный P&L", f"{total_pnl:+.2f}%")
        st.markdown("---")
        
        for pos in open_pos:
            mon = monitor_results.get(pos['id'])
            if not mon:
                st.error(f"❌ Нет данных: {pos['coin1']}/{pos['coin2']}"); continue
            with st.container():
                st.markdown("---")
                if mon['exit_signal']:
                    (st.error if 'STOP' in mon['exit_signal'] else st.success)(mon['exit_signal'])
                
                hdr1, hdr2, hdr3 = st.columns([4, 2, 1])
                de = '🟢' if pos['direction'] == 'LONG' else '🔴'
                pn = f"{pos['coin1']}/{pos['coin2']}"
                with hdr1: st.subheader(f"{de} {pos['direction']} | {pn} | #{pos['id']}")
                with hdr2:
                    st.caption(f"Основание: {pos.get('signal_basis','?')} | {pos.get('entry_readiness','?')}")
                    st.caption(f"Вход: {to_msk_full(pos['entry_time'])} МСК | ${pos.get('recommended_size', 100):.0f}")
                with hdr3:
                    if st.button(f"❌ Закрыть", key=f"cl_{pos['id']}", use_container_width=True):
                        close_position(pos['id'], mon['price1_now'], mon['price2_now'],
                                       mon['z_dynamic'], 'MANUAL',
                                       z_static=mon['z_static'], best_pnl=mon.get('best_pnl'))
                        st.success(f"#{pos['id']} закрыта | P&L: {mon['pnl_pct']:+.2f}%"); st.rerun()
                
                # v5.0: DUAL Z-SCORE
                z1, z2, z3 = st.columns(3)
                z1.metric("🎯 Z Static (РЕАЛЬНЫЙ)", f"{mon['z_static']:+.2f}",
                         help="Z-score по entry_hr — отражает P&L реальной позиции")
                z2.metric("📐 Z Dynamic (Kalman)", f"{mon['z_dynamic']:+.2f}",
                         help="Z-score по текущему Kalman HR — здоровье коинтеграции")
                z3.metric("⚡ Z Drift", f"{mon['z_drift']:.2f}",
                         delta="⚠️ HR дрейф" if mon['z_drift'] > 1.0 else "✅ OK",
                         help="Расхождение static vs dynamic. >1.5 = HR сильно уплыл")
                
                # KPIs
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("P&L", f"{mon['pnl_pct']:+.2f}%")
                c2.metric("Best P&L", f"{mon.get('best_pnl', 0):+.2f}%")
                c3.metric("HR", f"{mon['hr_now']:.4f}", f"entry: {mon['hr_entry']:.4f}")
                c4.metric(pos['coin1'], f"${mon['price1_now']:.4f}")
                c5.metric(pos['coin2'], f"${mon['price2_now']:.4f}")
                c6.metric("В позиции", f"{mon['hours_in']:.0f}ч", f"HL: {mon['halflife_hours']:.0f}ч")
                
                # Quality
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Hurst", f"{mon.get('hurst',0.5):.3f}", "🟢" if mon.get('hurst',0.5)<0.45 else "🔴")
                q2.metric("P-value", f"{mon.get('pvalue',1.0):.4f}", "✅" if mon.get('pvalue',1.0)<0.05 else "⚠️")
                q3.metric("ρ", f"{mon.get('correlation',0):.3f}", "🟢" if mon.get('correlation',0)>=0.5 else "⚠️")
                q4.metric("Z-win", f"{mon.get('z_window',30)}")
                for w in mon.get('quality_warnings', []): st.warning(w)
                
                # Charts
                with st.expander("📈 Графики (Static vs Dynamic Z)", expanded=False):
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                       subplot_titles=['Z-Score: Static (зелёный) vs Dynamic (серый)', 'Спред'],
                                       row_heights=[0.6, 0.4])
                    # Static Z (основной)
                    fig.add_trace(go.Scatter(x=mon['timestamps'], y=mon['zscore_series_static'],
                        name='Z Static', line=dict(color='#4caf50', width=2.5)), row=1, col=1)
                    # Dynamic Z (вспомогательный)
                    fig.add_trace(go.Scatter(x=mon['timestamps'], y=mon['zscore_series_dynamic'],
                        name='Z Dynamic', line=dict(color='#78909c', width=1, dash='dot')), row=1, col=1)
                    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5, row=1, col=1)
                    edt = datetime.fromisoformat(pos['entry_time'])
                    fig.add_trace(go.Scatter(x=[edt], y=[pos['entry_z']], mode='markers',
                        marker=dict(size=14, color='yellow', symbol='star'), name='Entry'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=mon['timestamps'], y=mon['spread_static'],
                        name='Static Spread', line=dict(color='#ffa726', width=1.5)), row=2, col=1)
                    fig.update_layout(height=450, template='plotly_dark', margin=dict(l=50,r=30,t=30,b=30))
                    st.plotly_chart(fig, use_container_width=True)

# ═══════ TAB 2: PHANTOM TRACKING ═══════
with tab2:
    st.subheader("👻 Phantom Tracking — «Не режу ли я прибыль?»")
    st.caption("Отслеживает закрытые сделки ещё 24ч после выхода, показывая max P&L, который можно было получить")
    
    if not phantom_pos:
        st.info("📭 Нет позиций на phantom-отслеживании. Они появятся через 24ч после закрытия сделки.")
    else:
        for pos in phantom_pos:
            with st.spinner(f"👻 {pos['coin1']}/{pos['coin2']}..."):
                ph = update_phantom_tracking(pos, exchange)
            
            if ph:
                pn = f"{pos['coin1']}/{pos['coin2']}"
                exit_pnl = pos.get('pnl_pct', 0)
                phantom_max = ph['phantom_max']
                cut_profit = phantom_max - exit_pnl if phantom_max > exit_pnl else 0
                
                with st.container():
                    st.markdown("---")
                    h1, h2, h3, h4 = st.columns(4)
                    h1.metric(f"👻 {pn}", f"Exit P&L: {exit_pnl:+.2f}%")
                    h2.metric("Phantom Now", f"{ph['phantom_pnl']:+.2f}%",
                             delta=f"{ph['phantom_pnl'] - exit_pnl:+.2f}% vs exit")
                    h3.metric("Phantom MAX", f"{phantom_max:+.2f}%",
                             delta=f"упущено: {cut_profit:+.2f}%" if cut_profit > 0 else "✅ не резал")
                    h4.metric("Best во время сделки", f"{pos.get('best_pnl_during_trade', 0):+.2f}%")
                    
                    if cut_profit > 0.5:
                        st.warning(f"⚠️ После закрытия пара дала ещё **+{cut_profit:.2f}%**. Возможно, стоит увеличить время удержания.")
                    elif cut_profit > 0:
                        st.info(f"ℹ️ Небольшой дополнительный рост: +{cut_profit:.2f}%. В пределах нормы.")
                    else:
                        st.success("✅ Правильно закрыли — пара не выросла дальше.")

# ═══════ TAB 3: PERFORMANCE TRACKER ═══════
with tab3:
    if not closed_pos:
        st.info("📭 Нет закрытых позиций")
    else:
        st.subheader("📋 Performance Tracker")
        pnls = [p.get('pnl_pct', 0) for p in closed_pos]; wins = [x for x in pnls if x > 0]
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Сделок", len(closed_pos))
        sc2.metric("Win Rate", f"{len(wins)/len(closed_pos)*100:.0f}%")
        sc3.metric("Total P&L", f"{sum(pnls):+.2f}%")
        sc4.metric("Avg P&L", f"{np.mean(pnls):+.2f}%")
        pw = sum(x for x in pnls if x > 0); pl = abs(sum(x for x in pnls if x < 0))
        sc5.metric("PF", f"{pw/pl:.2f}" if pl > 0 else "∞")
        st.markdown("---")
        
        rows = []
        for p in reversed(closed_pos):
            cut = ''
            if p.get('phantom_max_pnl') is not None and p.get('pnl_pct') is not None:
                delta = p['phantom_max_pnl'] - p['pnl_pct']
                cut = f"+{delta:.2f}%" if delta > 0.1 else "✅"
            rows.append({
                '#': p['id'], 'Пара': f"{p['coin1']}/{p['coin2']}", 'Dir': p['direction'],
                'TF': p['timeframe'], 'Основание': p.get('signal_basis', ''),
                'Готовность': p.get('entry_readiness', ''), 'Size $': p.get('recommended_size', 100),
                'Entry Z': f"{p['entry_z']:+.2f}", 'Exit Z': f"{p.get('exit_z',0):+.2f}",
                'Z Static': f"{p.get('exit_z_static',0):+.2f}" if p.get('exit_z_static') else '',
                'P&L %': f"{p.get('pnl_pct',0):+.2f}",
                'Best P&L': f"{p.get('best_pnl_during_trade',0):+.2f}%",
                'Упущено': cut,
                'Причина': p.get('exit_reason', ''),
                'Вход МСК': to_msk_full(p.get('entry_time','')),
                'Выход МСК': to_msk_full(p.get('exit_time','')),
                'Вход ЧЧ:ММ': to_msk(p.get('entry_time','')),
                'Выход ЧЧ:ММ': to_msk(p.get('exit_time','')),
            })
        df_t = pd.DataFrame(rows)
        st.dataframe(df_t, use_container_width=True, hide_index=True)
        
        if len(pnls) > 1:
            st.markdown("---"); st.subheader("📈 Equity Curve")
            cum = np.cumsum(pnls[::-1])
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=cum, mode='lines+markers', line=dict(color='#4fc3f7', width=2),
                marker=dict(size=6, color=['#4caf50' if x>0 else '#f44336' for x in pnls[::-1]])))
            fig_eq.add_hline(y=0, line_dash='dash', line_color='gray')
            fig_eq.update_layout(height=300, template='plotly_dark', xaxis_title='#', yaxis_title='Cumulative %',
                                 margin=dict(l=50,r=30,t=30,b=30))
            st.plotly_chart(fig_eq, use_container_width=True)
        
        st.download_button("📥 CSV", df_t.to_csv(index=False),
                          f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

# ═══════ TAB 4: PATTERN ANALYSIS ═══════
with tab4:
    if not closed_pos:
        st.info("📭 Нет данных")
    else:
        st.subheader("📊 Pattern Analysis")
        patterns = analyze_patterns(closed_pos)
        
        def rpt(title, data, icon="📊"):
            if not data: return
            st.markdown(f"### {icon} {title}")
            rows = []
            for k, s in sorted(data.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
                rows.append({'Группа': k, 'N': s['count'], 'WR': f"{s['win_rate']:.0f}%",
                    'Total': f"{s['total_pnl']:+.2f}%", 'Avg': f"{s['avg_pnl']:+.3f}%",
                    'Best': f"{s['best']:+.2f}%", 'Worst': f"{s['worst']:+.2f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
        rpt("По основанию + готовности", patterns.get('by_basis_readiness', {}), "🎯")
        cp1, cp2 = st.columns(2)
        with cp1:
            rpt("По основанию", patterns.get('by_basis', {}), "📡")
            rpt("По готовности", patterns.get('by_readiness', {}), "🚦")
        with cp2:
            rpt("По направлению", patterns.get('by_direction', {}), "↕️")
            rpt("По причине выхода", patterns.get('by_exit_reason', {}), "🚪")
        rpt("По таймфрейму", patterns.get('by_timeframe', {}), "⏰")
        
        bd = patterns.get('by_basis_readiness', {})
        if bd:
            st.markdown("---"); st.subheader("💡 Выводы")
            best = max(bd.items(), key=lambda x: x[1]['avg_pnl'])
            worst = min(bd.items(), key=lambda x: x[1]['avg_pnl'])
            if best[1]['count'] >= 2:
                st.success(f"✅ Лучшая: **{best[0]}** — avg {best[1]['avg_pnl']:+.3f}%, WR {best[1]['win_rate']:.0f}%")
            if worst[1]['count'] >= 2:
                st.error(f"❌ Худшая: **{worst[0]}** — avg {worst[1]['avg_pnl']:+.3f}%")

# ═══════ TAB 5: RALLY FILTER ═══════
with tab5:
    st.subheader("🛡️ Rally Filter — Variant A")
    st.caption("Блокировка + Alert + Cooldown | Оптимум (1.0/1.2/-0.5)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Alert", f"±{RALLY_CONFIG['alert_threshold']}%")
    p2.metric("Block", f"±{RALLY_CONFIG['block_threshold']}%")
    p3.metric("Resume", f"{RALLY_CONFIG['resume_threshold']}%")
    p4.metric("Cooldown", f"{RALLY_CONFIG['cooldown_bars']} баров")
    
    if rally_enabled:
        rs = load_rally_state()
        sc = {'NORMAL': '🟢', 'ALERT': '🟡', 'BLOCKED': '🔴', 'COOLDOWN': '⏳'}
        s1, s2, s3 = st.columns(3)
        s1.metric("Статус", f"{sc.get(rs.get('status','NORMAL'),'❓')} {rs.get('status','NORMAL')}")
        s2.metric("BTC", f"{rs.get('btc_move_pct',0):+.2f}%")
        s3.metric("Проверка", to_msk(rs.get('last_check','')) if rs.get('last_check') else '—')
        if st.button("🔄 Обновить"): check_rally_filter(exchange, '4h'); st.rerun()
    
    st.markdown("""---
### 📐 Схема
```
BTC < ±1.0%  →  🟢 NORMAL   → OK
BTC ≥ ±1.0%  →  🟡 ALERT    → осторожность
BTC ≥ ±1.2%  →  🔴 BLOCKED  → запрещено
BTC retrace ≤ -0.5%  →  ⏳ COOLDOWN (3 бара)
```""")

# Auto refresh
if auto_refresh: time.sleep(120); st.rerun()

st.divider()
st.caption("""
**Position Monitor v5.0** | Static Z-score + Phantom Tracking + Position Sizing + Rally Filter

Ключевые изменения v5.0:
• Z Static (по entry_hr) для exit сигналов — РЕШАЕТ проблему рассинхрона Z/PnL
• Z Dynamic (Kalman) — мониторинг здоровья коинтеграции
• Phantom Tracking — 24ч после закрытия → «не режу ли прибыль?»
• Position Sizing — от $25 до $100 в зависимости от Q, Conf, Readiness
""")
