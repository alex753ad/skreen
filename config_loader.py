"""
config_loader.py — Единый конфиг для всех приложений v27.
Загружает config.yaml, даёт дефолты если файл отсутствует.

Использование:
    from config_loader import CFG
    entry_z = CFG('strategy', 'entry_z')         # 1.8
    commission = CFG('strategy', 'commission_pct') # 0.10
    refresh = CFG('scanner', 'refresh_interval_min') # 10
    
    # С дефолтом:
    val = CFG('strategy', 'new_param', default=42)
"""
import os

_DEFAULTS = {
    'strategy': {
        'entry_z': 1.8, 'exit_z': 0.8, 'stop_z_offset': 2.0, 'min_stop_z': 4.0,
        'take_profit_pct': 1.5, 'stop_loss_pct': -5.0, 'max_hold_hours': 72,
        'micro_bt_max_bars': 6, 'min_hurst': 0.45, 'warn_hurst': 0.48,
        'min_correlation': 0.20, 'hr_naked_threshold': 0.15, 'max_pvalue': 0.15,
        'commission_pct': 0.10, 'slippage_pct': 0.05,
    },
    'scanner': {
        'coins_limit': 100, 'timeframe': '4h', 'lookback_days': 50,
        'exchange': 'okx', 'refresh_interval_min': 10,
    },
    'monitor': {
        'refresh_interval_sec': 120, 'exit_z_target': 0.5, 'pnl_stop_pct': -5.0,
        'trailing_z_bounce': 0.8, 'time_warning_ratio': 1.0, 'time_exit_ratio': 1.5,
        'time_critical_ratio': 2.0, 'overshoot_deep_z': 1.0,
        'pnl_trailing_threshold': 0.5, 'pnl_trailing_fraction': 0.4,
        'hurst_critical': 0.50, 'hurst_warning': 0.48, 'hurst_border': 0.45,
        'pvalue_warning': 0.10, 'correlation_warning': 0.20,
    },
    'backtester': {
        'n_bars': 300, 'max_bars': 50, 'min_bars': 2,
        'n_folds_wf': 3, 'train_pct': 0.65,
    },
    'z_velocity': {
        'lookback': 5, 'excellent_min_vel': 0.1, 'decel_threshold': 0.05,
    },
}

_config_data = None
_config_path = None


def _load():
    """Load config once (lazy singleton)."""
    global _config_data, _config_path
    if _config_data is not None:
        return
    
    _config_data = {}
    for section, vals in _DEFAULTS.items():
        _config_data[section] = dict(vals)
    
    # Try to load config.yaml
    paths = [
        'config.yaml',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'),
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                import yaml
                with open(path, 'r') as f:
                    user = yaml.safe_load(f) or {}
                _merge(user)
                _config_path = path
                return
            except ImportError:
                _merge(_parse_simple(path))
                _config_path = path
                return
            except Exception:
                pass


def _merge(user_cfg):
    """Merge user config over defaults."""
    for section, vals in user_cfg.items():
        if isinstance(vals, dict) and section in _config_data:
            _config_data[section].update(vals)
        elif isinstance(vals, dict):
            _config_data[section] = vals


def _parse_simple(path):
    """Parse YAML without PyYAML (handles simple flat sections)."""
    result = {}
    section = None
    with open(path, 'r') as f:
        for line in f:
            s = line.rstrip()
            if not s or s.lstrip().startswith('#'):
                continue
            indent = len(s) - len(s.lstrip())
            s = s.strip()
            if indent == 0 and s.endswith(':'):
                section = s[:-1]
                result[section] = {}
            elif ':' in s and section is not None:
                k, v = s.split(':', 1)
                k, v = k.strip(), v.strip()
                if '#' in v:
                    v = v[:v.index('#')].strip()
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                elif v.startswith("'") and v.endswith("'"):
                    v = v[1:-1]
                elif v.lower() in ('true', 'yes'):
                    v = True
                elif v.lower() in ('false', 'no'):
                    v = False
                else:
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                result[section][k] = v
    return result


def CFG(section, key=None, default=None):
    """
    Get config value.
    
    CFG('strategy', 'entry_z')       → 1.8
    CFG('strategy', 'entry_z', 2.0)  → 1.8 (from config) or 2.0 (if missing)
    CFG('strategy')                  → dict of all strategy params
    """
    _load()
    if key is None:
        return _config_data.get(section, {})
    return _config_data.get(section, {}).get(key, default)


def CFG_path():
    """Return path to loaded config file (or None)."""
    _load()
    return _config_path


def CFG_reload():
    """Force reload from disk."""
    global _config_data
    _config_data = None
    _load()


# ═══════════════════════════════════════════════════════
# PAIR MEMORY — v27: Track per-pair trade history
# ═══════════════════════════════════════════════════════

PAIR_MEMORY_FILE = 'pair_memory.json'

def pair_memory_load():
    """Load pair memory from disk."""
    import json
    try:
        with open(PAIR_MEMORY_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def pair_memory_save(data):
    """Save pair memory to disk."""
    import json
    try:
        with open(PAIR_MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass


def pair_memory_update(pair, pnl_pct, hold_hours, direction, entry_z, exit_z):
    """Update pair memory with a closed trade."""
    mem = pair_memory_load()
    if pair not in mem:
        mem[pair] = {'trades': 0, 'wins': 0, 'total_pnl': 0, 'pnls': [],
                     'avg_hold': 0, 'best_pnl': -999, 'worst_pnl': 999,
                     'last_trade': '', 'directions': {}}
    p = mem[pair]
    p['trades'] += 1
    if pnl_pct > 0:
        p['wins'] += 1
    p['total_pnl'] = round(p['total_pnl'] + pnl_pct, 4)
    p['pnls'] = (p.get('pnls', []) + [round(pnl_pct, 4)])[-20:]  # keep last 20
    p['avg_hold'] = round((p.get('avg_hold', 0) * (p['trades'] - 1) + hold_hours) / p['trades'], 1)
    p['best_pnl'] = max(p.get('best_pnl', -999), pnl_pct)
    p['worst_pnl'] = min(p.get('worst_pnl', 999), pnl_pct)
    from datetime import datetime, timezone, timedelta
    p['last_trade'] = datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d %H:%M')
    d = p.get('directions', {})
    d[direction] = d.get(direction, 0) + 1
    p['directions'] = d
    pair_memory_save(mem)
    return p


def pair_memory_get(pair):
    """Get pair memory stats, or None."""
    mem = pair_memory_load()
    return mem.get(pair)


def pair_memory_summary(pair):
    """One-line summary for display."""
    p = pair_memory_get(pair)
    if not p or p.get('trades', 0) == 0:
        return None
    wr = p['wins'] / p['trades'] * 100
    avg = p['total_pnl'] / p['trades']
    return (f"📝 {p['trades']} сделок, WR={wr:.0f}%, "
            f"avg={avg:+.2f}%, total={p['total_pnl']:+.2f}%, "
            f"hold={p['avg_hold']:.0f}ч")


# ═══════════════════════════════════════════════════════
# R7 ML SCORING — weighted feature scoring model
# ═══════════════════════════════════════════════════════

def ml_score(pair_data):
    """
    R7: ML-like scoring model for trade quality prediction.
    Returns: {'score': 0-100, 'grade': A/B/C/D/F, 'factors': {...}, 'recommendation': str}
    
    Uses logistic-weighted features calibrated from real trade outcomes.
    """
    import math
    
    factors = {}
    
    # 1. Z-score strength (0-20 pts)
    z = abs(pair_data.get('zscore', 0))
    entry_z = CFG('strategy', 'entry_z', 1.8)
    z_ratio = z / entry_z if entry_z > 0 else 0
    z_pts = min(20, z_ratio * 10)  # 2x threshold = 20 pts
    factors['z_strength'] = round(z_pts, 1)
    
    # 2. μBT Quick% (0-25 pts) — strongest predictor from real results
    mbt_q = pair_data.get('mbt_quick', 0)
    mbt_pts = mbt_q / 100 * 25
    factors['mbt_quick'] = round(mbt_pts, 1)
    
    # 3. Hurst quality (0-15 pts) — lower is better
    hurst = pair_data.get('hurst', 0.5)
    if hurst < 0.1:
        h_pts = 15
    elif hurst < 0.3:
        h_pts = 12
    elif hurst < 0.45:
        h_pts = 8
    else:
        h_pts = 0
    factors['hurst'] = h_pts
    
    # 4. Correlation (0-10 pts)
    corr = abs(pair_data.get('correlation', 0))
    c_pts = min(10, corr * 12)  # ρ=0.83 → 10 pts
    factors['correlation'] = round(c_pts, 1)
    
    # 5. Statistical tests (0-10 pts)
    stat_pts = 0
    if pair_data.get('adf_passed'): stat_pts += 3
    if pair_data.get('johansen_coint'): stat_pts += 4
    if pair_data.get('fdr_passed'): stat_pts += 3
    factors['statistics'] = stat_pts
    
    # 6. Regime + MTF (0-10 pts)
    regime_pts = 0
    if pair_data.get('regime') == 'MEAN_REVERT': regime_pts += 5
    if pair_data.get('mtf_confirmed'): regime_pts += 5
    factors['regime_mtf'] = regime_pts
    
    # 7. Pair Memory bonus/penalty (-5 to +10 pts)
    mem = pair_memory_get(pair_data.get('pair', ''))
    mem_pts = 0
    if mem and mem.get('trades', 0) >= 2:
        wr = mem['wins'] / mem['trades']
        avg_pnl = mem['total_pnl'] / mem['trades']
        if wr >= 0.8 and avg_pnl > 0:
            mem_pts = min(10, avg_pnl * 10)
        elif wr < 0.3:
            mem_pts = -5
    factors['pair_memory'] = round(mem_pts, 1)
    
    # 8. Risk penalty — naked HR, high uncertainty
    risk_pen = 0
    if pair_data.get('hr_naked'): risk_pen -= 10
    if pair_data.get('hr_uncertainty', 0) > 0.3: risk_pen -= 5
    if pair_data.get('cusum_risk') in ('HIGH', 'CRITICAL'): risk_pen -= 5
    factors['risk_penalty'] = risk_pen
    
    total = max(0, min(100, sum(factors.values())))
    
    # Grade
    if total >= 80: grade = 'A'
    elif total >= 65: grade = 'B'
    elif total >= 50: grade = 'C'
    elif total >= 35: grade = 'D'
    else: grade = 'F'
    
    # Recommendation
    if grade in ('A', 'B'):
        rec = 'ВХОД — сильный сигнал'
    elif grade == 'C':
        rec = 'УСЛОВНО — уменьшить размер'
    elif grade == 'D':
        rec = 'РИСКОВАННО — минимальный размер'
    else:
        rec = 'ПРОПУСТИТЬ'
    
    return {
        'score': round(total, 1),
        'grade': grade,
        'factors': factors,
        'recommendation': rec,
    }


# ═══════════════════════════════════════════════════════
# R10 RISK MANAGER — position sizing & limits
# ═══════════════════════════════════════════════════════

def risk_position_size(ml_result, portfolio_usdt=1000, open_positions=0):
    """
    R10: Calculate position size based on ML score and risk limits.
    
    Returns: {'size_usdt': float, 'size_pct': float, 'reason': str, 'allowed': bool}
    """
    max_positions = CFG('risk', 'max_positions', 5)
    max_per_trade_pct = CFG('risk', 'max_per_trade_pct', 20)
    min_per_trade_pct = CFG('risk', 'min_per_trade_pct', 5)
    max_total_exposure_pct = CFG('risk', 'max_total_exposure_pct', 80)
    
    # Check position limit
    if open_positions >= max_positions:
        return {'size_usdt': 0, 'size_pct': 0, 
                'reason': f'⛔ Лимит позиций: {open_positions}/{max_positions}', 'allowed': False}
    
    # Check total exposure
    current_exposure = open_positions * max_per_trade_pct  # rough estimate
    remaining_pct = max_total_exposure_pct - current_exposure
    if remaining_pct <= 0:
        return {'size_usdt': 0, 'size_pct': 0,
                'reason': f'⛔ Exposure limit: {current_exposure}%/{max_total_exposure_pct}%', 'allowed': False}
    
    # Size based on ML grade
    grade = ml_result.get('grade', 'F')
    score = ml_result.get('score', 0)
    
    if grade == 'A':
        size_pct = max_per_trade_pct
    elif grade == 'B':
        size_pct = max_per_trade_pct * 0.75
    elif grade == 'C':
        size_pct = max_per_trade_pct * 0.5
    elif grade == 'D':
        size_pct = min_per_trade_pct
    else:
        return {'size_usdt': 0, 'size_pct': 0,
                'reason': f'⛔ Grade F — не торговать', 'allowed': False}
    
    # Cap by remaining exposure
    size_pct = min(size_pct, remaining_pct)
    size_pct = max(size_pct, min_per_trade_pct)
    
    size_usdt = portfolio_usdt * size_pct / 100
    
    return {
        'size_usdt': round(size_usdt, 1),
        'size_pct': round(size_pct, 1),
        'reason': f'Grade {grade} ({score:.0f}pt): {size_pct:.0f}% = {size_usdt:.0f} USDT',
        'allowed': True,
    }


# ═══════════════════════════════════════════════════════
# 3.3 PATTERN ANALYSIS — discover what works from history
# ═══════════════════════════════════════════════════════

TRADE_HISTORY_FILE = 'trade_history.csv'

def pattern_analysis():
    """
    3.3: Analyze trade history patterns.
    Returns dict with patterns by Z-range, direction, hold time, time of day, pair.
    """
    import csv, io
    from datetime import datetime
    
    try:
        with open(TRADE_HISTORY_FILE, 'r') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
    except Exception:
        return {'error': 'No trade history', 'n_trades': 0}
    
    if len(trades) < 3:
        return {'error': f'Only {len(trades)} trades — need ≥3', 'n_trades': len(trades)}
    
    result = {'n_trades': len(trades)}
    
    # Parse trades
    parsed = []
    for t in trades:
        try:
            pnl = float(t.get('pnl_pct', 0))
            ez = float(t.get('entry_z', 0))
            d = t.get('direction', '')
            pair = t.get('pair', '')
            
            # Hold hours
            try:
                et = datetime.fromisoformat(str(t.get('entry_time', '')).replace('+03:00', ''))
                xt = datetime.fromisoformat(str(t.get('exit_time', '')).replace('+03:00', ''))
                hold_h = (xt - et).total_seconds() / 3600
                entry_hour = et.hour
            except Exception:
                hold_h = 0
                entry_hour = 12
            
            parsed.append({
                'pnl': pnl, 'ez': ez, 'dir': d, 'pair': pair,
                'hold_h': hold_h, 'entry_hour': entry_hour,
                'is_auto': 'AUTO' in str(t.get('notes', '')),
            })
        except Exception:
            continue
    
    if not parsed:
        return {'error': 'No parseable trades', 'n_trades': 0}
    
    # 1. Direction pattern
    longs = [t for t in parsed if t['dir'] == 'LONG']
    shorts = [t for t in parsed if t['dir'] == 'SHORT']
    result['by_direction'] = {
        'LONG': {'n': len(longs), 
                 'wr': sum(1 for t in longs if t['pnl'] > 0) / max(1, len(longs)) * 100,
                 'avg': sum(t['pnl'] for t in longs) / max(1, len(longs))},
        'SHORT': {'n': len(shorts),
                  'wr': sum(1 for t in shorts if t['pnl'] > 0) / max(1, len(shorts)) * 100,
                  'avg': sum(t['pnl'] for t in shorts) / max(1, len(shorts))},
    }
    
    # 2. Z-range pattern
    z_ranges = {'1.5-2.0': [], '2.0-2.5': [], '2.5-3.0': [], '3.0+': []}
    for t in parsed:
        az = abs(t['ez'])
        if az >= 3.0: z_ranges['3.0+'].append(t)
        elif az >= 2.5: z_ranges['2.5-3.0'].append(t)
        elif az >= 2.0: z_ranges['2.0-2.5'].append(t)
        else: z_ranges['1.5-2.0'].append(t)
    result['by_z_range'] = {}
    for rng, ts in z_ranges.items():
        if ts:
            result['by_z_range'][rng] = {
                'n': len(ts),
                'wr': sum(1 for t in ts if t['pnl'] > 0) / len(ts) * 100,
                'avg': sum(t['pnl'] for t in ts) / len(ts),
            }
    
    # 3. Hold time pattern
    quick = [t for t in parsed if t['hold_h'] <= 2]
    medium = [t for t in parsed if 2 < t['hold_h'] <= 8]
    long_ = [t for t in parsed if t['hold_h'] > 8]
    result['by_hold'] = {}
    for name, group in [('≤2h', quick), ('2-8h', medium), ('>8h', long_)]:
        if group:
            result['by_hold'][name] = {
                'n': len(group),
                'wr': sum(1 for t in group if t['pnl'] > 0) / len(group) * 100,
                'avg': sum(t['pnl'] for t in group) / len(group),
            }
    
    # 4. Time of day pattern (Moscow)
    morning = [t for t in parsed if 6 <= t['entry_hour'] < 12]
    afternoon = [t for t in parsed if 12 <= t['entry_hour'] < 18]
    evening = [t for t in parsed if 18 <= t['entry_hour'] or t['entry_hour'] < 6]
    result['by_time'] = {}
    for name, group in [('06-12 МСК', morning), ('12-18 МСК', afternoon), ('18-06 МСК', evening)]:
        if group:
            result['by_time'][name] = {
                'n': len(group),
                'wr': sum(1 for t in group if t['pnl'] > 0) / len(group) * 100,
                'avg': sum(t['pnl'] for t in group) / len(group),
            }
    
    # 5. Top pairs
    pair_stats = {}
    for t in parsed:
        if t['pair'] not in pair_stats:
            pair_stats[t['pair']] = {'pnls': [], 'n': 0}
        pair_stats[t['pair']]['pnls'].append(t['pnl'])
        pair_stats[t['pair']]['n'] += 1
    result['by_pair'] = {}
    for pair, stats in sorted(pair_stats.items(), key=lambda x: sum(x[1]['pnls']), reverse=True):
        result['by_pair'][pair] = {
            'n': stats['n'],
            'total': round(sum(stats['pnls']), 3),
            'avg': round(sum(stats['pnls']) / stats['n'], 3),
            'wr': round(sum(1 for p in stats['pnls'] if p > 0) / stats['n'] * 100, 0),
        }
    
    # 6. Auto vs Manual
    auto = [t for t in parsed if t['is_auto']]
    manual = [t for t in parsed if not t['is_auto']]
    result['auto_vs_manual'] = {
        'auto': {'n': len(auto),
                 'wr': sum(1 for t in auto if t['pnl'] > 0) / max(1, len(auto)) * 100,
                 'avg': sum(t['pnl'] for t in auto) / max(1, len(auto))},
        'manual': {'n': len(manual),
                   'wr': sum(1 for t in manual if t['pnl'] > 0) / max(1, len(manual)) * 100,
                   'avg': sum(t['pnl'] for t in manual) / max(1, len(manual))},
    }
    
    # 7. Best entry conditions
    winners = [t for t in parsed if t['pnl'] > 0]
    losers = [t for t in parsed if t['pnl'] <= 0]
    if winners and losers:
        import statistics
        result['winner_profile'] = {
            'avg_z': round(statistics.mean([abs(t['ez']) for t in winners]), 2),
            'avg_hold': round(statistics.mean([t['hold_h'] for t in winners]), 1),
        }
        result['loser_profile'] = {
            'avg_z': round(statistics.mean([abs(t['ez']) for t in losers]), 2),
            'avg_hold': round(statistics.mean([t['hold_h'] for t in losers]), 1),
        }
    
    return result


def pattern_summary():
    """One-line pattern insights for display."""
    p = pattern_analysis()
    if p.get('error'):
        return p['error']
    
    lines = [f"📊 {p['n_trades']} сделок проанализировано"]
    
    # Best direction
    bd = p.get('by_direction', {})
    if bd.get('SHORT', {}).get('avg', 0) > bd.get('LONG', {}).get('avg', 0):
        lines.append(f"📈 SHORT лучше: avg {bd['SHORT']['avg']:+.2f}% vs LONG {bd.get('LONG',{}).get('avg',0):+.2f}%")
    elif bd.get('LONG', {}).get('n', 0) > 0:
        lines.append(f"📈 LONG лучше: avg {bd['LONG']['avg']:+.2f}%")
    
    # Best Z range
    bz = p.get('by_z_range', {})
    if bz:
        best_z = max(bz.items(), key=lambda x: x[1]['avg'])
        lines.append(f"🎯 Лучший Z: {best_z[0]} (avg {best_z[1]['avg']:+.2f}%, WR={best_z[1]['wr']:.0f}%)")
    
    # Best hold time
    bh = p.get('by_hold', {})
    if bh:
        best_h = max(bh.items(), key=lambda x: x[1]['avg'])
        lines.append(f"⏱ Лучший hold: {best_h[0]} (avg {best_h[1]['avg']:+.2f}%)")
    
    return " | ".join(lines)
