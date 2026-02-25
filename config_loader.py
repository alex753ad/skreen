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
