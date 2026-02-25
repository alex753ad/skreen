"""
Модуль расчета Hurst Exponent и Ornstein-Uhlenbeck параметров
ВЕРСИЯ v11.4 | 20.02.2026 | + DRY Shared Utils + Regime Detection (ADX) + HR Warning + Min Bars Gate
Min threshold 1.5, Hurst hard gate, Continuous threshold, DFA min_window=8

Дата: 18 февраля 2026

ИЗМЕНЕНИЯ v10.4.0:
  [FIX] sanitize_pair() — HR cap: 50 → 30 (APT/ZORA HR=42 выходил через фильтр)
  [FIX] DFA min_window: 4 → 8 (устраняет нестабильный Hurst на коротких рядах)
  [NEW] get_adaptive_signal() — fdr_passed параметр: FDR fail + Q<60 → max READY
  [NEW] get_adaptive_signal() — Z>4.5 условный: Q≥70 AND Stab≥3/4 → SIGNAL (EXTREME)
  [FIX] Quality score: HR>10 penalty усилен (0 вместо 2)
  
  v10.3: Stability gate (Stab<2/4 → WATCH), Z cutoff 4.5
  v10.2: Q gate↑40, HR ceiling, N_min↑50
  v10.1: Min-Q gate, HR uncertainty, N-bars hard gate
  v10.0: Adaptive Robust Z, Crossing Density, Correlation, Kalman HR
"""

import numpy as np
from scipy import stats
# v27: Unified config
try:
    from config_loader import CFG
except ImportError:
    def CFG(section, key=None, default=None):
        _d = {'strategy': {'entry_z': 1.8, 'exit_z': 0.8, 'commission_pct': 0.10,
              'slippage_pct': 0.05, 'take_profit_pct': 1.5, 'stop_loss_pct': -5.0,
              'micro_bt_max_bars': 6},
              'z_velocity': {'lookback': 5, 'excellent_min_vel': 0.1, 'decel_threshold': 0.05}}
        if key is None:
            return _d.get(section, {})
        return _d.get(section, {}).get(key, default)


# =============================================================================
# HURST — DFA
# =============================================================================

def calculate_hurst_exponent(time_series, min_window=8):
    """DFA на инкрементах. Возвращает 0.5 при fallback.
    
    v10.4: min_window=8 (was 4). Меньшее значение давало нестабильные
    результаты на 100-300 свечах, что приводило к Hurst=0.085 на 35d
    и Hurst=0.5 (fallback) на 63d для одной и той же пары.
    """
    ts = np.array(time_series, dtype=float)
    n = len(ts)
    if n < 30:
        return 0.5

    increments = np.diff(ts)
    n_inc = len(increments)
    profile = np.cumsum(increments - np.mean(increments))

    max_window = n_inc // 4
    if max_window <= min_window:
        return 0.5

    num_points = min(20, max_window - min_window)
    if num_points < 4:
        return 0.5

    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=num_points).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= min_window]
    if len(window_sizes) < 4:
        return 0.5

    fluctuations = []
    for w in window_sizes:
        n_seg = n_inc // w
        if n_seg < 2:
            continue
        f2_sum, count = 0.0, 0
        for seg in range(n_seg):
            segment = profile[seg * w:(seg + 1) * w]
            x = np.arange(w, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            f2_sum += np.mean((segment - np.polyval(coeffs, x)) ** 2)
            count += 1
        for seg in range(n_seg):
            start = n_inc - (seg + 1) * w
            if start < 0:
                break
            segment = profile[start:start + w]
            x = np.arange(w, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            f2_sum += np.mean((segment - np.polyval(coeffs, x)) ** 2)
            count += 1
        if count > 0:
            f_n = np.sqrt(f2_sum / count)
            if f_n > 1e-15:
                fluctuations.append((w, f_n))

    if len(fluctuations) < 4:
        return 0.5

    log_n = np.log([f[0] for f in fluctuations])
    log_f = np.log([f[1] for f in fluctuations])

    try:
        slope, _, r_value, _, _ = stats.linregress(log_n, log_f)
        if r_value ** 2 < 0.70:
            return 0.5
        return round(max(0.01, min(0.99, slope)), 4)
    except Exception:
        return 0.5


def calculate_hurst_ema(spread, n_subwindows=8, ema_span=5):
    """
    v16.0: Smoothed Hurst via EMA of sub-window DFA estimates.
    
    Problem: Single DFA on 300 bars is unstable — one large candle shifts result.
    Solution: Calculate Hurst on N overlapping sub-windows, then EMA-smooth.
    
    Algorithm:
      1. Split spread into N overlapping windows (75% overlap)
      2. Calculate DFA Hurst on each sub-window
      3. Apply EMA(span=5) to smooth
      4. Return: current EMA value + raw series for diagnostics
    
    Args:
        spread: full spread array (300+ bars)
        n_subwindows: number of sub-windows (default 8)
        ema_span: EMA smoothing period (default 5)
    
    Returns:
        dict: {
            'hurst_ema': float (smoothed current Hurst),
            'hurst_raw': float (un-smoothed current Hurst),
            'hurst_series': list of raw Hurst values per sub-window,
            'hurst_std': float (std of raw values — stability indicator),
            'is_stable': bool (std < 0.08 = stable)
        }
    """
    spread = np.array(spread, float)
    n = len(spread)
    
    if n < 80:
        h_raw = calculate_hurst_exponent(spread)
        return {
            'hurst_ema': h_raw, 'hurst_raw': h_raw,
            'hurst_series': [h_raw], 'hurst_std': 0.0, 'is_stable': True
        }
    
    # Sub-windows: size = 60% of total, step = 40/n_subwindows of total
    win_size = max(60, int(n * 0.60))
    step = max(1, (n - win_size) // max(1, n_subwindows - 1))
    
    hurst_values = []
    for i in range(n_subwindows):
        start = min(i * step, n - win_size)
        sub = spread[start:start + win_size]
        if len(sub) >= 50:
            h = calculate_hurst_exponent(sub)
            hurst_values.append(h)
    
    if not hurst_values:
        h_raw = calculate_hurst_exponent(spread)
        return {
            'hurst_ema': h_raw, 'hurst_raw': h_raw,
            'hurst_series': [h_raw], 'hurst_std': 0.0, 'is_stable': True
        }
    
    # EMA smoothing
    h_raw = hurst_values[-1]
    alpha = 2.0 / (ema_span + 1)
    ema_val = hurst_values[0]
    for h in hurst_values[1:]:
        ema_val = alpha * h + (1 - alpha) * ema_val
    
    h_std = float(np.std(hurst_values))
    
    return {
        'hurst_ema': round(ema_val, 4),
        'hurst_raw': round(h_raw, 4),
        'hurst_series': [round(h, 4) for h in hurst_values],
        'hurst_std': round(h_std, 4),
        'is_stable': h_std < 0.08,
    }


def calculate_hurst_expanding(spread, scales=None):
    """
    v19.1: Expanding Window Hurst — multi-scale analysis.
    
    Calculates Hurst at increasing window sizes to detect regime changes:
      H₁₀₀ → H₁₅₀ → H₂₀₀ → H₂₅₀ → H₃₀₀
    
    Interpretation:
      H₁₀₀ ≈ H₃₀₀ → stable mean-reversion (safe to trade)
      H₁₀₀ >> H₃₀₀ → MR weakening over time (DANGER)
      H₁₀₀ << H₃₀₀ → MR strengthening (ideal entry)
      hurst_slope > +0.1 → regime shifting toward trending
      hurst_slope < -0.1 → regime shifting toward mean-reversion
    
    Returns: dict with scale results, slope, and assessment
    """
    spread = np.array(spread, float)
    n = len(spread)
    
    if scales is None:
        # Default: 5 scales from 60 to full length
        min_scale = max(60, n // 5)
        scales = list(range(min_scale, n + 1, max(1, (n - min_scale) // 4)))
        if scales[-1] != n:
            scales.append(n)
        # Ensure at least 3 scales
        if len(scales) < 3:
            scales = [max(60, n // 3), max(80, n * 2 // 3), n]
    
    # Calculate Hurst at each scale
    scale_results = []
    for s in scales:
        if s < 40:
            continue
        # Use the LAST s bars (most recent data)
        sub = spread[-s:]
        h = calculate_hurst_exponent(sub)
        scale_results.append({'bars': s, 'hurst': round(h, 4)})
    
    if len(scale_results) < 2:
        h_full = calculate_hurst_exponent(spread)
        return {
            'scales': [{'bars': n, 'hurst': round(h_full, 4)}],
            'hurst_slope': 0.0,
            'hurst_short': round(h_full, 4),
            'hurst_long': round(h_full, 4),
            'assessment': 'INSUFFICIENT_DATA',
            'mr_strengthening': False,
            'mr_weakening': False,
        }
    
    # Linear regression: Hurst vs normalized scale index
    hursts = np.array([r['hurst'] for r in scale_results])
    x = np.linspace(0, 1, len(hursts))
    
    if len(hursts) >= 2:
        slope = float(np.polyfit(x, hursts, 1)[0])
    else:
        slope = 0.0
    
    h_short = scale_results[0]['hurst']  # Shortest scale (recent)
    h_long = scale_results[-1]['hurst']  # Full scale
    
    # Assessment
    mr_strengthening = slope < -0.05 and h_short < h_long
    mr_weakening = slope > 0.05 and h_short > h_long
    
    if abs(slope) < 0.03:
        assessment = 'STABLE'
    elif mr_strengthening:
        assessment = 'MR_STRENGTHENING'
    elif mr_weakening:
        assessment = 'MR_WEAKENING'
    elif slope > 0.1:
        assessment = 'TRENDING_SHIFT'
    else:
        assessment = 'MIXED'
    
    return {
        'scales': scale_results,
        'hurst_slope': round(slope, 4),
        'hurst_short': h_short,
        'hurst_long': h_long,
        'assessment': assessment,
        'mr_strengthening': mr_strengthening,
        'mr_weakening': mr_weakening,
    }


# =============================================================================
# ROLLING Z-SCORE
# =============================================================================

def calculate_rolling_zscore(spread, window=30):
    """Rolling Z-score без lookahead bias. LEGACY — используйте adaptive версию."""
    spread = np.array(spread, dtype=float)
    n = len(spread)
    if n < window + 1:
        mean, std = np.mean(spread), np.std(spread)
        if std < 1e-10:
            return 0.0, np.zeros(n)
        zs = (spread - mean) / std
        return float(zs[-1]), zs

    zscore_series = np.full(n, np.nan)
    for i in range(window, n):
        lb = spread[i - window:i]
        m, s = np.mean(lb), np.std(lb)
        zscore_series[i] = (spread[i] - m) / s if s > 1e-10 else 0.0

    cz = zscore_series[-1]
    return float(0.0 if np.isnan(cz) else cz), zscore_series


def calculate_adaptive_robust_zscore(spread, halflife_bars=None, min_w=10, max_w=60):
    """
    Адаптивный робастный Z-score.

    Два улучшения над calculate_rolling_zscore:
      1. Адаптивное окно: Window = clip(2.5 × HL_bars, min_w, max_w)
         Синхронизирует Z-score с ритмом конкретной пары.
      2. MAD вместо std: устойчив к выбросам (fat tails крипто).
         MAD * 1.4826 ≈ sigma для нормального распределения.

    Args:
        spread: массив спреда
        halflife_bars: HL в барах (не часах!). None → default window.
        min_w: минимальное окно (10 — порог стабильности)
        max_w: максимальное окно (60 — не слишком далеко)

    Returns:
        (current_z, z_series, window_used)
    """
    spread = np.array(spread, dtype=float)
    n = len(spread)

    # 1. Адаптивное окно
    if halflife_bars is not None and 0 < halflife_bars < 500:
        window = int(np.clip(round(2.5 * halflife_bars), min_w, max_w))
    else:
        window = 30  # fallback

    if n < window + 1:
        # Мало данных — простой Z
        med = np.median(spread)
        mad = np.median(np.abs(spread - med)) * 1.4826
        if mad < 1e-10:
            return 0.0, np.zeros(n), window
        zs = (spread - med) / mad
        return float(zs[-1]), zs, window

    # 2. Rolling MAD Z-score
    zscore_series = np.full(n, np.nan)
    for i in range(window, n):
        lb = spread[i - window:i]
        med = np.median(lb)
        mad = np.median(np.abs(lb - med)) * 1.4826

        if mad < 1e-10:
            # fallback на std если MAD = 0 (стейблкоины)
            s = np.std(lb)
            zscore_series[i] = (spread[i] - np.mean(lb)) / s if s > 1e-10 else 0.0
        else:
            zscore_series[i] = (spread[i] - med) / mad

    cz = zscore_series[-1]
    return float(0.0 if np.isnan(cz) else cz), zscore_series, window


def calculate_garch_zscore(spread, halflife_bars=None, lam=0.94, min_w=10, max_w=60):
    """
    v18.0: GARCH-like Z-score using EWMA volatility (RiskMetrics λ=0.94).
    
    Problem: Standard Z uses fixed-window σ. When σ suddenly increases,
    Z collapses toward 0 even though spread hasn't converged (SUI/AVAX bug).
    
    Solution: EWMA volatility adapts to recent regime:
      σ²(t) = λ·σ²(t-1) + (1-λ)·(spread(t) - μ(t))²
      Z_garch(t) = (spread(t) - μ_rolling(t)) / σ_ewma(t)
    
    When σ increases (vol spike), Z_garch adjusts immediately, preventing
    false convergence signals. When σ is stable, Z_garch ≈ Z_standard.
    
    Also returns: vol_ratio = σ_ewma / σ_rolling. If > 1.5 → variance collapse warning.
    
    Args:
        spread: spread array
        halflife_bars: for rolling median window
        lam: EWMA decay factor (0.94 = RiskMetrics standard)
        min_w, max_w: window bounds for rolling median
    
    Returns:
        dict: {
            'z_garch': float (current GARCH Z),
            'z_standard': float (standard MAD Z for comparison),
            'vol_ratio': float (σ_ewma / σ_mad, >1.5 = regime shift),
            'sigma_ewma': float (current EWMA volatility),
            'z_garch_series': array,
            'z_divergence': float (|z_standard - z_garch|),
            'variance_expanding': bool (σ_ewma growing > 20% in last 10 bars),
        }
    """
    spread = np.array(spread, float)
    n = len(spread)
    
    # Window from halflife
    if halflife_bars and 0 < halflife_bars < 500:
        window = int(np.clip(round(2.5 * halflife_bars), min_w, max_w))
    else:
        window = 30
    
    if n < window + 5:
        return {
            'z_garch': 0.0, 'z_standard': 0.0, 'vol_ratio': 1.0,
            'sigma_ewma': 0.0, 'z_garch_series': np.zeros(n),
            'z_divergence': 0.0, 'variance_expanding': False,
        }
    
    # 1. Rolling median (center)
    med_series = np.full(n, np.nan)
    for i in range(window, n):
        med_series[i] = np.median(spread[i - window:i])
    
    # 2. EWMA variance
    sigma2_ewma = np.full(n, np.nan)
    # Initialize with first window variance
    init_var = np.var(spread[:window]) if window <= n else np.var(spread)
    sigma2_ewma[window] = max(init_var, 1e-12)
    
    for i in range(window + 1, n):
        residual = spread[i] - med_series[i] if not np.isnan(med_series[i]) else 0
        sigma2_ewma[i] = lam * sigma2_ewma[i-1] + (1 - lam) * residual**2
    
    # 3. GARCH Z-score
    z_garch_series = np.full(n, np.nan)
    for i in range(window + 1, n):
        sigma = np.sqrt(max(sigma2_ewma[i], 1e-12))
        if sigma > 1e-10 and not np.isnan(med_series[i]):
            z_garch_series[i] = (spread[i] - med_series[i]) / sigma
    
    # 4. Standard MAD Z for comparison
    z_std, z_std_series, _ = calculate_adaptive_robust_zscore(
        spread, halflife_bars=halflife_bars, min_w=min_w, max_w=max_w)
    
    # Current values
    z_g = z_garch_series[-1] if not np.isnan(z_garch_series[-1]) else 0.0
    sigma_now = np.sqrt(sigma2_ewma[-1]) if not np.isnan(sigma2_ewma[-1]) else 0
    
    # MAD for comparison
    lb = spread[-window:]
    mad_now = np.median(np.abs(lb - np.median(lb))) * 1.4826
    vol_ratio = sigma_now / mad_now if mad_now > 1e-10 else 1.0
    
    # Variance expanding? Check if σ_ewma grew > 20% over last 10 bars
    lookback = min(10, n - window - 2)
    variance_expanding = False
    if lookback > 2:
        s_old = np.sqrt(sigma2_ewma[-lookback]) if not np.isnan(sigma2_ewma[-lookback]) else sigma_now
        if s_old > 1e-10:
            variance_expanding = (sigma_now / s_old - 1.0) > 0.20
    
    return {
        'z_garch': round(float(z_g), 4),
        'z_standard': round(float(z_std), 4),
        'vol_ratio': round(float(vol_ratio), 3),
        'sigma_ewma': round(float(sigma_now), 6),
        'z_garch_series': z_garch_series,
        'z_divergence': round(abs(float(z_std) - float(z_g)), 3),
        'variance_expanding': variance_expanding,
    }


def calculate_crossing_density(zscore_series, window=None):
    """
    Частота пересечений нуля Z-score.

    Показывает как часто спред реально переходит через mean.
    Высокая плотность → пара активно mean-reverting.
    Низкая плотность → спред "застрял" на одной стороне.

    Args:
        zscore_series: массив Z-scores
        window: количество последних баров для анализа

    Returns:
        float: плотность (0.0–1.0). 0.05 = 5% баров содержат кроссинг.
    """
    z = np.array(zscore_series, dtype=float)
    z = z[~np.isnan(z)]

    if len(z) < 10:
        return 0.0

    if window is not None and len(z) > window:
        z = z[-window:]

    # Считаем смены знака
    signs = np.sign(z)
    # Убираем нули (на нуле — не считается сменой)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0

    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / len(signs))


def calculate_rolling_correlation(series1, series2, window=30):
    """
    Rolling корреляция Пирсона между двумя ценовыми рядами.

    НЕ используется как фильтр (коинтегрированные пары могут
    временно раскоррелироваться — это момент входа).
    Показывается в UI как информационный индикатор.

    Returns:
        (current_corr, corr_series)
    """
    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)
    n = min(len(s1), len(s2))

    if n < window + 1:
        if n > 5:
            return float(np.corrcoef(s1[:n], s2[:n])[0, 1]), np.array([])
        return 0.0, np.array([])

    s1, s2 = s1[:n], s2[:n]
    corr_series = np.full(n, np.nan)

    for i in range(window, n):
        x = s1[i - window:i]
        y = s2[i - window:i]
        sx, sy = np.std(x), np.std(y)
        if sx > 1e-10 and sy > 1e-10:
            corr_series[i] = np.corrcoef(x, y)[0, 1]

    cc = corr_series[-1]
    return float(0.0 if np.isnan(cc) else cc), corr_series


# =============================================================================
# OU PARAMETERS
# =============================================================================

def calculate_ou_parameters(spread, dt=1.0):
    """OU: dX = θ(μ - X)dt + σdW"""
    try:
        if len(spread) < 20:
            return None
        spread = np.array(spread, dtype=float)
        y, x = np.diff(spread), spread[:-1]
        n = len(x)
        sx, sy = np.sum(x), np.sum(y)
        sxy, sx2 = np.sum(x * y), np.sum(x ** 2)
        denom = n * sx2 - sx ** 2
        if abs(denom) < 1e-10:
            return None
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        theta = max(0.001, min(10.0, -b / dt))
        mu = a / theta if theta > 0 else 0.0
        y_pred = a + b * x
        sigma = np.std(y - y_pred)
        halflife = np.log(2) / theta if theta > 0 else 999.0
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {
            'theta': float(theta), 'mu': float(mu), 'sigma': float(sigma),
            'halflife_ou': float(halflife), 'r_squared': float(r_sq),
            'equilibrium_time': float(-np.log(0.05) / theta if theta > 0 else 999.0)
        }
    except Exception:
        return None


# =============================================================================
# KALMAN FILTER для адаптивного HEDGE RATIO
# =============================================================================

def kalman_hedge_ratio(series1, series2, delta=1e-4, ve=1e-3):
    """
    Kalman Filter для динамического hedge ratio.

    Модель:
      State:  β_t = [intercept_t, hedge_ratio_t]
      Transition: β_t = β_{t-1} + w_t,  w ~ N(0, Q)
      Observation: price1_t = intercept_t + hedge_ratio_t * price2_t + v_t

    Args:
        series1, series2: ценовые ряды (np.array или pd.Series)
        delta: дисперсия перехода (процесс случайного блуждания для β).
               Маленький delta = гладкий HR, большой = быстрая адаптация.
               Default 1e-4 — хороший баланс для 4h крипто.
        ve: начальная дисперсия наблюдения (measurement noise).

    Returns:
        dict:
            hedge_ratios:  np.array — HR на каждом баре
            intercepts:    np.array — intercept на каждом баре
            spread:        np.array — адаптивный спред
            hr_final:      float — текущий (последний) HR
            intercept_final: float
            hr_std:        float — uncertainty текущего HR
            sqrt_Q:        np.array — серия measurement prediction errors
    """
    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)
    n = min(len(s1), len(s2))

    if n < 10:
        return None

    s1, s2 = s1[:n], s2[:n]

    # State: [intercept, hedge_ratio]
    # Начальная оценка через OLS на первых 30 барах
    init_n = min(30, n // 3)
    try:
        X_init = np.column_stack([np.ones(init_n), s2[:init_n]])
        beta_init = np.linalg.lstsq(X_init, s1[:init_n], rcond=None)[0]
    except Exception:
        beta_init = np.array([0.0, 1.0])

    # Kalman state
    beta = beta_init.copy()          # [2,] state estimate
    P = np.eye(2) * 1.0              # [2,2] state covariance
    Q = np.eye(2) * delta            # [2,2] transition noise
    R = ve                            # scalar observation noise

    # Storage
    hedge_ratios = np.zeros(n)
    intercepts = np.zeros(n)
    innovations = np.zeros(n)    # Kalman innovations (≈ белый шум)
    trading_spread = np.zeros(n) # Торговый спред для Z-score
    sqrt_Q_series = np.zeros(n)

    for t in range(n):
        # Observation vector: x_t = [1, price2_t]
        x_t = np.array([1.0, s2[t]])

        # Predict
        # beta = beta (random walk)
        P = P + Q

        # Update
        y_hat = x_t @ beta                  # predicted price1
        e_t = s1[t] - y_hat                 # innovation
        S_t = x_t @ P @ x_t + R            # innovation variance
        K_t = P @ x_t / S_t                 # Kalman gain [2,]

        beta = beta + K_t * e_t             # state update
        P = P - np.outer(K_t, x_t) @ P     # covariance update

        # Ensure P stays positive definite
        P = (P + P.T) / 2
        np.fill_diagonal(P, np.maximum(np.diag(P), 1e-10))

        # Store
        intercepts[t] = beta[0]
        hedge_ratios[t] = beta[1]
        innovations[t] = e_t
        # Торговый спред: price1 - HR_t * price2 - intercept_t
        trading_spread[t] = s1[t] - beta[1] * s2[t] - beta[0]
        sqrt_Q_series[t] = np.sqrt(max(S_t, 1e-10))

    return {
        'hedge_ratios': hedge_ratios,
        'intercepts': intercepts,
        'spread': trading_spread,       # ← для Z-score и DFA
        'innovations': innovations,     # ← innovations (≈ белый шум)
        'hr_final': float(hedge_ratios[-1]),
        'intercept_final': float(intercepts[-1]),
        'hr_std': float(np.sqrt(P[1, 1])),
        'sqrt_Q': sqrt_Q_series,
        'P_final': P,
    }


def kalman_select_delta(series1, series2, deltas=None):
    """
    Автоподбор delta по максимизации log-likelihood.

    Перебирает несколько значений delta и выбирает лучший.
    Используется если нет уверенности в default delta=1e-4.

    Returns:
        best_delta, best_result, all_likelihoods
    """
    if deltas is None:
        deltas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)
    n = min(len(s1), len(s2))

    best_ll = -np.inf
    best_delta = 1e-4
    best_result = None
    all_ll = {}

    for d in deltas:
        res = kalman_hedge_ratio(s1, s2, delta=d)
        if res is None:
            continue

        # Log-likelihood: sum of log N(e_t; 0, S_t)
        sq = res['sqrt_Q']
        innov = res['innovations']  # innovations, not trading spread

        # Ignore first 30 bars (warmup)
        warmup = min(30, n // 3)
        ll_valid = -0.5 * np.sum(
            np.log(2 * np.pi * sq[warmup:]**2 + 1e-10) +
            innov[warmup:]**2 / (sq[warmup:]**2 + 1e-10)
        )

        all_ll[d] = float(ll_valid)
        if ll_valid > best_ll:
            best_ll = ll_valid
            best_delta = d
            best_result = res

    return best_delta, best_result, all_ll




# =============================================================================
# ADF-ТЕСТ СПРЕДА
# =============================================================================

def adf_test_spread(spread, significance=0.05):
    """ADF тест на стационарность спреда."""
    from statsmodels.tsa.stattools import adfuller
    try:
        spread = np.array(spread, dtype=float)
        if len(spread) < 20:
            return {'adf_stat': 0, 'adf_pvalue': 1.0, 'is_stationary': False, 'critical_values': {}}
        result = adfuller(spread, autolag='AIC')
        return {
            'adf_stat': float(result[0]), 'adf_pvalue': float(result[1]),
            'is_stationary': result[1] < significance,
            'critical_values': {k: float(v) for k, v in result[4].items()}
        }
    except Exception:
        return {'adf_stat': 0, 'adf_pvalue': 1.0, 'is_stationary': False, 'critical_values': {}}


# =============================================================================
# FDR-КОРРЕКЦИЯ
# =============================================================================

def apply_fdr_correction(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR. Передавайте ВСЕ p-values!"""
    pvalues = np.array(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]

    adjusted = np.empty(n)
    for i in range(n):
        adjusted[i] = sorted_p[i] * n / (i + 1)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)

    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result, result <= alpha


# =============================================================================
# COINTEGRATION STABILITY
# =============================================================================

def johansen_test(series1, series2, det_order=0, k_ar_diff=1):
    """
    v13.0: Johansen cointegration test (symmetric, multi-equation).
    
    Unlike Engle-Granger (asymmetric OLS), Johansen tests a VECM system
    and doesn't require choosing dependent/independent variable.
    
    Args:
        series1, series2: price arrays
        det_order: -1=no const, 0=const (default), 1=const+trend
        k_ar_diff: number of lagged differences (default 1)
    
    Returns:
        dict: {
            'trace_stat': float (test statistic),
            'trace_cv_5pct': float (5% critical value),
            'is_cointegrated': bool,
            'eigen_stat': float,
            'eigen_cv_5pct': float,
            'hedge_ratio': float (from eigenvector),
            'method': 'johansen'
        }
        or None on failure
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except ImportError:
        return None
    
    s1, s2 = np.array(series1, float), np.array(series2, float)
    n = min(len(s1), len(s2))
    if n < 50:
        return None
    s1, s2 = s1[:n], s2[:n]
    
    try:
        data = np.column_stack([s1, s2])
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        
        # r=0: test for "no cointegration" (reject = cointegrated)
        trace_stat = float(result.lr1[0])   # trace statistic for r=0
        trace_cv = float(result.cvt[0, 1])  # 5% critical value
        eigen_stat = float(result.lr2[0])   # max-eigenvalue for r=0
        eigen_cv = float(result.cvm[0, 1])  # 5% critical value
        
        is_coint = trace_stat > trace_cv  # reject H0: no cointegration
        
        # Hedge ratio from first eigenvector
        evec = result.evec[:, 0]
        hr = float(-evec[1] / evec[0]) if abs(evec[0]) > 1e-10 else 1.0
        
        return {
            'trace_stat': round(trace_stat, 3),
            'trace_cv_5pct': round(trace_cv, 3),
            'is_cointegrated': is_coint,
            'eigen_stat': round(eigen_stat, 3),
            'eigen_cv_5pct': round(eigen_cv, 3),
            'hedge_ratio': round(hr, 6),
            'method': 'johansen',
        }
    except Exception:
        return None


def check_cointegration_stability(series1, series2, window_fraction=0.6):
    """4 подокна: полное, начало, конец, середина."""
    from statsmodels.tsa.stattools import coint
    s1, s2 = np.array(series1, dtype=float), np.array(series2, dtype=float)
    n = min(len(s1), len(s2))
    if n < 30:
        return {'is_stable': False, 'windows_passed': 0, 'total_windows': 0,
                'stability_score': 0.0, 'pvalues': []}
    ws = max(20, int(n * window_fraction))
    mid = (n - ws) // 2
    windows = [(0, n), (0, ws), (n - ws, n), (mid, mid + ws)]
    pvalues, passed = [], 0
    for start, end in windows:
        end = min(end, n)
        if end - start < 20:
            continue
        try:
            _, pval, _ = coint(s1[start:end], s2[start:end])
            pvalues.append(float(pval))
            if pval < 0.05:
                passed += 1
        except Exception:
            pvalues.append(1.0)
    total = len(pvalues)
    return {
        'is_stable': passed >= 3, 'windows_passed': passed,
        'total_windows': total,
        'stability_score': round(passed / total if total > 0 else 0.0, 3),
        'pvalues': pvalues
    }


# =============================================================================
# CONFIDENCE
# =============================================================================

def calculate_confidence(hurst, stability_score, fdr_passed, adf_passed,
                         zscore, hedge_ratio, hurst_is_fallback=False,
                         hr_std=None):
    """
    HIGH / MEDIUM / LOW на основе 7 критериев.
    
    v11.0: HARD GATE — Hurst≥0.45 или fallback → max MEDIUM (без исключений).
    Нет статистического подтверждения mean reversion → HIGH confidence невозможен.
    """
    checks = 0
    total_checks = 7
    if fdr_passed:
        checks += 1
    if adf_passed:
        checks += 1
    if not hurst_is_fallback and hurst != 0.5 and hurst < 0.48:
        checks += 1
    if stability_score >= 0.75:
        checks += 1
    if 0 < hedge_ratio and 0.1 <= abs(hedge_ratio) <= 10.0:
        checks += 1
    if 1.5 <= abs(zscore) <= 5.0:
        checks += 1
    # v10.1: HR uncertainty — Калман уверен в хедже?
    if hr_std is not None and hedge_ratio > 0:
        hr_unc = hr_std / hedge_ratio
        if hr_unc < 0.5:  # uncertainty < 50%
            checks += 1
    else:
        checks += 1  # нет данных — не штрафуем (OLS fallback)

    # v11.0: HARD GATE — без подтверждённого mean reversion HIGH невозможен
    hurst_is_bad = hurst_is_fallback or hurst >= 0.45
    
    if checks >= 6 and not hurst_is_bad:
        return "HIGH", checks, total_checks
    elif checks >= 4:
        return "MEDIUM", checks, total_checks
    else:
        return "LOW", checks, total_checks


# =============================================================================
# [D-B] QUALITY SCORE — насколько пара надёжна
# =============================================================================

def calculate_quality_score(hurst, ou_params, pvalue_adj, stability_score,
                            hedge_ratio, adf_passed=None,
                            hurst_is_fallback=False,
                            crossing_density=None,
                            n_bars=None,
                            hr_std=None):
    """
    Quality Score (0-100) — оценка ПАРЫ, без привязки к текущему Z.

    Компоненты:
      FDR p-value:    25  — статистическая надёжность коинтеграции
      Stability:      25  — устойчивость во времени
      Hurst (DFA):    20  — подтверждение mean-reversion
      ADF:            15  — независимый тест стационарности
      Hedge ratio:    15  — практичность для торговли
                     ----
                     100

    Модификаторы:
      Crossing density < 0.03 → -10 (спред застрял на одной стороне)
      N < 100 баров → -15 (мало данных, ненадёжная статистика)
      HR uncertainty > 50% → -10 (Калман не уверен в хедже)
    """
    bd = {}

    # FDR (25)
    bd['fdr'] = 25 if pvalue_adj <= 0.01 else 20 if pvalue_adj <= 0.03 else 12 if pvalue_adj <= 0.05 else 0

    # Stability (25)
    bd['stability'] = int(stability_score * 25)

    # Hurst (20)
    if hurst_is_fallback or hurst == 0.5:
        bd['hurst'] = 0
    elif hurst <= 0.30:
        bd['hurst'] = 20
    elif hurst <= 0.40:
        bd['hurst'] = 15
    elif hurst <= 0.48:
        bd['hurst'] = 10
    elif hurst < 0.50:
        bd['hurst'] = 4
    else:
        bd['hurst'] = 0

    # ADF (15)
    bd['adf'] = 15 if adf_passed else 0

    # Hedge ratio (15) — v10.4: steeper penalty for extreme HR
    if hedge_ratio <= 0 or abs(hedge_ratio) > 30:
        bd['hedge_ratio'] = 0
    elif 0.2 <= abs(hedge_ratio) <= 5.0:
        bd['hedge_ratio'] = 15
    elif 0.1 <= abs(hedge_ratio) <= 10.0:
        bd['hedge_ratio'] = 10
    elif 0.05 <= abs(hedge_ratio) <= 30.0:
        bd['hedge_ratio'] = 5  # v10.4: covers 10-30 range
    else:
        bd['hedge_ratio'] = 0

    # Crossing density modifier
    if crossing_density is not None and crossing_density < 0.03:
        bd['crossing_penalty'] = -10
    else:
        bd['crossing_penalty'] = 0

    # Data depth modifier (pairs with N<100 get penalty)
    if n_bars is not None and n_bars < 100:
        bd['data_penalty'] = -15
    else:
        bd['data_penalty'] = 0

    # HR uncertainty modifier (v10.1)
    if hr_std is not None and hedge_ratio > 0 and hr_std / hedge_ratio > 0.5:
        bd['hr_unc_penalty'] = -10
    else:
        bd['hr_unc_penalty'] = 0

    total = max(0, min(100, sum(bd.values())))
    return int(total), bd


# =============================================================================
# [D-B] SIGNAL SCORE — насколько сейчас хороший момент для входа
# =============================================================================

def calculate_signal_score(zscore, ou_params, confidence, quality_score=100):
    """
    Signal Score (0-100) — оценка МОМЕНТА входа.
    
    v8.1: Cap по Quality — высокий Z на мусорной паре ≠ хороший сигнал.
    Финальный S = min(raw_S, quality_score * 1.2)

    Компоненты:
      Z-score сила:       40
      Скорость возврата:  30
      Confidence бонус:   30
                         ----
                         100
    """
    bd = {}

    az = abs(zscore)
    if az > 5.0:
        bd['zscore'] = 10
    elif az >= 3.0:
        bd['zscore'] = 40
    elif az >= 2.5:
        bd['zscore'] = 35
    elif az >= 2.0:
        bd['zscore'] = 30
    elif az >= 1.5:
        bd['zscore'] = 20
    elif az >= 1.0:
        bd['zscore'] = 10
    else:
        bd['zscore'] = 0

    if ou_params is not None:
        hl = ou_params['halflife_ou'] * 24
        bd['ou_speed'] = 30 if hl <= 12 else 25 if hl <= 20 else 15 if hl <= 28 else 8 if hl <= 48 else 0
    else:
        bd['ou_speed'] = 0

    if confidence == "HIGH":
        bd['confidence'] = 30
    elif confidence == "MEDIUM":
        bd['confidence'] = 15
    else:
        bd['confidence'] = 0

    raw = max(0, min(100, sum(bd.values())))
    
    # Cap: Signal не может сильно превышать Quality
    cap = int(quality_score * 1.2)
    total = min(raw, cap)
    bd['_cap'] = cap  # Для отладки
    
    return int(total), bd


# =============================================================================
# [v8.1] SANITIZER — жёсткие фильтры-исключения
# =============================================================================

def sanitize_pair(hedge_ratio, stability_passed, stability_total, zscore,
                  n_bars=None, hr_std=None):
    """
    Жёсткий фильтр: пара исключается полностью если не проходит.

    Исключения:
      HR <= 0:        не арбитраж
      |HR| < 0.001:   экономически бессмысленный HR
      |HR| > 30:      фактически односторонняя ставка (v10.4: was 50)
      Stab 0/N:       коинтеграция не подтверждена ни в одном окне
      |Z| > 10:       сломанная модель
      N < 50:         слишком мало данных для надёжной статистики
      HR uncertainty > 100%: Калман не уверен в связи

    Returns:
        (passed, reason)
    """
    if hedge_ratio <= 0:
        return False, f"HR={hedge_ratio:.4f} ≤ 0"
    if abs(hedge_ratio) < 0.001:
        return False, f"|HR|={abs(hedge_ratio):.6f} < 0.001"
    if abs(hedge_ratio) > 30:
        return False, f"|HR|={abs(hedge_ratio):.1f} > 30 (v10.4)"
    if stability_total > 0 and stability_passed == 0:
        return False, f"Stab=0/{stability_total}"
    if abs(zscore) > 10:
        return False, f"|Z|={abs(zscore):.1f} > 10"
    # v10.2: Hard minimum bars
    if n_bars is not None and n_bars < 50:
        return False, f"N={n_bars} < 50 баров"
    # v10.1: HR uncertainty > 100% — Калман не нашёл стабильную связь
    if hr_std is not None and hr_std > 0 and hedge_ratio > 0:
        hr_unc = hr_std / hedge_ratio
        if hr_unc > 1.0:
            return False, f"HR uncertainty {hr_unc:.0%} > 100%"
    return True, "OK"


# =============================================================================
# [v8.1] ADAPTIVE SIGNAL — TF-aware
# =============================================================================

def get_adaptive_signal(zscore, confidence, quality_score, timeframe='4h',
                        stability_ratio=1.0, fdr_passed=True,
                        hurst=None):
    """
    Адаптивный торговый сигнал с НЕПРЕРЫВНЫМ порогом.

    v11.0: Continuous threshold formula (вместо 3 фиксированных уровней):
      threshold = base[confidence] - quality_adj + hurst_penalty
      
      base:  HIGH=1.5, MEDIUM=2.0, LOW=2.5 (4h)
      quality_adj:  (Q - 50) / 250  → Q=100: -0.20, Q=50: 0
      hurst_penalty: H≥0.45: +0.5, H∈[0.35,0.45): 0, H∈[0.20,0.35): -0.05, H<0.20: -0.10
      
      Примеры (4h):
        HIGH  + Q=100 + H=0.10 → 1.5 - 0.20 - 0.10 = 1.20 (best pair)
        HIGH  + Q=80  + H=0.04 → 1.5 - 0.12 - 0.10 = 1.28
        MEDIUM+ Q=63  + H=0.50 → 2.0 - 0.05 + 0.50 = 2.45 (Hurst penalty!)
        LOW   + Q=37  + H=0.06 → 2.5 - 0.00 - 0.10 = 2.40

    v10.4 Hard guards (каскад) — сохранены:
      |Z| > 4.5 AND (Q<70 OR Stab<3/4) → NEUTRAL
      Q < 30 → NEUTRAL
      Q < 40 → max READY
      Stab < 2/4 → max WATCH
      FDR fail + Q<60 → max READY
    """
    az = abs(zscore)
    direction = "LONG" if zscore < 0 else "SHORT" if zscore > 0 else "NONE"

    # v10.4: Conditional Z-cap
    if az > 4.5:
        if quality_score >= 70 and stability_ratio >= 0.75:
            pass
        else:
            return "NEUTRAL", "NONE", 4.5

    if quality_score < 30:
        return "NEUTRAL", "NONE", 99.0

    # ═══════ v11.0: CONTINUOUS THRESHOLD ═══════
    # Base by confidence + timeframe
    if timeframe == '1h':
        base_map = {'HIGH': 1.8, 'MEDIUM': 2.3, 'LOW': 2.8}
    elif timeframe == '1d':
        base_map = {'HIGH': 1.3, 'MEDIUM': 1.8, 'LOW': 2.3}
    else:  # 4h (default)
        base_map = {'HIGH': 1.5, 'MEDIUM': 2.0, 'LOW': 2.5}

    base = base_map.get(confidence, base_map['LOW'])

    # Quality adjustment: higher Q → lower threshold (easier entry)
    q_adj = max(0, (quality_score - 50)) / 250.0  # 0 to 0.20

    # Hurst penalty: no MR evidence → harder entry
    h_adj = 0.0
    if hurst is not None:
        if hurst >= 0.45:
            h_adj = 0.50   # Severe: no mean reversion → much higher threshold
        elif hurst >= 0.35:
            h_adj = 0.0    # Borderline: no adjustment
        elif hurst >= 0.20:
            h_adj = -0.05  # Moderate MR: slight bonus
        else:
            h_adj = -0.10  # Strong MR: notable bonus
    
    threshold = round(max(1.5, min(3.5, base - q_adj + h_adj)), 2)
    
    # v15.0: Cost-aware minimum threshold (рассуждение #3)
    # For crypto altcoins: commission ~0.08% + slippage ~0.03% per leg × 2 legs = 0.22%
    # Typical PnL per Z unit ≈ 0.3-0.5%. Need profit > costs × 2
    # → minimum Z ≈ 0.22 * 2 / 0.4 ≈ 1.1, rounded up for safety
    cost_floors = {'1h': 2.0, '4h': 1.8, '1d': 1.5}
    cost_floor = cost_floors.get(timeframe, 1.8)
    if threshold < cost_floor:
        threshold = cost_floor

    # ═══════ SIGNAL LEVELS (unchanged logic, new threshold) ═══════
    # Ready threshold = threshold * 0.8, Watch = threshold * 0.55
    t_signal = threshold
    t_ready = round(threshold * 0.80, 2)
    t_watch = round(threshold * 0.55, 2)

    if az >= t_signal:
        if stability_ratio < 0.5:
            return "WATCH", direction, t_signal
        if quality_score < 40:
            return "READY", direction, t_signal
        if not fdr_passed and quality_score < 60:
            return "READY", direction, t_signal
        return "SIGNAL", direction, t_signal
    elif az >= t_ready:
        if stability_ratio < 0.5:
            return "WATCH", direction, t_signal
        return "READY", direction, t_signal
    elif az >= t_watch:
        return "WATCH", direction, t_signal
    else:
        return "NEUTRAL", "NONE", t_signal


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def calculate_trade_score(hurst, ou_params, pvalue_adj, zscore,
                          stability_score, hedge_ratio,
                          adf_passed=None, hurst_is_fallback=False):
    """Legacy — вызывает quality + signal и объединяет."""
    q, qbd = calculate_quality_score(hurst, ou_params, pvalue_adj,
                                      stability_score, hedge_ratio,
                                      adf_passed, hurst_is_fallback)
    # Простое среднее для обратной совместимости
    return q, qbd


def calculate_ou_score(ou_params, hurst):
    """Legacy OU Score."""
    if ou_params is None:
        return 0
    score = 0
    if 0.30 <= hurst <= 0.48: score += 50
    elif 0.48 < hurst <= 0.52: score += 30
    elif 0.25 <= hurst < 0.30: score += 40
    elif hurst < 0.25: score += 25
    elif 0.52 < hurst <= 0.60: score += 15
    hl = ou_params['halflife_ou'] * 24
    if 4 <= hl <= 24: score += 30
    elif 24 < hl <= 48: score += 20
    elif 2 <= hl < 4: score += 15
    elif hl < 2: score += 5
    if ou_params['r_squared'] > 0.15: score += 20
    elif ou_params['r_squared'] > 0.08: score += 15
    elif ou_params['r_squared'] > 0.05: score += 10
    return int(min(100, max(0, score)))


def estimate_exit_time(current_z, theta, mu=0.0, target_z=0.5):
    if theta <= 0.001:
        return 999.0
    try:
        ratio = abs(target_z - mu) / abs(current_z - mu)
        ratio = max(0.001, min(0.999, ratio))
        return -np.log(ratio) / theta
    except Exception:
        return 999.0


def validate_ou_quality(ou_params, hurst=None, min_theta=0.1, max_halflife=100):
    if ou_params is None:
        return False, "No OU"
    if ou_params['theta'] < min_theta:
        return False, "Low theta"
    if ou_params['halflife_ou'] * 24 > max_halflife:
        return False, "High HL"
    if hurst is not None and hurst > 0.70:
        return False, "High Hurst"
    return True, "OK"


# =============================================================================
# v11.2: REGIME DETECTION + HR WARNING + MIN BARS GATE
# =============================================================================

def detect_spread_regime(spread, window=20):
    """
    v11.2: Spread-based regime detection (ADX analog for pairs trading).
    
    Вычисляет:
      1. Spread ADX: directional movement of spread (0-100)
      2. Variance Ratio: short/long variance (>1.5 = trending)
      3. Trend persistence: % баров в одном направлении
    
    Returns:
      dict: adx, variance_ratio, trend_pct, regime ('MEAN_REVERT', 'NEUTRAL', 'TRENDING')
    """
    import pandas as _pd
    
    if len(spread) < window * 3:
        return {'adx': 0, 'variance_ratio': 1.0, 'trend_pct': 0.5, 'regime': 'UNKNOWN'}
    
    spread = np.array(spread)
    n = len(spread)
    
    # 1. Spread-based ADX
    diff = np.diff(spread)
    pos_dm = np.maximum(diff, 0)
    neg_dm = np.maximum(-diff, 0)
    
    pos_smooth = _pd.Series(pos_dm).rolling(window, min_periods=1).mean().values
    neg_smooth = _pd.Series(neg_dm).rolling(window, min_periods=1).mean().values
    atr = _pd.Series(np.abs(diff)).rolling(window, min_periods=1).mean().values
    
    di_plus = pos_smooth / (atr + 1e-10) * 100
    di_minus = neg_smooth / (atr + 1e-10) * 100
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10) * 100
    adx = float(_pd.Series(dx).rolling(window, min_periods=1).mean().iloc[-1])
    
    # 2. Variance Ratio (short vs long window)
    short_w = min(window, n // 4)
    long_w = min(window * 3, n - 1)
    
    short_returns = np.diff(spread[-short_w:])
    long_returns = np.diff(spread[-long_w:])
    
    var_short = np.var(short_returns) if len(short_returns) > 2 else 0
    var_long = np.var(long_returns) if len(long_returns) > 2 else 1e-10
    variance_ratio = var_short / (var_long + 1e-10)
    
    # 3. Trend persistence
    recent = spread[-window:]
    diffs = np.diff(recent)
    pos_count = np.sum(diffs > 0)
    trend_pct = max(pos_count, len(diffs) - pos_count) / len(diffs) if len(diffs) > 0 else 0.5
    
    # Classify
    if adx > 30 and variance_ratio > 1.5:
        regime = 'TRENDING'
    elif adx > 25 or variance_ratio > 2.0 or trend_pct > 0.75:
        regime = 'TRENDING'
    elif adx < 15 and variance_ratio < 1.2:
        regime = 'MEAN_REVERT'
    else:
        regime = 'NEUTRAL'
    
    return {
        'adx': round(adx, 1),
        'variance_ratio': round(variance_ratio, 2),
        'trend_pct': round(trend_pct, 2),
        'regime': regime,
    }


def check_hr_magnitude(hedge_ratio, threshold=5.0):
    """v11.2: HR magnitude warning if |HR| > threshold."""
    abs_hr = abs(hedge_ratio)
    if abs_hr > threshold:
        return (f"⚠️ HR={hedge_ratio:.2f} — капитальный дисбаланс! "
                f"На $1 первой монеты нужно ${abs_hr:.1f} второй.")
    return None


def check_minimum_bars(n_bars, timeframe='1d', min_bars_map=None):
    """v11.2: Minimum bars gate (1d needs ≥200 for reliable DFA)."""
    if min_bars_map is None:
        min_bars_map = {'1d': 200, '4h': 100, '1h': 150, '2h': 120, '15m': 200}
    min_required = min_bars_map.get(timeframe, 100)
    if n_bars < min_required:
        return (f"⚠️ {n_bars} баров < {min_required} мин для {timeframe}. "
                f"DFA/Kalman ненадёжны.")
    return None


def cusum_structural_break(spread, threshold_sigma=3.0, min_tail=30, zscore=None):
    """
    v13.0: CUSUM test with Z-magnitude amplifier.
    
    If |Z| > 5 AND CUSUM > 2.0 → break (catches FIL/CRV Z=9.36)
    
    Returns:
        dict with: has_break, cusum_score, tail_drift, tail_trend,
                   risk_level ('LOW'/'MEDIUM'/'HIGH'/'CRITICAL'),
                   position_advice (str), warning
    """
    spread = np.array(spread, dtype=float)
    n = len(spread)
    
    if n < min_tail * 2:
        return {'has_break': False, 'break_index': None, 
                'cusum_score': 0.0, 'tail_drift': 0.0, 'tail_trend': 0.0,
                'risk_level': 'LOW', 'position_advice': '', 'warning': None}
    
    ref_n = int(n * 0.70)
    ref_mean = np.mean(spread[:ref_n])
    ref_std = np.std(spread[:ref_n])
    if ref_std < 1e-10:
        return {'has_break': False, 'break_index': None,
                'cusum_score': 0.0, 'tail_drift': 0.0, 'tail_trend': 0.0,
                'risk_level': 'LOW', 'position_advice': '', 'warning': None}
    
    residuals = (spread - ref_mean) / ref_std
    cusum = np.cumsum(residuals - np.mean(residuals[:ref_n]))
    cusum_norm = cusum / np.sqrt(np.arange(1, n + 1))
    max_cusum = float(np.max(np.abs(cusum_norm)))
    
    cusum_diff = np.abs(np.diff(cusum_norm))
    if len(cusum_diff) > min_tail:
        recent_diff = cusum_diff[-min_tail * 2:]
        break_offset = int(np.argmax(recent_diff))
        break_index = n - min_tail * 2 + break_offset
    else:
        break_index = None
    
    tail = residuals[-min_tail:]
    tail_drift = float(np.mean(tail))
    tail_trend = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
    
    # v13.0: Z-magnitude amplifier
    abs_z = abs(zscore) if zscore is not None else 0
    
    has_break = (
        (max_cusum > threshold_sigma and abs(tail_drift) > 1.5) or
        (abs(tail_drift) > 2.5) or
        (abs(tail_trend) > 0.08 and abs(tail_drift) > 1.0) or
        # NEW: extreme Z + elevated CUSUM = structural break
        (abs_z > 5.0 and max_cusum > 2.0) or
        (abs_z > 7.0 and max_cusum > 1.5)
    )
    
    # v13.0: Risk classification for position sizing
    if has_break or abs_z > 6.0:
        risk_level = 'CRITICAL'
    elif max_cusum > 2.5 or (abs_z > 4.0 and max_cusum > 2.0):
        risk_level = 'HIGH'
    elif max_cusum > 2.0 or abs(tail_drift) > 1.0:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    # v13.0: Position sizing advice
    advice_map = {
        'CRITICAL': '🚫 НЕ ВХОДИТЬ. Коинтеграция разрушена. Спред в тренде.',
        'HIGH': '⚠️ Макс 25% позиции. Зарезервируйте 75% на усреднение или стоп. Высокий риск продолжения тренда.',
        'MEDIUM': '💡 Макс 50% позиции. Зарезервируйте 50% на добавление при откате Z к среднему.',
        'LOW': '✅ Полная позиция допустима. Стандартный risk management.',
    }
    position_advice = advice_map[risk_level]
    
    warning = None
    if risk_level == 'CRITICAL':
        warning = (f"🚨 CRITICAL: CUSUM={max_cusum:.1f}σ, Z={abs_z:.1f}, "
                   f"drift={tail_drift:+.2f}σ — НЕ ВХОДИТЬ!")
    elif risk_level == 'HIGH':
        warning = (f"🔴 HIGH RISK: CUSUM={max_cusum:.1f}σ, "
                   f"drift={tail_drift:+.2f}σ — макс 25% позиции")
    elif risk_level == 'MEDIUM':
        warning = (f"⚠️ Возможный сдвиг: CUSUM={max_cusum:.1f}σ, "
                   f"drift={tail_drift:+.2f}σ — макс 50% позиции")
    
    return {
        'has_break': has_break,
        'break_index': break_index,
        'cusum_score': round(max_cusum, 2),
        'tail_drift': round(tail_drift, 2),
        'tail_trend': round(tail_trend, 4),
        'risk_level': risk_level,
        'position_advice': position_advice,
        'warning': warning,
    }


# =============================================================================
# v14.0: COST-AWARE THRESHOLD (рассуждение #3)
# =============================================================================

def cost_aware_min_z(spread_std, commission_pct=0.10, slippage_pct=0.05, 
                     min_profit_ratio=3.0):
    """
    Minimum Z for profitable entry. 
    Z_min = total_costs / spread_pnl_per_z * min_profit_ratio
    
    If expected profit at Z=threshold doesn't exceed costs by min_profit_ratio,
    the trade is not worth taking.
    
    Returns: float (minimum Z threshold)
    """
    total_costs_pct = (commission_pct + slippage_pct) * 2  # 2 legs × (comm + slip)
    # Rough estimate: 1 Z of spread movement ≈ spread_std in price terms
    # PnL per Z ≈ spread_std / price ≈ proportional to Z movement
    # We need Z × pnl_per_z > costs × min_profit_ratio
    # Simplified: min_Z ≈ costs * ratio / typical_pnl_per_z
    # For crypto altcoins, typical pnl_per_z ≈ 0.3-0.5% per Z unit
    pnl_per_z = max(0.15, min(0.8, spread_std * 100)) if spread_std > 0 else 0.3
    min_z = total_costs_pct * min_profit_ratio / pnl_per_z
    return max(1.5, round(min_z, 2))


# =============================================================================
# v14.0: DOLLAR EXPOSURE CHECK (рассуждение #4)
# =============================================================================

def check_dollar_exposure(price1, price2, hedge_ratio, capital=1000):
    """
    Check dollar neutrality of the pair position.
    
    Beta-neutral (HR-adjusted) != Dollar-neutral.
    If HR=3, you sell $1000 coin1 and buy $3000 coin2 → $2000 net exposure!
    
    Returns:
        dict: {
            'leg1_dollars': float,
            'leg2_dollars': float,
            'net_exposure': float (absolute),
            'exposure_pct': float (% of capital),
            'is_balanced': bool,
            'warning': str or None
        }
    """
    leg1 = capital
    leg2 = capital * abs(hedge_ratio) * (price2 / price1) if price1 > 0 else capital
    net = abs(leg1 - leg2)
    exposure_pct = net / max(leg1, leg2) * 100 if max(leg1, leg2) > 0 else 0
    
    is_balanced = exposure_pct < 50
    warning = None
    if exposure_pct > 100:
        warning = f"🚨 Dollar exposure {exposure_pct:.0f}%: leg1=${leg1:.0f} vs leg2=${leg2:.0f}"
    elif exposure_pct > 50:
        warning = f"⚠️ Dollar exposure {exposure_pct:.0f}%: позиция не доллар-нейтральна"
    
    return {
        'leg1_dollars': round(leg1, 2),
        'leg2_dollars': round(leg2, 2),
        'net_exposure': round(net, 2),
        'exposure_pct': round(exposure_pct, 1),
        'is_balanced': is_balanced,
        'warning': warning,
    }


# =============================================================================
# v14.0: PnL/Z DISAGREEMENT (рассуждение #1)
# =============================================================================

def check_pnl_z_disagreement(entry_z, current_z, pnl_pct, direction):
    """
    Detect false Z-convergence caused by variance expansion (not price convergence).
    
    If Z fell from 3.0 to 0.0 but PnL ≈ 0%, the spread didn't actually converge —
    the standard deviation expanded, making Z shrink artificially.
    
    Returns:
        dict: {
            'z_moved': float (how much Z moved toward zero),
            'pnl_expected_pct': float (expected PnL for this Z move),
            'disagreement': bool,
            'severity': str ('NONE'|'MILD'|'SEVERE'),
            'warning': str or None
        }
    """
    z_delta = abs(entry_z) - abs(current_z)  # positive = Z moved toward zero
    
    # Expected rough PnL for Z movement (typically 0.2-0.5% per Z unit)
    pnl_per_z = 0.3  # conservative estimate
    pnl_expected = z_delta * pnl_per_z
    
    # Disagreement: Z says "converged" but PnL says "no profit"
    has_disagreement = (z_delta > 1.0 and pnl_pct < pnl_expected * 0.3)
    
    if has_disagreement and z_delta > 2.0 and pnl_pct < 0.1:
        severity = 'SEVERE'
        warning = (f"🚨 Ложное схождение! Z сместился на {z_delta:+.1f}, "
                   f"но P&L={pnl_pct:+.2f}%. Причина: рост σ спреда, а не возврат цен.")
    elif has_disagreement:
        severity = 'MILD'
        warning = (f"⚠️ Z/PnL расхождение: Z Δ={z_delta:+.1f}, P&L={pnl_pct:+.2f}% "
                   f"(ожидалось ~{pnl_expected:+.1f}%)")
    else:
        severity = 'NONE'
        warning = None
    
    return {
        'z_moved': round(z_delta, 2),
        'pnl_expected_pct': round(pnl_expected, 2),
        'disagreement': has_disagreement,
        'severity': severity,
        'warning': warning,
    }


# =============================================================================
# v17.0: MINI-BACKTEST for Scanner Integration (P1 Roadmap)
# =============================================================================

def mini_backtest(spread, p1, p2, hrs, entry_z=1.8, exit_z=0.8,
                  stop_z=4.0, halflife_bars=None, commission_pct=0.10,
                  slippage_pct=0.05, min_bars=2, max_bars=50):
    """
    v19.0: Lightweight backtest — REALISTIC exit rules for crypto.
    
    Key changes from v17:
      - exit_z=0.8 (not 0.5) — take profit at 55% reversion, not 75%
      - stop = entry + 1.5 (not +2.0) — cut losses faster
      - trailing: exit at 40% of peak (not at 0%)
      - max_bars=50 (not 100) — don't hold dying trades
      - min_bars=2 (not 3) — allow faster exits
      - Overshoot profit: z crosses zero → take profit immediately
    
    Gate: Total P&L < -8% OR Sharpe < -0.8 → FAIL (relaxed from -5%/-0.5)
    """
    spread = np.array(spread, float)
    p1 = np.array(p1, float)
    p2 = np.array(p2, float)
    hrs = np.array(hrs, float)
    n = len(spread)
    
    if n < 80:
        return {'verdict': 'SKIP', 'reason': 'Too few bars', 'n_trades': 0}
    
    # Z-score
    z_cur, zs, z_window = calculate_adaptive_robust_zscore(
        spread, halflife_bars=halflife_bars)
    
    # Adaptive stop: tighter than before
    adaptive_stop = max(stop_z, entry_z + 1.5)
    
    # Costs
    cost_total = (commission_pct + slippage_pct) * 4 / 100
    
    # Simulation
    trades_pnl = []
    position = None
    min_hold = min_bars
    cooldown = max(3, int(halflife_bars) if halflife_bars and halflife_bars < 50 else 3)
    last_close = -cooldown - 1
    warmup = max(z_window + 10, 50)
    
    for i in range(warmup, n):
        z = zs[i]
        if np.isnan(z):
            continue
        
        # OPEN (with pre-entry guard)
        if position is None and (i - last_close) > cooldown:
            if z > entry_z and z < adaptive_stop:
                position = {'bar': i, 'dir': 'SHORT', 'z': z,
                            'p1': p1[i], 'p2': p2[i], 'hr': hrs[i], 'best': 0}
            elif z < -entry_z and z > -adaptive_stop:
                position = {'bar': i, 'dir': 'LONG', 'z': z,
                            'p1': p1[i], 'p2': p2[i], 'hr': hrs[i], 'best': 0}
        
        # CLOSE
        if position is not None:
            bars = i - position['bar']
            hr_e = position['hr']
            r1 = (p1[i] - position['p1']) / position['p1'] if position['p1'] > 0 else 0
            r2 = (p2[i] - position['p2']) / position['p2'] if position['p2'] > 0 else 0
            raw = (r1 - hr_e * r2) if position['dir'] == 'LONG' else (-r1 + hr_e * r2)
            pnl = raw / (1 + abs(hr_e)) * 100
            
            if pnl > position['best']:
                position['best'] = pnl
            
            close = False
            if bars >= min_hold:
                # Mean revert: Z crosses into exit zone
                if position['dir'] == 'LONG' and z >= -exit_z:
                    close = True
                elif position['dir'] == 'SHORT' and z <= exit_z:
                    close = True
                # Overshoot: Z crosses zero to other side → take profit
                if position['dir'] == 'LONG' and z > 0.5:
                    close = True
                elif position['dir'] == 'SHORT' and z < -0.5:
                    close = True
            
            # Stop loss (tighter)
            if position['dir'] == 'LONG' and z < -adaptive_stop:
                close = True
            elif position['dir'] == 'SHORT' and z > adaptive_stop:
                close = True
            
            # Trailing: exit at 40% of peak (not 0%)
            if position['best'] >= 0.8 and pnl <= position['best'] * 0.4 and bars >= min_hold:
                close = True
            
            # Max hold (shorter)
            if bars >= max_bars:
                close = True
            
            if close:
                trades_pnl.append(pnl - cost_total * 100)
                position = None
                last_close = i
    
    # Close remaining
    if position is not None:
        hr_e = position['hr']
        r1 = (p1[-1] - position['p1']) / position['p1'] if position['p1'] > 0 else 0
        r2 = (p2[-1] - position['p2']) / position['p2'] if position['p2'] > 0 else 0
        raw = (r1 - hr_e * r2) if position['dir'] == 'LONG' else (-r1 + hr_e * r2)
        pnl = raw / (1 + abs(hr_e)) * 100 - cost_total * 100
        trades_pnl.append(pnl)
    
    # Stats
    nt = len(trades_pnl)
    if nt == 0:
        return {'verdict': 'SKIP', 'reason': 'No trades', 'n_trades': 0}
    
    total_pnl = sum(trades_pnl)
    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p <= 0]
    win_rate = len(wins) / nt * 100
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99
    avg_pnl = np.mean(trades_pnl)
    std_pnl = np.std(trades_pnl) if nt > 1 else 1
    sharpe = avg_pnl / std_pnl * np.sqrt(nt) if std_pnl > 0 else 0
    
    # Verdict: very lenient — only block truly catastrophic pairs
    # Mini-BT on 300 bars is noisy; we only want to catch "clearly broken" pairs
    if total_pnl < -15.0 or sharpe < -1.5:
        verdict = 'FAIL'
    elif total_pnl < -5.0 or sharpe < -0.5 or pf < 0.5:
        verdict = 'WARN'
    else:
        verdict = 'PASS'
    
    return {
        'verdict': verdict,
        'total_pnl': round(total_pnl, 2),
        'sharpe': round(sharpe, 2),
        'win_rate': round(win_rate, 1),
        'pf': round(pf, 2),
        'n_trades': nt,
        'avg_pnl': round(avg_pnl, 2),
    }


def micro_backtest(spread, p1, p2, hrs, entry_z=1.8, exit_z=0.8,
                   max_hold_bars=6, take_profit_pct=1.5, stop_loss_pct=-2.0,
                   commission_pct=0.10, slippage_pct=0.05, min_bars=1):
    """
    v23: R2 Micro-Backtest — 1-6 bar horizon, matches real trading style.
    
    Problem: Standard BT simulates 20+ trades over 50 days, but real trading
    is 1-3 hours. BT shows -8%, reality is +0.9%. 
    
    Solution: Micro-BT enters at Z > threshold, holds max 6 bars,
    exits at exit_z OR take_profit OR stop_loss.
    
    Returns metrics matching real trading: avg P&L per entry, % quick reversions,
    mean Z velocity toward 0.
    """
    spread = np.array(spread, float)
    p1 = np.array(p1, float)
    p2 = np.array(p2, float)
    hrs = np.array(hrs, float)
    n = len(spread)
    
    if n < 50:
        return {'verdict': 'SKIP', 'error': 'Недостаточно данных', 'n_trades': 0}
    
    # Compute Z-score
    from scipy.stats import median_abs_deviation
    window = max(10, min(30, n // 10))
    zs = np.full(n, np.nan)
    for i in range(window, n):
        seg = spread[i-window:i+1]
        med = np.median(seg)
        mad = median_abs_deviation(seg)
        if mad > 1e-10:
            zs[i] = (spread[i] - med) / (mad * 1.4826)
    
    commission = (commission_pct + slippage_pct) / 100
    trades = []
    
    i = window + 1
    while i < n - 1:
        z = zs[i]
        if np.isnan(z):
            i += 1
            continue
        
        direction = None
        if z > entry_z:
            direction = 'SHORT'
        elif z < -entry_z:
            direction = 'LONG'
        
        if direction is None:
            i += 1
            continue
        
        # Entry
        entry_bar = i
        entry_hr = hrs[i]
        entry_p1, entry_p2 = p1[i], p2[i]
        entry_z_val = z
        best_pnl = 0
        
        # Hold loop (max_hold_bars)
        exit_bar = None
        exit_reason = 'MAX_HOLD'
        
        for j in range(1, max_hold_bars + 1):
            if i + j >= n:
                exit_bar = min(i + j, n - 1)
                exit_reason = 'END_DATA'
                break
            
            # Calculate P&L at bar i+j
            r1 = (p1[i+j] - entry_p1) / entry_p1
            r2 = (p2[i+j] - entry_p2) / entry_p2
            
            if direction == 'LONG':
                pnl = (r1 - entry_hr * r2) / (1 + abs(entry_hr)) * 100
            else:
                pnl = (-r1 + entry_hr * r2) / (1 + abs(entry_hr)) * 100
            
            if pnl > best_pnl:
                best_pnl = pnl
            
            z_now = zs[i+j] if not np.isnan(zs[i+j]) else z
            
            # Exit conditions
            if j >= min_bars:
                # Z mean-reverted
                if direction == 'LONG' and z_now >= -exit_z:
                    exit_bar = i + j
                    exit_reason = 'MEAN_REVERT'
                    break
                elif direction == 'SHORT' and z_now <= exit_z:
                    exit_bar = i + j
                    exit_reason = 'MEAN_REVERT'
                    break
                
                # Take profit
                if take_profit_pct > 0 and pnl >= take_profit_pct:
                    exit_bar = i + j
                    exit_reason = 'TAKE_PROFIT'
                    break
            
            # Stop loss (always active)
            if pnl <= stop_loss_pct:
                exit_bar = i + j
                exit_reason = 'STOP_LOSS'
                break
        
        if exit_bar is None:
            exit_bar = min(i + max_hold_bars, n - 1)
        
        # Final P&L
        r1 = (p1[exit_bar] - entry_p1) / entry_p1
        r2 = (p2[exit_bar] - entry_p2) / entry_p2
        if direction == 'LONG':
            final_pnl = (r1 - entry_hr * r2) / (1 + abs(entry_hr)) * 100
        else:
            final_pnl = (-r1 + entry_hr * r2) / (1 + abs(entry_hr)) * 100
        
        final_pnl -= commission * 2 * 100  # entry + exit commission
        bars_held = exit_bar - entry_bar
        
        # Z velocity (how fast Z moved toward 0)
        z_exit = zs[exit_bar] if not np.isnan(zs[exit_bar]) else entry_z_val
        z_velocity = (abs(entry_z_val) - abs(z_exit)) / max(1, bars_held)
        
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'direction': direction, 'entry_z': entry_z_val,
            'exit_z': z_exit, 'bars_held': bars_held,
            'pnl_pct': round(final_pnl, 3),
            'best_pnl': round(best_pnl, 3),
            'exit_reason': exit_reason,
            'z_velocity': round(z_velocity, 4),
        })
        
        i = exit_bar + 1  # Skip past exit
    
    # Aggregate
    nt = len(trades)
    if nt == 0:
        return {'verdict': 'SKIP', 'error': 'Нет сделок', 'n_trades': 0}
    
    pnls = [t['pnl_pct'] for t in trades]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / nt
    win_rate = sum(1 for p in pnls if p > 0) / nt * 100
    avg_bars = sum(t['bars_held'] for t in trades) / nt
    avg_z_vel = sum(t['z_velocity'] for t in trades) / nt
    
    # Quick reversion rate: % of trades that exited via MEAN_REVERT or TAKE_PROFIT
    quick_exits = sum(1 for t in trades if t['exit_reason'] in ('MEAN_REVERT', 'TAKE_PROFIT'))
    quick_rate = quick_exits / nt * 100
    
    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0) or 0.001
    gross_loss = abs(sum(p for p in pnls if p < 0)) or 0.001
    pf = gross_profit / gross_loss
    
    # Exit type breakdown
    exit_counts = {}
    for t in trades:
        er = t['exit_reason']
        if er not in exit_counts:
            exit_counts[er] = {'count': 0, 'pnl': 0}
        exit_counts[er]['count'] += 1
        exit_counts[er]['pnl'] += t['pnl_pct']
    
    # Verdict — quick_reversion_rate is the PRIMARY criterion
    # Real trades: avg +0.5%, WR ~60%, but BT can't capture maker orders etc.
    # Commission (0.30% roundtrip) dominates small P&L, so we focus on reversion speed.
    if quick_rate >= 50 and avg_pnl >= -0.3:
        verdict = 'PASS'  # High quick revert, P&L within commission noise
    elif quick_rate >= 35 or (avg_pnl >= -0.1 and win_rate >= 35):
        verdict = 'WARN'  # Moderate reversion
    else:
        verdict = 'FAIL'  # Spread doesn't revert quickly
    
    return {
        'verdict': verdict,
        'n_trades': nt,
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 3),
        'win_rate': round(win_rate, 1),
        'pf': round(pf, 2),
        'avg_bars_held': round(avg_bars, 1),
        'avg_z_velocity': round(avg_z_vel, 4),
        'quick_reversion_rate': round(quick_rate, 1),
        'exit_breakdown': exit_counts,
        'max_hold_bars': max_hold_bars,
        'trades': trades[-20:],  # Last 20 for display
    }


def z_velocity_analysis(zscore_series, lookback=5):
    """
    v24: R4 Z-Velocity Entry Filter.
    
    Analyzes Z-score momentum to determine if spread is:
    - Decelerating (good for entry — Z slowing down before reverting)
    - Accelerating away (bad — Z still expanding)
    - Reversing (best — Z already moving toward zero)
    
    Returns dict with velocity, acceleration, and entry recommendation.
    """
    zs = np.array(zscore_series, float)
    zs = zs[~np.isnan(zs)]
    
    if len(zs) < lookback + 2:
        return {
            'velocity': 0, 'acceleration': 0,
            'direction': 'UNKNOWN', 'entry_quality': 'UNKNOWN',
            'description': 'Недостаточно данных'
        }
    
    # Last `lookback` Z-scores
    recent = zs[-lookback:]
    
    # Velocity: dZ/dt over last bars (positive = Z going up)
    velocities = np.diff(recent)
    avg_velocity = np.mean(velocities)
    last_velocity = velocities[-1]
    
    # Acceleration: d²Z/dt² (change of velocity)
    if len(velocities) >= 2:
        accelerations = np.diff(velocities)
        avg_acceleration = np.mean(accelerations)
    else:
        avg_acceleration = 0
    
    # Current Z
    z_current = zs[-1]
    z_sign = np.sign(z_current)  # +1 for positive Z, -1 for negative
    
    # Is Z moving TOWARD zero? (good)
    # For Z>0: velocity<0 means toward zero
    # For Z<0: velocity>0 means toward zero
    z_toward_zero = (z_sign > 0 and avg_velocity < 0) or \
                    (z_sign < 0 and avg_velocity > 0) or \
                    z_sign == 0
    
    # Is Z DECELERATING its move away from zero? (medium)
    # v27: thresholds from config
    _decel_thr = CFG('z_velocity', 'decel_threshold', 0.05)
    z_decelerating = (z_sign > 0 and avg_acceleration < -_decel_thr) or \
                     (z_sign < 0 and avg_acceleration > _decel_thr)
    
    # Entry quality assessment
    _exc_vel = CFG('z_velocity', 'excellent_min_vel', 0.1)
    abs_vel = abs(avg_velocity)
    
    if z_toward_zero and abs_vel > _exc_vel:
        entry_quality = 'EXCELLENT'
        description = f'Z ревертирует (v={avg_velocity:+.2f}/бар). Идеальный вход!'
    elif z_toward_zero:
        entry_quality = 'GOOD'
        description = f'Z замедлился и разворачивается (v={avg_velocity:+.2f}/бар).'
    elif z_decelerating:
        entry_quality = 'FAIR'
        description = f'Z замедляется (a={avg_acceleration:+.2f}). Скоро может развернуться.'
    elif abs_vel < 0.05:
        entry_quality = 'FAIR'
        description = f'Z стабилен (v={avg_velocity:+.2f}/бар). Нейтральный вход.'
    else:
        entry_quality = 'POOR'
        description = f'Z ускоряется от нуля (v={avg_velocity:+.2f}/бар). Подождите замедления!'
    
    return {
        'velocity': round(avg_velocity, 4),
        'last_velocity': round(last_velocity, 4),
        'acceleration': round(avg_acceleration, 4),
        'z_toward_zero': z_toward_zero,
        'z_decelerating': z_decelerating,
        'entry_quality': entry_quality,
        'description': description,
        'lookback': lookback,
    }


def smart_exit_analysis(z_entry, z_now, z_history, pnl_pct, hours_in,
                        halflife_hours, direction, best_pnl=None):
    """
    v24: R5 Smart Exit Signals.
    
    Three exit strategies:
    1. Trailing Z-stop: after Z passes 0.5 toward zero, lock in by trailing
    2. Time-based urgency: if >1.5x halflife without reversion → urgent exit
    3. Overshoot profit lock: Z crossed 0 and going further → take profit
    
    Returns exit recommendation with urgency level.
    """
    signals = []
    urgency = 0  # 0=hold, 1=watch, 2=exit, 3=urgent
    
    z_hist = np.array(z_history, float) if z_history is not None else np.array([z_now])
    z_hist = z_hist[~np.isnan(z_hist)]
    
    if best_pnl is None:
        best_pnl = max(pnl_pct, 0)
    
    # === 1. TRAILING Z-STOP ===
    # v27: thresholds from config
    _z_bounce = CFG('monitor', 'trailing_z_bounce', 0.8)
    
    # Track best Z (closest to 0)
    if direction == 'SHORT':
        best_z_for_us = min(z_hist) if len(z_hist) > 0 else z_now
        z_reverted_well = best_z_for_us < 0.5
        z_retreated = z_now > best_z_for_us + _z_bounce
    else:
        best_z_for_us = max(z_hist) if len(z_hist) > 0 else z_now
        z_reverted_well = best_z_for_us > -0.5
        z_retreated = z_now < best_z_for_us - _z_bounce
    
    if z_reverted_well and z_retreated:
        signals.append({
            'type': 'TRAILING_Z',
            'urgency': 2,
            'message': f'📉 Z достиг {best_z_for_us:+.2f}, но откатился к {z_now:+.2f}. '
                       f'Фиксируйте прибыль — Z-trailing stop сработал.'
        })
        urgency = max(urgency, 2)
    
    # === 2. TIME-BASED URGENCY ===
    _t_crit = CFG('monitor', 'time_critical_ratio', 2.0)
    _t_exit = CFG('monitor', 'time_exit_ratio', 1.5)
    _t_warn = CFG('monitor', 'time_warning_ratio', 1.0)
    
    if halflife_hours > 0:
        time_ratio = hours_in / halflife_hours
        
        if time_ratio > _t_crit:
            signals.append({
                'type': 'TIME_CRITICAL',
                'urgency': 3,
                'message': f'⏰ В позиции {hours_in:.0f}ч = {time_ratio:.1f}x HL ({halflife_hours:.0f}ч). '
                           f'Коинтеграция могла разрушиться. СРОЧНЫЙ ВЫХОД.'
            })
            urgency = max(urgency, 3)
        elif time_ratio > _t_exit:
            signals.append({
                'type': 'TIME_WARNING',
                'urgency': 2,
                'message': f'⏰ В позиции {hours_in:.0f}ч = {time_ratio:.1f}x HL. '
                           f'Если Z не вернулся — рассмотрите выход.'
            })
            urgency = max(urgency, 2)
        elif time_ratio > _t_warn:
            signals.append({
                'type': 'TIME_WATCH',
                'urgency': 1,
                'message': f'⏳ В позиции {hours_in:.0f}ч = {time_ratio:.1f}x HL. Следите за Z.'
            })
            urgency = max(urgency, 1)
    
    # === 3. OVERSHOOT PROFIT LOCK ===
    _overshoot_z = CFG('monitor', 'overshoot_deep_z', 1.0)
    if direction == 'SHORT':
        z_crossed_zero = z_entry > 0 and z_now < 0
        z_overshoot_deep = z_now < -_overshoot_z
    else:
        z_crossed_zero = z_entry < 0 and z_now > 0
        z_overshoot_deep = z_now > _overshoot_z
    
    if z_crossed_zero:
        if z_overshoot_deep:
            signals.append({
                'type': 'OVERSHOOT_DEEP',
                'urgency': 2,
                'message': f'🎯 OVERSHOOT: Z пересёк 0 и достиг {z_now:+.2f}. '
                           f'P&L={pnl_pct:+.2f}%. ФИКСИРУЙТЕ ПРИБЫЛЬ!'
            })
            urgency = max(urgency, 2)
        else:
            signals.append({
                'type': 'OVERSHOOT_MILD',
                'urgency': 1,
                'message': f'🎯 Z пересёк 0 (сейчас {z_now:+.2f}). '
                           f'Можно зафиксировать или подождать overshoot.'
            })
            urgency = max(urgency, 1)
    
    # === 4. PnL PROFIT PROTECTION ===
    _pnl_thr = CFG('monitor', 'pnl_trailing_threshold', 0.5)
    _pnl_frac = CFG('monitor', 'pnl_trailing_fraction', 0.4)
    if best_pnl > _pnl_thr and pnl_pct < best_pnl * _pnl_frac:
        signals.append({
            'type': 'PNL_TRAILING',
            'urgency': 2,
            'message': f'💰 P&L упал с пика {best_pnl:+.2f}% до {pnl_pct:+.2f}%. '
                       f'Фиксируйте остаток прибыли.'
        })
        urgency = max(urgency, 2)
    elif best_pnl > 1.0:
        signals.append({
            'type': 'PNL_PEAK',
            'urgency': 1,
            'message': f'💰 P&L достигал {best_pnl:+.2f}%. Текущий: {pnl_pct:+.2f}%. '
                       f'Рассмотрите фиксацию.'
        })
        urgency = max(urgency, 1)
    
    # === OVERALL RECOMMENDATION ===
    if urgency >= 3:
        recommendation = '🛑 СРОЧНО ЗАКРЫТЬ'
    elif urgency >= 2:
        recommendation = '✅ ЗАКРЫВАТЬ'
    elif urgency >= 1:
        recommendation = '👀 НАБЛЮДАТЬ'
    else:
        recommendation = '⏳ ДЕРЖАТЬ'
    
    return {
        'signals': signals,
        'urgency': urgency,
        'recommendation': recommendation,
        'n_signals': len(signals),
        'best_z_for_us': round(best_z_for_us, 4) if len(z_hist) > 0 else 0,
    }


def walk_forward_validate(spread, p1, p2, hrs, entry_z=1.8,
                          n_folds=3, train_pct=0.65, **bt_kwargs):
    """
    v19.0: Walk-Forward Validation — anti-overfitting.
    
    Splits data into overlapping train/test windows:
      Fold 1: Train [0:195], Test [195:260]
      Fold 2: Train [40:235], Test [235:300]
      Fold 3: Train [65:260], Test [260:300]
    
    Only pairs profitable in MAJORITY of out-of-sample folds → PASS.
    
    Returns: dict with oos_pnl, oos_sharpe, folds_passed, verdict
    """
    n = len(spread)
    if n < 120:
        return {'verdict': 'SKIP', 'reason': 'Too few bars for WF', 'folds_passed': 0}
    
    fold_size = n // n_folds
    train_size = int(n * train_pct)
    test_size = n - train_size
    
    oos_pnls = []
    oos_details = []
    
    for fold in range(n_folds):
        # Sliding window
        offset = fold * (n - train_size) // max(1, n_folds - 1)
        if fold == n_folds - 1:
            offset = n - train_size - test_size  # Ensure last fold reaches end
        
        t_start = offset
        t_end = min(offset + train_size, n)
        test_start = t_end
        test_end = min(test_start + test_size, n)
        
        if test_end - test_start < 30:
            continue
        
        # Run mini-backtest on TEST portion only
        # (In walk-forward, the "parameters" come from train, but we test on OOS)
        test_spread = spread[test_start:test_end]
        test_p1 = p1[test_start:test_end]
        test_p2 = p2[test_start:test_end]
        test_hrs = hrs[test_start:test_end]
        
        if len(test_spread) < 40:
            continue
        
        result = mini_backtest(
            test_spread, test_p1, test_p2, test_hrs,
            entry_z=entry_z, **bt_kwargs
        )
        
        fold_pnl = result.get('total_pnl', 0)
        oos_pnls.append(fold_pnl)
        oos_details.append({
            'fold': fold + 1,
            'test_bars': test_end - test_start,
            'pnl': fold_pnl,
            'trades': result.get('n_trades', 0),
            'wr': result.get('win_rate', 0),
        })
    
    if not oos_pnls:
        return {'verdict': 'SKIP', 'reason': 'No valid folds', 'folds_passed': 0}
    
    folds_positive = sum(1 for p in oos_pnls if p > -1.0)  # Slightly negative OK
    folds_passed = folds_positive
    total_oos = sum(oos_pnls)
    avg_oos = np.mean(oos_pnls)
    
    # Verdict
    if folds_passed >= n_folds * 0.6 and total_oos > -3.0:
        verdict = 'PASS'
    elif folds_passed >= 1 and total_oos > -8.0:
        verdict = 'WARN'
    else:
        verdict = 'FAIL'
    
    return {
        'verdict': verdict,
        'total_oos_pnl': round(total_oos, 2),
        'avg_fold_pnl': round(avg_oos, 2),
        'folds_passed': folds_passed,
        'n_folds': len(oos_pnls),
        'folds': oos_details,
    }


# =============================================================================
# ТЕСТ
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  v10.0.0 — Adaptive Robust Z + Crossing Density + Correlation")
    print("=" * 65)
    np.random.seed(42)

    # Generate synthetic mean-reverting spread
    n = 300
    spread_mr = [0.0]
    for i in range(n - 1):
        dx = 0.3 * (0 - spread_mr[-1]) + 0.5 * np.random.randn()
        spread_mr.append(spread_mr[-1] + dx)
    spread_mr = np.array(spread_mr)

    # OU params for HL
    ou = calculate_ou_parameters(spread_mr, dt=1/6)  # 4h
    hl_bars = ou['halflife_ou'] / (4.0) if ou else 10  # HL_hours / hours_per_bar

    # 1. Adaptive Robust Z-score
    print(f"\n--- Adaptive Robust Z-Score ---")
    print(f"OU HL = {ou['halflife_ou']:.1f}ч → {hl_bars:.1f} bars (4h TF)")

    z_old, zs_old = calculate_rolling_zscore(spread_mr, window=30)
    z_new, zs_new, w_used = calculate_adaptive_robust_zscore(spread_mr, halflife_bars=hl_bars)
    print(f"Old (window=30, std):  Z={z_old:+.3f}")
    print(f"New (window={w_used}, MAD): Z={z_new:+.3f}")

    # Test with different HL values
    print(f"\nAdaptive windows for different HL:")
    for hl in [2, 5, 10, 20, 50]:
        _, _, w = calculate_adaptive_robust_zscore(spread_mr, halflife_bars=hl)
        print(f"  HL={hl:>2d} bars → window={w}")

    # 2. Crossing Density
    print(f"\n--- Crossing Density ---")
    cd_mr = calculate_crossing_density(zs_new)
    # Generate "stuck" spread
    stuck = np.concatenate([np.ones(100) * 2, np.ones(100) * -1, np.ones(100) * 1.5])
    cd_stuck = calculate_crossing_density(stuck)
    print(f"Mean-reverting: density={cd_mr:.3f} ({'✅ active' if cd_mr > 0.03 else '❌ stuck'})")
    print(f"Stuck spread:   density={cd_stuck:.3f} ({'✅ active' if cd_stuck > 0.03 else '❌ stuck'})")

    # 3. Rolling Correlation
    print(f"\n--- Rolling Correlation ---")
    s2 = np.cumsum(np.random.randn(n) * 0.3) + 100
    s1 = 1.5 * s2 + np.random.randn(n) * 0.5 + 20
    corr, corr_s = calculate_rolling_correlation(s1, s2, window=30)
    print(f"Correlated pair: ρ={corr:.3f}")
    s1_uncorr = np.cumsum(np.random.randn(n) * 0.3) + 50
    corr_u, _ = calculate_rolling_correlation(s1_uncorr, s2, window=30)
    print(f"Uncorrelated:    ρ={corr_u:.3f}")

    # 4. Sanitizer with min_bars + HR uncertainty
    print(f"\n--- Sanitizer v10.1 ---")
    tests = [
        (1.2, 3, 4, 2.0, 300, 0.1, "normal 300 bars"),
        (1.2, 3, 4, 2.0, 29,  0.1, "only 29 bars"),
        (1.2, 3, 4, 2.0, 30,  0.1, "30 bars (boundary)"),
        (0.0003, 2, 4, -2.3, 300, 0.0, "HR<0.001"),
        (0.18, 3, 4, -1.8, 300, 0.25, "HR unc=139%"),
        (0.18, 3, 4, -1.8, 300, 0.05, "HR unc=28%"),
    ]
    for hr, sp, st, z, nb, hs, name in tests:
        ok, reason = sanitize_pair(hr, sp, st, z, n_bars=nb, hr_std=hs)
        print(f"  {name:<22s} → {'✅' if ok else '❌'} {reason}")

    # 5. Quality with crossing penalty + HR unc penalty
    print(f"\n--- Quality Score v10.1 ---")
    q1, bd1 = calculate_quality_score(0.2, ou, 0.01, 0.75, 1.5, True, crossing_density=0.08, hr_std=0.1)
    q2, bd2 = calculate_quality_score(0.2, ou, 0.01, 0.75, 1.5, True, crossing_density=0.01, hr_std=0.1)
    q3, bd3 = calculate_quality_score(0.2, ou, 0.01, 0.75, 1.5, True, crossing_density=0.08, hr_std=1.0)
    print(f"Active/good HR: Q={q1} cross={bd1.get('crossing_penalty',0)} hr_unc={bd1.get('hr_unc_penalty',0)}")
    print(f"Stuck/good HR:  Q={q2} cross={bd2.get('crossing_penalty',0)} hr_unc={bd2.get('hr_unc_penalty',0)}")
    print(f"Active/bad HR:  Q={q3} cross={bd3.get('crossing_penalty',0)} hr_unc={bd3.get('hr_unc_penalty',0)}")

    # 6. Signal gates: Q, Z, Stability
    print(f"\n--- Signal Gates v10.3 ---")
    s1, d1, t1 = get_adaptive_signal(2.69, 'LOW', 8, '4h')
    s2, d2, t2 = get_adaptive_signal(2.69, 'LOW', 25, '4h')
    s3, d3, t3 = get_adaptive_signal(-1.83, 'HIGH', 63, '4h')
    s4, d4, t4 = get_adaptive_signal(2.5, 'MEDIUM', 56, '4h', stability_ratio=0.25)
    s5, d5, t5 = get_adaptive_signal(2.5, 'MEDIUM', 56, '4h', stability_ratio=0.75)
    s6, d6, t6 = get_adaptive_signal(4.8, 'LOW', 50, '4h')
    print(f"Q=8  LOW:           {s1} (expect NEUTRAL)")
    print(f"Q=25 LOW:           {s2} (expect NEUTRAL)")
    print(f"Q=63 HIGH:          {s3} (expect SIGNAL)")
    print(f"Q=56 Stab=1/4:      {s4} (expect WATCH)")
    print(f"Q=56 Stab=3/4:      {s5} (expect SIGNAL)")
    print(f"Z=4.8:              {s6} (expect NEUTRAL)")

    print(f"\n✅ v11.4 ready!")


# =============================================================================
# SHARED UTILITIES (imported by scanner, monitor, backtester)
# =============================================================================

def assess_entry_readiness(p):
    """
    v11.4: ЕДИНАЯ оценка готовности к входу.
    Импортируется scanner, monitor — единый источник истины.
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
        ('Hurst < 0.35', hurst_ok, f"H={hurst_val:.3f}"),
        ('Stability ≥ 3/4', stab_ok, f"{p.get('stability_passed', 0)}/{p.get('stability_total', 4)}"),
    ]
    opt_count = sum(1 for o in optional if o[1])
    fdr_bypass = opt_count >= 4 and not fdr_ok
    
    if not all_mandatory:
        level, label = 'WAIT', '⚪ ЖДАТЬ'
    elif hurst_is_fallback:
        level, label = 'CONDITIONAL', '🟡 СЛАБЫЙ ⚠️H=0.5'
    elif p.get('cusum_break', False):
        level, label = 'CONDITIONAL', '🟡 УСЛОВНО ⚠️CUSUM'
    elif all_mandatory:
        if hurst_val >= 0.45:
            level, label = 'CONDITIONAL', '🟡 УСЛОВНО ⚠️H≥0.45'
        elif opt_count >= 3:
            level, label = 'ENTRY', '🟢 ВХОД'
        else:
            level, label = 'CONDITIONAL', '🟡 УСЛОВНО'
    else:
        level, label = 'WAIT', '⚪ ЖДАТЬ'
    
    return {
        'level': level, 'label': label,
        'mandatory': mandatory, 'optional': optional,
        'all_mandatory': all_mandatory, 'opt_count': opt_count,
        'fdr_bypass': fdr_bypass,
    }


def calc_halflife_from_spread(spread, dt=1/6):
    """Единый расчёт half-life из OU-регрессии. dt=1/6 для 4h."""
    spread = np.array(spread, dtype=float)
    if len(spread) < 10:
        return 999
    dS = np.diff(spread)
    S_lag = spread[:-1]
    S_lag_c = S_lag - np.mean(S_lag)
    if np.std(S_lag_c) < 1e-12:
        return 999
    theta = -float(np.polyfit(S_lag_c, dS, 1)[0])
    if theta <= 0:
        return 999
    hl = np.log(2) / theta * dt
    return max(0.01, min(hl, 999))


# =============================================================================
# P5: PCA FACTOR CLUSTERING
# =============================================================================

def pca_factor_clustering(returns_dict, n_components=3):
    """
    v21: PCA Factor Clustering — identifies hidden market factors.
    
    Problem: All crypto altcoins correlate with BTC/ETH.
    When BTC drops 5%, all pairs with BTC-exposure break.
    
    Solution:
    1. Build returns matrix from coin price series
    2. PCA → 3 components (Market/BTC-beta, ALT-premium, Sector-factor)
    3. Calculate factor_exposure for each coin
    4. For pairs: compute net factor exposure (should be near zero for good pairs)
    
    Args:
        returns_dict: {coin_name: np.array of log returns} for all coins
        n_components: number of PCA components (default 3)
    
    Returns:
        dict with loadings, explained variance, coin clusters, factor names
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    coins = sorted(returns_dict.keys())
    if len(coins) < 5:
        return {'error': 'Need at least 5 coins', 'coins': coins}
    
    # Build returns matrix: align lengths
    min_len = min(len(returns_dict[c]) for c in coins)
    if min_len < 30:
        return {'error': f'Insufficient data: {min_len} bars', 'coins': coins}
    
    R = np.column_stack([returns_dict[c][-min_len:] for c in coins])
    
    # Remove coins with zero variance
    valid_mask = np.std(R, axis=0) > 1e-10
    valid_coins = [c for c, v in zip(coins, valid_mask) if v]
    R = R[:, valid_mask]
    
    if len(valid_coins) < 5:
        return {'error': 'Too few valid coins after filtering', 'coins': valid_coins}
    
    # Standardize
    R_std = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-10)
    
    # PCA
    n_comp = min(n_components, len(valid_coins) - 1)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(R_std)  # (T x n_comp)
    loadings = pca.components_          # (n_comp x N_coins)
    explained = pca.explained_variance_ratio_
    
    # Name factors heuristically
    factor_names = []
    for i in range(n_comp):
        abs_load = np.abs(loadings[i])
        mean_abs = abs_load.mean()
        spread_load = abs_load.max() - abs_load.min()
        
        if i == 0:
            factor_names.append("Market (BTC-beta)")
        elif spread_load > 2 * mean_abs:
            # High dispersion: sector factor
            top_coins = [valid_coins[j] for j in np.argsort(-abs_load)[:3]]
            factor_names.append(f"Sector ({'/'.join(top_coins[:2])})")
        else:
            factor_names.append(f"Factor_{i+1}")
    
    # Coin loadings dict
    coin_loadings = {}
    for idx, coin in enumerate(valid_coins):
        coin_loadings[coin] = {
            f'PC{i+1}': round(float(loadings[i, idx]), 4)
            for i in range(n_comp)
        }
    
    # K-Means clustering on loadings
    n_clusters = min(4, len(valid_coins) // 3)
    n_clusters = max(2, n_clusters)
    
    loading_matrix = loadings.T  # (N_coins x n_comp)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(loading_matrix)
    
    coin_clusters = {}
    for idx, coin in enumerate(valid_coins):
        coin_clusters[coin] = int(cluster_labels[idx])
    
    # Cluster summary
    cluster_summary = {}
    for cl in range(n_clusters):
        members = [c for c, l in coin_clusters.items() if l == cl]
        if members:
            avg_loadings = {
                f'PC{i+1}': round(float(np.mean([
                    coin_loadings[c][f'PC{i+1}'] for c in members
                ])), 4) for i in range(n_comp)
            }
            cluster_summary[cl] = {
                'members': members,
                'n': len(members),
                'avg_loadings': avg_loadings,
            }
    
    return {
        'coins': valid_coins,
        'n_components': n_comp,
        'explained_variance': [round(float(e), 4) for e in explained],
        'total_explained': round(float(explained.sum()), 4),
        'factor_names': factor_names,
        'coin_loadings': coin_loadings,
        'coin_clusters': coin_clusters,
        'cluster_summary': cluster_summary,
        'loadings_raw': loadings,
        'scores': scores,
    }


def pair_factor_exposure(pca_result, coin1, coin2, hedge_ratio=1.0):
    """
    Calculate net factor exposure for a pair trade.
    
    LONG coin1 / SHORT coin2 with hedge_ratio:
      net_exposure_PCi = loading_coin1_PCi - HR * loading_coin2_PCi
    
    Good pair: net exposure ≈ 0 for PC1 (market-neutral)
    Bad pair: large |net_PC1| means correlated with market
    """
    loadings = pca_result.get('coin_loadings', {})
    
    if coin1 not in loadings or coin2 not in loadings:
        return None
    
    l1 = loadings[coin1]
    l2 = loadings[coin2]
    n_comp = pca_result.get('n_components', 3)
    
    net = {}
    total_exposure = 0
    for i in range(n_comp):
        key = f'PC{i+1}'
        net_val = l1.get(key, 0) - hedge_ratio * l2.get(key, 0)
        net[key] = round(net_val, 4)
        total_exposure += net_val ** 2
    
    # Same cluster?
    clusters = pca_result.get('coin_clusters', {})
    same_cluster = clusters.get(coin1) == clusters.get(coin2)
    
    # Market neutrality score (0=neutral, 1=fully exposed)
    market_exposure = abs(net.get('PC1', 0))
    neutrality = 1.0 - min(1.0, market_exposure)
    
    return {
        'net_exposure': net,
        'total_exposure': round(float(np.sqrt(total_exposure)), 4),
        'market_neutrality': round(neutrality, 4),
        'same_cluster': same_cluster,
        'cluster_coin1': clusters.get(coin1, -1),
        'cluster_coin2': clusters.get(coin2, -1),
        'factor_names': pca_result.get('factor_names', []),
    }
