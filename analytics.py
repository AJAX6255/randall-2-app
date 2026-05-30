#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analytics.py

Deterministic analytical primitives for macro-financial time-series calculations.
Implemented as vectorized, robust functions using numpy and pandas.
All metrics avoid LLM-side calculations and are mathematically reproducible.
"""

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# NORMALIZATION & SCALING
# -----------------------------------------------------------------------------

def normalize_series(series: pd.Series, dates: pd.Series, anchor_date: str | pd.Timestamp) -> pd.Series:
    """
    Normalizes a series to equal 100 on a specific anchor date.
    Formula: (x_t / x_anchor) * 100
    """
    temp_df = pd.DataFrame({"date": pd.to_datetime(dates), "val": series.astype(float)})
    anchor_dt = pd.to_datetime(anchor_date)
    
    # Find closest date in dataset to the anchor date
    available_dates = temp_df["date"]
    time_deltas = (available_dates - anchor_dt).abs()
    if time_deltas.empty:
        return pd.Series(100.0, index=series.index)
        
    closest_idx = time_deltas.idxmin()
    anchor_val = temp_df.loc[closest_idx, "val"]
    
    if pd.isna(anchor_val) or anchor_val == 0:
        # Fallback: search for first non-zero, non-NaN value near anchor
        valid_vals = temp_df.dropna()
        if not valid_vals.empty:
            closest_valid_idx = (valid_vals["date"] - anchor_dt).abs().idxmin()
            anchor_val = valid_vals.loc[closest_valid_idx, "val"]
            
    if pd.isna(anchor_val) or anchor_val == 0:
        anchor_val = 1.0  # Avoid division by zero/NaN issues
        
    return (series / anchor_val) * 100.0

def zscore_series(series: pd.Series, window: int = 90) -> pd.Series:
    """
    Calculates rolling Z-score.
    Formula: (x_t - mean_window) / std_window
    """
    rolling_mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    rolling_std = series.rolling(window, min_periods=max(5, window // 4)).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    z = (series - rolling_mean) / rolling_std
    return z.ffill().fillna(0.0)

def percentile_scaling(series: pd.Series, window: int = 90) -> pd.Series:
    """
    Calculates rolling percentile rank of the current value relative to the window.
    Outputs values between 0.0 and 100.0.
    """
    def get_rank(win):
        if len(win) == 0 or pd.isna(win[-1]):
            return np.nan
        sorted_win = sorted(win[~np.isnan(win)])
        if len(sorted_win) <= 1:
            return 50.0
        val = win[-1]
        idx = sorted_win.index(val)
        return (idx / (len(sorted_win) - 1)) * 100.0

    return series.rolling(window, min_periods=max(5, window // 4)).apply(get_rank, raw=True).ffill().fillna(50.0)

def log_scaling(series: pd.Series) -> pd.Series:
    """
    Applies natural log scaling to the series. Shifts values if they are negative.
    """
    min_val = series.min()
    shift = 0.0
    if min_val <= 0:
        shift = abs(min_val) + 1.0
    return np.log(series + shift)

# -----------------------------------------------------------------------------
# VOLATILITY & BETA
# -----------------------------------------------------------------------------

def rolling_volatility(returns: pd.Series, window: int = 30, annualization_factor: float = 252.0) -> pd.Series:
    """
    Calculates rolling annualized volatility of returns.
    """
    vol = returns.rolling(window, min_periods=max(3, window // 4)).std()
    return vol * np.sqrt(annualization_factor)

def rolling_beta(asset_returns: pd.Series, market_returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculates rolling beta of an asset returns series relative to a market returns series.
    Formula: Cov(asset, market) / Var(market)
    """
    df = pd.DataFrame({"asset": asset_returns, "market": market_returns})
    rolling_cov = df["asset"].rolling(window).cov(df["market"])
    rolling_var = df["market"].rolling(window).var()
    
    # Avoid division by zero
    rolling_var = rolling_var.replace(0, np.nan)
    beta = rolling_cov / rolling_var
    return beta.ffill().fillna(1.0)

# -----------------------------------------------------------------------------
# CORRELATION & RELATIONSHIPS
# -----------------------------------------------------------------------------

def rolling_correlation(series1: pd.Series, series2: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculates rolling Pearson correlation between two series.
    """
    return series1.rolling(window, min_periods=max(5, window // 4)).corr(series2).ffill().fillna(0.0)

def detect_divergence(series1: pd.Series, series2: pd.Series, window: int = 20) -> pd.Series:
    """
    Detects divergence in trends by comparing rolling slopes of linear regressions.
    Returns:
    - 1: Bullish divergence (Asset 1 down, Asset 2 up)
    - -1: Bearish divergence (Asset 1 up, Asset 2 down)
    - 0: Converging/No strong divergence
    """
    def get_slope(y):
        n = len(y)
        if n < 5 or pd.isna(y).any():
            return 0.0
        x = np.arange(n)
        slope, _ = np.polyfit(x, y, 1)
        return slope

    slope1 = series1.rolling(window, min_periods=window).apply(get_slope, raw=True)
    slope2 = series2.rolling(window, min_periods=window).apply(get_slope, raw=True)

    # Normalize slopes to signs
    sign1 = np.sign(slope1)
    sign2 = np.sign(slope2)

    # Divergence occurs when signs are opposite
    divergence = np.zeros(len(series1))
    divergence[(sign1 < 0) & (sign2 > 0)] = 1.0   # Bullish divergence
    divergence[(sign1 > 0) & (sign2 < 0)] = -1.0  # Bearish divergence

    return pd.Series(divergence, index=series1.index)

def lead_lag_analysis(series1: pd.Series, series2: pd.Series, max_lag: int = 15) -> int:
    """
    Performs cross-correlation analysis to find the lag (in periods) that maximizes
    the correlation. Returns positive value if series1 leads series2, and negative
    if series2 leads series1.
    """
    s1 = (series1 - series1.mean()) / (series1.std() or 1.0)
    s2 = (series2 - series2.mean()) / (series2.std() or 1.0)
    
    best_lag = 0
    max_corr = -1.0
    
    # Align and compute correlation for different lag shifts
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = s1.iloc[:lag].corr(s2.shift(-lag).iloc[:lag])
        elif lag > 0:
            corr = s1.iloc[lag:].corr(s2.shift(lag).iloc[lag:])
        else:
            corr = s1.corr(s2)
            
        if not pd.isna(corr) and abs(corr) > max_corr:
            max_corr = abs(corr)
            best_lag = lag
            
    return best_lag

# -----------------------------------------------------------------------------
# ADVANCED QUANTITATIVE PRIMITIVES
# -----------------------------------------------------------------------------

def compute_drawdown(series: pd.Series) -> pd.Series:
    """
    Calculates percentage drawdown of a price series from its running maximum.
    """
    running_max = series.cummax()
    # Avoid division by zero
    running_max = running_max.replace(0, np.nan)
    drawdown = (series - running_max) / running_max * 100.0
    return drawdown.fillna(0.0)

def compute_relative_strength(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Computes relative strength ratio of series1 divided by series2.
    """
    denom = series2.replace(0, np.nan)
    return (series1 / denom).ffill().fillna(1.0)

def rolling_pca(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Extracts the first principal component (projection) from a multi-column dataframe
    using SVD on a rolling window basis. Represents systemic macro factor.
    """
    pc_values = []
    
    for i in range(len(df)):
        if i < window:
            pc_values.append(0.0)
            continue
            
        sub_df = df.iloc[i - window + 1:i + 1].copy()
        
        # 1. Fill missing and standardize variables
        sub_df = sub_df.ffill().bfill().fillna(0.0)
        mean = sub_df.mean()
        std = sub_df.std().replace(0, 1.0)
        norm_df = (sub_df - mean) / std
        
        # 2. Singular Value Decomposition
        try:
            U, S, Vt = np.linalg.svd(norm_df.values, full_matrices=False)
            # The first singular vector in Vt corresponds to the direction of highest variance
            first_component = Vt[0, :]
            
            # 3. Project current standardized point onto first component
            latest_point = norm_df.iloc[-1].values
            projection = np.dot(latest_point, first_component)
            pc_values.append(projection)
        except Exception:
            pc_values.append(0.0)
            
    return pd.Series(pc_values, index=df.index)

def volatility_adjustment(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Adjusts the series scale by dividing by its rolling standard deviation.
    """
    std = series.rolling(window, min_periods=5).std().replace(0, 1.0).ffill().fillna(1.0)
    return series / std

def volatility_compression(series: pd.Series, threshold: float = 2.0, window: int = 30) -> pd.Series:
    """
    Compresses outlier spikes exceeding (threshold * rolling std) to limit display distortion.
    """
    rolling_mean = series.rolling(window, min_periods=5).mean().ffill().fillna(series.iloc[0] if not series.empty else 0)
    rolling_std = series.rolling(window, min_periods=5).std().ffill().fillna(1.0)
    
    upper_bound = rolling_mean + threshold * rolling_std
    lower_bound = rolling_mean - threshold * rolling_std
    
    return series.clip(lower_bound, upper_bound)

def rolling_smoothing(series: pd.Series, window: int = 10, method: str = 'exponential') -> pd.Series:
    """
    Smoothes a series using SMA or EMA.
    """
    if method == 'exponential':
        return series.ewm(span=window, adjust=False).mean()
    return series.rolling(window, min_periods=max(1, window // 2)).mean().ffill()

# -----------------------------------------------------------------------------
# REGIME DETECTION & SEGMENTATION
# -----------------------------------------------------------------------------

def liquidity_regime_detection(stablecoin_mkt_cap: pd.Series, sofr_spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Determines liquidity regime based on stablecoin growth and SOFR funding spreads.
    Output categories: 'Liquidity Expansion', 'Liquidity Drain', 'Systemic Funding Stress'
    """
    stablecoin_pct = stablecoin_mkt_cap.pct_change(window).fillna(0.0)
    
    regime = []
    for s_pct, spread in zip(stablecoin_pct, sofr_spread):
        if spread > 0.15:  # Over 15bps spread
            regime.append("Systemic Funding Stress")
        elif s_pct < -0.01:  # Contraction of stablecoin cap
            regime.append("Liquidity Drain")
        else:
            regime.append("Liquidity Expansion")
            
    return pd.Series(regime, index=stablecoin_mkt_cap.index)

def detect_regimes(master_df: pd.DataFrame) -> pd.Series:
    """
    Performs Oil Supply vs. Demand Shock and risk regime classification.
    Inputs:
    - SPY (S&P 500)
    - VIX (Volatility Index)
    - DGS10 (10Y Yield)
    - GLD (Gold)
    
    Regime classifications:
    - 'Oil Supply Shock': Equities fall, Volatility rises, Yields rise (inflation fear)
    - 'Oil Demand Shock': Equities rise, Volatility falls, Yields rise (growth expansion)
    - 'Flight to Safety': Yields fall, Gold/VIX rise, Equities fall
    - 'Expansionary': Equities rise, VIX falls, Yields stable/falling
    - 'Neutral': Standard conditions
    """
    # Calculate percentage changes
    spy_ret = master_df["SPY"].pct_change(5).fillna(0.0)
    vix_ret = master_df["VIX"].pct_change(5).fillna(0.0)
    dgs_chg = master_df["DGS10"].diff(5).fillna(0.0)
    gld_ret = master_df["GLD"].pct_change(5).fillna(0.0)
    
    regimes = []
    for spy, vix, dgs, gld in zip(spy_ret, vix_ret, dgs_chg, gld_ret):
        if spy < -0.01 and vix > 0.05 and dgs > 0.05:
            regimes.append("Oil Supply Shock")
        elif spy > 0.01 and vix < 0.00 and dgs > 0.02:
            regimes.append("Oil Demand Shock")
        elif spy < -0.01 and vix > 0.05 and dgs < -0.05 and gld > 0.01:
            regimes.append("Flight to Safety")
        elif spy > 0.01 and vix < 0.00:
            regimes.append("Expansionary")
        else:
            regimes.append("Neutral")
            
    return pd.Series(regimes, index=master_df.index)

def regime_segmentation(stress_composite: pd.Series) -> pd.Series:
    """
    Segments a 0-100 stress score into qualitative categories.
    """
    conditions = [
        stress_composite < 25.0,
        (stress_composite >= 25.0) & (stress_composite < 50.0),
        (stress_composite >= 50.0) & (stress_composite < 75.0),
        stress_composite >= 75.0
    ]
    labels = ["Low Stress", "Moderate Stress", "High Stress", "Crisis"]
    return pd.Series(np.select(conditions, labels, default="Unknown"), index=stress_composite.index)

def macro_event_alignment(series: pd.Series, dates: pd.Series, event_date: str) -> pd.DataFrame:
    """
    Aligns time series dates relative to a specific event date.
    Returns a DataFrame with offset index (e.g. -10, 0, +10 days) and normalized values.
    """
    event_dt = pd.to_datetime(event_date)
    temp_df = pd.DataFrame({"date": pd.to_datetime(dates), "value": series.astype(float)})
    temp_df.sort_values("date", inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    
    # Find event index
    time_deltas = (temp_df["date"] - event_dt).abs()
    if time_deltas.empty:
        return pd.DataFrame()
    event_idx = time_deltas.idxmin()
    
    # Calculate offset
    temp_df["offset"] = temp_df.index - event_idx
    # Normalize values relative to event day value
    event_val = temp_df.loc[event_idx, "value"]
    if pd.isna(event_val) or event_val == 0:
        event_val = 1.0
        
    temp_df["aligned_val"] = (temp_df["value"] / event_val) * 100.0
    
    return temp_df[["offset", "aligned_val", "date"]]

def calculate_stress_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the systemic stress composite index.
    Combines VIX, JNK/TLT, Stablecoin Mkt Cap, and SOFR_Spread.
    Returns a DataFrame containing 'date', 'stress_composite', and 'stress_regime'.
    """
    df_out = pd.DataFrame({"date": df["date"]})
    
    # 1. VIX
    vix = df["VIX"]
    v_min, v_max = vix.min(), vix.max()
    vix_score = (vix - v_min) / (v_max - v_min) * 100.0 if v_max != v_min else pd.Series(0.0, index=df.index)
    
    # 2. Credit Stress (JNK/TLT ratio) - falling ratio means more stress
    credit_ratio = df["JNK"] / df["TLT"]
    cr_min, cr_max = credit_ratio.min(), credit_ratio.max()
    credit_score = (cr_max - credit_ratio) / (cr_max - cr_min) * 100.0 if cr_max != cr_min else pd.Series(0.0, index=df.index)
    
    # 3. Stablecoin Liquidity (contraction means more stress)
    stable = df["Stablecoin Mkt Cap"]
    s_min, s_max = stable.min(), stable.max()
    stable_score = (s_max - stable) / (s_max - s_min) * 100.0 if s_max != s_min else pd.Series(0.0, index=df.index)
    
    # 4. Funding Spread (SOFR - FFR) - higher spread means more stress
    sofr_spread = df["SOFR_Spread"]
    sf_min, sf_max = sofr_spread.min(), sofr_spread.max()
    funding_score = (sofr_spread - sf_min) / (sf_max - sf_min) * 100.0 if sf_max != sf_min else pd.Series(0.0, index=df.index)
    
    # Composite
    df_out["stress_composite"] = (vix_score + credit_score + stable_score + funding_score) / 4.0
    
    # Regime segmentation
    conditions = [
        df_out["stress_composite"] < 25.0,
        (df_out["stress_composite"] >= 25.0) & (df_out["stress_composite"] < 50.0),
        (df_out["stress_composite"] >= 50.0) & (df_out["stress_composite"] < 75.0),
        df_out["stress_composite"] >= 75.0
    ]
    labels = ["Low Stress", "Moderate Stress", "High Stress", "Crisis"]
    df_out["stress_regime"] = np.select(conditions, labels, default="Unknown")
    
    return df_out

