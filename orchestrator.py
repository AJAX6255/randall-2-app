#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
orchestrator.py

Orchestration engine for the macro analytical dashboard.
Interprets semantic user prompts (via Gemini Flash or local regex fallbacks)
and executes deterministic analytical operations on the dataset.
"""

import os
import json
import re
import google.generativeai as genai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import prompts
import analytics

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------------------------------------------------------
# SEMANTIC INTENT PARSER
# -----------------------------------------------------------------------------

def parse_intent(user_prompt: str) -> dict:
    """
    Translates user query into a structured execution plan.
    Uses Gemini Flash if configured, else falls back to a deterministic regex parser.
    """
    if not GEMINI_API_KEY:
        return run_fallback_parser(user_prompt)
        
    try:
        # Configure model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Build prompt
        full_prompt = f"{prompts.INTENT_SYSTEM_PROMPT}\n\nUSER PROMPT: '{user_prompt}'\nJSON EXECUTION PLAN:"
        
        # Invoke API
        response = model.generate_content(full_prompt)
        plan = json.loads(response.text.strip())
        
        # Verify default window
        if "window" not in plan or plan["window"] is None:
            plan["window"] = 60
            
        return plan
        
    except Exception as e:
        print(f"Gemini API Error: {e}. Running fallback parser.")
        return run_fallback_parser(user_prompt)

def run_fallback_parser(user_prompt: str) -> dict:
    """
    Simple keyword parser in case Gemini API is offline or key is missing.
    """
    p_lower = user_prompt.lower()
    
    # Defaults
    plan = {
        "y_cols": ["SPY"],
        "transform": "none",
        "secondary_asset": None,
        "window": 60,
        "normalize": False,
        "anchor_date": None,
        "smooth_window": None,
        "show_regimes": False,
        "analysis_text": "Parsed using local rules. (Set GEMINI_API_KEY in .env for advanced semantic querying)."
    }

    # Column extraction
    asset_map = {
        "btc": "BTC", "bitcoin": "BTC", "crypto": "BTC",
        "spy": "SPY", "stocks": "SPY", "equities": "SPY",
        "gld": "GLD", "gold": "GLD",
        "vix": "VIX", "volatility": "VIX",
        "tlt": "TLT", "bonds": "TLT",
        "vtip": "VTIP", "inflation": "VTIP",
        "dram": "DRAM", "semis": "DRAM",
        "jnk": "JNK", "junk": "JNK",
        "emb": "EMB", "emerging": "EMB",
        "sofr": "SOFR",
        "ffr": "FFR", "fed funds": "FFR",
        "dgs10": "DGS10", "yield": "DGS10",
        "sofr_spread": "SOFR_Spread", "spread": "SOFR_Spread",
        "stablecoin": "Stablecoin Mkt Cap", "llama": "Stablecoin Mkt Cap"
    }

    found_cols = []
    for word, col in asset_map.items():
        if word in p_lower:
            if col not in found_cols:
                found_cols.append(col)
                
    if found_cols:
        plan["y_cols"] = found_cols[:4]  # Limit to 4 primary assets

    # Transform detection
    if "correlation" in p_lower or "corr" in p_lower:
        plan["transform"] = "correlation"
        if len(found_cols) >= 2:
            plan["y_cols"] = [found_cols[0]]
            plan["secondary_asset"] = found_cols[1]
    elif "beta" in p_lower:
        plan["transform"] = "beta"
        if len(found_cols) >= 2:
            plan["y_cols"] = [found_cols[0]]
            plan["secondary_asset"] = found_cols[1]
    elif "zscore" in p_lower or "z-score" in p_lower:
        plan["transform"] = "zscore"
    elif "volatility" in p_lower or "vol" in p_lower:
        plan["transform"] = "volatility"
    elif "drawdown" in p_lower:
        plan["transform"] = "drawdown"
    elif "relative strength" in p_lower or "ratio" in p_lower or "compare" in p_lower:
        if len(found_cols) >= 2:
            plan["transform"] = "relative_strength"
            plan["y_cols"] = [found_cols[0]]
            plan["secondary_asset"] = found_cols[1]
    elif "pca" in p_lower or "factor" in p_lower:
        plan["transform"] = "pca"

    # Context modifiers
    if "normalize" in p_lower or "norm" in p_lower:
        plan["normalize"] = True

    if "regime" in p_lower or "shading" in p_lower:
        plan["show_regimes"] = True

    # Anchor date extraction
    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", user_prompt)
    if date_match:
        plan["anchor_date"] = date_match.group(1)
        plan["normalize"] = True

    # Smoothing detection
    smooth_match = re.search(r"smooth(?:ing)?\s*(\d+)", p_lower)
    if smooth_match:
        plan["smooth_window"] = int(smooth_match.group(1))

    return plan

# -----------------------------------------------------------------------------
# DETERMINISTIC PLAN EXECUTION
# -----------------------------------------------------------------------------

def execute_plan(plan: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """
    Executes the analytical plan on the master dataframe.
    Returns: (output_dataframe, columns_to_plot, y_axis_label)
    """
    df_out = df.copy()
    y_cols = plan.get("y_cols", ["SPY"])
    transform = plan.get("transform", "none")
    window = plan.get("window", 60) or 60
    normalize = plan.get("normalize", False)
    anchor_date = plan.get("anchor_date", None)
    
    # Standardize selected cols
    y_cols = [c for c in y_cols if c in df_out.columns]
    if not y_cols:
        y_cols = ["SPY"]

    plot_cols = []
    y_label = "Value"

    # 1. Apply Primitives based on Transform
    if transform == "none":
        for col in y_cols:
            if normalize:
                col_name = f"{col}_normalized"
                df_out[col_name] = analytics.normalize_series(df_out[col], df_out["date"], anchor_date or df_out["date"].iloc[0])
                plot_cols.append(col_name)
                y_label = "Index (Base = 100)"
            else:
                plot_cols.append(col)
                
    elif transform == "zscore":
        for col in y_cols:
            col_name = f"{col}_zscore"
            df_out[col_name] = analytics.zscore_series(df_out[col], window)
            plot_cols.append(col_name)
        y_label = "Z-Score (Standard Deviations)"

    elif transform == "volatility":
        for col in y_cols:
            # Volatility is calculated on returns
            returns = df_out[col].pct_change().fillna(0.0)
            col_name = f"{col}_vol"
            df_out[col_name] = analytics.rolling_volatility(returns, window) * 100.0  # Show as %
            plot_cols.append(col_name)
        y_label = "Annualized Volatility (%)"

    elif transform == "drawdown":
        for col in y_cols:
            col_name = f"{col}_drawdown"
            df_out[col_name] = analytics.compute_drawdown(df_out[col])
            plot_cols.append(col_name)
        y_label = "Drawdown (%)"

    elif transform in ["correlation", "beta", "relative_strength"]:
        sec_asset = plan.get("secondary_asset")
        if not sec_asset or sec_asset not in df_out.columns:
            # Fallback if secondary asset missing
            sec_asset = "SPY" if y_cols[0] != "SPY" else "TLT"
            
        prim_asset = y_cols[0]
        
        if transform == "correlation":
            col_name = f"{prim_asset}_{sec_asset}_corr"
            df_out[col_name] = analytics.rolling_correlation(df_out[prim_asset], df_out[sec_asset], window)
            plot_cols.append(col_name)
            y_label = f"{window}-Day Rolling Correlation"
            
        elif transform == "beta":
            p_ret = df_out[prim_asset].pct_change().fillna(0.0)
            s_ret = df_out[sec_asset].pct_change().fillna(0.0)
            col_name = f"{prim_asset}_{sec_asset}_beta"
            df_out[col_name] = analytics.rolling_beta(p_ret, s_ret, window)
            plot_cols.append(col_name)
            y_label = f"Beta (vs {sec_asset})"
            
        elif transform == "relative_strength":
            col_name = f"{prim_asset}_{sec_asset}_ratio"
            df_out[col_name] = analytics.compute_relative_strength(df_out[prim_asset], df_out[sec_asset])
            plot_cols.append(col_name)
            y_label = f"Ratio ({prim_asset} / {sec_asset})"

    elif transform == "pca":
        # Run PCA on selected columns
        col_name = "systemic_factor"
        df_out[col_name] = analytics.rolling_pca(df_out[y_cols], window)
        plot_cols.append(col_name)
        y_label = "Systemic Risk Factor (1st PC)"

    # Ensure dates and output columns contain no infs/nans
    df_out[plot_cols] = df_out[plot_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    return df_out, plot_cols, y_label
