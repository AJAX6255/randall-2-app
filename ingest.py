#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ingest.py

Fetches macro-financial indicators from external APIs (FRED, Yahoo Finance, Llama.fi)
with conditional caching using db.py. Minimizes network requests and aligns datasets.
"""

import os
import datetime
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import db

# Load environment configuration
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

TODAY = datetime.date.today()
END_DATE = TODAY - datetime.timedelta(days=1)
START_DATE = END_DATE - datetime.timedelta(days=365)  # 1 year of history

# Definitions
FRED_SERIES = {
    "SOFR": "SOFR",
    "FFR": "DFF",
    "DGS10": "DGS10"
}

YF_TICKERS = {
    "SPY": "SPY",
    "VIX": "^VIX",
    "GLD": "GLD",
    "VTIP": "VTIP",
    "TLT": "TLT",
    "CRCL": "CRCL",
    "DRAM": "DRAM",
    "JNK": "JNK",
    "EMB": "EMB",
    "BTC": "BTC-USD"
}

# -----------------------------------------------------------------------------
# HEALTH CHECKS
# -----------------------------------------------------------------------------

def check_fred_api() -> tuple[bool, str]:
    if not FRED_API_KEY:
        return False, "FRED API key is missing from environment variables."
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "DFF",
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": "2024-01-01",
            "observation_end": "2024-01-02"
        }
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            return True, "FRED API: OK"
        return False, f"FRED API: HTTP error {res.status_code}"
    except Exception as e:
        return False, f"FRED API: Exception ({str(e)[:40]})"

def check_stablecoins_api() -> tuple[bool, str]:
    try:
        res = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=5)
        if res.status_code == 200:
            return True, "Stablecoins Llama: OK"
        return False, f"Stablecoins Llama: HTTP error {res.status_code}"
    except Exception as e:
        return False, f"Stablecoins Llama: Exception ({str(e)[:40]})"

def check_yfinance_api() -> tuple[bool, str]:
    try:
        spy = yf.download("SPY", period="1d", progress=False)
        if not spy.empty:
            return True, "Yahoo Finance: OK"
        return False, "Yahoo Finance: Empty dataset"
    except Exception as e:
        return False, f"Yahoo Finance: Exception ({str(e)[:40]})"

# -----------------------------------------------------------------------------
# FETCH LOGIC
# -----------------------------------------------------------------------------

def is_cache_fresh(series_id: str, max_age_hours: int = 12) -> bool:
    """Check if the cache has been updated recently."""
    last_updated = db.get_last_updated(series_id)
    if not last_updated:
        return False
    age = datetime.datetime.now() - last_updated
    return age.total_seconds() < (max_age_hours * 3600)

def fetch_fred_series(series_name: str, series_id: str) -> pd.DataFrame:
    """Fetch time series from Federal Reserve Economic Data (FRED)."""
    if not FRED_API_KEY:
        print(f"Skipping FRED {series_name}: No FRED_API_KEY in .env")
        return pd.DataFrame()
    
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": START_DATE.strftime("%Y-%m-%d"),
        "observation_end": END_DATE.strftime("%Y-%m-%d")
    }
    
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.rename(columns={"value": series_name}, inplace=True)
        return df[["date", series_name]]
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
        return pd.DataFrame()

def fetch_stablecoin_marketcap() -> pd.DataFrame:
    """Fetch total stablecoin market cap history from Llama.fi."""
    try:
        url = "https://stablecoins.llama.fi/stablecoincharts/all"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        records = []
        for entry in data:
            date_val = datetime.datetime.fromtimestamp(int(entry["date"]))
            total_circulating = entry.get("totalCirculatingUSD", {})
            total_usd = sum(total_circulating.values())
            records.append({
                "date": date_val,
                "Stablecoin Mkt Cap": total_usd
            })
        
        df = pd.DataFrame(records)
        df = df[
            (df["date"] >= pd.Timestamp(START_DATE)) &
            (df["date"] <= pd.Timestamp(END_DATE))
        ]
        df.set_index("date", inplace=True)
        df = df.resample("B").last()  # Business days resample
        df.ffill(inplace=True)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching stablecoin marketcap: {e}")
        return pd.DataFrame()

def fetch_yf_series(ticker: str, column_name: str) -> pd.DataFrame:
    """Fetch adjusted close price data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            return pd.DataFrame()
            
        df = df.reset_index()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.columns = [str(col).lower() for col in df.columns]
        
        date_col = 'date' if 'date' in df.columns else df.columns[0]
        
        # Check for typical close candidates
        close_candidates = ['close', 'adj close', 'adjclose']
        close_col = None
        for candidate in close_candidates:
            if candidate in df.columns:
                close_col = candidate
                break
        if close_col is None:
            close_col = df.columns[1]
            
        df = df[[date_col, close_col]].copy()
        df.columns = ["date", column_name]
        df["date"] = pd.to_datetime(df["date"])
        
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
            
        return df
    except Exception as e:
        print(f"Error fetching Yahoo Finance ticker {ticker}: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# ORCHESTRATED INGESTION & PIPELINE ALIGNMENT
# -----------------------------------------------------------------------------

def ingest_all(force: bool = False):
    """
    Verify/fetch all series and store them in the DuckDB/Parquet cache.
    """
    db.init_db()

    # 1. FRED Series
    for name, fred_id in FRED_SERIES.items():
        if force or not is_cache_fresh(name):
            print(f"Fetching FRED series: {name} ({fred_id})...")
            df = fetch_fred_series(name, fred_id)
            if not df.empty:
                db.save_series(name, df)

    # 2. Stablecoins
    if force or not is_cache_fresh("stablecoin_mkt_cap"):
        print("Fetching Stablecoins Market Cap from Llama.fi...")
        df = fetch_stablecoin_marketcap()
        if not df.empty:
            db.save_series("stablecoin_mkt_cap", df)

    # 3. Yahoo Finance Assets
    for name, ticker in YF_TICKERS.items():
        if force or not is_cache_fresh(name):
            print(f"Fetching Yahoo Finance ticker: {ticker}...")
            df = fetch_yf_series(ticker, name)
            if not df.empty:
                db.save_series(name, df)

def get_aligned_dataset() -> pd.DataFrame:
    """
    Load all cached assets using Polars lazy frames, merge them onto
    a master business-day index, and forward fill.
    
    Returns:
        pd.DataFrame: Aligned historical dataset.
    """
    # Force ingest if cache is empty
    all_cached = db.get_all_cached_series()
    if not all_cached:
        print("No cache detected. Running initial ingestion...")
        ingest_all(force=True)
        all_cached = db.get_all_cached_series()

    # Define business day skeleton
    date_range = pd.bdate_range(start=START_DATE, end=END_DATE)
    master_df = pd.DataFrame({"date": date_range})

    # Merge FRED
    for name in FRED_SERIES.keys():
        df = db.get_series(name)
        if not df.empty:
            master_df = master_df.merge(df, on="date", how="left")

    # Merge Stablecoins
    stable_df = db.get_series("stablecoin_mkt_cap")
    if not stable_df.empty:
        # Standardize name for merge
        stable_df.rename(columns={"stablecoin_mkt_cap": "Stablecoin Mkt Cap"}, inplace=True)
        master_df = master_df.merge(stable_df, on="date", how="left")

    # Merge Yahoo Finance
    for name in YF_TICKERS.keys():
        df = db.get_series(name)
        if not df.empty:
            master_df = master_df.merge(df, on="date", how="left")

    # Sort, align and forward fill
    master_df.sort_values("date", inplace=True)
    master_df.ffill(inplace=True)
    master_df.bfill(inplace=True)  # Backfill initial dates if gaps exist at startup

    # Inject calculated baseline indicators
    master_df["SOFR_Spread"] = master_df["SOFR"] - master_df["FFR"]

    return master_df
