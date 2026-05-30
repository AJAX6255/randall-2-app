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
import time
import random
from functools import wraps
import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests
from dotenv import load_dotenv
import db

# Load environment configuration
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

TODAY = datetime.date.today()
END_DATE = TODAY - datetime.timedelta(days=1)
START_DATE = END_DATE - datetime.timedelta(days=365)  # 1 year of history

def get_date_range(timeframe: str) -> tuple[datetime.date, datetime.date]:
    """Helper to convert a timeframe string into start and end dates."""
    today = datetime.date.today()
    end_date = today - datetime.timedelta(days=1)
    if timeframe == "2yr":
        days = 2 * 365
    elif timeframe == "1yr":
        days = 365
    elif timeframe == "6mo":
        days = 180
    elif timeframe == "1mo":
        days = 30
    else:
        days = 365
    start_date = end_date - datetime.timedelta(days=days)
    return start_date, end_date

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
# RETRY & BROWSER WORKAROUNDS
# -----------------------------------------------------------------------------

def retry_with_backoff(retries=5, delay=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "rate limit" in err_msg or "too many requests" in err_msg or "429" in err_msg or "forbidden" in err_msg or "status code 403" in err_msg:
                        wait = delay * (2 ** i) + random.uniform(0, 1)
                        print(f"Rate limited or forbidden. Retrying in {wait:.1f}s... (Attempt {i+1}/{retries})")
                        time.sleep(wait)
                    else:
                        wait = delay + random.uniform(0, 1)
                        print(f"Fetch failed: {e}. Retrying in {wait:.1f}s... (Attempt {i+1}/{retries})")
                        time.sleep(wait)
            print("Max retries exceeded for yfinance fetch.")
            return pd.DataFrame()
        return wrapper
    return decorator

@retry_with_backoff(retries=5, delay=3)
def _download_yf_with_retry(ticker: str, start=None, end=None, period=None) -> pd.DataFrame:
    session = curl_requests.Session(impersonate="chrome")
    if period:
        return yf.download(ticker, period=period, progress=False, session=session)
    else:
        return yf.download(ticker, start=start, end=end, progress=False, session=session)

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
        spy = _download_yf_with_retry("SPY", period="1d")
        if spy is not None and not spy.empty:
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

def fetch_fred_series(series_name: str, series_id: str, start_date: datetime.date = None, end_date: datetime.date = None) -> pd.DataFrame:
    """Fetch time series from Federal Reserve Economic Data (FRED)."""
    if not FRED_API_KEY:
        print(f"Skipping FRED {series_name}: No FRED_API_KEY in .env")
        return pd.DataFrame()
    
    s_date = start_date or START_DATE
    e_date = end_date or END_DATE
    
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": s_date.strftime("%Y-%m-%d"),
        "observation_end": e_date.strftime("%Y-%m-%d")
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

def fetch_stablecoin_marketcap(start_date: datetime.date = None, end_date: datetime.date = None) -> pd.DataFrame:
    """Fetch total stablecoin market cap history from Llama.fi."""
    s_date = start_date or START_DATE
    e_date = end_date or END_DATE
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
            (df["date"] >= pd.Timestamp(s_date)) &
            (df["date"] <= pd.Timestamp(e_date))
        ]
        df.set_index("date", inplace=True)
        df = df.resample("B").last()  # Business days resample
        df.ffill(inplace=True)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching stablecoin marketcap: {e}")
        return pd.DataFrame()

def fetch_yf_series(ticker: str, column_name: str, start_date: datetime.date = None, end_date: datetime.date = None) -> pd.DataFrame:
    """Fetch adjusted close price data from Yahoo Finance."""
    s_date = start_date or START_DATE
    e_date = end_date or END_DATE
    try:
        df = _download_yf_with_retry(ticker, start=s_date, end=e_date)
        if df is None or df.empty:
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

def ingest_all(force: bool = False, start_date: datetime.date = None, end_date: datetime.date = None):
    """
    Verify/fetch all series and store them in the DuckDB/Parquet cache.
    """
    db.init_db()

    s_date = start_date or START_DATE
    e_date = end_date or END_DATE

    # 1. FRED Series
    for name, fred_id in FRED_SERIES.items():
        if force or not is_cache_fresh(name):
            print(f"Fetching FRED series: {name} ({fred_id})...")
            df = fetch_fred_series(name, fred_id, start_date=s_date, end_date=e_date)
            if not df.empty:
                db.save_series(name, df)

    # 2. Stablecoins
    if force or not is_cache_fresh("stablecoin_mkt_cap"):
        print("Fetching Stablecoins Market Cap from Llama.fi...")
        df = fetch_stablecoin_marketcap(start_date=s_date, end_date=e_date)
        if not df.empty:
            db.save_series("stablecoin_mkt_cap", df)

    # 3. Yahoo Finance Assets
    for name, ticker in YF_TICKERS.items():
        if force or not is_cache_fresh(name):
            print(f"Fetching Yahoo Finance ticker: {ticker}...")
            df = fetch_yf_series(ticker, name, start_date=s_date, end_date=e_date)
            if not df.empty:
                db.save_series(name, df)

def get_aligned_dataset(start_date: datetime.date = None, end_date: datetime.date = None) -> pd.DataFrame:
    """
    Load all cached assets using Polars lazy frames, merge them onto
    a master business-day index, and forward fill.
    
    Returns:
        pd.DataFrame: Aligned historical dataset.
    """
    # Verify if cache is empty or doesn't cover the requested date range
    all_cached = db.get_all_cached_series()
    s_date = start_date or START_DATE
    e_date = end_date or END_DATE

    cache_needs_sync = False
    if not all_cached:
        cache_needs_sync = True
    else:
        try:
            spy_df = db.get_series("SPY")
            if not spy_df.empty:
                cached_min = spy_df["date"].min().date()
                cached_max = spy_df["date"].max().date()
                if cached_min > s_date or cached_max < e_date:
                    cache_needs_sync = True
            else:
                cache_needs_sync = True
        except Exception:
            cache_needs_sync = True

    if cache_needs_sync:
        print(f"Cache is empty or doesn't cover requested range ({s_date} to {e_date}). Running initial sync...")
        ingest_all(force=True, start_date=s_date, end_date=e_date)
        all_cached = db.get_all_cached_series()

    # Define business day skeleton
    date_range = pd.bdate_range(start=s_date, end=e_date)
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
