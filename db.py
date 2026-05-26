#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
db.py

Handles local caching, metadata, and structured time-series storage
using DuckDB and Parquet files. Enables lazy loading via Polars.
"""

import os
from datetime import datetime
import pandas as pd
import polars as pl
import duckdb

CACHE_DIR = "data_cache"
DB_PATH = os.path.join(CACHE_DIR, "macro_data.duckdb")

def init_db():
    """Verify data_cache directory and database file exist and are initialized."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    conn = duckdb.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                series_id VARCHAR PRIMARY KEY,
                last_updated TIMESTAMP
            )
        """)
    finally:
        conn.close()

def clean_series_id(series_id: str) -> str:
    """Sanitize the series ID for SQL table naming."""
    return series_id.lower().replace("-", "_").replace("^", "").replace(".", "_")

def save_series(series_id: str, df: pd.DataFrame):
    """
    Save a time-series DataFrame to Parquet and update DuckDB state.
    
    Args:
        series_id (str): Unique identifier for the series
        df (pd.DataFrame): Time-series data containing a 'date' column and a value column
    """
    init_db()
    if df.empty:
        return

    # Clean and standardize DataFrame
    df = df.copy()
    if "date" not in df.columns:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
    
    # Identify value column
    val_cols = [col for col in df.columns if col != "date"]
    if not val_cols:
        raise ValueError(f"No value column found in DataFrame for series: {series_id}")
    
    value_col = val_cols[0]
    # Standardize column names internally
    df.rename(columns={value_col: "value"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    # Write to Parquet cache using Polars
    parquet_path = os.path.join(CACHE_DIR, f"{series_id}.parquet")
    pl.from_pandas(df).write_parquet(parquet_path)

    # Sync to DuckDB
    table_name = f"series_{clean_series_id(series_id)}"
    conn = duckdb.connect(DB_PATH)
    try:
        # Load from parquet into DuckDB table
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_path}')")
        
        # Update metadata
        conn.execute("INSERT OR REPLACE INTO cache_metadata VALUES (?, ?)", (series_id, datetime.now()))
    finally:
        conn.close()

def get_series(series_id: str) -> pd.DataFrame:
    """
    Load a time-series from Parquet cache using Polars.
    
    Args:
        series_id (str): Unique identifier for the series
        
    Returns:
        pd.DataFrame: DataFrame containing 'date' and [series_id] columns
    """
    init_db()
    parquet_path = os.path.join(CACHE_DIR, f"{series_id}.parquet")
    if not os.path.exists(parquet_path):
        return pd.DataFrame(columns=["date", series_id])

    # Scan and collect using Polars lazy pipelines
    lf = pl.scan_parquet(parquet_path)
    df = lf.collect().to_pandas()
    df.rename(columns={"value": series_id}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_last_updated(series_id: str) -> datetime or None:
    """
    Get the timestamp of the last database sync for a specific series.
    """
    init_db()
    conn = duckdb.connect(DB_PATH)
    try:
        row = conn.execute("SELECT last_updated FROM cache_metadata WHERE series_id = ?", (series_id,)).fetchone()
        if row:
            return row[0]
        return None
    finally:
        conn.close()

def get_all_cached_series() -> list:
    """Get list of all series IDs stored in the cache metadata."""
    init_db()
    conn = duckdb.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT series_id FROM cache_metadata").fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()
