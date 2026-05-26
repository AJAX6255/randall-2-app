# Contributing to Semantic Macro-Financial OS

Thank you for contributing! To maintain the quality, security, and predictability of this institutional-grade macro workspace, please adhere to the following design patterns and requirements.

---

## 🏛️ Core Design Principles

1.  **Strict Separation of Concerns**:
    *   **LLM (Gemini Flash)**: Handles intent parsing, tool selection, visualization structure, and state planning.
    *   **Python Engine (Pandas, Polars, DuckDB)**: Handles all mathematics, vector calculations, and data alignment.
    *   **Vega-Lite/Altair**: Handles chart rendering, interactive brushing, and layering.
    *   *Rule*: **Never let the LLM generate dynamic python code or execute statistical math on the client side.**

2.  **Determinism & Reproducibility**:
    *   Any query output must be mathematically reproducible. Avoid stochastic calculations.
    *   Avoid using random seeds, hard-coded magic values, or client-side statistical estimates.

---

## 🔧 Development Guidelines

### How to Add a New Quantitative Primitive

If you want to introduce a new transformation (e.g., *Moving Average Convergence Divergence (MACD)*):

1.  **Define the Primitive**:
    In [analytics.py](./analytics.py), implement your primitive as a vectorized, type-hinted function operating on pandas/polars data structures:
    ```python
    def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
        """Calculates MACD and Signal series."""
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
    ```

2.  **Expose in the Orchestrator**:
    In [orchestrator.py](./orchestrator.py), update the parser fallback rules (`run_fallback_parser`) and execution map (`execute_plan`) to handle the new transform:
    ```python
    elif transform == "macd":
        for col in y_cols:
            col_name = f"{col}_macd"
            df_out[col_name], df_out[f"{col}_macd_sig"] = analytics.compute_macd(df_out[col])
            plot_cols.extend([col_name, f"{col}_macd_sig"])
        y_label = "MACD / Signal"
    ```

3.  **Update LLM System Prompt**:
    In [prompts.py](./prompts.py), add the new transform description and schemas to `INTENT_SYSTEM_PROMPT` so the Gemini model understands how and when to invoke it.

---

### How to Add a New Asset Data Source

1.  **Update Definitions**:
    In [ingest.py](./ingest.py), append your asset definition to the mapping constants:
    *   If economic (FRED): Add to `FRED_SERIES`.
    *   If financial (Yahoo Finance): Add to `YF_TICKERS`.
2.  **Add to Ingestion Pipeline**:
    *   Verify or update `ingest_all` to check cache freshness and download/sync the asset.
    *   Update `get_aligned_dataset()` to merge the new asset onto the business day timeline.
3.  **Sync Local DB Cache**:
    *   Run `python -c "import ingest; ingest.ingest_all(force=True)"` to synchronize local storage.

---

## 🧼 Code Quality & Style

*   **Code Linting**: Always verify changes using `pyflakes`:
    ```bash
    pyflakes app.py analytics.py visuals.py db.py ingest.py orchestrator.py
    ```
*   **Format**: Maintain clear PEP-8 compliant style.
*   **Type Hints**: Use python type annotations for all new function declarations.
*   **Errors**: Do not silently suppress exceptions. Wrap calculations in try-except statements when appropriate, falling back safely to standard values (`nan`, `0.0`, etc.) to prevent streamlit crash cycles.
