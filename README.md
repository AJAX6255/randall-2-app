# 🏛️ Semantic Macro-Financial Operating System

A conversational, AI-assisted macro-financial exploration environment built on a Bloomberg-style dark-mode visualization layer, deterministic quantitative primitives, and LLM-orchestrated intent parsing.

This system follows a strict architectural boundary: **The LLM orchestrates; the Python engine calculates.** The LLM (Gemini 1.5 Flash) interprets natural language prompts and generates structured JSON execution plans, while high-performance Python libraries (Polars, DuckDB, Pandas, NumPy) execute the mathematical operations.

---

## 🏗️ Architecture & Philosophy

```text
       User Interface (Streamlit)
                   ↓
   Conversational Intent Parser (Gemini 1.5 Flash)
                   ↓
      Execution Plan Generator (JSON Plan)
                   ↓
   Deterministic Analytical Engine (Pandas, Polars, NumPy)
                   ↓
     Bloomberg-Style Visuals (Altair / Vega-Lite)
```

### Key Separation of Concerns
- **Orchestration Layer**: interprets user inputs, maps queries to assets, selects rolling windows, and plans transformations.
- **Computation Layer**: Vectorized quantitative operations implemented in pure Python. **No LLM-side calculations.**
- **Storage Layer**: Hybrid DuckDB and Parquet storage system using Polars lazy-loading pipelines to align historical time-series datasets.
- **Visualization Layer**: Interactive, layered Altair charts supporting Bloomberg-style dark-mode layouts, hover cross-hairs, interactive tooltips, and background regime shading.

---

## 📂 Repository Structure

The codebase is organized as follows:

*   [app.py](file:///c:/Users/Allan/Documents/Dev/randall-2-app/app.py): The main Streamlit dashboard workspace. Implements the UI layout, template shortcuts, metric displays, conversational chat panels, and handles user state.
*   [orchestrator.py](file:///c:/Users/Allan/Documents/Dev/randall-2-app/orchestrator.py): The core orchestration engine. Parses user prompts (via Gemini Flash or local regex fallbacks) into JSON plans, and executes these plans deterministically on the dataset.
*   [analytics.py](file:///c:/Users/Allan/Documents/Dev/randall-2-app/analytics.py): Mathematical primitives and vectorized routines for all calculations (normalization, z-scores, rolling PCA, correlation, relative strength, drawdowns, and regime detection).
*   [db.py](file:///c:/Users/Allan/Documents/Dev/randall-2-app/db.py): Interacts with the local caching system (DuckDB database + Parquet files) to load and save series using Polars.
*   [ingest.py](file:///c:/Users/Allan/Documents/Dev/randall-2-app/ingest.py): Multi-source data pipeline fetching from FRED (Federal Reserve Economic Data), Yahoo Finance, and Llama.fi (Stablecoin market caps).
*   [visuals.py](file:///c:/Users/Allan/Documents/Dev/randall-2-app/visuals.py): Altair visualization configurations, including the custom Bloomberg obsidian theme, layered charting configurations, and custom stress indicator visualizations.
*   [prompts.py](file:///c:/Users/Allan/Documents/Dev/randall-2-app/prompts.py): System prompt configurations defining constraints for structural JSON intent translation.
*   [requirements.txt](file:///c:/Users/Allan/Documents/Dev/randall-2-app/requirements.txt): Python dependencies for the project.

---

## 📈 Quantitative Primitives

All quantitative transforms are defined in [analytics.py](file:///c:/Users/Allan/Documents/Dev/randall-2-app/analytics.py):
*   **Normalization & Scaling**: Normalizes series to `100.0` at specific event/anchor dates (`normalize_series`), rolling Z-scores (`zscore_series`), rolling percentile ranks (`percentile_scaling`), log scaling (`log_scaling`), and outlier compression (`volatility_compression`).
*   **Risk & Volatility**: Rolling annualized volatility (`rolling_volatility`), rolling market beta (`rolling_beta`), and max drawdown calculations (`compute_drawdown`).
*   **Relationships & Factor Models**: Rolling Pearson correlations (`rolling_correlation`), divergence detection based on regression slope comparison (`detect_divergence`), lead-lag cross-correlation analysis (`lead_lag_analysis`), and systemic risk factor extraction via SVD (`rolling_pca`).
*   **Regime Detection**: Multi-variate conditions detecting Stagflation Supply Shocks, Demand Expansion Shocks, Flight to Safety, and Expansionary regimes (`detect_regimes`).

---

## ⚙️ Ingestion & Cached Data Sources

The ingestion pipeline ([ingest.py](file:///c:/Users/Allan/Documents/Dev/randall-2-app/ingest.py)) fetches economic data and financial assets:
1.  **FRED economic indicators**: SOFR (Securities Financing Overnight Rate), FFR (Federal Funds Rate), and DGS10 (10-Year Treasury Constant Maturity Yield).
2.  **Stablecoin Market Cap**: Unified historical stablecoin supply history from Llama.fi.
3.  **Yahoo Finance assets**: Equities (`SPY`), Volatility (`^VIX`), Gold (`GLD`), Inflation-Protected Treasuries (`VTIP`), Long Bonds (`TLT`), Emerging Debt (`EMB`), Junk Bonds (`JNK`), Semiconductors (`DRAM`), and Cryptocurrencies (`BTC-USD`).

All retrieved data is serialized as `.parquet` files and mirrored to `data_cache/macro_data.duckdb` for rapid SQL query access.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (Recommended Python 3.13)
- API Keys:
    - **FRED API Key**: Get one from [FRED API](https://fred.stlouisfed.org/api_key.html).
    - **Gemini API Key** *(Optional)*: Get one from Google AI Studio to enable conversational features. If omitted, the orchestrator defaults to a local regex parser.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/randall-2-app.git
    cd randall-2-app
    ```

2.  **Set up the environment**:
    Create a virtual environment and activate it:
    ```bash
    # Windows PowerShell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    
    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Create a `.env` file in the root directory (based on `.env` settings):
    ```env
    FRED_API_KEY=your_fred_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

5.  **Synchronize and Warm the Cache**:
    Run data ingestion to warm the database (downloads ~1 year of history for all assets):
    ```bash
    python -c "import ingest; ingest.ingest_all(force=True)"
    ```

6.  **Launch the OS Workspace**:
    Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

---

## 💻 Usage & Conversational Queries

You can interact with the system via the conversational chat panel in the UI. Here are some query examples:

*   **Correlations**: `"Compare BTC and GLD rolling 60-day correlation"`
*   **Z-Scores**: `"Show VIX zscore with window 30"`
*   **Relative Strength / Compare**: `"Show ratio of SPY vs TLT and apply smoothing 15"`
*   **Volatility**: `"Show rolling volatility for DRAM and JNK"`
*   **Z-Scores with Regimes**: `"Show BTC zscore and overlay regime shading"`

If your Gemini API key is not configured, the system fallback parser will recognize keyword triggers (like `btc`, `gld`, `correlation`, `zscore`, etc.) and construct the plan locally.

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](file:///c:/Users/Allan/Documents/Dev/randall-2-app/LICENSE) file for details.
