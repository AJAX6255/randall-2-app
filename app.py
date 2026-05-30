#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app.py

Semantic Macro-Financial Analytical Operating System.
Main UI workspace coordinating system health, conversational queries,
quantitative primitives, and interactive data visualization.
"""

import streamlit as st
import pandas as pd
import datetime
import os
from dotenv import load_dotenv

# Import custom modules
import db
import ingest
import analytics
import orchestrator
import visuals

# -----------------------------------------------------------------------------
# PAGE CONFIG & PREMIUM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Semantic Macro OS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling override for a premium responsive Bloomberg-style appearance
st.markdown("""
<style>
    /* Custom headers */
    h1, h2, h3 {
        font-family: 'Inter', system-ui, sans-serif !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
    }
    
    /* Metric boxes */
    div[data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-size: 2rem;
        color: var(--primary-color) !important;
    }
    
    /* Chat inputs */
    .stChatInput {
        border-color: var(--border-color) !important;
    }
    
    /* Custom card container styling */
    .macro-card {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    
    /* High contrast radio selectors (Timeframe Selector) */
    div[data-testid="stRadio"] {
        background-color: var(--secondary-background-color) !important;
        padding: 6px 14px !important;
        border-radius: 6px !important;
        border: 1px solid var(--border-color) !important;
    }
    div[data-testid="stRadio"] label {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] label p {
        color: var(--text-color) !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stRadio"] label:hover p {
        color: var(--primary-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INITIALIZE STATE
# -----------------------------------------------------------------------------
if "active_plan" not in st.session_state:
    st.session_state.active_plan = {
        "y_cols": ["SPY", "VIX"],
        "transform": "none",
        "secondary_asset": None,
        "window": 60,
        "normalize": False,
        "anchor_date": None,
        "smooth_window": None,
        "show_regimes": True,
        "analysis_text": "System ready. Initializing with baseline equity and volatility assets."
    }

if "active_timeframe" not in st.session_state:
    st.session_state.active_timeframe = "1yr"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to the Semantic Macro OS. Ask me to compare assets, z-score signals, calculate rolling correlations, or analyze macro stress regimes."}
    ]

# Callback for chat input to avoid double-rendering and state race conditions
def handle_chat_submit():
    user_query = st.session_state.user_chat_input
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        new_plan = orchestrator.parse_intent(user_query)
        st.session_state.active_plan = new_plan
        analysis = new_plan.get("analysis_text", "Execution plan prepared.")
        st.session_state.chat_history.append({"role": "assistant", "content": analysis})

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL PANEL & HEALTH
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Macro Terminal")
    st.markdown("---")
    
    # 1. API Status Indicators
    st.subheader("🔧 System Connections")
    
    # Helper to render custom glowing LED status indicators
    def render_led_indicator(color: str, text: str):
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; padding: 2px 0;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {color}; box-shadow: 0 0 8px {color};"></div>
                <span style="font-family: 'Courier New', monospace; font-size: 0.85rem; font-weight: bold; color: var(--text-color);">{text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.spinner("Verifying connections..."):
        fred_ok, fred_msg = ingest.check_fred_api()
        stable_ok, stable_msg = ingest.check_stablecoins_api()
        yahoo_ok, yahoo_msg = ingest.check_yfinance_api()
        
        # Render connections
        render_led_indicator("#00e676" if fred_ok else "#ef4444", fred_msg)
        render_led_indicator("#00e676" if stable_ok else "#ef4444", stable_msg)
        render_led_indicator("#00e676" if yahoo_ok else "#ef4444", yahoo_msg)
        
        # Gemini LLM status detection
        gemini_status = st.session_state.active_plan.get("parse_method", "local")
        if not orchestrator.GEMINI_API_KEY:
            gemini_color = "#ef4444"
            gemini_text = "Gemini LLM: Not Configured"
        elif gemini_status == "gemini" or (gemini_status == "local" and orchestrator.GEMINI_API_KEY):
            gemini_color = "#00e676"
            gemini_text = "Gemini LLM: Active"
        elif gemini_status == "fallback":
            gemini_color = "#eab308"
            gemini_text = "Gemini LLM: Error (Fallback)"
        else:
            gemini_color = "#ef4444"
            gemini_text = "Gemini LLM: Not Configured"
            
        render_led_indicator(gemini_color, gemini_text)
            
    st.markdown("---")

    # 2. Database Controls
    st.subheader("💾 Caching Controls")
    if st.button("Force Database Sync", use_container_width=True):
        with st.spinner("Downloading fresh financial time-series..."):
            start_date, end_date = ingest.get_date_range(st.session_state.active_timeframe)
            ingest.ingest_all(force=True, start_date=start_date, end_date=end_date)
            st.success("Database fully synchronized!")
            st.rerun()

    st.markdown("---")


    st.subheader("👁️ Regime Overlays")
    show_regimes_check = st.checkbox(
        "Shade Macro Regimes on Chart", 
        value=st.session_state.active_plan.get("show_regimes", True)
    )
    
    selected_regime_types = []
    if show_regimes_check:
        selected_regime_types = st.multiselect(
            "Regimes to Highlight:",
            options=["Oil Supply Shock", "Oil Demand Shock", "Flight to Safety", "Expansionary", "Neutral"],
            default=["Oil Supply Shock", "Oil Demand Shock", "Flight to Safety", "Expansionary"]
        )

# -----------------------------------------------------------------------------
# MAIN APP BODY
# -----------------------------------------------------------------------------
st.title("🏛️ Semantic Macro-Financial Operating System")
st.markdown("LLM-orchestrated quantitative time-series exploration environment.")

# Load baseline dataset
try:
    start_date, end_date = ingest.get_date_range(st.session_state.active_timeframe)
    master_df = ingest.get_aligned_dataset(start_date=start_date, end_date=end_date)
except Exception as e:
    st.error(f"Failed to fetch initial dataset: {e}")
    st.stop()

# 1. Quick Stats Metric Header
with st.container():
    # Calculate simple stress components to show in metrics
    vix_val = master_df["VIX"].iloc[-1]
    sofr_val = master_df["SOFR"].iloc[-1]
    btc_val = master_df["BTC"].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("VIX Index (Volatility)", f"{vix_val:.2f}", f"{(vix_val - master_df['VIX'].iloc[-5]):.2f} (5d)")
    with col2:
        st.metric("SOFR Rate (Funding)", f"{sofr_val:.4f}%", f"{(sofr_val - master_df['SOFR'].iloc[-5]):.4f}% (5d)")
    with col3:
        st.metric("Bitcoin (BTC-USD)", f"${btc_val:,.0f}", f"{(btc_val / master_df['BTC'].iloc[-5] - 1.0)*100:.2f}% (5d)")
    with col4:
        # Calculate stress composite index
        stress_df = analytics.calculate_stress_composite(master_df)
        latest_stress = stress_df["stress_composite"].iloc[-1]
        st.metric("Systemic Stress Indicator", f"{latest_stress:.1f} / 100")

# 2. Main Exploration Panel (Layered Visualizer)
st.markdown('<div class="macro-card">', unsafe_allow_html=True)
plan = st.session_state.active_plan

# Timeframe Selector & Title Row
col_title, col_tf = st.columns([5, 3])
with col_title:
    st.subheader(f"Exploring: {', '.join(plan.get('y_cols', []))}")
with col_tf:
    selected_timeframe = st.radio(
        "Timeframe Selector:",
        options=["2yr", "1yr", "6mo", "1mo"],
        index=["2yr", "1yr", "6mo", "1mo"].index(st.session_state.active_timeframe),
        horizontal=True
    )

# Poll and trigger resync on timeframe change
if selected_timeframe != st.session_state.active_timeframe:
    st.session_state.active_timeframe = selected_timeframe
    with st.spinner(f"Resyncing historical data for {selected_timeframe} timeframe..."):
        start_date, end_date = ingest.get_date_range(selected_timeframe)
        ingest.ingest_all(force=True, start_date=start_date, end_date=end_date)
    st.success(f"Synchronized cache for {selected_timeframe} timeframe!")
    st.rerun()

# Execute transformations on the master dataset
try:
    plot_df, plot_cols, y_label = orchestrator.execute_plan(plan, master_df)
    
    # Calculate regimes if requested
    regimes_df = None
    if show_regimes_check:
        raw_regimes = analytics.detect_regimes(master_df)
        filtered_regimes = raw_regimes.apply(lambda r: r if r in selected_regime_types else "Neutral")
        regimes_df = pd.DataFrame({
            "date": master_df["date"],
            "regime": filtered_regimes
        })

    # Render dynamic layered chart
    chart = visuals.build_macro_chart(
        df=plot_df,
        y_cols=plot_cols,
        title=f"Exploring: {', '.join(plan.get('y_cols', []))}",
        y_axis_title=y_label,
        smooth_window=plan.get("smooth_window"),
        show_regimes=show_regimes_check,
        regime_df=regimes_df,
        normalize=plan.get("normalize", False),
        reference_y=plan.get("reference_y"),
        show_bollinger=plan.get("show_bollinger", False),
        show_crossovers=plan.get("show_crossovers", False),
        top_k_stress=plan.get("top_k_stress"),
        horizontal_lines=plan.get("horizontal_lines", []),
        vertical_lines=plan.get("vertical_lines", []),
        bands=plan.get("bands", [])
    )
    
    st.altair_chart(chart, use_container_width=True)
    
except Exception as e:
    st.error(f"Analytical pipeline execution error: {e}")
    st.exception(e)

st.markdown('</div>', unsafe_allow_html=True)

# 3. Conversational Exploration Sidebar/Chat Interface
st.markdown("### 💬 Conversational Macro Interpreter")
chat_col, spec_col = st.columns([2, 1])

with chat_col:
    # Display chat messages
    for msg in st.session_state.chat_history[-4:]:  # Show recent conversations
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"**{role_icon} {msg['role'].capitalize()}:** {msg['content']}")

    # Prompt input
    st.chat_input(
        "Enter analysis intent: (e.g., 'Compare BTC and GLD rolling 60-day correlation')",
        key="user_chat_input",
        on_submit=handle_chat_submit
    )

with spec_col:
    # Display the current JSON execution spec generated by the LLM
    st.markdown("**Active Execution Plan (Structured JSON)**")
    st.json(st.session_state.active_plan)

# 4. Underlying Stress Indicator & Regimes table
st.markdown("---")
with st.expander("📊 Systemic Stress & Macro Regime Reference Matrix", expanded=False):
    st.markdown("""
    ### Systemic Stress Model Logic
    The stress composite combines market fear metrics (VIX), credit stress ratios (JNK/TLT), stablecoin liquidity metrics, and funding spreads (SOFR/FFR).
    
    ### Dynamic Regime Classifications
    Our deterministic model classifies macro conditions based on multi-variate trend alignments:
    - **Oil Supply Shock**: Equities fall, Volatility rises, Yields rise (Stagflationary worries).
    - **Oil Demand Shock**: Equities rise, Volatility falls, Yields rise (Growth expansion).
    - **Flight to Safety**: Yields drop, Gold/VIX rise, Equities contract (Risk-off panic).
    - **Expansionary**: Equities rise, Volatility declines (Standard bull market).
    """)
    
    # Calculate stress composite
    stress_df = analytics.calculate_stress_composite(master_df)
    
    # Plot stress composite chart
    stress_chart = visuals.build_stress_bands_chart(stress_df)
    st.altair_chart(stress_chart, use_container_width=True)

    st.markdown("### 📊 Systemic Stress & Macro Regime Reference Matrix")
    
    # Calculate stress history table for the user
    all_regimes = analytics.detect_regimes(master_df)
    stress_hist = pd.DataFrame({
        "Date": master_df["date"],
        "VIX Value": master_df["VIX"],
        "SOFR Rate (%)": master_df["SOFR"],
        "Stablecoin Cap ($)": master_df["Stablecoin Mkt Cap"],
        "Stress Score": stress_df["stress_composite"].round(1),
        "Stress Regime": stress_df["stress_regime"],
        "Regime": all_regimes
    })
    
    # Filter the history table based on the selected regimes
    if show_regimes_check and selected_regime_types:
        stress_hist = stress_hist[stress_hist["Regime"].isin(selected_regime_types)]
        
    # We display up to 100 rows to ensure that a scrollbar is visible
    stress_hist_subset = stress_hist.tail(100)
    
    # Custom color-coding styler for the Streamlit dataframe
    def style_stress_matrix(df: pd.DataFrame) -> pd.DataFrame:
        def color_stress_regime(val):
            if val == "Crisis":
                return 'background-color: #ef4444; color: white; font-weight: bold;'
            elif val == "High Stress":
                return 'background-color: #f97316; color: white;'
            elif val == "Moderate Stress":
                return 'background-color: #eab308; color: black;'
            elif val == "Low Stress":
                return 'background-color: #22c55e; color: white;'
            return ''

        def color_macro_regime(val):
            if val == "Oil Supply Shock":
                return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
            elif val == "Oil Demand Shock":
                return 'background-color: #14532d; color: #86efac; font-weight: bold;'
            elif val == "Flight to Safety":
                return 'background-color: #7c2d12; color: #fdba74; font-weight: bold;'
            elif val == "Expansionary":
                return 'background-color: #1e3a8a; color: #93c5fd; font-weight: bold;'
            return ''

        return df.style.map(
            color_stress_regime, subset=["Stress Regime"]
        ).map(
            color_macro_regime, subset=["Regime"]
        ).background_gradient(
            cmap="PuBu", subset=["Stress Score"]
        )

    st.dataframe(style_stress_matrix(stress_hist_subset), use_container_width=True)

# -----------------------------------------------------------------------------
# 5. REFERENCE ASSET SUITE
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("📊 Reference Asset Suite")
st.markdown("Tracked macro-financial benchmarks, liquidity flows, and sector ETF price indicators.")

import altair as alt

# Helper to plot standard reference line charts
def plot_ref_chart(df, y_col, title_label, color_hex, y_axis_label="Price"):
    chart = alt.Chart(df).mark_line(color=color_hex, strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y(f"{y_col}:Q", scale=alt.Scale(zero=False), title=y_axis_label),
        tooltip=["date:T", alt.Tooltip(f"{y_col}:Q", format=".2f", title=title_label)]
    ).properties(height=220).interactive()
    return chart

tab_rates, tab_liquidity, tab_assets = st.tabs([
    "🏦 Rates & Funding Spreads", 
    "💧 Stablecoins & Capital Flows", 
    "📈 Core ETFs & Macro Benchmarks"
])

with tab_rates:
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("10-Year Treasury Yield (DGS10)")
        st.altair_chart(plot_ref_chart(master_df, "DGS10", "DGS10 Yield", "#00e5ff", "Yield (%)"), use_container_width=True)
    with col_r2:
        st.subheader("SOFR vs FFR (Zoomed Spread View)")
        rates_df = master_df[["date", "SOFR", "FFR"]].melt(id_vars="date")
        chart_rates = alt.Chart(rates_df).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", scale=alt.Scale(zero=False), title="Rate (%)"),
            color=alt.Color("variable:N", scale=alt.Scale(domain=["SOFR", "FFR"], range=["#ff3d00", "#2979ff"]), legend=alt.Legend(title="Benchmark")),
            tooltip=["date:T", "variable:N", alt.Tooltip("value:Q", format=".4f")]
        ).properties(height=220).interactive()
        st.altair_chart(chart_rates, use_container_width=True)
        
    st.subheader("SOFR Spread (SOFR - FFR)")
    st.altair_chart(plot_ref_chart(master_df, "SOFR_Spread", "Spread", "#ffea00", "Spread (%)"), use_container_width=True)

with tab_liquidity:
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.subheader("Stablecoin Market Cap (USD)")
        st.altair_chart(plot_ref_chart(master_df, "Stablecoin Mkt Cap", "Mkt Cap", "#00e676", "Market Cap ($)"), use_container_width=True)
    with col_l2:
        st.subheader("Stablecoin Market Cap (% Change)")
        master_df["Stablecoin % Change"] = (master_df["Stablecoin Mkt Cap"].pct_change() * 100.0).fillna(0.0)
        st.altair_chart(plot_ref_chart(master_df, "Stablecoin % Change", "% Change", "#00e676", "% Change"), use_container_width=True)
        
    st.subheader("CRCL ETF Daily Price")
    st.altair_chart(plot_ref_chart(master_df, "CRCL", "CRCL Price", "#00e5ff"), use_container_width=True)

with tab_assets:
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.subheader("SPY ETF (S&P 500)")
        st.altair_chart(plot_ref_chart(master_df, "SPY", "SPY Price", "#00e676"), use_container_width=True)
        st.subheader("GLD ETF (Gold)")
        st.altair_chart(plot_ref_chart(master_df, "GLD", "GLD Price", "#ffea00"), use_container_width=True)
        st.subheader("TLT ETF (Long-Term Treasuries)")
        st.altair_chart(plot_ref_chart(master_df, "TLT", "TLT Price", "#2979ff"), use_container_width=True)
        st.subheader("JNK ETF (High-Yield Bonds)")
        st.altair_chart(plot_ref_chart(master_df, "JNK", "JNK Price", "#ff3d00"), use_container_width=True)
    with col_a2:
        st.subheader("VIX Index (Volatility)")
        st.altair_chart(plot_ref_chart(master_df, "VIX", "VIX Index", "#ff3d00", "Index Value"), use_container_width=True)
        st.subheader("VTIP ETF (Inflation-Protected Treasuries)")
        st.altair_chart(plot_ref_chart(master_df, "VTIP", "VTIP Price", "#d500f9"), use_container_width=True)
        st.subheader("DRAM ETF (Memory/AI Data Center)")
        st.altair_chart(plot_ref_chart(master_df, "DRAM", "DRAM Price", "#ff9100"), use_container_width=True)
        st.subheader("EMB ETF (Emerging Market Bonds)")
        st.altair_chart(plot_ref_chart(master_df, "EMB", "EMB Price", "#d500f9"), use_container_width=True)

