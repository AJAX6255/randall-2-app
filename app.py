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

# Custom styling override for a premium dark terminal appearance
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
    }
    
    /* Custom headers */
    h1, h2, h3 {
        font-family: 'Inter', system-ui, sans-serif !important;
        font-weight: 700 !important;
        color: #f7fafc !important;
    }
    
    /* Metric boxes */
    div[data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-size: 2rem;
        color: #00e5ff !important;
    }
    
    /* Chat inputs */
    .stChatInput {
        border-color: #2d3748 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0e121a !important;
        border-right: 1px solid #1f2633;
    }
    
    /* Custom card container styling */
    .macro-card {
        background-color: #121722;
        border: 1px solid #1f2633;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to the Semantic Macro OS. Ask me to compare assets, z-score signals, calculate rolling correlations, or analyze macro stress regimes."}
    ]

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL PANEL & HEALTH
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Macro Terminal")
    st.markdown("---")
    
    # 1. API Status Indicators
    st.subheader("🔧 System Connections")
    with st.spinner("Verifying connections..."):
        fred_ok, fred_msg = ingest.check_fred_api()
        stable_ok, stable_msg = ingest.check_stablecoins_api()
        yahoo_ok, yahoo_msg = ingest.check_yfinance_api()
        
        if fred_ok:
            st.success(f"● {fred_msg}")
        else:
            st.error(f"● {fred_msg}")
            
        if stable_ok:
            st.success(f"● {stable_msg}")
        else:
            st.error(f"● {stable_msg}")
            
        if yahoo_ok:
            st.success(f"● {yahoo_msg}")
        else:
            st.error(f"● {yahoo_msg}")
            
    st.markdown("---")

    # 2. Database Controls
    st.subheader("💾 Caching Controls")
    if st.button("Force Database Sync", use_container_width=True):
        with st.spinner("Downloading fresh financial time-series..."):
            ingest.ingest_all(force=True)
            st.success("Database fully synchronized!")
            st.rerun()

    st.markdown("---")

    # 3. Macro Regime Quick-Select Templates
    st.subheader("💡 Macro Templates")
    templates = {
        "Risk-On Expansion": {
            "y_cols": ["SPY", "JNK", "DRAM", "BTC"],
            "transform": "none",
            "normalize": True,
            "show_regimes": False,
            "analysis_text": "Template: Visualizing risk-on parameters (Equities, Junk Bonds, Semiconductors, and Bitcoin) normalized to the start."
        },
        "Flight to Safety": {
            "y_cols": ["TLT", "GLD", "VIX", "SPY"],
            "transform": "none",
            "normalize": True,
            "show_regimes": True,
            "analysis_text": "Template: Flight to safety indicators (Long Treasuries, Gold, VIX Volatility Index, and Equities) shaded with macro stress regimes."
        },
        "Credit & Funding Stress": {
            "y_cols": ["SOFR_Spread", "JNK", "EMB", "TLT"],
            "transform": "none",
            "normalize": True,
            "show_regimes": True,
            "analysis_text": "Template: Credit spreads, emerging debt and funding rate indicators mapped together to trace liquidity tightening."
        },
        "Crypto Leverage Bubble": {
            "y_cols": ["BTC", "Stablecoin Mkt Cap", "VIX"],
            "transform": "none",
            "normalize": True,
            "show_regimes": False,
            "analysis_text": "Template: Tracking cryptocurrency pricing against stablecoin supply expansion and broader equity volatility."
        }
    }

    for name, spec in templates.items():
        if st.button(name, use_container_width=True):
            st.session_state.active_plan = spec
            st.session_state.chat_history.append({"role": "user", "content": f"Load template: {name}"})
            st.session_state.chat_history.append({"role": "assistant", "content": f"Loaded template configuration: {name}"})
            st.rerun()

# -----------------------------------------------------------------------------
# MAIN APP BODY
# -----------------------------------------------------------------------------
st.title("🏛️ Semantic Macro-Financial Operating System")
st.markdown("LLM-orchestrated quantitative time-series exploration environment.")

# Load baseline dataset
try:
    master_df = ingest.get_aligned_dataset()
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
        vix_score = (master_df["VIX"] - master_df["VIX"].min()) / (master_df["VIX"].max() - master_df["VIX"].min()) * 100
        latest_stress = vix_score.iloc[-1]
        st.metric("Systemic Stress Indicator", f"{latest_stress:.1f} / 100")

# 2. Main Exploration Panel (Layered Visualizer)
st.markdown('<div class="macro-card">', unsafe_allow_html=True)
plan = st.session_state.active_plan

# Execute transformations on the master dataset
try:
    plot_df, plot_cols, y_label = orchestrator.execute_plan(plan, master_df)
    
    # Calculate regimes if requested
    regimes_df = None
    if plan.get("show_regimes", False):
        regimes_df = pd.DataFrame({
            "date": master_df["date"],
            "regime": analytics.detect_regimes(master_df)
        })

    # Render dynamic layered chart
    chart = visuals.build_macro_chart(
        df=plot_df,
        y_cols=plot_cols,
        title=f"Exploring: {', '.join(plan.get('y_cols', []))}",
        y_axis_title=y_label,
        smooth_window=plan.get("smooth_window"),
        show_regimes=plan.get("show_regimes", False),
        regime_df=regimes_df,
        normalize=plan.get("normalize", False)
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
    user_query = st.chat_input("Enter analysis intent: (e.g., 'Compare BTC and GLD rolling 60-day correlation')")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("Interpreting intent & compiling execution plan..."):
            new_plan = orchestrator.parse_intent(user_query)
            st.session_state.active_plan = new_plan
            
            # Append analysis response
            analysis = new_plan.get("analysis_text", "Execution plan prepared.")
            st.session_state.chat_history.append({"role": "assistant", "content": analysis})
            st.rerun()

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
    
    # Calculate stress history table for the user
    stress_hist = pd.DataFrame({
        "Date": master_df["date"],
        "VIX Value": master_df["VIX"],
        "SOFR Rate (%)": master_df["SOFR"],
        "Stablecoin Cap ($)": master_df["Stablecoin Mkt Cap"],
        "Regime": analytics.detect_regimes(master_df)
    }).tail(10)
    
    st.dataframe(stress_hist, use_container_width=True)
