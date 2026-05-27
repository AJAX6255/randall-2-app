# Prompt for Google Antigravity Agentic Development System

You are an elite principal engineer, quantitative systems architect, full-stack TypeScript/Python developer, data visualization specialist, and AI systems designer.

Your task is to design and implement a production-grade:

> Semantic Macro-Financial Analytical Operating System

This is NOT a conventional dashboard.

The goal is to create a conversational, multimodal, AI-assisted macro-financial exploration environment capable of:
- semantic querying
- adaptive visualization
- macro regime analysis
- dynamic normalization
- cross-asset comparison
- conversational analytical workflows
- LLM-orchestrated transformations
- stateful analytical memory

The system should feel closer to:
- Bloomberg Terminal + AI copilot
- macro risk intelligence system
- semantic analytical workspace

rather than:
- a static BI dashboard
- generic chatbot
- ordinary financial plotting app

---

# PRIMARY DESIGN PHILOSOPHY

The architecture MUST follow this principle:

```text
LLM orchestrates deterministic analytical primitives.
LLM does NOT perform arbitrary calculations directly.
```

This means:
- all statistics
- normalization
- rolling transforms
- z-scores
- correlations
- regime analysis
- smoothing
- lead/lag analysis
- event anchoring

must be implemented as deterministic Python analytical functions.

The LLM:
- interprets user intent
- selects analytical tools
- constructs transformation plans
- builds visualization specifications
- manages conversational state

---

# TARGET USER EXPERIENCE

Users should be able to type or speak commands like:

- "Compare BTC behavior to gold during stress regimes"
- "Normalize all assets to the Fed pivot"
- "Show where stablecoin contraction led credit widening"
- "Find periods where BTC diverged from SPY"
- "Expand the crisis regime and compress quiet periods"
- "Overlay liquidity contraction with Treasury volatility"
- "Compare this region to the COVID liquidity event"

The system should:
- understand intent
- identify relevant data
- choose transformations
- generate optimized Vega-Lite visualizations
- preserve conversational analytical context

The user should also be able to:
- brush regions on charts
- select time windows visually
- speak contextual commands
- refine visualizations conversationally
- save analytical sessions

---

# CORE ARCHITECTURE

Design the system as:

```text
User Interface
    ↓
Semantic Intent Layer
    ↓
Transformation Planning Engine
    ↓
Deterministic Analytical Engine
    ↓
Visualization Specification Engine
    ↓
Vega-Lite Rendering Layer
```

---

# REQUIRED TECHNOLOGY STACK

## Frontend

Use:
- Next.js
- TypeScript
- Tailwind CSS
- Vega-Lite / react-vega
- Zustand or Redux for state management
- WebSocket support for conversational interaction

The frontend must support:
- responsive dashboard layout
- semantic chat sidebar
- voice interaction hooks
- region brushing
- synchronized charts
- adaptive layouts
- multi-panel macro dashboards

---

## Visualization Layer

Visualization system MUST use:
- Vega-Lite
- declarative specifications
- layered charts
- semantic chart composition

DO NOT use imperative matplotlib-style chart generation.

The LLM should generate:
- Vega-Lite specifications
- compositional chart layers
- transformation instructions

Required visualization capabilities:
- layered macro charts
- regime shading
- threshold overlays
- rolling normalization
- z-score transformations
- synchronized axes
- linked brushing
- event annotations
- adaptive scaling
- multi-series overlays
- interactive tooltips
- semantic chart updates

---

# BACKEND

Use:
- Python
- FastAPI
- Polars
- DuckDB
- Pydantic
- Async architecture

The backend should expose deterministic analytical tools.

---

# REQUIRED ANALYTICAL PRIMITIVES

Implement deterministic functions for:

```python
normalize_series()
zscore_series()
rolling_correlation()
rolling_beta()
rolling_volatility()
detect_divergence()
detect_regimes()
compute_drawdown()
compute_relative_strength()
rolling_pca()
lead_lag_analysis()
volatility_adjustment()
liquidity_regime_detection()
macro_event_alignment()
rolling_smoothing()
percentile_scaling()
log_scaling()
volatility_compression()
regime_segmentation()
```

All primitives must:
- be independently testable
- support vectorized execution
- operate efficiently on large datasets
- expose clean tool interfaces

---

# DATA LAYER

Implement a modular data ingestion architecture.

Initial supported sources:
- Yahoo Finance
- FRED
- CoinGecko
- Stablecoin metrics
- Treasury data
- credit spreads
- VIX data
- custom CSV/Parquet uploads

Storage architecture:
- DuckDB
- Parquet caching
- lazy Polars pipelines

The system must support:
- time-series alignment
- asynchronous updates
- cached transformations
- efficient historical retrieval

---

# LLM ORCHESTRATION LAYER

Use:
- Gemini Flash models
- structured JSON tool calling
- LangGraph orchestration

DO NOT generate arbitrary Python dynamically.

Instead:

```text
User intent
    ↓
Structured semantic plan
    ↓
Tool invocation
    ↓
Deterministic execution
    ↓
Visualization specification
```

The LLM should:
- infer analytical intent
- choose transformations
- select visualization structures
- maintain conversational context
- generate Vega-Lite specifications
- preserve reproducibility

---

# MEMORY SYSTEM

Implement:
- conversational session memory
- analytical state persistence
- preferred normalization tracking
- prior comparison recall
- saved dashboard states
- semantic analytical history

Recommended stack:
- Postgres
- pgvector
- LangGraph state persistence

---

# MULTIMODAL INTERACTION

Design the architecture for future support of:
- voice commands
- multimodal prompts
- chart-region selection
- visual-semantic interaction

Examples:

User brushes chart region and says:
- "Normalize everything from here"
- "Compare this to COVID"
- "Find similar liquidity structures"

The system should combine:
- visual state
- conversational state
- active chart context
- selected regions
- analytical history

---

# REQUIRED VISUALIZATION ARCHITECTURE

Professional macro dashboard standards must be followed.

Preferred chart architecture:

```text
Background regime shading
+ threshold rules
+ faint raw signal
+ smoothed primary signal
+ area under curve
+ semantic annotations
+ hover interactions
```

Charts should resemble:
- institutional macro dashboards
- risk-monitoring systems
- hedge-fund analytical tooling

NOT retail trading applications.

---

# APPLICATION FEATURES

The final application must include:

## Dashboard Workspace
- multi-panel chart grid
- synchronized timelines
- semantic control sidebar
- conversational analytics panel
- saved workspace states

## Semantic Analytics
- natural-language transformations
- conversational chart refinement
- dynamic normalization
- event anchoring
- adaptive overlays

## Cross-Asset Intelligence
- rolling correlation analysis
- divergence detection
- stress-regime comparison
- liquidity overlays
- risk contagion analysis

## Regime Intelligence
- automatic regime segmentation
- volatility clustering
- crisis-state identification
- persistence analysis
- macro-event alignment

## Adaptive Visualization
- context-aware scaling
- semantic chart restructuring
- automatic overlay selection
- intelligent comparative layouts

---

# IMPLEMENTATION REQUIREMENTS

Generate:

1. Full system architecture
2. Directory structure
3. Backend architecture
4. Frontend architecture
5. API specifications
6. Tool schemas
7. LangGraph orchestration plan
8. Vega-Lite rendering architecture
9. State management plan
10. Memory architecture
11. Deployment architecture
12. Docker configuration
13. Environment configuration
14. CI/CD recommendations
15. Performance optimization strategy
16. Security considerations
17. Authentication architecture
18. Scaling strategy
19. Cost optimization strategy
20. Incremental development roadmap

---

# DEVELOPMENT PHASES

## Phase 1
Core deterministic dashboard:
- data ingestion
- analytical engine
- Vega-Lite rendering
- multi-panel dashboard

## Phase 2
Conversational orchestration:
- Gemini integration
- semantic transformation planning
- tool calling
- conversational refinement

## Phase 3
Advanced macro intelligence:
- regime detection
- divergence analysis
- adaptive normalization
- semantic overlays

## Phase 4
Multimodal analytical workspace:
- voice interaction
- visual-semantic interaction
- adaptive chart restructuring
- analytical memory systems

---

# CRITICAL DESIGN CONSTRAINTS

The application MUST:
- remain deterministic
- preserve reproducibility
- expose transformation transparency
- avoid hallucinated calculations
- support inspection of analytical pipelines
- preserve auditability
- support institutional-quality workflows

The LLM is:
- orchestration layer
- semantic interpreter
- workflow planner

NOT:
- uncontrolled code generator
- direct statistical engine

---

# FINAL OUTPUT REQUIREMENTS

Produce:

1. Complete architectural specification
2. End-to-end system design
3. Production-grade repository structure
4. Backend implementation plan
5. Frontend implementation plan
6. Vega-Lite semantic rendering system
7. Tool-calling orchestration architecture
8. LangGraph workflow graphs
9. Example conversational analytical flows
10. Example Vega-Lite generation flows
11. Example semantic transformation pipelines
12. Complete MVP implementation roadmap
13. Recommended deployment topology
14. Cost estimates
15. Scalability considerations
16. Technical debt minimization strategy
17. Future extensibility plan

The resulting system should represent:

> a semantic operating system for macro-financial exploration.

Once running, the dashboard will display:

1. **10-Year Treasury Yield (DGS10)** - Shows the yield on 10-year US Treasury bonds
2. **SOFR vs FFR (Zoomed Spread View)** - Compares Secured Overnight Financing Rate with Federal Funds Rate
3. **SOFR Spread (SOFR - FFR)** - Visualizes the spread between SOFR and Federal Funds Rate
4. **Stablecoin Market Cap (USD)** - Tracks the total market capitalization of stablecoins
5. **Stablecoin Market Cap (% Change)** - Shows percentage changes in stablecoin market cap
6. **Individual ETF Charts** - Displays price movements for:
   - VIX Index (Volatility)
   - SPY ETF (S&P 500)
   - GLD ETF (Gold)
   - VTIP ETF (Inflation-Protected Treasuries)
   - TLT ETF (Long-Term Treasuries)
   - CRCL ETF (Circle Stablecoin)
   - DRAM ETF (Memory/AI Data Center)
   - JNK ETF (High-Yield Bonds)
   - EMB ETF (Emerging Market Bonds)