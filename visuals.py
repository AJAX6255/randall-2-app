#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visuals.py

Generates production-grade, institutional-quality Altair / Vega-Lite charts.
Supports Bloomberg-style dark mode styling, layered graphs, regime shading,
faint raw signals paired with smoothed indicators, and linked interactive brushing.
"""

import altair as alt
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# THEME CONFIGURATION
# -----------------------------------------------------------------------------

def get_bloomberg_theme():
    """
    Returns custom theme config representing a professional macro risk dashboard:
    Sleek dark background, clean grey grids, clear readable typography.
    """
    return {
        "config": {
            "view": {
                "stroke": "transparent",
                "width": 800,
                "height": 400
            },
            "background": "#0b0e14",  # Deep obsidian dark blue/grey
            "axis": {
                "grid": True,
                "gridColor": "#1f2633",
                "domainColor": "#2d3748",
                "tickColor": "#2d3748",
                "labelColor": "#a0aec0",
                "titleColor": "#e2e8f0",
                "labelFont": "Inter, system-ui, sans-serif",
                "titleFont": "Inter, system-ui, sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 13,
                "titlePadding": 10
            },
            "legend": {
                "labelColor": "#cbd5e0",
                "titleColor": "#cbd5e0",
                "labelFont": "Inter, system-ui, sans-serif",
                "titleFont": "Inter, system-ui, sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 12,
                "padding": 12,
                "symbolType": "stroke"
            },
            "line": {
                "strokeWidth": 2,
                "interpolate": "monotone"
            }
        }
    }

# Register theme
alt.themes.register("bloomberg_dark", get_bloomberg_theme)
alt.themes.enable("bloomberg_dark")

# Custom axis style helper
axis_style = alt.Axis(
    gridColor="#1a202c",
    labelColor="#a0aec0",
    titleColor="#e2e8f0",
    labelFontSize=11,
    titleFontSize=12,
    tickCount=8
)

COLOR_PALETTE = [
    "#00e5ff",  # Neon Cyan
    "#ff3d00",  # Neon Red/Orange
    "#00e676",  # Neon Green
    "#d500f9",  # Neon Purple
    "#ffea00",  # Neon Yellow
    "#2979ff",  # Neon Blue
    "#ff9100"   # Amber
]

# -----------------------------------------------------------------------------
# CHART GENERATOR
# -----------------------------------------------------------------------------

def build_macro_chart(
    df: pd.DataFrame,
    y_cols: list[str],
    title: str = "Market Indicators",
    y_axis_title: str = "Value",
    smooth_window: int = None,
    show_regimes: bool = False,
    regime_df: pd.DataFrame = None,
    normalize: bool = False,
    log_scale: bool = False,
    brush_selection = None
) -> alt.Chart:
    """
    Creates a layered, interactive macro chart.
    
    Args:
        df: Pandas DataFrame containing 'date' and the target series.
        y_cols: Columns to plot on the Y-axis.
        title: Title of the chart.
        y_axis_title: Label for the Y-axis.
        smooth_window: Optional rolling window to create a bold smoothed signal overlaid on a faint raw signal.
        show_regimes: If True, draws background shading bands indicating stress/volatility regimes.
        regime_df: DataFrame containing 'date' and 'regime' for background shading.
        normalize: If True, indicates that the series are normalized (adds appropriate titles).
        log_scale: If True, applies log scaling to the y-axis.
        brush_selection: An optional Altair brush selection object for linked zooming/brushing.
    """
    df_reset = df.reset_index()
    if "date" not in df_reset.columns:
        # Fallback if index wasn't date
        df_reset.rename(columns={df_reset.columns[0]: "date"}, inplace=True)
    
    df_reset["date"] = pd.to_datetime(df_reset["date"])

    # Melt data for easy multi-series color mapping in Altair
    melted_df = df_reset.melt(
        id_vars=["date"],
        value_vars=y_cols,
        var_name="Series",
        value_name="Value"
    )

    # 1. Base selection for cross-hair hover interaction
    hover_selection = alt.selection_point(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty=False
    )

    # Determine Y Scale type
    y_scale = alt.Scale(type="log" if log_scale else "linear", zero=False)

    # X axis spec
    x_scale_spec = alt.X("date:T", axis=axis_style, title=None)
    if brush_selection is not None:
        x_scale_spec = x_scale_spec.filter(brush_selection)

    # 2. Raw Signal Layer (faint if smoothed line is active, otherwise standard)
    raw_stroke_width = 1.0 if smooth_window else 2.0
    raw_opacity = 0.35 if smooth_window else 0.85
    
    raw_line = alt.Chart(melted_df).mark_line(
        strokeWidth=raw_stroke_width,
        opacity=raw_opacity
    ).encode(
        x=x_scale_spec,
        y=alt.Y("Value:Q", scale=y_scale, axis=axis_style, title=y_axis_title),
        color=alt.Color("Series:N", scale=alt.Scale(range=COLOR_PALETTE), legend=alt.Legend(title="Assets")),
        tooltip=[
            "date:T",
            "Series:N",
            alt.Tooltip("Value:Q", format=".2f", title="Price/Value")
        ]
    )

    layers = [raw_line]

    # 3. Optional Smoothed Layer
    if smooth_window and smooth_window > 1:
        # Compute smoothed column and melt
        smooth_df = df_reset[["date"]].copy()
        for col in y_cols:
            smooth_df[col] = df_reset[col].ewm(span=smooth_window, adjust=False).mean()
            
        melted_smooth_df = smooth_df.melt(
            id_vars=["date"],
            value_vars=y_cols,
            var_name="Series",
            value_name="Value"
        )
        
        smooth_line = alt.Chart(melted_smooth_df).mark_line(
            strokeWidth=2.5,
            opacity=1.0
        ).encode(
            x=x_scale_spec,
            y=alt.Y("Value:Q", scale=y_scale, axis=None),
            color=alt.Color("Series:N"),
            tooltip=[
                "date:T",
                "Series:N",
                alt.Tooltip("Value:Q", format=".2f", title="Smoothed")
            ]
        )
        layers.append(smooth_line)

    # 4. Optional Background Shading for Macro Regimes
    if show_regimes and regime_df is not None and not regime_df.empty:
        reg_df = regime_df.copy()
        reg_df["date"] = pd.to_datetime(reg_df["date"])
        reg_df["next_date"] = reg_df["date"].shift(-1).fillna(reg_df["date"] + pd.Timedelta(days=1))
        
        # Color mapping for regimes
        regime_colors = {
            "Oil Supply Shock": "rgba(244, 67, 54, 0.20)",   # Red
            "Oil Demand Shock": "rgba(76, 175, 80, 0.15)",   # Green
            "Flight to Safety": "rgba(255, 152, 0, 0.18)",   # Orange
            "Expansionary": "rgba(0, 229, 255, 0.12)",       # Cyan
            "Neutral": "rgba(45, 55, 72, 0.05)"              # Muted grey
        }
        
        # Find unique regimes in data to populate correct domain
        unique_regimes = reg_df["regime"].unique().tolist()
        domain = [r for r in regime_colors.keys() if r in unique_regimes]
        color_range = [regime_colors[r] for r in domain]

        shading = alt.Chart(reg_df).mark_rect(opacity=0.6).encode(
            x=alt.X("date:T"),
            x2="next_date:T",
            color=alt.Color(
                "regime:N",
                scale=alt.Scale(domain=domain, range=color_range),
                legend=alt.Legend(title="Macro Regimes", orient="bottom")
            )
        )
        # Place shading at the bottom of layer list so lines draw over it
        layers.insert(0, shading)

    # 5. Interactive cross-hair guidelines
    rule = alt.Chart(melted_df).mark_rule(color="#4a5568", strokeWidth=1.0).encode(
        x="date:T",
        opacity=alt.condition(hover_selection, alt.value(0.5), alt.value(0.0))
    ).add_params(hover_selection)

    points = raw_line.mark_point(size=60, fill="#0b0e14").encode(
        opacity=alt.condition(hover_selection, alt.value(1.0), alt.value(0.0))
    )

    layers.extend([rule, points])

    # Combined layered chart
    chart = alt.layer(*layers).properties(
        title=alt.TitleParams(
            text=title,
            anchor="start",
            color="white",
            fontSize=16,
            fontWeight="bold"
        )
    ).interactive()

    return chart

# -----------------------------------------------------------------------------
# SPECIFIC COMPOSITE CHARTS
# -----------------------------------------------------------------------------

def build_stress_bands_chart(stress_df: pd.DataFrame) -> alt.Chart:
    """
    Builds the flagship stress composite index chart with colored background bands
    and a solid neon-cyan value line.
    """
    df = stress_df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    min_dt = df["date"].min()
    max_dt = df["date"].max()

    # Bands data
    bands_data = pd.DataFrame([
        {"start": min_dt, "end": max_dt, "ymin": 0,  "ymax": 25,  "regime": "Low Stress"},
        {"start": min_dt, "end": max_dt, "ymin": 25, "ymax": 50,  "regime": "Moderate Stress"},
        {"start": min_dt, "end": max_dt, "ymin": 50, "ymax": 75,  "regime": "High Stress"},
        {"start": min_dt, "end": max_dt, "ymin": 75, "ymax": 100, "regime": "Crisis"}
    ])

    bands = alt.Chart(bands_data).mark_rect().encode(
        x=alt.X("start:T", axis=axis_style, title=None),
        x2="end:T",
        y=alt.Y("ymin:Q", axis=None),
        y2="ymax:Q",
        color=alt.Color(
            "regime:N",
            scale=alt.Scale(
                domain=["Crisis", "High Stress", "Moderate Stress", "Low Stress"],
                range=[
                    "rgba(239, 68, 68, 0.35)",   # Crisis (Red)
                    "rgba(249, 115, 22, 0.25)",  # High Stress (Orange)
                    "rgba(234, 179, 8, 0.18)",   # Moderate (Yellow)
                    "rgba(34, 197, 94, 0.10)"    # Low (Green)
                ]
            ),
            legend=alt.Legend(title="Stress Regime", orient="right")
        )
    )

    lines = alt.Chart(df).mark_line(
        color="#00e5ff",
        strokeWidth=2.5
    ).encode(
        x=alt.X("date:T", axis=axis_style),
        y=alt.Y("stress_composite:Q", axis=axis_style, scale=alt.Scale(domain=[0, 100]), title="Composite Score"),
        tooltip=[
            "date:T",
            alt.Tooltip("stress_composite:Q", format=".2f", title="Stress Index"),
            alt.Tooltip("stress_regime:N", title="Regime")
        ]
    )

    return alt.layer(bands, lines).properties(
        title=alt.TitleParams(
            text="Systemic Market Stress Composite Index",
            anchor="start",
            fontSize=15
        )
    ).interactive()
