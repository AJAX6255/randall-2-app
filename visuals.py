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
# Commented out global enablement to allow Streamlit's default responsive Altair theme to adapt to Light/Dark modes
# alt.themes.enable("bloomberg_dark")

# Custom axis style helper (omit hardcoded colors to allow responsive rendering)
axis_style = alt.Axis(
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
    brush_selection = None,
    reference_y: float = None,
    show_bollinger: bool = False,
    show_crossovers: bool = False,
    top_k_stress: int = None,
    horizontal_lines: list[dict] = None,
    vertical_lines: list[dict] = None,
    bands: list[dict] = None
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
        reference_y: Optional horizontal reference line at a specific value.
        show_bollinger: If True, renders Bollinger Bands (20-day rolling mean +/- 2 std dev).
        show_crossovers: If True, renders 50/200 SMA golden/death cross markers.
        top_k_stress: Optional integer to highlight top-K single-day absolute percent returns.
        horizontal_lines: List of dicts specifying horizontal levels and labels to render.
        vertical_lines: List of dicts specifying vertical dates and labels to render.
        bands: List of dicts specifying horizontal range bands and labels to render.
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
    
    # Styled drawdown area if Y axis is Drawdown
    if "drawdown" in y_axis_title.lower() or "drawdown" in title.lower():
        gradient_fill = alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(255, 61, 0, 0.4)", offset=0),
                alt.GradientStop(color="rgba(255, 61, 0, 0.0)", offset=1)
            ],
            x1=1, y1=0, x2=1, y2=1
        )
        raw_line = alt.Chart(melted_df).mark_area(
            line={"color": "#ff3d00", "strokeWidth": 2},
            color=gradient_fill,
            opacity=raw_opacity
        ).encode(
            x=x_scale_spec,
            y=alt.Y("Value:Q", scale=y_scale, axis=axis_style, title=y_axis_title),
            color=alt.Color("Series:N", scale=alt.Scale(range=COLOR_PALETTE), legend=alt.Legend(title="Assets")),
            tooltip=[
                "date:T",
                "Series:N",
                alt.Tooltip("Value:Q", format=".2f", title="Drawdown (%)")
            ]
        )
    else:
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

    # Bollinger Bands
    if show_bollinger:
        bb_chart = alt.Chart(melted_df).transform_window(
            window=[
                {"op": "mean", "field": "Value", "as": "sma20"},
                {"op": "stdev", "field": "Value", "as": "std20"}
            ],
            frame=[-19, 0],
            groupby=["Series"],
            sort=[{"field": "date"}]
        ).transform_calculate(
            upper_band="datum.sma20 + 2 * datum.std20",
            lower_band="datum.sma20 - 2 * datum.std20"
        )
        
        bb_band = bb_chart.mark_area(opacity=0.12).encode(
            x=x_scale_spec,
            y=alt.Y("lower_band:Q", scale=y_scale),
            y2="upper_band:Q",
            color=alt.Color("Series:N")
        )
        
        bb_sma = bb_chart.mark_line(strokeWidth=1, strokeDash=[2, 2]).encode(
            x=x_scale_spec,
            y=alt.Y("sma20:Q"),
            color=alt.Color("Series:N")
        )
        layers.insert(0, bb_band)
        layers.append(bb_sma)

    # SMA Crossovers (Golden & Death Crosses)
    if show_crossovers:
        crossover_chart = alt.Chart(melted_df).transform_window(
            sma50="mean(Value)",
            frame=[-49, 0],
            groupby=["Series"],
            sort=[{"field": "date"}]
        ).transform_window(
            sma200="mean(Value)",
            frame=[-199, 0],
            groupby=["Series"],
            sort=[{"field": "date"}]
        ).transform_window(
            window=[
                {"op": "lag", "field": "sma50", "as": "prev_sma50"},
                {"op": "lag", "field": "sma200", "as": "prev_sma200"}
            ],
            groupby=["Series"],
            sort=[{"field": "date"}]
        ).transform_calculate(
            crossover="datum.prev_sma50 <= datum.prev_sma200 && datum.sma50 > datum.sma200 ? 'Bullish' : (datum.prev_sma50 >= datum.prev_sma200 && datum.sma50 < datum.sma200 ? 'Bearish' : null)"
        ).transform_filter(
            "datum.crossover != null"
        )
        
        crossover_points = crossover_chart.mark_point(size=120, filled=True).encode(
            x=x_scale_spec,
            y=alt.Y("Value:Q"),
            color=alt.Color("crossover:N", scale=alt.Scale(domain=["Bullish", "Bearish"], range=["#00e676", "#ff3d00"]), legend=alt.Legend(title="Crossover")),
            shape=alt.Shape("crossover:N", scale=alt.Scale(domain=["Bullish", "Bearish"], range=["triangle-up", "triangle-down"]), legend=None),
            tooltip=["date:T", "Series:N", alt.Tooltip("Value:Q", format=".2f"), "crossover:N"]
        )
        layers.append(crossover_points)

    # Reference Y Line
    if reference_y is not None:
        ref_line = alt.Chart().mark_rule(
            color="#cbd5e0",
            strokeDash=[4, 4],
            strokeWidth=1.5
        ).encode(
            y=alt.datum(reference_y)
        )
        layers.append(ref_line)

    # Custom Horizontal Lines with Labels
    if horizontal_lines:
        for line in horizontal_lines:
            val = line.get("value")
            label = line.get("label", "")
            if val is not None:
                h_line = alt.Chart().mark_rule(
                    color="#eab308",
                    strokeDash=[4, 4],
                    strokeWidth=1.5
                ).encode(
                    y=alt.datum(val)
                )
                layers.append(h_line)
                
                if label:
                    h_label = alt.Chart().mark_text(
                        align="left",
                        dx=8,
                        dy=-5,
                        color="#eab308",
                        fontSize=10,
                        fontWeight="bold"
                    ).encode(
                        x=alt.value(10),
                        y=alt.datum(val),
                        text=alt.datum(label)
                    )
                    layers.append(h_label)

    # Custom Vertical Lines with Labels
    if vertical_lines:
        for line in vertical_lines:
            date_val = line.get("date")
            label = line.get("label", "")
            if date_val:
                iso_date = pd.to_datetime(date_val).isoformat()
                v_line = alt.Chart().mark_rule(
                    color="#718096",
                    strokeDash=[4, 4],
                    strokeWidth=1.5
                ).encode(
                    x=alt.datum(iso_date)
                )
                layers.append(v_line)
                
                if label:
                    v_label = alt.Chart().mark_text(
                        align="left",
                        angle=270,
                        dx=8,
                        dy=12,
                        color="#718096",
                        fontSize=10,
                        fontWeight="bold"
                    ).encode(
                        x=alt.datum(iso_date),
                        y=alt.value(20),
                        text=alt.datum(label)
                    )
                    layers.append(v_label)

    # Custom Horizontal Bands with Labels
    if bands:
        for band in bands:
            ymin = band.get("ymin")
            ymax = band.get("ymax")
            label = band.get("label", "")
            if ymin is not None and ymax is not None:
                band_rect = alt.Chart().mark_rect(
                    color="#00e5ff",
                    opacity=0.08
                ).encode(
                    y=alt.datum(ymin),
                    y2=alt.datum(ymax)
                )
                layers.insert(0, band_rect)
                
                if label:
                    band_label = alt.Chart().mark_text(
                        align="left",
                        dx=8,
                        dy=12,
                        color="#00e5ff",
                        fontSize=10,
                        fontWeight="bold",
                        opacity=0.7
                    ).encode(
                        x=alt.value(10),
                        y=alt.datum(ymin),
                        text=alt.datum(label)
                    )
                    layers.append(band_label)

    # Top-K Stress Highlights
    if top_k_stress is not None and top_k_stress > 0:
        stress_marker_chart = alt.Chart(melted_df).transform_window(
            window=[{"op": "lag", "field": "Value", "as": "prev_value"}],
            groupby=["Series"],
            sort=[{"field": "date"}]
        ).transform_calculate(
            daily_change="datum.prev_value ? abs(datum.Value - datum.prev_value) / datum.prev_value : 0"
        ).transform_window(
            window=[{"op": "rank", "as": "rank"}],
            sort=[{"field": "daily_change", "order": "descending"}],
            groupby=["Series"]
        ).transform_filter(
            f"datum.rank <= {top_k_stress}"
        )
        
        stress_markers = stress_marker_chart.mark_point(
            size=150, strokeWidth=2, color="#ff3d00"
        ).encode(
            x=x_scale_spec,
            y=alt.Y("Value:Q"),
            tooltip=["date:T", "Series:N", alt.Tooltip("Value:Q", format=".2f"), alt.Tooltip("daily_change:Q", format=".2%", title="Daily Abs Return")]
        )
        layers.append(stress_markers)



    # 5. Interactive cross-hair guidelines
    rule = alt.Chart(melted_df).mark_rule(color="#4a5568", strokeWidth=1.0).encode(
        x="date:T",
        opacity=alt.condition(hover_selection, alt.value(0.5), alt.value(0.0))
    ).add_params(hover_selection)

    points = raw_line.mark_point(size=60).encode(
        opacity=alt.condition(hover_selection, alt.value(1.0), alt.value(0.0))
    )

    layers.extend([rule, points])

    # Combined layered chart
    chart = alt.layer(*layers).properties(
        title=alt.TitleParams(
            text=title,
            anchor="start",
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
