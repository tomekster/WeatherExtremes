"""
Browser viewer for ERA5 daily max 2m temperature and per-DOY percentile thresholds.

Right: map of Earth. Hovering previews the last year of raw data for the grid cell
under the cursor; clicking pins that location (later hovers are ignored until the
next click, or until "Follow cursor" is pressed). Middle: the selected point.
Left: once a point is pinned, controls for the displayed year and for every
parameter of ExceedancesCalculator.compute_exceedances (src/synthetic_vienna/
exceedances.py). The two middle columns then show, side by side, the original
data and the data with the reference mean seasonal cycle subtracted before the
thresholds are computed (Brunner & Voigt 2024, Nat. Commun. 15, 2087): each with
the aggregated series, per-DOY thresholds and exceedance days for the chosen
year, and below it a years × DOY exceedance grid with the yearly exceedance rate.
A TemperatureModel (src/synthetic_vienna/temperature_model.py) is fitted to the
selected point; one seeded realization of it goes through the same computation and
is drawn on the temperature/anomaly plots and as dashed trend lines on the
exceedance % plots. The fitted parameters are listed under the map as editable fields: editing one
regenerates the synthetic data (same seed) and updates its plots.

Point data is retrieved with scripts/load_vienna_data.select_point_subset, i.e.
the same logic as load_vienna_subset() (1961–2020, no-leap calendar, °C).

Run from the repo root (the zarr path in load_vienna_data is relative):
    python -m src.app [--port 8051]
then open http://127.0.0.1:8051/ (over SSH, forward the port first).
"""
import argparse
import copy
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cartopy.feature as cfeature
import cftime
import dash
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import ALL, Input, Output, Patch, State, dcc, html
from dash.exceptions import PreventUpdate

from scripts.load_vienna_data import open_t2max_celsius, select_point_subset
from src.synthetic_vienna.exceedances import ExceedancesCalculator
from statsmodels.tsa.arima_process import ArmaProcess

from src.synthetic_vienna.temperature_model import ModelParams, TemperatureModel, ar_variance_factor
from src.synthetic_vienna.trends import fit_all_trends, yearly_percentage

VAR = "daily_max_2m_temperature"
GRID_LATS = np.arange(-90.0, 90.01, 0.25)
GRID_LONS = np.arange(-180.0, 180.0, 0.25)
FIRST_YEAR, LAST_YEAR = 1961, 2020  # range returned by select_point_subset

DEFAULTS = dict(percentile=0.95, boosting_window=1, aggregation_window=1, agg_method="mean",
                reference_window=[1961, 1990], analysis_window=[FIRST_YEAR, LAST_YEAR], pot_run_length=3)
# Controls and map stay in view while the figure columns scroll
STICKY = {"position": "sticky", "top": 0, "maxHeight": "98vh"}
TREND_COLORS = {"Linear (OLS)": "#1f77b4", "Theil–Sen": "#ff7f0e", "NHPP": "#9467bd", "POT": "#2ca02c"}
SYNTH_COLOR = "#e377c2"
MODEL_HINT = "Click a point on the map to fit the synthetic temperature model there."
MODEL_K, MODEL_L = 3, 1  # TemperatureModel harmonics and AR order (same defaults as the synthetic scripts)

# First day-of-year of each month in the no-leap calendar, for x-axis ticks
MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_STARTS = (np.cumsum([0] + MONTH_LENGTHS[:-1]) + 1).tolist()
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOYS = np.arange(1, 366)

PIN_TRACE = 3  # index of the pin marker trace in the map figure
VARIANTS = ("original", "deseasonalized")  # compute_exceedances without / with subtract_seasonal_cycle


def snap_to_grid(lat, lon):
    """Snap to the 0.25° grid so lat_lon_to_grid_index yields in-range indices (lon wraps at ±180)."""
    lat = float(np.clip(round(lat * 4) / 4, -90.0, 90.0))
    lon = (round(lon * 4) / 4 + 180.0) % 360.0 - 180.0
    return lat, lon


def format_coord(lat, lon):
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"


def full_years(year_range):
    """[y0, y1] -> (Jan 1 of y0, Dec 31 of y1) as no-leap dates, the window format compute_exceedances expects."""
    return cftime.DatetimeNoLeap(year_range[0], 1, 1), cftime.DatetimeNoLeap(year_range[1], 12, 31)


def outlines(geometries, polygons):
    """Flatten (multi)polygon exteriors or (multi)lines into x/y lists separated by None, for one Scatter trace."""
    xs, ys = [], []
    for geom in geometries:
        for part in getattr(geom, "geoms", [geom]):
            x, y = (part.exterior if polygons else part).xy
            xs += list(x) + [None]
            ys += list(y) + [None]
    return xs, ys


def build_map_figure():
    land_x, land_y = outlines(cfeature.NaturalEarthFeature("physical", "land", "110m").geometries(), polygons=True)
    coast_x, coast_y = outlines(cfeature.COASTLINE.with_scale("110m").geometries(), polygons=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=land_x, y=land_y, mode="lines", fill="toself", fillcolor="#e8e2cf",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=coast_x, y=coast_y, mode="lines", line=dict(color="#555", width=0.7),
                             hoverinfo="skip", showlegend=False))
    # Invisible heatmap at data resolution: makes every grid cell hoverable/clickable
    fig.add_trace(go.Heatmap(x=GRID_LONS, y=GRID_LATS, z=np.zeros((len(GRID_LATS), len(GRID_LONS)), dtype=np.uint8),
                             colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
                             hovertemplate="%{y:.2f}°, %{x:.2f}°<extra></extra>"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(color="red", size=11, line=dict(color="white", width=2))))
    fig.update_xaxes(range=[-180, 180], dtick=60, showgrid=True, gridcolor="#d5e3ee", zeroline=False,
                     ticksuffix="°", fixedrange=True, constrain="domain")
    fig.update_yaxes(range=[-90, 90], dtick=30, showgrid=True, gridcolor="#d5e3ee", zeroline=False,
                     ticksuffix="°", scaleanchor="x", fixedrange=True, constrain="domain")
    fig.update_layout(plot_bgcolor="#b9d3e6", margin=dict(l=40, r=10, t=40, b=30),
                      title=dict(text="Hover to preview, click to select", font=dict(size=14)),
                      hovermode="closest", uirevision="map")
    return fig


def base_year_figure(title, subtitle=None, ytitle="Daily max 2m temperature (°C)"):
    fig = go.Figure()
    fig.update_xaxes(tickvals=MONTH_STARTS, ticktext=MONTH_LABELS, range=[1, 365])
    fig.update_yaxes(title=ytitle)
    fig.update_layout(title=dict(text=title, font=dict(size=15), subtitle=dict(text=subtitle or "")),
                      template="plotly_white", margin=dict(l=60, r=15, t=70, b=60), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0))
    return fig


def preview_figure(values, lat, lon, year):
    fig = base_year_figure(f"{format_coord(lat, lon)} — {year} (hover preview, click to select)")
    fig.add_trace(go.Scatter(x=DOYS, y=values, mode="lines", line=dict(color="#1f77b4", width=1.2),
                             name="Daily max 2m temperature", hovertemplate="%{y:.1f} °C"))
    return fig


def analysis_figure(raw, ec, lat, lon, year, params, seasonal_cycle, synth_ec):
    """One year of data, thresholds and exceedances.

    If ec was run with subtract_seasonal_cycle, everything is plotted as anomalies from ec.seasonal_cycle;
    otherwise *seasonal_cycle* (the reference mean seasonal cycle) is drawn for orientation.
    synth_ec holds the same computation on the TemperatureModel realization; its (aggregated) series is drawn.
    """
    deseasonalized = ec.seasonal_cycle is not None
    an_idx = year - params["analysis_window"][0]
    agg = ec.analysis_agg[an_idx]
    exceeded = ec.exceedances[an_idx]
    ref0, ref1 = params["reference_window"]
    agg_label = (f"{params['aggregation_window']}-day {params['agg_method']}"
                 if params["aggregation_window"] > 1 else "no aggregation")
    variant = "Seasonal cycle removed" if deseasonalized else "Original"
    title = f"{variant}: {int(exceeded.sum())} exceedance days in {year}"
    subtitle = (f"{format_coord(lat, lon)} · p{params['percentile'] * 100:g}, ref {ref0}–{ref1}, {agg_label}, "
                f"boost {params['boosting_window']}")

    fig = base_year_figure(title, subtitle, ytitle="Anomaly (°C)" if deseasonalized else
                           "Daily max 2m temperature (°C)")
    if params["aggregation_window"] > 1:
        fig.add_trace(go.Scatter(x=DOYS, y=raw - ec.seasonal_cycle if deseasonalized else raw, mode="lines",
                                 line=dict(color="#b0b0b0", width=1), name="Raw daily",
                                 hovertemplate="%{y:.1f} °C"))
    fig.add_trace(go.Scatter(x=DOYS, y=synth_ec.analysis_agg[an_idx], mode="lines",
                             line=dict(color=SYNTH_COLOR, width=1.1), opacity=0.85, name="Synthetic (TemperatureModel)",
                             hovertemplate="%{y:.1f} °C"))
    if not deseasonalized:
        fig.add_trace(go.Scatter(x=DOYS, y=seasonal_cycle, mode="lines", line=dict(color="#555", width=1.2, dash="dot"),
                                 name="Seasonal cycle", hovertemplate="%{y:.1f} °C"))
    fig.add_trace(go.Scatter(x=DOYS, y=agg, mode="lines", line=dict(color="#1f77b4", width=1.4),
                             name="Aggregated" if params["aggregation_window"] > 1 else "Daily",
                             hovertemplate="%{y:.1f} °C"))
    fig.add_trace(go.Scatter(x=DOYS, y=ec.thresholds, mode="lines", line=dict(color="#d62728", width=1.6, dash="dash"),
                             name=f"p{params['percentile'] * 100:g} threshold", hovertemplate="%{y:.1f} °C"))
    fig.add_trace(go.Scatter(x=DOYS[exceeded], y=agg[exceeded], mode="markers",
                             marker=dict(color="#d62728", size=6), name="Exceedance", hoverinfo="skip"))
    return fig


def empty_overview(text):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=10, r=10, t=10, b=10),
                      annotations=[dict(text=text, showarrow=False, font=dict(size=13, color="#666"))])
    return fig


def monthly_percentage(exceedances):
    """% of exceedance days in each calendar month, pooled over all years; shape (12,)."""
    return np.array([100 * exceedances[:, start - 1:start - 1 + length].mean()
                     for start, length in zip(MONTH_STARTS, MONTH_LENGTHS)])


def overview_figure(ec, year, params, pct_max, month_pct_max, baseline=None, title_prefix=""):
    """Years × DOY exceedance grid, with the percentage of exceedance days per year alongside (x-axis 0–pct_max)
    and per calendar month, pooled over all years, below it (y-axis 0–month_pct_max).

    If *baseline* (an exceedance array of the same shape) is given, cells are coloured by how they changed
    relative to it: red = exceedance in both, green = new exceedance, black = no longer an exceedance.
    """
    y0, y1 = params["analysis_window"]
    years = np.arange(y0, y1 + 1)
    current = ec.exceedances
    pct = 100 * current.mean(axis=1)
    month_pct = monthly_percentage(current)

    # Cell categories: 0 none, 1 exceedance (in both), 2 new exceedance, 3 lost exceedance
    if baseline is None:
        category = current.astype(np.uint8)
    else:
        category = np.select([current & baseline, current & ~baseline, ~current & baseline], [1, 2, 3], 0)
    labels = np.array(["no exceedance", "exceedance", "new exceedance", "no longer an exceedance"])[category]
    # Each category gets an equal quarter of the colour scale (zmin/zmax put integers at quarter centres)
    colorscale = [[0, "#ffffff"], [0.25, "#ffffff"], [0.25, "#d62728"], [0.5, "#d62728"],
                  [0.5, "#2ca02c"], [0.75, "#2ca02c"], [0.75, "#000000"], [1, "#000000"]]

    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True, column_widths=[0.82, 0.18],
                        row_heights=[0.78, 0.22], horizontal_spacing=0.02, vertical_spacing=0.03,
                        specs=[[{}, {}], [{}, None]])
    fig.add_trace(go.Heatmap(x=DOYS, y=years, z=category, zmin=-0.5, zmax=3.5, showscale=False,
                             colorscale=colorscale, xgap=0, ygap=1, text=labels,
                             hovertemplate="%{y}, day %{x}: %{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=pct, y=years, orientation="h", marker_color="#d62728",
                         hovertemplate="%{y}: %{x:.1f}% of days<extra></extra>"), row=1, col=2)
    # Monthly bars span their month's days, so they line up with the grid columns above
    fig.add_trace(go.Bar(x=[start - 0.5 + length / 2 for start, length in zip(MONTH_STARTS, MONTH_LENGTHS)],
                         y=month_pct, width=[0.85 * length for length in MONTH_LENGTHS], marker_color="#d62728",
                         customdata=MONTH_LABELS,
                         hovertemplate="%{customdata}: %{y:.1f}% of days<extra></extra>"), row=2, col=1)
    # Rate expected if the analysis climate matched the reference period
    expected = 100 * (1 - params["percentile"])
    fig.add_vline(x=expected, line=dict(color="#555", dash="dot", width=1), row=1, col=2)
    fig.add_hline(y=expected, line=dict(color="#555", dash="dot", width=1), row=2, col=1)
    # Outline the year shown in the main plot
    fig.add_shape(type="rect", x0=0.5, x1=365.5, y0=year - 0.5, y1=year + 0.5,
                  line=dict(color="#1f77b4", width=2), row=1, col=1)

    fig.update_xaxes(tickvals=MONTH_STARTS, ticktext=MONTH_LABELS, range=[0.5, 365.5], showgrid=False, col=1)
    fig.update_xaxes(title=dict(text="% exceedance days", font=dict(size=11)), range=[0, pct_max], row=1, col=2)
    fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=1)
    fig.update_yaxes(title=dict(text="% days", font=dict(size=11)), range=[0, month_pct_max], row=2, col=1)
    # In-reference-period frequency should equal the expected rate; Brunner & Voigt (2024) show that
    # without removing the seasonal cycle a wide boosting window pushes it below
    r0, r1 = max(params["reference_window"][0], y0), min(params["reference_window"][1], y1)
    ref_note = (f"ref-period freq. {100 * ec.exceedances[r0 - y0:r1 - y0 + 1].mean():.2f}% (expected {expected:g}%)"
                if r0 <= r1 else f"expected rate {expected:g}%")
    change_note = None
    if baseline is not None:
        change_note = (f"vs original: <span style='color:#2ca02c'>■</span> {int((category == 2).sum())} new, "
                       f"■ {int((category == 3).sum())} no longer exceedances, "
                       f"<span style='color:#d62728'>■</span> {int((category == 1).sum())} in both")
    fig.update_layout(template="plotly_white", showlegend=False, bargap=0.15,
                      title=dict(text=f"{title_prefix}Exceedances {y0}–{y1} · {ref_note}", font=dict(size=13),
                                 subtitle=dict(text=change_note or "")),
                      margin=dict(l=50, r=15, t=55 if change_note else 40, b=40))
    return fig


def trend_figure(ec, params, pct_max, run_length, synth_ec):
    """Year (x) vs yearly % of exceedance days (y) with linear, Theil–Sen, NHPP and POT trend fits.

    The same fits on the TemperatureModel realization (synth_ec) are drawn dashed; its points are not shown.
    """
    y0, y1 = params["analysis_window"]
    years = np.arange(y0, y1 + 1)
    variant = "Seasonal cycle removed" if ec.seasonal_cycle is not None else "Original"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=yearly_percentage(ec.exceedances), mode="markers",
                             marker=dict(color="#d62728", size=6, opacity=0.7), name="Yearly % exceedance days",
                             hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    for source, exceedances, dash_style in (("", ec.exceedances, "solid"), ("Synthetic ", synth_ec.exceedances, "dash")):
        for fit in fit_all_trends(years, exceedances, run_length):
            color = next(c for prefix, c in TREND_COLORS.items() if fit.name.startswith(prefix))
            fig.add_trace(go.Scatter(x=years, y=fit.fitted, mode="lines",
                                     line=dict(color=color, width=2, dash=dash_style),
                                     name=f"{source}{fit.name}: {fit.summary}",
                                     hovertemplate=f"{source}{fit.name}<br>%{{x}}: %{{y:.2f}}%<extra></extra>"))
    fig.add_hline(y=100 * (1 - params["percentile"]), line=dict(color="#555", dash="dot", width=1))
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="% exceedance days", range=[0, pct_max])
    fig.update_layout(template="plotly_white", title=dict(text=f"{variant}: trend in exceedance frequency",
                                                          font=dict(size=13)),
                      margin=dict(l=55, r=15, t=40, b=40), hovermode="closest",
                      legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0, font=dict(size=11)))
    return fig


def params_to_values(p):
    """ModelParams -> {field name: value} for the editable form (rounded so the fields stay readable)."""
    values = {"intercept": round(float(p.intercept), 4), "trend": round(float(p.trend), 6)}
    for k, (s, c) in enumerate(zip(p.sin_coeffs, p.cos_coeffs), start=1):
        values[f"sin_{k}"], values[f"cos_{k}"] = round(float(s), 4), round(float(c), 4)
    for j, a in enumerate(p.ar_coeffs, start=1):
        values[f"ar_{j}"] = round(float(a), 4)
    values["innovation_std"] = round(float(p.innovation_std), 4)
    return values


def values_to_params(values, K, L):
    """Form values -> ModelParams; raises ValueError with a readable message if they define an invalid model."""
    missing = [name for name, v in values.items() if v is None]
    if missing:
        raise ValueError(f"Missing value for: {', '.join(missing)}")
    ar_coeffs = np.array([values[f"ar_{j}"] for j in range(1, L + 1)], dtype=float)
    if not ArmaProcess(np.r_[1, -ar_coeffs]).isstationary:
        raise ValueError(f"AR coefficients {ar_coeffs.tolist()} define a non-stationary process")
    if values["innovation_std"] <= 0:
        raise ValueError("Innovation std must be positive")
    return ModelParams(
        intercept=float(values["intercept"]),
        trend=float(values["trend"]),
        sin_coeffs=np.array([values[f"sin_{k}"] for k in range(1, K + 1)], dtype=float),
        cos_coeffs=np.array([values[f"cos_{k}"] for k in range(1, K + 1)], dtype=float),
        ar_coeffs=ar_coeffs,
        innovation_std=float(values["innovation_std"]),
    )


def param_label(name):
    if name == "intercept":
        return "Intercept (°C)"
    if name == "trend":
        return "Trend (°C/yr)"
    if name == "innovation_std":
        return "Innovation std (°C)"
    kind, idx = name.split("_")
    return {"sin": f"Harmonic {idx}: sin (°C)", "cos": f"Harmonic {idx}: cos (°C)", "ar": f"AR coefficient lag {idx}"}[kind]


def model_form(lat, lon, values, K, L):
    """Editable TemperatureModel parameters; a change (Enter / leaving the field) regenerates the synthetic data."""
    cell = {"padding": "2px 8px", "borderBottom": "1px solid #eee", "fontSize": "13px"}
    rows = [html.Tr([html.Td(param_label(name), style={**cell, "color": "#555"}),
                     html.Td(dcc.Input(id={"type": "model-param", "name": name}, type="number", value=value,
                                       step="any", debounce=True, style={"width": "190px"}), style=cell)])
            for name, value in values.items()]
    return html.Div([
        html.Div(f"TemperatureModel fitted at {format_coord(lat, lon)}",
                 style={"fontWeight": 600, "fontSize": "14px", "marginBottom": "2px"}),
        html.Div(f"Daily max 2m temperature {FIRST_YEAR}–{LAST_YEAR}: intercept + linear trend + {K} annual "
                 f"harmonics + AR({L}) noise. The synthetic series in the plots is one seeded realization of this "
                 "model, processed with the same settings as the data. Edit a value and press Enter (or leave the "
                 "field) to regenerate it; the random seed stays fixed, so only your edit changes the series. "
                 "“Resample” draws a new realization of the current model.",
                 style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"}),
        html.Table([html.Tbody(rows)], style={"borderCollapse": "collapse"}),
    ])


def derived_summary(p, edited, draw):
    noise_std = p.innovation_std * np.sqrt(ar_variance_factor(p.ar_coeffs))
    amplitudes = ", ".join(f"{a:.2f}" for a in p.harmonic_amplitudes)
    source = "edited" if edited else "fitted"
    return (f"Using {source} parameters, realization #{draw + 1} · trend {10 * p.trend:+.3f} °C/decade · "
            f"harmonic amplitudes {amplitudes} °C · stationary noise std {noise_std:.2f} °C")


def labelled(label, component):
    return html.Div([html.Div(label, style={"fontWeight": 600, "fontSize": "13px", "marginBottom": "2px"}),
                     component], style={"marginBottom": "18px"})


def year_marks(y0, y1, step=10):
    marks = {y: str(y) for y in range(y0, y1 + 1) if y % step == 0}
    marks.update({y0: str(y0), y1: str(y1)})
    return marks


def odd_marks(lo, hi):
    return {v: str(v) for v in range(lo, hi + 1, 2) if v in (1, 3, 5, 7, 11, 15, 21, 31)}


def controls_panel():
    slider = dict(updatemode="mouseup", tooltip={"placement": "bottom"})
    return html.Div(id="controls", style={"display": "none"}, children=[
        html.Div(id="mode-label", style={"fontSize": "13px", "marginBottom": "6px"}),
        html.Button("Follow cursor", id="unpin", n_clicks=0, style={"marginBottom": "18px"}),
        labelled("Displayed year", dcc.Slider(id="year", min=FIRST_YEAR, max=LAST_YEAR, step=1, value=LAST_YEAR,
                                              marks=year_marks(FIRST_YEAR, LAST_YEAR), **slider)),
        html.Hr(),
        html.Div("compute_exceedances parameters", style={"fontSize": "12px", "color": "#666", "marginBottom": "10px"}),
        labelled("Percentile", dcc.Slider(id="percentile", min=0.50, max=0.99, step=0.01,
                                          value=DEFAULTS["percentile"],
                                          marks={v: f"{v:g}" for v in (0.5, 0.6, 0.7, 0.8, 0.9, 0.99)}, **slider)),
        labelled("Reference window (years)",
                 dcc.RangeSlider(id="reference_window", min=FIRST_YEAR, max=LAST_YEAR, step=1,
                                 value=DEFAULTS["reference_window"], marks=year_marks(FIRST_YEAR, LAST_YEAR),
                                 allowCross=False, **slider)),
        labelled("Analysis window (years)",
                 dcc.RangeSlider(id="analysis_window", min=FIRST_YEAR, max=LAST_YEAR, step=1,
                                 value=DEFAULTS["analysis_window"], marks=year_marks(FIRST_YEAR, LAST_YEAR),
                                 allowCross=False, **slider)),
        labelled("Aggregation window (days)",
                 dcc.Slider(id="aggregation_window", min=1, max=31, step=2, value=DEFAULTS["aggregation_window"],
                            marks=odd_marks(1, 31), **slider)),
        labelled("Aggregation method",
                 dcc.RadioItems(id="agg_method", options=["mean", "max", "min"], value=DEFAULTS["agg_method"],
                                inline=True, inputStyle={"marginRight": "4px", "marginLeft": "10px"})),
        labelled("Boosting window (days)",
                 dcc.Slider(id="boosting_window", min=1, max=31, step=2, value=DEFAULTS["boosting_window"],
                            marks=odd_marks(1, 31), **slider)),
        html.Hr(),
        html.Div("Trend fits", style={"fontSize": "12px", "color": "#666", "marginBottom": "10px"}),
        labelled("POT declustering run length r (days)",
                 dcc.Slider(id="pot_run_length", min=1, max=10, step=1, value=DEFAULTS["pot_run_length"],
                            marks={v: str(v) for v in (1, 3, 5, 10)}, **slider)),
    ])


def create_app():
    ds = open_t2max_celsius()

    @lru_cache(maxsize=256)
    def get_point(lat, lon):
        """Full 1961–2020 point series as an in-memory Dataset (a few hundred KB)."""
        return select_point_subset(ds, lat, lon)[[VAR]].load()

    @lru_cache(maxsize=256)
    def get_fit(lat, lon):
        """TemperatureModel fitted to the point's full series."""
        model = TemperatureModel()
        model.fit(get_point(lat, lon)[VAR].values, L=MODEL_L, K=MODEL_K, with_autocorrelation=True)
        return model

    @lru_cache(maxsize=512)
    def get_synthetic(lat, lon, values_key, draw):
        """Realization number *draw* of the model with the given parameter values, as a Dataset shaped like the point.

        The seed depends only on the grid cell and the draw number, so slider changes keep the realization and
        parameter edits change it only through the parameters (same random innovations); "Resample" bumps draw.
        """
        model = copy.copy(get_fit(lat, lon))
        model.model_params = values_to_params(dict(values_key), model.K, model.L)
        cell_seed = int(round((lat + 90) * 4)) * len(GRID_LONS) + int(round((lon + 180) * 4))
        np.random.seed((cell_seed + draw * len(GRID_LATS) * len(GRID_LONS)) % 2**32)
        synthetic = get_point(lat, lon).copy(deep=True)
        synthetic[VAR].values = model.generate(with_autocorrelation=True)
        return synthetic

    def year_values(point, year):
        return point[VAR].sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values

    app = dash.Dash(__name__, title="Global Temperature Viewer")
    app.layout = html.Div([
        html.Div([
            controls_panel(),
            html.Div(id="controls-hint", children="Click a point on the map to select it and show the "
                                                  "year and threshold controls.",
                     style={"fontSize": "13px", "color": "#666"}),
        ], style={"flex": "0 0 300px", "padding": "12px", "overflowY": "auto", **STICKY}),
        *[html.Div([
            dcc.Graph(id=f"selection-{variant}", config={"displayModeBar": False}, style={"height": "44vh"},
                      figure=base_year_figure("Hover over the map") if variant == "original" else
                      empty_overview("Click a point on the map to compare with the seasonal cycle removed")),
            dcc.Graph(id=f"overview-{variant}", config={"displayModeBar": False}, style={"height": "60vh"},
                      figure=empty_overview("Click a point on the map to see exceedances across all years")),
            dcc.Graph(id=f"synth-overview-{variant}", config={"displayModeBar": False}, style={"height": "60vh"},
                      figure=empty_overview("Click a point on the map to see exceedances of the synthetic model")),
            dcc.Graph(id=f"trends-{variant}", config={"displayModeBar": False}, style={"height": "60vh"},
                      figure=empty_overview("Click a point on the map to fit trends")),
        ], style={"flex": "4 1 0", "minWidth": 0}) for variant in VARIANTS],
        html.Div([
            dcc.Graph(id="map", figure=build_map_figure(), clear_on_unhover=False,
                      config={"displayModeBar": False}, style={"height": "50vh"}),
            html.Div([
                html.Div(id="model-form", children=MODEL_HINT),
                html.Div([
                    html.Button("Reset to fitted values", id="reset-model", n_clicks=0, style={"marginRight": "8px"}),
                    html.Button("Resample synthetic data", id="resample-model", n_clicks=0),
                ], id="model-buttons", style={"display": "none", "marginTop": "8px"}),
                html.Div(id="model-status", style={"fontSize": "12px", "marginTop": "6px"}),
            ], style={"padding": "4px 12px"}),
        ], style={"flex": "5 1 0", "minWidth": 0, "overflowY": "auto", **STICKY}),
        dcc.Store(id="pinned", data=None),
        dcc.Store(id="active-cell", data=None),
        # {"cell": [lat, lon], "values": {param name: value}} of the model used for the synthetic data
        dcc.Store(id="model-store", data=None),
    ], style={"display": "flex", "alignItems": "flex-start", "fontFamily": "sans-serif", "gap": "8px"})

    @app.callback(Output("pinned", "data"),
                  Input("map", "clickData"), Input("unpin", "n_clicks"),
                  prevent_initial_call=True)
    def update_pin(click_data, _):
        if dash.ctx.triggered_id == "unpin" or not click_data:
            return None
        point = click_data["points"][0]
        return snap_to_grid(point["y"], point["x"])

    @app.callback(Output("map", "figure"), Output("controls", "style"), Output("controls-hint", "style"),
                  Output("mode-label", "children"), Output("model-buttons", "style"),
                  Input("pinned", "data"))
    def show_pin(pinned):
        patch = Patch()
        patch["data"][PIN_TRACE]["x"] = [pinned[1]] if pinned else []
        patch["data"][PIN_TRACE]["y"] = [pinned[0]] if pinned else []
        hint_style = {"fontSize": "13px", "color": "#666", "display": "none" if pinned else "block"}
        label = f"Selected: {format_coord(*pinned)} — click the map to select another point" if pinned else ""
        buttons_style = {"display": "block" if pinned else "none", "marginTop": "8px"}
        return patch, {"display": "block" if pinned else "none"}, hint_style, label, buttons_style

    @app.callback(Output("model-form", "children"), Output("model-store", "data"),
                  Input("pinned", "data"), Input("reset-model", "n_clicks"))
    def render_model_form(pinned, _):
        """Fit the model at the selected point and fill the form (and the store) with the fitted values."""
        if not pinned:
            return MODEL_HINT, None
        lat, lon = pinned
        model = get_fit(lat, lon)
        values = params_to_values(model.model_params)
        return model_form(lat, lon, values, model.K, model.L), {"cell": [lat, lon], "values": values}

    @app.callback(Output("model-store", "data", allow_duplicate=True),
                  Input({"type": "model-param", "name": ALL}, "value"),
                  State({"type": "model-param", "name": ALL}, "id"), State("pinned", "data"),
                  prevent_initial_call=True)
    def edit_model(values, ids, pinned):
        """Hand edits in the form become the parameters of the synthetic model."""
        if not pinned or not ids:
            raise PreventUpdate
        return {"cell": list(pinned), "values": {i["name"]: v for i, v in zip(ids, values)}}

    @app.callback(Output("year", "min"), Output("year", "max"), Output("year", "value"), Output("year", "marks"),
                  Input("analysis_window", "value"), Input("year", "value"))
    def limit_year_to_analysis_window(analysis_window, year):
        # Exceedances only exist inside the analysis window, so the displayed year must lie in it
        y0, y1 = analysis_window
        return y0, y1, int(np.clip(year, y0, y1)), year_marks(y0, y1)

    # Decide hover-vs-pinned in the browser: a server callback that returned "no update" for hovers
    # while pinned would race with (and discard) the in-flight response for the pin itself.
    app.clientside_callback(
        """
        function(hoverData, pinned) {
            const triggered = dash_clientside.callback_context.triggered.map(t => t.prop_id);
            if (pinned) {
                return triggered.includes("pinned.data") ? {lat: pinned[0], lon: pinned[1], pinned: true}
                                                         : dash_clientside.no_update;
            }
            if (!hoverData) { return dash_clientside.no_update; }
            const p = hoverData.points[0];
            return {lat: p.y, lon: p.x, pinned: false};
        }
        """,
        Output("active-cell", "data"),
        Input("map", "hoverData"), Input("pinned", "data"),
    )

    param_ids = ["percentile", "boosting_window", "aggregation_window", "agg_method",
                 "reference_window", "analysis_window"]

    @app.callback(*[Output(f"{kind}-{variant}", "figure") for variant in VARIANTS
                    for kind in ("selection", "overview", "synth-overview", "trends")],
                  Output("model-status", "children"),
                  Input("active-cell", "data"), Input("year", "value"), Input("pot_run_length", "value"),
                  Input("model-store", "data"), Input("resample-model", "n_clicks"),
                  *[Input(p, "value") for p in param_ids])
    def show_selection(active, year, pot_run_length, model_store, draw, *param_values):
        if not active:
            raise PreventUpdate
        lat, lon = snap_to_grid(active["lat"], active["lon"])
        point = get_point(lat, lon)
        if not active["pinned"]:
            hint = empty_overview("Click a point on the map to see exceedances across all years")
            synth_hint = empty_overview("Click a point on the map to see exceedances of the synthetic model")
            trend_hint = empty_overview("Click a point on the map to fit trends")
            return (preview_figure(year_values(point, LAST_YEAR), lat, lon, LAST_YEAR), hint, synth_hint, trend_hint,
                    empty_overview("Click a point on the map to compare with the seasonal cycle removed"), hint,
                    synth_hint, trend_hint, "")

        params = dict(zip(param_ids, param_values))
        year = int(np.clip(year, *params["analysis_window"]))
        # Model parameters: the (possibly hand-edited) values in the store if they belong to this cell, else the fit
        fit = get_fit(lat, lon)
        fitted_values = params_to_values(fit.model_params)
        values = model_store["values"] if model_store and model_store["cell"] == [lat, lon] else fitted_values
        try:
            model_params = values_to_params(values, fit.K, fit.L)
        except (ValueError, KeyError) as err:
            n_figures = len(VARIANTS) * 4
            return (*[dash.no_update] * n_figures,
                    html.Span(f"Invalid parameters, plots not updated: {err}", style={"color": "#d62728"}))
        synthetic = get_synthetic(lat, lon, tuple(sorted(values.items())), draw or 0)

        def run(data, variant):
            ec = ExceedancesCalculator()
            ec.compute_exceedances(
                data=data,
                percentile=params["percentile"],
                boosting_window=params["boosting_window"],
                aggregation_window=params["aggregation_window"],
                agg_method=params["agg_method"],
                reference_window=full_years(params["reference_window"]),
                analysis_window=full_years(params["analysis_window"]),
                subtract_seasonal_cycle=(variant == "deseasonalized"),
            )
            return ec

        calculators = {variant: run(point, variant) for variant in VARIANTS}
        synth_calculators = {variant: run(synthetic, variant) for variant in VARIANTS}
        seasonal_cycle = calculators["deseasonalized"].seasonal_cycle
        raw = year_values(point, year)
        # Same x-range for both yearly-rate bar charts so the variants are directly comparable
        pct_max = 1.05 * max(100 * ec.exceedances.mean(axis=1).max()
                             for ec in [*calculators.values(), *synth_calculators.values()])
        pct_max = max(pct_max, 1.1 * 100 * (1 - params["percentile"]))
        # Likewise one y-range for all four monthly bar charts
        month_pct_max = 1.1 * max(monthly_percentage(ec.exceedances).max()
                                  for ec in [*calculators.values(), *synth_calculators.values()])
        month_pct_max = max(month_pct_max, 1.1 * 100 * (1 - params["percentile"]))
        figures = []
        for variant in VARIANTS:
            figures += [analysis_figure(raw, calculators[variant], lat, lon, year, params, seasonal_cycle,
                                        synth_calculators[variant]),
                        overview_figure(calculators[variant], year, params, pct_max, month_pct_max,
                                        baseline=calculators["original"].exceedances
                                        if variant == "deseasonalized" else None),
                        # Synthetic grid: in the deseasonalized column, changes are relative to the synthetic original
                        overview_figure(synth_calculators[variant], year, params, pct_max, month_pct_max,
                                        baseline=synth_calculators["original"].exceedances
                                        if variant == "deseasonalized" else None,
                                        title_prefix="Synthetic: "),
                        trend_figure(calculators[variant], params, pct_max, pot_run_length,
                                     synth_calculators[variant])]
        status = html.Span(derived_summary(model_params, edited=values != fitted_values, draw=draw or 0),
                           style={"color": "#555"})
        return (*figures, status)

    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8051)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    app = create_app()
    print(f"\nStarting app at http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False)
