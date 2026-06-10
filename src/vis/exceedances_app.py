#!/usr/bin/env python3
"""
Interactive exceedances explorer.

Tab 1 – ERA5 Exceedances
    Global heatmap of mean seasonal exceedances from experiment zarr stores.
    Hover over any grid cell to see a time series in the right-hand panel.

Tab 2 – Synthetic (optional, shown when --synthetic-zarr is provided)
    Time series of annual/seasonal exceedance counts for 8 fixed locations
    across 5 synthetic warming slopes × 5 noise variances.
    Left panel: multi-line time series (one line per location) with optional
    trend overlays.  Right panel: detected OLS slope per location as a bar
    chart so the known input warming rate can be compared against the
    recovered exceedance trend.

Example
-------
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate

    python src/vis/exceedances_app.py \\
        --experiments-dir experiments/ \\
        --synthetic-zarr  data/synthetic/synthetic_extremes.zarr
"""
import argparse
import os
import re
from datetime import timedelta

import cftime
import numpy as np
import xarray as xr
import zarr

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Season definitions (shared)
# ---------------------------------------------------------------------------

SEASONS = {
    "Annual": dict(months=list(range(1, 13)), label="Annual",
                   note=""),
    "DJF":    dict(months=[12, 1, 2],          label="Winter (DJF)",
                   note="December belongs to year N-1. X-axis shows year N."),
    "MAM":    dict(months=[3, 4, 5],           label="Spring (MAM)",
                   note=""),
    "JJA":    dict(months=[6, 7, 8],           label="Summer (JJA)",
                   note=""),
    "SON":    dict(months=[9, 10, 11],         label="Autumn (SON)",
                   note=""),
}

# Mapping season key → index in seasonal_counts dim-2 (DJF=0 … SON=3)
_SYN_SEASON_IDX = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}

# Distinct colours for the 8 synthetic locations
_LOC_COLOURS = [
    "#f97316", "#60a5fa", "#4ade80", "#e879f9",
    "#facc15", "#f87171", "#34d399", "#a78bfa",
]

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

_DARK   = "#0e1117"
_GRID   = "#2a2a2a"
_ACCENT = "#f97316"


# ---------------------------------------------------------------------------
# ERA5 tab: discovery
# ---------------------------------------------------------------------------

def _discover_experiments(experiments_dir: str) -> dict[str, str]:
    found = {}
    for root, dirs, _ in os.walk(experiments_dir):
        for d in sorted(dirs):
            if d.startswith("exceedances_") and d.endswith(".zarr"):
                zarr_path = os.path.join(root, d)
                label = _make_label(root, d)
                found[label] = zarr_path
    return found


def _make_label(exp_dir: str, zarr_name: str) -> str:
    dirname = os.path.basename(exp_dir)
    perc_str = zarr_name.removeprefix("exceedances_").removesuffix(".zarr")
    try:
        perc = float(perc_str.replace("_", "."))
        perc_label = f"p={perc:.2f}"
    except ValueError:
        perc_label = perc_str
    parts = []
    for pattern, key in [
        (r"ref(\d{4}-\d{4})", "ref"),
        (r"an(\d{4}-\d{4})",  "an"),
        (r"agg(\d+\w+)",      "agg"),
        (r"boost(\d+)",       "boost"),
    ]:
        m = re.search(pattern, dirname)
        if m:
            parts.append(f"{key}={m.group(1)}")
    return " | ".join([perc_label] + parts) if parts else f"{perc_label} | {dirname}"


# ---------------------------------------------------------------------------
# ERA5 tab: precomputation
# ---------------------------------------------------------------------------

def _load_or_compute_monthly(exc_path: str) -> tuple:
    """Return (years, lat_vals, lon_vals, monthly) from cache or zarr.

    monthly shape: (n_years, 12, nlat, nlon), dtype int8.
    Axis 1 is 0-indexed month (0 = Jan, 11 = Dec).
    """
    cache_path = os.path.join(exc_path, "_monthly_cache.npz")

    if os.path.exists(cache_path):
        print(f"  Loading cache: {cache_path}")
        d = np.load(cache_path)
        return d["years"], d["lat_vals"], d["lon_vals"], d["monthly"]

    print(f"  Computing monthly sums: {exc_path}")
    ds = xr.open_zarr(exc_path, consolidated=False)
    exc = ds["data"]

    lat_vals = exc.latitude.values.astype(np.float32)
    lon_vals = exc.longitude.values.astype(np.float32)
    nlat, nlon = len(lat_vals), len(lon_vals)

    time_vals = exc.time.values
    epoch = cftime.DatetimeNoLeap(1900, 1, 1)
    time_cf = [epoch + timedelta(days=int(d)) for d in time_vals]
    day_years  = np.array([t.year  for t in time_cf], dtype=np.int32)
    day_months = np.array([t.month for t in time_cf], dtype=np.int8)

    unique_years = np.unique(day_years)
    monthly = np.zeros((len(unique_years), 12, nlat, nlon), dtype=np.int8)

    for i, yr in enumerate(tqdm(unique_years, desc="    years", leave=False)):
        yr_mask = day_years == yr
        yr_data = exc.isel(time=yr_mask).values
        yr_months = day_months[yr_mask]
        for m in range(1, 13):
            m_mask = yr_months == m
            if m_mask.any():
                monthly[i, m - 1] = yr_data[m_mask].sum(axis=0).astype(np.int8)

    np.savez_compressed(cache_path, years=unique_years,
                        lat_vals=lat_vals, lon_vals=lon_vals, monthly=monthly)
    print(f"  Cache saved: {cache_path}")
    return unique_years, lat_vals, lon_vals, monthly


def compute_seasonal(years: np.ndarray, monthly: np.ndarray,
                     season_key: str) -> tuple[np.ndarray, np.ndarray]:
    if season_key == "Annual":
        return years, monthly.sum(axis=1).astype(np.int16)

    months = SEASONS[season_key]["months"]

    if season_key == "DJF":
        sums = (
            monthly[:-1, 11].astype(np.int16)
            + monthly[1:,  0].astype(np.int16)
            + monthly[1:,  1].astype(np.int16)
        )
        return years[1:], sums
    else:
        idxs = [m - 1 for m in months]
        sums = monthly[:, idxs, :, :].sum(axis=1).astype(np.int16)
        return years, sums


# ---------------------------------------------------------------------------
# ERA5 tab: figure helpers
# ---------------------------------------------------------------------------

def _regression_slope(s_years: np.ndarray, sums: np.ndarray) -> np.ndarray:
    x = s_years.astype(np.float64) - s_years.mean()
    denom = float((x ** 2).sum())
    numer = (x[:, None, None] * sums.astype(np.float64)).sum(axis=0)
    return (numer / denom).astype(np.float32)


def _map_figure(s_years, sums, lat_vals, lon_vals,
                season_key: str, display_mode: str) -> go.Figure:
    season_label = SEASONS[season_key]["label"]

    if display_mode == "mean":
        z          = sums.mean(axis=0)
        colorscale = "YlOrRd"
        zmid       = None
        cb_title   = "Mean days"
        hover_fmt  = "lon: %{x:.2f}°  lat: %{y:.2f}°<br>mean: %{z:.1f} days<extra></extra>"
        title      = f"Mean exceedance-days — {season_label}"
    else:
        z          = _regression_slope(s_years, sums)
        colorscale = "RdBu_r"
        zmid       = 0.0
        cb_title   = "Slope (days/yr)"
        hover_fmt  = "lon: %{x:.2f}°  lat: %{y:.2f}°<br>slope: %{z:.3f} days/yr<extra></extra>"
        title      = f"Trend slope — {season_label}"

    heatmap_kw = dict(
        z=z, x=lon_vals, y=lat_vals,
        colorscale=colorscale,
        colorbar=dict(title=cb_title, thickness=14, len=0.75),
        hovertemplate=hover_fmt,
    )
    if zmid is not None:
        heatmap_kw["zmid"] = zmid

    fig = go.Figure(go.Heatmap(**heatmap_kw))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Longitude", showgrid=False),
        yaxis=dict(title="Latitude", showgrid=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=60, r=10, t=50, b=50),
        plot_bgcolor=_DARK, paper_bgcolor=_DARK, font=dict(color="white"),
    )
    return fig


def _empty_ts_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Hover over the map",
        xaxis_title="Year", yaxis_title="Exceedance-days",
        plot_bgcolor=_DARK, paper_bgcolor=_DARK, font=dict(color="white"),
        margin=dict(l=55, r=15, t=50, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Synthetic tab: data loading
# ---------------------------------------------------------------------------

def _load_synthetic(path: str) -> dict | None:
    if not os.path.exists(path):
        print(f"  Synthetic zarr not found: {path}")
        return None
    print(f"  Loading synthetic data: {path}")
    grp = zarr.open_group(path, mode="r")
    return dict(
        slopes          = grp["slope"][:],
        variances       = grp["variance"][:],
        locations       = [b.decode() for b in grp["location"][:]],
        years           = grp["year"][:],
        annual_counts   = grp["annual_counts"][:],    # (sl, var, loc, yr)
        seasonal_counts = grp["seasonal_counts"][:],  # (sl, var, seas, loc, yr)
        annual_trend    = grp["annual_trend"][:],     # (sl, var, loc, 2)
        seasonal_trend  = grp["seasonal_trend"][:],   # (sl, var, seas, loc, 2)
    )


# ---------------------------------------------------------------------------
# Synthetic tab: figure helpers
# ---------------------------------------------------------------------------

def _syn_counts(sd: dict, si: int, vi: int, season_key: str) -> np.ndarray:
    """Return exceedance counts of shape (n_loc, n_years) for given indices."""
    if season_key == "Annual":
        return sd["annual_counts"][si, vi]          # (n_loc, n_years)
    else:
        s_idx = _SYN_SEASON_IDX[season_key]
        return sd["seasonal_counts"][si, vi, s_idx]  # (n_loc, n_years)


def _syn_trend(sd: dict, si: int, vi: int, season_key: str) -> np.ndarray:
    """Return OLS [slope, intercept] of shape (n_loc, 2)."""
    if season_key == "Annual":
        return sd["annual_trend"][si, vi]            # (n_loc, 2)
    else:
        s_idx = _SYN_SEASON_IDX[season_key]
        return sd["seasonal_trend"][si, vi, s_idx]   # (n_loc, 2)


def _syn_timeseries_figure(sd: dict, si: int, vi: int,
                            season_key: str, show_trend: bool) -> go.Figure:
    counts   = _syn_counts(sd, si, vi, season_key)  # (n_loc, n_years)
    trend    = _syn_trend(sd, si, vi, season_key)   # (n_loc, 2)
    years    = sd["years"].astype(float)
    locs     = sd["locations"]
    slope_v  = sd["slopes"][si]
    var_v    = sd["variances"][vi]
    sea_lbl  = SEASONS[season_key]["label"]

    traces = []
    for li, (loc, col) in enumerate(zip(locs, _LOC_COLOURS)):
        traces.append(go.Scatter(
            x=sd["years"], y=counts[li],
            mode="lines+markers",
            name=loc,
            marker=dict(size=3, color=col),
            line=dict(color=col, width=1.5),
            legendgroup=loc,
        ))
        if show_trend:
            sl, ic = float(trend[li, 0]), float(trend[li, 1])
            t_line = sl * years + ic
            traces.append(go.Scatter(
                x=sd["years"], y=t_line,
                mode="lines",
                name=f"{loc} trend ({sl:+.3f} d/yr)",
                line=dict(color=col, width=1.5, dash="dash"),
                legendgroup=loc,
                showlegend=True,
            ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{sea_lbl} exceedance-days  |  slope={slope_v:.3f} °C/yr  "
              f"variance={var_v:.2f} °C²",
        xaxis=dict(title="Year", showgrid=True, gridcolor=_GRID),
        yaxis=dict(title="Exceedance-days", showgrid=True, gridcolor=_GRID),
        legend=dict(orientation="v", font=dict(size=10)),
        plot_bgcolor=_DARK, paper_bgcolor=_DARK, font=dict(color="white"),
        margin=dict(l=60, r=10, t=60, b=50),
    )
    return fig


def _syn_slope_bar_figure(sd: dict, si: int, vi: int, season_key: str) -> go.Figure:
    trend   = _syn_trend(sd, si, vi, season_key)   # (n_loc, 2)
    locs    = sd["locations"]
    sea_lbl = SEASONS[season_key]["label"]
    true_slope = float(sd["slopes"][si])

    slopes_det = trend[:, 0].astype(float)   # detected exceedance slope (d/yr)

    fig = go.Figure(go.Bar(
        x=slopes_det,
        y=locs,
        orientation="h",
        marker=dict(color=_LOC_COLOURS[:len(locs)]),
        text=[f"{s:+.3f}" for s in slopes_det],
        textposition="outside",
    ))
    fig.add_vline(
        x=0.0,
        line=dict(color="white", width=1, dash="dot"),
    )
    fig.update_layout(
        title=dict(
            text=(f"Detected trend ({sea_lbl})<br>"
                  f"<sup>True warming: {true_slope:.3f} °C/yr</sup>"),
            font=dict(size=13),
        ),
        xaxis=dict(title="Exceedance slope (days/yr)",
                   showgrid=True, gridcolor=_GRID, zeroline=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor=_DARK, paper_bgcolor=_DARK, font=dict(color="white"),
        margin=dict(l=130, r=60, t=80, b=50),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app(experiments: dict[str, str], data: dict,
              syn_data: dict | None) -> dash.Dash:

    labels     = list(experiments.keys())
    first_path = experiments[labels[0]]
    years0, lat_vals0, lon_vals0, monthly0 = data[first_path]
    s_years0, sums0 = compute_seasonal(years0, monthly0, "Annual")

    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    season_options = [{"label": v["label"], "value": k} for k, v in SEASONS.items()]

    # ---- ERA5 tab content ----
    era5_controls = html.Div(
        style={"display": "flex", "alignItems": "center",
               "padding": "8px 16px", "gap": "20px", "flexWrap": "wrap"},
        children=[
            html.Label("Experiment:", style={"color": "#aaa", "flex": "0 0 auto"}),
            dcc.Dropdown(
                id="experiment-dropdown",
                options=[{"label": l, "value": l} for l in labels],
                value=labels[0],
                clearable=False,
                style={"flex": "1", "minWidth": "260px", "maxWidth": "560px",
                       "color": "black"},
            ),
            html.Label("Season:", style={"color": "#aaa", "flex": "0 0 auto"}),
            dcc.RadioItems(
                id="season-radio",
                options=season_options,
                value="Annual",
                inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "16px", "color": "white",
                            "cursor": "pointer"},
            ),
            html.Label("Map shows:", style={"color": "#aaa", "flex": "0 0 auto"}),
            dcc.RadioItems(
                id="display-mode",
                options=[
                    {"label": "Mean",          "value": "mean"},
                    {"label": "Trend (slope)", "value": "slope"},
                ],
                value="mean",
                inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "16px", "color": "white",
                            "cursor": "pointer"},
            ),
        ],
    )

    era5_tab = dcc.Tab(
        label="ERA5 Exceedances",
        style={"backgroundColor": _DARK, "color": "white"},
        selected_style={"backgroundColor": "#1e2230", "color": "white"},
        children=[
            era5_controls,
            html.Div(
                id="season-note",
                style={"color": "#facc15", "fontSize": "12px",
                       "padding": "0 16px 4px", "display": "none"},
            ),
            html.Div(
                style={"display": "flex", "flex": "1", "overflow": "hidden",
                       "height": "calc(100vh - 130px)"},
                children=[
                    dcc.Graph(
                        id="world-map",
                        figure=_map_figure(s_years0, sums0, lat_vals0, lon_vals0,
                                           "Annual", "mean"),
                        style={"flex": "7", "minWidth": 0},
                        config={"scrollZoom": True},
                    ),
                    dcc.Graph(
                        id="timeseries",
                        figure=_empty_ts_figure(),
                        style={"flex": "3", "minWidth": 0},
                        config={"displayModeBar": False},
                    ),
                ],
            ),
        ],
    )

    # ---- Synthetic tab content ----
    if syn_data is not None:
        sd = syn_data
        slope_options = [
            {"label": f"{v:.3f} °C/yr", "value": i}
            for i, v in enumerate(sd["slopes"])
        ]
        var_options = [
            {"label": f"{v:.2f} °C²", "value": i}
            for i, v in enumerate(sd["variances"])
        ]

        syn_controls = html.Div(
            style={"display": "flex", "alignItems": "center",
                   "padding": "8px 16px", "gap": "20px", "flexWrap": "wrap"},
            children=[
                html.Label("Warming slope:", style={"color": "#aaa", "flex": "0 0 auto"}),
                dcc.Dropdown(
                    id="syn-slope-dropdown",
                    options=slope_options,
                    value=0,
                    clearable=False,
                    style={"width": "160px", "color": "black"},
                ),
                html.Label("Noise variance:", style={"color": "#aaa", "flex": "0 0 auto"}),
                dcc.Dropdown(
                    id="syn-var-dropdown",
                    options=var_options,
                    value=0,
                    clearable=False,
                    style={"width": "140px", "color": "black"},
                ),
                html.Label("Season:", style={"color": "#aaa", "flex": "0 0 auto"}),
                dcc.RadioItems(
                    id="syn-season-radio",
                    options=season_options,
                    value="Annual",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "16px", "color": "white",
                                "cursor": "pointer"},
                ),
                dcc.Checklist(
                    id="syn-show-trend",
                    options=[{"label": "Show trend lines", "value": "trend"}],
                    value=["trend"],
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"color": "white", "cursor": "pointer"},
                ),
            ],
        )

        syn_tab = dcc.Tab(
            label="Synthetic",
            style={"backgroundColor": _DARK, "color": "white"},
            selected_style={"backgroundColor": "#1e2230", "color": "white"},
            children=[
                syn_controls,
                html.Div(
                    style={"display": "flex", "flex": "1", "overflow": "hidden",
                           "height": "calc(100vh - 110px)"},
                    children=[
                        dcc.Graph(
                            id="syn-timeseries",
                            figure=_syn_timeseries_figure(sd, 0, 0, "Annual", True),
                            style={"flex": "7", "minWidth": 0},
                            config={"displayModeBar": False},
                        ),
                        dcc.Graph(
                            id="syn-slope-bar",
                            figure=_syn_slope_bar_figure(sd, 0, 0, "Annual"),
                            style={"flex": "3", "minWidth": 0},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        )
        tabs = [era5_tab, syn_tab]
    else:
        tabs = [era5_tab]

    app.layout = html.Div(
        style={"backgroundColor": _DARK, "height": "100vh",
               "display": "flex", "flexDirection": "column",
               "fontFamily": "sans-serif"},
        children=[
            html.H2("Exceedances Explorer",
                    style={"color": "white", "margin": "8px 16px 4px",
                           "flex": "0 0 auto"}),
            dcc.Tabs(
                id="main-tabs",
                value="tab-era5",
                style={"flex": "0 0 auto"},
                colors={"border": _GRID, "primary": _ACCENT,
                        "background": _DARK},
                children=[
                    dcc.Tab(label="ERA5 Exceedances", value="tab-era5",
                            style={"backgroundColor": _DARK, "color": "white"},
                            selected_style={"backgroundColor": "#1e2230",
                                            "color": "white"},
                            children=era5_tab.children),
                ] + ([
                    dcc.Tab(label="Synthetic", value="tab-syn",
                            style={"backgroundColor": _DARK, "color": "white"},
                            selected_style={"backgroundColor": "#1e2230",
                                            "color": "white"},
                            children=syn_tab.children),
                ] if syn_data is not None else []),
            ),
        ],
    )

    # ---- ERA5 callbacks ----

    @app.callback(
        Output("world-map",   "figure"),
        Output("season-note", "children"),
        Output("season-note", "style"),
        Input("experiment-dropdown", "value"),
        Input("season-radio",        "value"),
        Input("display-mode",        "value"),
    )
    def update_map(label, season_key, display_mode):
        path = experiments[label]
        years, lat_vals, lon_vals, monthly = data[path]
        s_years, sums = compute_seasonal(years, monthly, season_key)

        note_text  = SEASONS[season_key]["note"]
        note_style = {"color": "#facc15", "fontSize": "12px",
                      "padding": "0 16px 4px",
                      "display": "block" if note_text else "none"}

        return (_map_figure(s_years, sums, lat_vals, lon_vals,
                            season_key, display_mode),
                note_text, note_style)

    @app.callback(
        Output("timeseries", "figure"),
        Input("world-map",   "hoverData"),
        State("experiment-dropdown", "value"),
        State("season-radio",        "value"),
        State("display-mode",        "value"),
    )
    def update_timeseries(hover_data, label, season_key, display_mode):
        if hover_data is None:
            return _empty_ts_figure()

        path = experiments[label]
        years, lat_vals, lon_vals, monthly = data[path]
        s_years, sums = compute_seasonal(years, monthly, season_key)

        pt      = hover_data["points"][0]
        lat_idx = int(np.argmin(np.abs(lat_vals - pt["y"])))
        lon_idx = int(np.argmin(np.abs(lon_vals - pt["x"])))
        ts      = sums[:, lat_idx, lon_idx].astype(np.float64)

        lat_label = f"{abs(lat_vals[lat_idx]):.2f}°{'N' if lat_vals[lat_idx] >= 0 else 'S'}"
        lon_label = f"{abs(lon_vals[lon_idx]):.2f}°{'E' if lon_vals[lon_idx] >= 0 else 'W'}"

        season_meta = SEASONS[season_key]

        if season_key == "DJF":
            x_vals  = [f"{y-1}/{y}" for y in s_years]
            x_title = "Winter (Dec year N-1 / Jan–Feb year N)"
        else:
            x_vals  = s_years
            x_title = "Year"

        traces = [go.Scatter(
            x=x_vals, y=ts,
            mode="lines+markers", name="Exceedances",
            marker=dict(size=5, color=_ACCENT),
            line=dict(color=_ACCENT, width=1.5),
        )]

        if display_mode == "slope":
            x_num  = s_years.astype(np.float64)
            x_c    = x_num - x_num.mean()
            slope  = float((x_c * ts).sum() / (x_c ** 2).sum())
            interc = ts.mean() - slope * x_num.mean()
            trend  = slope * x_num + interc
            traces.append(go.Scatter(
                x=x_vals, y=trend,
                mode="lines", name=f"Slope: {slope:+.3f} days/yr",
                line=dict(color="#60a5fa", width=2, dash="dash"),
            ))

        subtitle = season_meta["note"]
        title_text = (
            f"{lat_label}, {lon_label} — {season_meta['label']}"
            + (f"<br><sup style='color:#facc15'>{subtitle}</sup>" if subtitle else "")
        )

        fig = go.Figure(traces)
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=12)),
            xaxis_title=x_title,
            yaxis_title="Exceedance-days",
            legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
            plot_bgcolor=_DARK, paper_bgcolor=_DARK, font=dict(color="white"),
            margin=dict(l=55, r=15, t=60, b=70),
            xaxis=dict(showgrid=True, gridcolor=_GRID, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor=_GRID),
        )
        return fig

    # ---- Synthetic callbacks ----

    if syn_data is not None:
        sd = syn_data

        @app.callback(
            Output("syn-timeseries", "figure"),
            Output("syn-slope-bar",  "figure"),
            Input("syn-slope-dropdown", "value"),
            Input("syn-var-dropdown",   "value"),
            Input("syn-season-radio",   "value"),
            Input("syn-show-trend",     "value"),
        )
        def update_synthetic(si, vi, season_key, show_trend_val):
            show_trend = "trend" in (show_trend_val or [])
            ts_fig  = _syn_timeseries_figure(sd, int(si), int(vi),
                                             season_key, show_trend)
            bar_fig = _syn_slope_bar_figure(sd, int(si), int(vi), season_key)
            return ts_fig, bar_fig

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiments-dir", default="experiments/",
                   help="Directory to scan for experiment results (default: experiments/)")
    p.add_argument("--synthetic-zarr",
                   default="data/synthetic/synthetic_extremes.zarr",
                   help="Path to synthetic_extremes.zarr "
                        "(default: data/synthetic/synthetic_extremes.zarr). "
                        "Tab is hidden if the file does not exist.")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="127.0.0.1")
    return p


def main() -> None:
    args = build_parser().parse_args()

    experiments = _discover_experiments(args.experiments_dir)
    if not experiments:
        print(f"No exceedances zarr stores found under {args.experiments_dir}")
        return

    print(f"Found {len(experiments)} experiment(s):")
    for label, path in experiments.items():
        print(f"  [{label}]  {path}")

    print("\nPrecomputing / loading monthly sums ...")
    data = {}
    for label, path in experiments.items():
        print(f"[{label}]")
        years, lat_vals, lon_vals, monthly = _load_or_compute_monthly(path)
        data[path] = (years, lat_vals, lon_vals, monthly)
        print(f"  {len(years)} years, {monthly.nbytes // 1_000_000} MB")

    print("\nLoading synthetic data ...")
    syn_data = _load_synthetic(args.synthetic_zarr)
    if syn_data:
        print(f"  {len(syn_data['locations'])} locations, "
              f"{len(syn_data['slopes'])} slopes × "
              f"{len(syn_data['variances'])} variances, "
              f"{len(syn_data['years'])} years")

    app = build_app(experiments, data, syn_data)
    print(f"\nStarting app at http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
