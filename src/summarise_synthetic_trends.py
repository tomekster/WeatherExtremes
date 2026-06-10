#!/usr/bin/env python3
"""
Summarise detected exceedance trends from the synthetic experiment.

Reads   data/synthetic/synthetic_extremes.zarr
Writes  data/synthetic/synthetic_trends_summary.csv
        data/synthetic/synthetic_trends_summary.pdf

One row per (location × warming slope × noise variance).
Columns: Location, Slope (°C/yr), Variance (°C²), then one detected-trend
column (days/yr) for each period: Annual, DJF, MAM, JJA, SON.

Usage
-----
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate
    python src/summarise_synthetic_trends.py
"""
import os
import numpy as np
import pandas as pd
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT_ZARR  = "data/synthetic/synthetic_extremes.zarr"
OUTPUT_CSV  = "data/synthetic/synthetic_trends_summary.csv"
OUTPUT_PDF  = "data/synthetic/synthetic_trends_summary.pdf"

SEASONS     = ["DJF", "MAM", "JJA", "SON"]
ROWS_PER_PAGE = 40


def _build_dataframe() -> pd.DataFrame:
    grp = zarr.open_group(INPUT_ZARR, mode="r")

    slopes    = grp["slope"][:]
    variances = grp["variance"][:]
    locations = [b.decode() for b in grp["location"][:]]
    ann_trend = grp["annual_trend"][:]    # (5, 5, 8, 2)
    sea_trend = grp["seasonal_trend"][:] # (5, 5, 4, 8, 2)

    rows = []
    for si, slope in enumerate(slopes):
        for vi, var in enumerate(variances):
            for li, loc in enumerate(locations):
                row = {
                    "Location":          loc,
                    "Slope (°C/yr)":     round(float(slope), 3),
                    "Variance (°C²)":    round(float(var),   2),
                    "Annual (days/yr)":  round(float(ann_trend[si, vi, li, 0]), 4),
                }
                for s_idx, s_name in enumerate(SEASONS):
                    row[f"{s_name} (days/yr)"] = round(
                        float(sea_trend[si, vi, s_idx, li, 0]), 4
                    )
                rows.append(row)

    return pd.DataFrame(rows)


def _save_pdf(df: pd.DataFrame, path: str) -> None:
    cols      = list(df.columns)
    n_rows    = len(df)
    n_pages   = -(-n_rows // ROWS_PER_PAGE)   # ceiling division

    # Column widths (relative): location wider, rest equal
    col_widths = [2.8] + [1.1] * (len(cols) - 1)
    total_w    = sum(col_widths)

    # Alternating row colours
    row_colours_even = "#f0f4fa"
    row_colours_odd  = "#ffffff"
    header_colour    = "#1e3a5f"
    header_text      = "white"

    with PdfPages(path) as pdf:
        for page in range(n_pages):
            start = page * ROWS_PER_PAGE
            end   = min(start + ROWS_PER_PAGE, n_rows)
            chunk = df.iloc[start:end]

            fig_h = 1.0 + len(chunk) * 0.22   # dynamic height
            fig, ax = plt.subplots(figsize=(14, fig_h))
            fig.patch.set_facecolor("white")
            ax.axis("off")

            cell_text = chunk.values.tolist()
            cell_colours = [
                [row_colours_even if (start + i) % 2 == 0 else row_colours_odd]
                * len(cols)
                for i in range(len(chunk))
            ]

            tbl = ax.table(
                cellText=cell_text,
                colLabels=cols,
                cellLoc="center",
                colWidths=[w / total_w for w in col_widths],
                loc="upper center",
                cellColours=cell_colours,
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)

            # Style header row
            for col_idx in range(len(cols)):
                cell = tbl[0, col_idx]
                cell.set_facecolor(header_colour)
                cell.set_text_props(color=header_text, fontweight="bold")
                cell.set_height(0.06)

            # Left-align location column
            for row_idx in range(1, len(chunk) + 1):
                tbl[row_idx, 0].get_text().set_ha("left")
                tbl[row_idx, 0].PAD = 0.02

            fig.suptitle(
                "Synthetic Experiment — Detected Exceedance Trends (days/yr)\n"
                f"Page {page + 1} of {n_pages}  •  rows {start + 1}–{end} of {n_rows}",
                fontsize=9, y=0.98, color="#333333",
            )

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # PDF metadata
        info = pdf.infodict()
        info["Title"]   = "Synthetic Exceedance Trends Summary"
        info["Subject"] = "WeatherExtremes2 synthetic experiment results"


def main() -> None:
    df = _build_dataframe()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV → {OUTPUT_CSV}  ({len(df)} rows × {len(df.columns)} columns)")

    df_pdf = df.sort_values(["Location", "Slope (°C/yr)", "Variance (°C²)"]).reset_index(drop=True)
    _save_pdf(df_pdf, OUTPUT_PDF)
    print(f"Saved PDF → {OUTPUT_PDF}  ({-(-len(df) // ROWS_PER_PAGE)} pages)")


if __name__ == "__main__":
    main()
