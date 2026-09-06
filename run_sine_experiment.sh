#!/usr/bin/env bash
# Run the 4-variant sine-wave synthetic experiment.
#
# Experiment matrix (2×2):
#   Rows : no autocorrelation (φ=0)  vs  AR(1) autocorrelation (φ=0.7)
#   Cols : constant variance          vs  increasing variance (σ²(Y) = σ_base²+k·Y)
#
# Each variant is processed with two aggregation windows: 3 days and 7 days.
#
# Usage:
#   bash run_sine_experiment.sh [options]
#
# Options:
#   --amplitude     FLOAT   Sine-wave amplitude [°C]        (default: 15)
#   --mean-temp     FLOAT   Sine-wave mean temperature [°C] (default: 10)
#   --autocorr      FLOAT   AR(1) φ coefficient             (default: 0.7)
#   --variance-trend FLOAT  Variance slope k [°C²/yr]       (default: 0.02)
#   --agg-windows   "W …"  Space-separated agg windows      (default: "3 7")
#   --perc-boosts   "B …"  Space-separated boost windows    (default: "31")
#   --percentile    FLOAT   Exceedance percentile            (default: 0.90)
#   --ref-periods   "Y …"  Reference period start/end pairs (default: "1950 1979")
#   --out-root      PATH    Root directory for experiment outputs
#                           (default: data/synthetic/experiments/sine)
#   --data-dir      PATH    Directory for generated t2max zarrs
#                           (default: data/synthetic)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
AMPLITUDE=15
MEAN_TEMP=10
AUTOCORR=0.7
VARIANCE_TREND=0.02
AGG_WINDOWS="3 7"
PERC_BOOSTS="31"
PERCENTILE=0.90
REF_PERIODS="1950 1979"
OUT_ROOT="data/synthetic/experiments/sine"
DATA_DIR="data/synthetic"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --amplitude)      AMPLITUDE="$2";       shift 2 ;;
        --mean-temp)      MEAN_TEMP="$2";        shift 2 ;;
        --autocorr)       AUTOCORR="$2";         shift 2 ;;
        --variance-trend) VARIANCE_TREND="$2";   shift 2 ;;
        --agg-windows)    AGG_WINDOWS="$2";      shift 2 ;;
        --perc-boosts)    PERC_BOOSTS="$2";      shift 2 ;;
        --percentile)     PERCENTILE="$2";       shift 2 ;;
        --ref-periods)    REF_PERIODS="$2";      shift 2 ;;
        --out-root)       OUT_ROOT="$2";         shift 2 ;;
        --data-dir)       DATA_DIR="$2";         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

source venv/bin/activate

mkdir -p "${DATA_DIR}" "${OUT_ROOT}"

echo "================================================================"
echo "  Sine-wave synthetic experiment"
echo "================================================================"
echo "  Sine:            amplitude=${AMPLITUDE}°C, mean=${MEAN_TEMP}°C"
echo "  Autocorr (AR1):  φ=${AUTOCORR}"
echo "  Variance trend:  k=${VARIANCE_TREND} °C²/yr"
echo "  Agg windows:     ${AGG_WINDOWS} days"
echo "  Perc boosts:     ${PERC_BOOSTS} DOYs"
echo "  Percentile:      ${PERCENTILE}"
echo "  Reference period(s): ${REF_PERIODS}"
echo "  Output root:     ${OUT_ROOT}"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# 4 variants: autocorr × variance_trend
#
# Each entry: "variant_name|autocorr_value|variance_trend_value"
# ---------------------------------------------------------------------------
VARIANTS=(
    "A_no_ar_const_var|0.0|0.0"
    "B_ar${AUTOCORR}_const_var|${AUTOCORR}|0.0"
    "C_no_ar_var_trend|0.0|${VARIANCE_TREND}"
    "D_ar${AUTOCORR}_var_trend|${AUTOCORR}|${VARIANCE_TREND}"
)

total=${#VARIANTS[@]}
run=0

for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r name ar vt <<< "${entry}"
    run=$(( run + 1 ))
    zarr="${DATA_DIR}/synthetic_t2max_sine_${name}.zarr"

    echo "--- [${run}/${total}] Variant: ${name} ---"
    echo "    autocorr=${ar}  variance_trend=${vt}"
    echo "    zarr: ${zarr}"
    echo ""

    # ---- Step 1: Generate synthetic t2max zarr ----
    echo "  [1/2] Generating data ..."
    python src/generate_synthetic_t2max.py \
        --mode           sine \
        --amplitude      "${AMPLITUDE}" \
        --mean-temp      "${MEAN_TEMP}" \
        --autocorr       "${ar}" \
        --variance-trend "${vt}" \
        --output         "${zarr}"

    # ---- Step 2: Run extremes battery ----
    echo ""
    echo "  [2/2] Running extremes battery ..."
    bash run_synthetic_battery.sh \
        --input        "${zarr}"         \
        --out-root     "${OUT_ROOT}/${name}" \
        --agg-windows  "${AGG_WINDOWS}"  \
        --perc-boosts  "${PERC_BOOSTS}"  \
        --percentile   "${PERCENTILE}"   \
        --ref-periods  "${REF_PERIODS}"

    echo ""
done

echo "================================================================"
echo "  All ${total} variants complete."
echo "  Results in ${OUT_ROOT}/"
echo ""
echo "  Variants:"
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r name ar vt <<< "${entry}"
    echo "    ${name}  →  ${OUT_ROOT}/${name}/"
done
echo "================================================================"
