#!/usr/bin/env bash
# Battery of synthetic extremes experiments.
#
# Reads every location from the input zarr and, for each one, calls
# run_synthetic_extremes.py with the full sweep of agg_windows × perc_boosts.
# If an output zarr already exists for a location, new (agg_window, perc_boost)
# combinations are appended; existing combinations are skipped automatically.
# Each location produces a single output zarr that contains all parameter
# combinations as dimensions:
#
#   data/synthetic/experiments/synthetic_extremes_<loc>.zarr
#     annual_counts   (n_agg, n_boost, n_sl, n_var, n_yr)
#     seasonal_counts (n_agg, n_boost, n_sl, n_var, 4, n_yr)
#     annual_trend    (n_agg, n_boost, n_sl, n_var, 2)
#     seasonal_trend  (n_agg, n_boost, n_sl, n_var, 4, 2)
#     thresholds      (n_agg, n_boost, n_sl, n_var, 365)
#
# Usage:
#   bash run_synthetic_battery.sh --input <zarr_path> [options]
#
# Required:
#   --input   PATH      Input synthetic t2max zarr
#
# Optional sweep parameters (override the defaults below):
#   --agg-windows  "1 3"      Space-separated list of aggregation windows (days)
#   --perc-boosts  "1 3"      Space-separated list of percentile boost windows (DOYs)
#   --percentile   0.99       Exceedance percentile threshold
#   --ref-periods  "1960 1989"           One or more reference periods (start end pairs)
#   --out-root     data/synthetic/experiments

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
INPUT=""
OUT_ROOT="data/synthetic/experiments"
AGG_WINDOWS="1 3 5 7"
PERC_BOOSTS="1 3 5 7"
PERCENTILE=0.95
REF_PERIODS="1960 1989 1990 2019"   # space-separated start end pairs; add more: "1960 1989 1970 1999"
AGG_METHOD="max"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)        INPUT="$2";        shift 2 ;;
        --out-root)     OUT_ROOT="$2";     shift 2 ;;
        --percentile)   PERCENTILE="$2";   shift 2 ;;
        --ref-periods)  REF_PERIODS="$2";  shift 2 ;;
        --agg-windows)  AGG_WINDOWS="$2";  shift 2 ;;
        --perc-boosts)  PERC_BOOSTS="$2";  shift 2 ;;
        --agg-method)   AGG_METHOD="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${INPUT}" ]]; then
    echo "Error: --input is required."
    echo "Usage: bash run_synthetic_battery.sh --input <zarr_path> [options]"
    exit 1
fi

if [[ ! -e "${INPUT}" ]]; then
    echo "Error: zarr not found: ${INPUT}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Discover all locations in the zarr
# ---------------------------------------------------------------------------
source venv/bin/activate

mapfile -t LOCATIONS < <(python - <<EOF
import zarr, sys
try:
    g = zarr.open_group("${INPUT}", mode="r")
    for b in g["location"][:]:
        print(b.decode())
except Exception as e:
    print(f"ERROR reading locations: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

if [[ ${#LOCATIONS[@]} -eq 0 ]]; then
    echo "Error: no locations found in ${INPUT}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_loc=${#LOCATIONS[@]}

echo "Input zarr   : ${INPUT}"
echo "Output root  : ${OUT_ROOT}"
echo "Locations    : ${LOCATIONS[*]}  (${n_loc})"
echo "Agg windows  : ${AGG_WINDOWS}"
echo "Agg method   : ${AGG_METHOD}"
echo "Boost windows: ${PERC_BOOSTS}"
echo "Percentile   : ${PERCENTILE}"
echo "Ref periods  : ${REF_PERIODS}"
echo "Total runs   : ${n_loc}  (one zarr per location, all param combos inside)"
echo ""

mkdir -p "${OUT_ROOT}"

# ---------------------------------------------------------------------------
# Run — one python call per location
# ---------------------------------------------------------------------------
run=0
for location in "${LOCATIONS[@]}"; do
    run=$(( run + 1 ))

    loc_tag="${location,,}"
    loc_tag="${loc_tag// /_}"
    output="${OUT_ROOT}/synthetic_extremes_${loc_tag}.zarr"

    echo ">>> [${run}/${n_loc}] Location: ${location}  →  ${output}"

    python src/run_synthetic_extremes.py \
        --input        "${INPUT}"        \
        --output       "${output}"       \
        --agg-windows  ${AGG_WINDOWS}    \
        --agg-method   "${AGG_METHOD}"   \
        --perc-boosts  ${PERC_BOOSTS}    \
        --percentile   "${PERCENTILE}"   \
        --ref-periods  ${REF_PERIODS}    \
        --location     "${location}"

    echo ""
done

echo "All ${n_loc} locations complete."
echo "Results in ${OUT_ROOT}/"
echo ""
echo "To visualise a location, e.g. '${LOCATIONS[0]}':"
_loc0_tag="${LOCATIONS[0],,}"
_loc0_tag="${_loc0_tag// /_}"
echo "  python src/vis/visualise_synthetic_experiments.py \\"
echo "    synthetic_extremes_${_loc0_tag}.zarr --season JJA"
