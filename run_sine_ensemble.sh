#!/usr/bin/env bash
# Generate 10 independent runs of the 4-variant sine-wave experiment and
# run the extremes battery (boost windows 3, 5, 7) on each run.
#
# Outputs
# -------
#   data/synthetic/ensemble/<variant>/run_<seed>/t2max.zarr
#   data/synthetic/experiments/sine_ensemble/<variant>/run_<seed>/synthetic_extremes_sine_wave.zarr
#
# Usage:
#   bash run_sine_ensemble.sh [--n-runs N] [--seeds "0 1 2 ..."]

set -euo pipefail

AMPLITUDE=15
MEAN_TEMP=10
AUTOCORR=0.7
VARIANCE_TREND=0.02
AGG_WINDOWS="3 7"
AGG_METHOD="max"
PERC_BOOSTS="3 5 7"
PERCENTILE=0.90
REF_PERIODS="1950 1979"
N_RUNS=10
SEEDS=""          # if empty, uses 0 .. N_RUNS-1
DATA_ROOT="data/synthetic/ensemble"
OUT_ROOT="data/synthetic/experiments/sine_ensemble"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-runs)     N_RUNS="$2";     shift 2 ;;
        --seeds)      SEEDS="$2";      shift 2 ;;
        --agg-method) AGG_METHOD="$2"; shift 2 ;;
        --out-root)   OUT_ROOT="$2";   shift 2 ;;
        --data-root)  DATA_ROOT="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Build seed list
if [[ -z "${SEEDS}" ]]; then
    SEEDS=$(seq 0 $(( N_RUNS - 1 )) | tr '\n' ' ')
fi
read -ra SEED_ARR <<< "${SEEDS}"

source venv/bin/activate
mkdir -p "${DATA_ROOT}" "${OUT_ROOT}"

VARIANTS=(
    "A_no_ar_const_var|0.0|0.0"
    "B_ar${AUTOCORR}_const_var|${AUTOCORR}|0.0"
    "C_no_ar_var_trend|0.0|${VARIANCE_TREND}"
    "D_ar${AUTOCORR}_var_trend|${AUTOCORR}|${VARIANCE_TREND}"
)

n_variants=${#VARIANTS[@]}
n_seeds=${#SEED_ARR[@]}
total=$(( n_variants * n_seeds ))
run=0

echo "================================================================"
echo "  Sine-wave ensemble experiment"
echo "================================================================"
echo "  Variants : ${n_variants}   Seeds : ${SEED_ARR[*]}"
echo "  Agg windows  : ${AGG_WINDOWS}"
echo "  Agg method   : ${AGG_METHOD}"
echo "  Perc boosts  : ${PERC_BOOSTS}"
echo "  Total runs   : ${total}"
echo "================================================================"
echo ""

for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r vname ar vt <<< "${entry}"

    for seed in "${SEED_ARR[@]}"; do
        run=$(( run + 1 ))
        t2max_zarr="${DATA_ROOT}/${vname}/run_${seed}/t2max.zarr"
        extremes_out="${OUT_ROOT}/${vname}/run_${seed}"

        echo "--- [${run}/${total}]  ${vname}  seed=${seed} ---"

        # Generate t2max zarr
        python src/generate_synthetic_t2max.py \
            --mode           sine \
            --amplitude      "${AMPLITUDE}" \
            --mean-temp      "${MEAN_TEMP}" \
            --autocorr       "${ar}" \
            --variance-trend "${vt}" \
            --seed           "${seed}" \
            --output         "${t2max_zarr}" 2>/dev/null

        # Run extremes battery
        bash run_synthetic_battery.sh \
            --input        "${t2max_zarr}"   \
            --out-root     "${extremes_out}" \
            --agg-windows  "${AGG_WINDOWS}"  \
            --agg-method   "${AGG_METHOD}"   \
            --perc-boosts  "${PERC_BOOSTS}"  \
            --percentile   "${PERCENTILE}"   \
            --ref-periods  "${REF_PERIODS}"  2>/dev/null

        echo "    done → ${extremes_out}/synthetic_extremes_sine_wave.zarr"
    done
done

echo ""
echo "================================================================"
echo "  All ${total} runs complete."
echo "  Results in ${OUT_ROOT}/"
echo "================================================================"
