#!/usr/bin/env bash
source venv/bin/activate

python src/main.py \
	--input   data/preprocessed/rechunked/precip_rechunked.zarr \
	--var     precip \
	--ref-start 1960-01-01  --ref-end 1989-12-31 \
	--an-start  1960-01-01  --an-end  2019-12-31 \
	--agg-window  1 \
	--agg-method  max \
	--perc-boost  1 \
	--percentile  0.95 \
	--output  experiments/
