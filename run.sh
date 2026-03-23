#source venv/bin/activate

python src/main.py \
	--input   data/preprocessed/rechunked/t2max_rechunked.zarr \
	--var     daily_max_2m_temperature \
	--ref-start 1960-01-01  --ref-end 1989-12-31 \
	--an-start  1960-01-01  --an-end  2019-12-31 \
	--agg-window  3 \
	--agg-method  max \
	--perc-boost  3 \
	--percentile  0.90 \
	--output  experiments/
