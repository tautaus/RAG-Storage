.PHONY: setup data embed index bench update analyze qa clean

setup:
\tpython -m pip install -U pip

data:
\tpython scripts/prepare_data.py --out data/raw

embed:
\tpython scripts/embed_corpus.py --cfg config/experiment.yaml

index:
\tpython scripts/build_indexes.py --cfg config/experiment.yaml

bench:
\tpython scripts/run_benchmarks.py --cfg config/experiment.yaml

update:
\tpython scripts/run_update_test.py --cfg config/experiment.yaml

analyze:
\tpython - <<'PY'\nimport pandas as pd, glob\npaths=glob.glob('results/logs/*.csv')\nprint('Found logs:', len(paths))\nPY

qa:
\tbash tools/static_analysis.sh

clean:
\trm -rf data/processed/* results/logs/* results/figures/*
