.PHONY: install data test results clean

install:
	pip install -r requirements.txt && pip install -e .

data:
	python scripts/download_data.py

test:
	python -m pytest -q || python -m unittest discover -s tests

# Full study: 14 origins x 9 models (~15 min on a 2-core laptop), then all analyses
results: data
	python scripts/run_all.py
	python scripts/build_dashboard.py

clean:
	rm -rf data/cache reports/tables/*.csv reports/figures/*.png reports/results.json
