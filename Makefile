PYTHON ?= python
PIP ?= $(PYTHON) -m pip
export PYTHONPATH := .

.PHONY: setup ingest features train evaluate insights promote drift pipeline test lint format api dashboard run mlflow-ui docker-build docker-up clean

setup:
	$(PIP) install -r requirements-dev.txt

ingest:
	$(PYTHON) scripts/01_ingest.py

features:
	$(PYTHON) scripts/02_features.py

train:
	$(PYTHON) scripts/03_train.py

evaluate:
	$(PYTHON) scripts/04_evaluate.py

insights:
	$(PYTHON) scripts/05_insights.py

promote:
	$(PYTHON) scripts/06_promote.py

drift:
	$(PYTHON) scripts/07_drift.py

pipeline:
	$(PYTHON) scripts/run_pipeline.py

test:
	$(PYTHON) -m pytest tests/unit --cov=src --cov-fail-under=80
	$(PYTHON) -m pytest tests/integration

lint:
	$(PYTHON) -m ruff check src tests scripts app
	$(PYTHON) -m black --check src tests scripts app
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m ruff check --fix src tests scripts app
	$(PYTHON) -m black src tests scripts app

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port $${PORT:-8000}

dashboard:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port $${STREAMLIT_SERVER_PORT:-8501}

run:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port $${PORT:-8000} &
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri ./mlruns --port 5000

docker-build:
	docker build -t ncae:local .

docker-up:
	docker compose up --build

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in map(pathlib.Path, ['artifacts','mlruns','data/interim','data/processed','data/features','.mypy_cache','.ruff_cache','.pytest_cache'])]"
