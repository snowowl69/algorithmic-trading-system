.PHONY: install test run backtest dashboard clean docker-build docker-up

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov

test:
	python -m pytest tests/ -v

run:
	python main.py

backtest:
	python main.py --backtest --days 30

dashboard:
	streamlit run dashboard/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t algo-trading-platform:latest .

docker-up:
	docker compose up -d
