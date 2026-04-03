# Algorithmic Trading System

A **modular, production-grade** algorithmic trading platform in Python.

## Architecture

```
algo_trading/
├── config/           Central settings, secrets, logging
├── core/             Shared models (Bar, Signal, Order, Position) + Event Bus
├── data_engine/      WebSocket market feeds (Alpaca / Binance) + FinBERT NLP
├── strategies/       Pluggable strategy framework
│   ├── momentum      MA crossover trend-following
│   ├── mean_reversion  Z-score mean-reversion
│   └── sentiment     News-driven extreme-sentiment triggers
├── risk/             Stop-loss, take-profit, Kelly sizing, circuit breaker
├── execution/        Broker order routing with slippage/fee model
├── backtest/         Event-driven historical replay engine
├── dashboard/        Real-time Streamlit monitoring UI
└── main.py           System orchestrator & CLI
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export ALPACA_API_KEY="your-key"
export ALPACA_API_SECRET="your-secret"
export NEWS_API_KEY="your-newsapi-key"

# 3. Run paper trading
python main.py

# 4. Run backtest (last 30 days)
python main.py --backtest --days 30

# 5. Launch live dashboard
python main.py --dashboard
# (or: streamlit run dashboard/app.py)
```

## Modes

| Flag | Description |
|------|-------------|
| *(none)* | Paper trading via Alpaca sandbox |
| `--live` | Live trading (real money — use with caution) |
| `--backtest` | Replay historical data through strategies |
| `--dashboard` | Launch the Streamlit monitoring UI |

## Key Design Decisions

- **Event Bus** — modules communicate via a publish/subscribe bus, keeping them fully decoupled.
- **Immutable models** — `Bar`, `Signal`, `SentimentResult` are frozen dataclasses for thread safety.
- **Thread-per-subsystem** — the WS feed, sentiment pipeline, and main loop each run in their own thread.
- **Risk-first** — every signal must pass through `RiskManager.approve_signal()` before execution.
- **Circuit breaker** — if daily drawdown exceeds the configured threshold, all trading halts immediately.
- **Kelly criterion** — position sizes are dynamically computed and capped to prevent over-leveraging.
- **Zero backtest/live divergence** — the backtest engine reuses the same strategy → risk → executor pipeline.
