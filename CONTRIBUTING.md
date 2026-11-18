# Contributing to the Algorithmic Trading Platform

Thank you for your interest in contributing! We welcome quantitative research strategies, risk management enhancements, execution model improvements, and performance optimizations.

---

## 🛠️ Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/algorithmic-trading-system.git
   cd algorithmic-trading-system
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```

3. **Verify the test suite passes**:
   ```bash
   python -m pytest tests/
   ```

---

## 🧠 Adding a New Strategy

All strategies inherit from `BaseStrategy` (`strategies/base_strategy.py`):

1. Create a new file in `strategies/my_new_strategy.py`.
2. Implement the required abstract methods:
   ```python
   from core.models import Bar, Signal, Side, SentimentResult
   from strategies.base_strategy import BaseStrategy

   class MyNewStrategy(BaseStrategy):
       def __init__(self, ticker: str, param: float = 1.0):
           super().__init__(ticker)
           self.param = param

       def on_bar(self, bar: Bar) -> Signal:
           # Generate your quantitative signal
           # Must return a Signal dataclass
           ...

       def on_sentiment(self, sentiment: SentimentResult) -> Signal:
           # Handle real-time NLP sentiment updates
           ...
   ```
3. Register the strategy in `main.py` and write tests in `tests/test_strategies.py`.

---

## 🛡️ Risk Management Guidelines

Every proposed feature or execution route must pass through the `RiskManager` before interacting with the broker or paper trading simulator. Pull requests that bypass risk checks will be rejected.

---

## 🧪 Testing Standards

- All new code must be accompanied by unit tests in `tests/`.
- Ensure all 60+ tests pass with `python -m pytest tests/`.
- Code should follow PEP 8 styling conventions.

---

## 📬 Submitting a Pull Request

1. Create a feature branch: `git checkout -b feature/my-quant-strategy`
2. Commit changes with clear, semantic commit messages: `git commit -m "feat(strategies): add adaptive RSI breakout strategy"`
3. Push to your fork: `git push origin feature/my-quant-strategy`
4. Open a Pull Request against the `main` branch.
