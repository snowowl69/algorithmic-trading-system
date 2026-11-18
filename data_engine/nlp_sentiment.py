"""
NLP Sentiment Pipeline — FinBERT-powered news analysis.

1. Periodically scrapes financial news headlines from NewsAPI (or RSS fallback).
2. Runs each headline through the ProsusAI/FinBERT transformer model.
3. Publishes ``SentimentResult`` objects on the event bus channel ``"nlp:sentiment"``.

The pipeline is designed to run in its own daemon thread so it never blocks
the main trading loop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import requests

from config.logging_config import get_logger
from config.settings import NLP, SYSTEM
from core.events import bus
from core.models import Sentiment, SentimentResult

log = get_logger(__name__)

# Lazy-loaded transformer objects (heavy imports)
_tokenizer = None
_model = None


def _load_finbert():
    """Load FinBERT model and tokenizer once on first use."""
    global _tokenizer, _model
    if _tokenizer is not None:
        return
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch  # noqa: F401 — ensure torch is available

        log.info("Loading FinBERT model '%s' …", NLP.model_name)
        _tokenizer = AutoTokenizer.from_pretrained(NLP.model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(NLP.model_name)
        _model.eval()
        log.info("FinBERT model loaded successfully")
    except ImportError:
        log.error(
            "transformers / torch not installed. "
            "Run: pip install transformers torch"
        )
        raise


# ═══════════════════════════════════════════════════════════════════════
#  News fetching
# ═══════════════════════════════════════════════════════════════════════

def _fetch_headlines(ticker: str, max_results: int = 20) -> List[Dict]:
    """
    Fetch recent headlines mentioning *ticker* from Finnhub.

    Returns a list of dicts with keys: ``title``, ``source``, ``publishedAt``.
    """
    from data_engine.news_feed import FINNHUB_API_KEY
    if not FINNHUB_API_KEY:
        log.warning("FINNHUB_API_KEY not set — returning empty headlines")
        return []

    # Get today and 3 days ago for company news range
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from_date = (datetime.now(timezone.utc) - __import__('datetime').timedelta(days=3)).strftime("%Y-%m-%d")

    url = f"https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": today,
        "token": FINNHUB_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json()
        if not isinstance(articles, list):
            return []
        
        results = []
        for a in articles[:max_results]:
            title = a.get("headline", "")
            if title:
                # Convert unix timestamp to ISO format string
                pub_dt = datetime.fromtimestamp(a.get("datetime", time.time()), tz=timezone.utc)
                results.append({
                    "title": title,
                    "source": a.get("source", "Finnhub"),
                    "publishedAt": pub_dt.isoformat(),
                })
        return results
    except requests.exceptions.RequestException as e:
        log.warning("Failed to fetch headlines for %s: %s (using synthetic data)", ticker, type(e).__name__)
        # Fallback to synthetic data to keep the simulation alive when offline/rate-limited
        return [
            {
                "title": f"{ticker} expected to exceed earnings estimates next quarter.",
                "source": "Synthetic News",
                "publishedAt": datetime.now(timezone.utc).isoformat()
            },
            {
                "title": f"Market concerns over {ticker} supply chain disruptions.",
                "source": "Synthetic News",
                "publishedAt": (datetime.now(timezone.utc) - __import__('datetime').timedelta(hours=2)).isoformat()
            }
        ]
    except Exception:
        log.warning("Failed to parse headlines for %s (using synthetic data)", ticker)
        return []


# ═══════════════════════════════════════════════════════════════════════
#  FinBERT inference
# ═══════════════════════════════════════════════════════════════════════

def analyze_sentiment(headlines: List[str]) -> List[Dict]:
    """
    Run a batch of headline strings through FinBERT.

    Returns a list of dicts::

        {"label": "positive"/"negative"/"neutral", "score": float}

    where *score* is mapped to the range –1 … +1 (negative for bearish).
    """
    _load_finbert()
    import torch

    if not headlines:
        return []

    inputs = _tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # FinBERT label order: positive(0), negative(1), neutral(2)
    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    results = []
    for i in range(len(headlines)):
        idx = int(torch.argmax(probs[i]).item())
        raw_score = float(probs[i][idx].item())
        label = label_map[idx]

        # Normalise to –1 … +1
        if label == "positive":
            score = raw_score
        elif label == "negative":
            score = -raw_score
        else:
            score = 0.0

        results.append({"label": label, "score": round(score, 4)})

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Continuous pipeline (runs in background thread)
# ═══════════════════════════════════════════════════════════════════════

class SentimentPipeline:
    """
    Daemon thread that periodically fetches news, scores sentiment, and
    publishes results on ``"nlp:sentiment"``.

    Recent results are stored in a bounded deque accessible via
    ``pipeline.recent_results``.
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        max_history: int = 200,
    ) -> None:
        self.tickers = tickers or SYSTEM.tickers
        self.recent_results: Deque[SentimentResult] = deque(maxlen=max_history)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Per-ticker aggregate sentiment (exponential moving average)
        self.ticker_sentiment: Dict[str, float] = {t: 0.0 for t in self.tickers}

    def start(self) -> None:
        if self._running.is_set():
            log.warning("SentimentPipeline already running")
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="sentiment-pipeline"
        )
        self._thread.start()
        log.info("SentimentPipeline started for %s", self.tickers)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=10)
        log.info("SentimentPipeline stopped")

    # ── internals ─────────────────────────────────────────────────────
    def _loop(self) -> None:
        while self._running.is_set():
            for ticker in self.tickers:
                if not self._running.is_set():
                    break
                try:
                    self._process_ticker(ticker)
                except Exception:
                    log.exception("Sentiment processing failed for %s", ticker)
            # Sleep between polling rounds
            for _ in range(NLP.poll_interval_seconds):
                if not self._running.is_set():
                    return
                time.sleep(1)

    def _process_ticker(self, ticker: str) -> None:
        headlines_raw = _fetch_headlines(ticker)
        if not headlines_raw:
            return

        texts = [h["title"] for h in headlines_raw]
        sentiments = analyze_sentiment(texts)

        alpha = 0.3  # EMA smoothing for aggregate score
        for headline_meta, sent in zip(headlines_raw, sentiments):
            label = sent["label"]
            score = sent["score"]

            if label == "positive":
                sentiment_enum = Sentiment.BULLISH
            elif label == "negative":
                sentiment_enum = Sentiment.BEARISH
            else:
                sentiment_enum = Sentiment.NEUTRAL

            try:
                pub_dt = datetime.fromisoformat(
                    headline_meta["publishedAt"].replace("Z", "+00:00")
                )
            except Exception:
                pub_dt = datetime.now(timezone.utc)

            result = SentimentResult(
                ticker=ticker,
                headline=headline_meta["title"],
                source=headline_meta["source"],
                published_at=pub_dt,
                sentiment=sentiment_enum,
                score=score,
            )
            self.recent_results.appendleft(result)
            bus.publish("nlp:sentiment", result)

            # Update rolling aggregate
            prev = self.ticker_sentiment.get(ticker, 0.0)
            self.ticker_sentiment[ticker] = alpha * score + (1 - alpha) * prev

        log.info(
            "Sentiment | %s | agg=%.3f | headlines=%d",
            ticker,
            self.ticker_sentiment[ticker],
            len(sentiments),
        )
