"""
Live Market News Aggregator — Indian Financial Markets.

Fetches real-time news from multiple RSS feeds covering:
- CNBC TV18, NDTV Profit, Moneycontrol, Economic Times, LiveMint,
  Business Standard, Zee Business, Bloomberg Quint, Reuters India,
  Financial Express, Business Today, The Hindu Business Line.

Provides categorization, keyword extraction, and ticker matching.
"""

from __future__ import annotations

import re
import time
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  RSS Feed Sources — Major Indian Financial News
# ═══════════════════════════════════════════════════════════════════════
RSS_FEEDS: dict[str, str] = {
    # ── TV Channels ──
    "CNBC TV18": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
    "Zee Business": "https://zeenews.india.com/rss/business.xml",
    "NDTV Profit": "https://feeds.feedburner.com/ndtvprofit-latest",
    # ── Business Newspapers / Portals ──
    "Moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
    "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET Commodities": "https://economictimes.indiatimes.com/markets/commodities/rssfeeds/19422778.cms",
    "LiveMint": "https://www.livemint.com/rss/markets",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "Financial Express": "https://www.financialexpress.com/market/feed/",
    "Business Today": "https://www.businesstoday.in/rss/markets",
    "Hindu Business Line": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    # ── International (India coverage) ──
    "Reuters India": "https://feeds.reuters.com/reuters/INbusinessNews",
    "Bloomberg Quint": "https://www.ndtvprofit.com/rss/feed",
}


# ── Ticker keyword mapping for auto-tagging ──────────────────────────
_TICKER_KEYWORDS: dict[str, list[str]] = {
    "RELIANCE": ["reliance", "ril", "jio", "mukesh ambani"],
    "TCS": ["tcs", "tata consultancy"],
    "INFY": ["infosys", "infy", "salil parekh"],
    "HDFCBANK": ["hdfc bank", "hdfcbank"],
    "ICICIBANK": ["icici bank", "icicibank"],
    "SBIN": ["sbi", "state bank of india"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "ITC": ["itc limited", "itc hotels"],
    "HINDUNILVR": ["hindustan unilever", "hul"],
    "WIPRO": ["wipro"],
    "LT": ["larsen", "l&t", "larsen & toubro"],
    "AXISBANK": ["axis bank"],
    "KOTAKBANK": ["kotak mahindra", "kotak bank"],
    "MARUTI": ["maruti suzuki", "maruti"],
    "TATAMOTORS": ["tata motors", "jaguar land rover", "jlr"],
    "ADANIENT": ["adani enterprises", "adani group", "gautam adani"],
    "BAJFINANCE": ["bajaj finance", "bajfinance"],
    "SUNPHARMA": ["sun pharma", "sun pharmaceutical"],
    "TITAN": ["titan company", "titan watch", "tanishq"],
    "NESTLEIND": ["nestle india", "nestle"],
    "GOLD": ["gold price", "gold futures", "mcx gold", "yellow metal"],
    "SILVER": ["silver price", "silver futures", "mcx silver"],
    "CRUDEOIL": ["crude oil", "brent crude", "wti crude", "oil price"],
    "NATURALGAS": ["natural gas", "gas futures"],
    "COPPER": ["copper price", "copper futures", "mcx copper"],
    # General market
    "NIFTY": ["nifty", "nifty 50", "nifty50"],
    "SENSEX": ["sensex", "bse sensex", "bse 30"],
    "BANKNIFTY": ["bank nifty", "banknifty"],
}

# ── Category patterns ─────────────────────────────────────────────────
_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "Earnings": ["quarterly results", "q1", "q2", "q3", "q4", "profit", "revenue", "earnings", "eps"],
    "IPO": ["ipo", "initial public offering", "listing", "allotment"],
    "Regulatory": ["sebi", "rbi", "regulation", "compliance", "policy"],
    "Global": ["fed", "us market", "wall street", "nasdaq", "dow jones", "global", "china", "recession"],
    "Commodities": ["gold", "silver", "crude", "oil", "copper", "commodity", "mcx"],
    "Forex": ["rupee", "dollar", "forex", "currency", "inr", "usd"],
    "Sector": ["banking", "it sector", "pharma", "auto", "fmcg", "infra", "realty", "metal"],
    "Technical": ["support", "resistance", "breakout", "moving average", "rsi", "macd"],
    "Macro": ["gdp", "inflation", "cpi", "fiscal", "budget", "rbi policy", "interest rate", "repo rate"],
}


@dataclass
class NewsItem:
    """Single news article."""
    title: str
    link: str
    source: str
    published: datetime
    description: str = ""
    tickers: List[str] = field(default_factory=list)
    category: str = "General"
    importance: str = "normal"  # low, normal, high, critical
    uid: str = ""

    def __post_init__(self):
        if not self.uid:
            self.uid = hashlib.md5(
                (self.title + self.source).encode()
            ).hexdigest()[:12]


def _clean_html(text: str) -> str:
    """Strip HTML tags from RSS description."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _parse_pub_date(date_str: str) -> datetime:
    """Parse various RSS date formats."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue
    return datetime.now(timezone.utc)


def _match_tickers(text: str) -> list[str]:
    """Find mentioned tickers in text."""
    text_lower = text.lower()
    matched = []
    for ticker, keywords in _TICKER_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                matched.append(ticker)
                break
    return matched


def _categorize(text: str) -> str:
    """Categorize news article."""
    text_lower = text.lower()
    for category, patterns in _CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return category
    return "General"


def _assess_importance(item: NewsItem) -> str:
    """Assess news importance based on content signals."""
    text = (item.title + " " + item.description).lower()
    critical_words = ["crash", "circuit breaker", "halt", "scam", "fraud",
                      "rbi policy", "emergency", "war", "default", "ban"]
    high_words = ["record high", "all-time", "surge", "plunge", "acquisition",
                  "merger", "takeover", "ipo", "delisting", "quarterly result",
                  "budget", "rate cut", "rate hike"]

    for w in critical_words:
        if w in text:
            return "critical"
    for w in high_words:
        if w in text:
            return "high"
    if len(item.tickers) >= 2:
        return "high"
    return "normal"


def fetch_feed(source_name: str, url: str, timeout: int = 10) -> list[NewsItem]:
    """Fetch and parse a single RSS feed."""
    items = []
    try:
        headers = {
            "User-Agent": "AlgoTradingBot/1.0 (Indian Market Monitor)",
            "Accept": "application/rss+xml, application/xml, text/xml",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        root = ElementTree.fromstring(resp.content)

        # Handle both RSS 2.0 and Atom formats
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # RSS 2.0
        for item_el in root.findall(".//item"):
            title = (item_el.findtext("title") or "").strip()
            link = (item_el.findtext("link") or "").strip()
            desc = _clean_html(item_el.findtext("description") or "")
            pub_str = item_el.findtext("pubDate") or ""

            if not title:
                continue

            pub_date = _parse_pub_date(pub_str) if pub_str else datetime.now(timezone.utc)
            full_text = title + " " + desc
            tickers = _match_tickers(full_text)
            category = _categorize(full_text)

            news = NewsItem(
                title=title,
                link=link,
                source=source_name,
                published=pub_date,
                description=desc[:300],
                tickers=tickers,
                category=category,
            )
            news.importance = _assess_importance(news)
            items.append(news)

        # Atom format fallback
        if not items:
            for entry in root.findall("atom:entry", ns):
                title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                link = link_el.get("href", "") if link_el is not None else ""
                desc = _clean_html(entry.findtext("atom:summary", namespaces=ns) or "")
                pub_str = entry.findtext("atom:updated", namespaces=ns) or ""

                if not title:
                    continue

                pub_date = _parse_pub_date(pub_str) if pub_str else datetime.now(timezone.utc)
                full_text = title + " " + desc
                tickers = _match_tickers(full_text)
                category = _categorize(full_text)

                news = NewsItem(
                    title=title, link=link, source=source_name,
                    published=pub_date, description=desc[:300],
                    tickers=tickers, category=category,
                )
                news.importance = _assess_importance(news)
                items.append(news)

    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", source_name, e)
    except ElementTree.ParseError as e:
        logger.warning("Failed to parse %s XML: %s", source_name, e)
    except Exception as e:
        logger.warning("Unexpected error fetching %s: %s", source_name, e)

    return items


def fetch_all_news(
    max_per_source: int = 15,
    timeout: int = 8,
) -> list[NewsItem]:
    """
    Fetch news from ALL configured RSS feeds.

    Returns a deduplicated, sorted (newest first) list of NewsItem.
    """
    all_items: list[NewsItem] = []
    seen_uids: set[str] = set()

    for source_name, url in RSS_FEEDS.items():
        try:
            items = fetch_feed(source_name, url, timeout=timeout)
            for item in items[:max_per_source]:
                if item.uid not in seen_uids:
                    seen_uids.add(item.uid)
                    all_items.append(item)
        except Exception as e:
            logger.warning("Error processing %s: %s", source_name, e)

    # Sort by publication date (newest first)
    all_items.sort(key=lambda x: x.published, reverse=True)
    return all_items


def news_to_dataframe(items: list[NewsItem]) -> "pd.DataFrame":
    """Convert news items to a Pandas DataFrame for dashboard display."""
    import pandas as pd

    if not items:
        return pd.DataFrame(columns=[
            "Time", "Source", "Category", "Importance", "Tickers", "Headline",
        ])

    rows = []
    for n in items:
        rows.append({
            "Time": n.published.strftime("%H:%M"),
            "Date": n.published.strftime("%Y-%m-%d"),
            "Source": n.source,
            "Category": n.category,
            "Importance": n.importance,
            "Tickers": ", ".join(n.tickers) if n.tickers else "—",
            "Headline": n.title,
            "Link": n.link,
            "Description": n.description,
        })

    return pd.DataFrame(rows)


# ── Full NSE + BSE Stock Universe ─────────────────────────────────────
# Major indices and all actively traded stocks
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
    "HCLTECH", "WIPRO", "ASIANPAINT", "MARUTI", "TATAMOTORS", "SUNPHARMA",
    "BAJFINANCE", "TITAN", "NESTLEIND", "ULTRACEMCO", "ADANIENT",
    "BAJAJFINSV", "NTPC", "TECHM", "POWERGRID", "TATASTEEL", "ONGC",
    "INDUSINDBK", "M&M", "JSWSTEEL", "ADANIPORTS", "HINDALCO",
    "COALINDIA", "GRASIM", "SBILIFE", "BAJAJ-AUTO", "BRITANNIA",
    "DIVISLAB", "DRREDDY", "CIPLA", "BPCL", "EICHERMOT", "HEROMOTOCO",
    "APOLLOHOSP", "HDFCLIFE", "TATACONSUM", "SHRIRAMFIN", "LTIM",
]

NIFTY_NEXT_50 = [
    "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "AUROPHARMA", "BANDHANBNK",
    "BANKBARODA", "BEL", "BERGEPAINT", "BIOCON", "BOSCHLTD",
    "CANBK", "CHOLAFIN", "COLPAL", "CONCOR", "DABUR",
    "DLF", "GAIL", "GODREJCP", "HAVELLS", "HINDPETRO",
    "ICICIGI", "ICICIPRULI", "IGL", "INDUSTOWER", "IOC",
    "IRCTC", "JINDALSTEL", "JUBLFOOD", "LICI", "LUPIN",
    "MARICO", "MCDOWELL-N", "MOTHERSON", "MUTHOOTFIN", "NAUKRI",
    "NMDC", "OBEROIRLTY", "OFSS", "PAGEIND", "PEL",
    "PETRONET", "PFC", "PIDILITIND", "PNB", "POLYCAB",
    "RECLTD", "SBICARD", "SIEMENS", "SRF", "TRENT",
]

NIFTY_MIDCAP_100 = [
    "AARTIIND", "ABB", "ABCAPITAL", "ABFRL", "ACC",
    "ALKEM", "ASTRAL", "ATUL", "AUBANK", "BALKRISIND",
    "BATAINDIA", "BHEL", "BSE", "CANFINHOME", "CDSL",
    "CESC", "COFORGE", "COROMANDEL", "CROMPTON", "CUB",
    "CUMMINSIND", "DALBHARAT", "DEEPAKNTR", "DEVYANI", "DIXON",
    "ESCORTS", "EXIDEIND", "FEDERALBNK", "FORTIS", "GLENMARK",
    "GMRINFRA", "GNFC", "GRANULES", "GSPL", "HAL",
    "HONAUT", "IDFCFIRSTB", "INDHOTEL", "IRFC", "ISEC",
    "JKCEMENT", "JSWENERGY", "KAJARIACER", "KEI", "KPITTECH",
    "L&TFH", "LICHSGFIN", "LODHA", "LTF", "MANAPPURAM",
    "MFSL", "MPHASIS", "MRF", "NAM-INDIA", "NATIONALUM",
    "NAVINFLUOR", "NHPC", "OIL", "PAYTM", "PERSISTENT",
    "PHOENIXLTD", "PIIND", "POLYCAB", "PRESTIGE", "PVRINOX",
    "RAIN", "RAMCOCEM", "RBLBANK", "RELIANCE", "SAIL",
    "SCHAEFFLER", "SJVN", "SOLARINDS", "SUMICHEM", "SUNDARMFIN",
    "SUPREMEIND", "SYNGENE", "TATACHEM", "TATACOMM", "TATAELXSI",
    "TATAPOWER", "THERMAX", "TORNTPHARM", "TORNTPOWER", "TVSMOTOR",
    "UBL", "UNIONBANK", "UPL", "VEDL", "VOLTAS",
    "YESBANK", "ZEEL", "ZOMATO", "ZYDUSLIFE",
]

BSE_ADDITIONAL = [
    "ADANIENSOL", "AFFLE", "ANGELONE", "APTUS", "ATHER",
    "CAMPUS", "CARTRADE", "CLEAN", "DELHIVERY", "ELECTCAST",
    "FIVESTAR", "FSN", "GLAND", "HAPPSTMNDS", "HOME",
    "IDEAFORGE", "IONEXCHANG", "IREDA", "JIOFIN", "KAYNES",
    "LATENTVIEW", "LXCHEM", "MAPMYINDIA", "MAZAGON", "MEDANTA",
    "METROBRAND", "NAZARA", "NYKAA", "OLECTRA", "POONAWALLA",
    "PRINCEPIPE", "RATEGAIN", "ROUTE", "SAPPHIRE", "SBICARD",
    "SONACOMS", "STARHEALTH", "SYRMA", "TIINDIA", "TRIDENT",
    "TRIVENI", "UTIAMC", "VAIBHAVGBL", "VALIANT", "ZOMATO",
]

MCX_COMMODITIES = [
    "GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", "COPPER",
    "ZINC", "ALUMINIUM", "LEAD", "NICKEL", "COTTON",
    "CPO", "MENTHAOIL",
]

# Combine ALL — deduplicated
_all_set = set()
ALL_STOCKS: list[str] = []
for _lst in [NIFTY_50, NIFTY_NEXT_50, NIFTY_MIDCAP_100, BSE_ADDITIONAL, MCX_COMMODITIES]:
    for _s in _lst:
        if _s not in _all_set:
            _all_set.add(_s)
            ALL_STOCKS.append(_s)


# ── yfinance symbol mapping for ALL stocks ────────────────────────────
def get_yfinance_symbol(ticker: str) -> str:
    """Convert ticker name to yfinance-compatible symbol."""
    mcx_map = {
        "GOLD": "GC=F", "SILVER": "SI=F", "CRUDEOIL": "CL=F",
        "NATURALGAS": "NG=F", "COPPER": "HG=F", "ZINC": "ZN=F",
        "ALUMINIUM": "ALI=F", "LEAD": "PB=F", "NICKEL": "NI=F",
        "COTTON": "CT=F", "CPO": "KPO=F", "MENTHAOIL": "MO=F",
    }
    indices = {"NIFTY": "^NSEI", "SENSEX": "^BSESN", "BANKNIFTY": "^NSEBANK"}

    if ticker in mcx_map:
        return mcx_map[ticker]
    if ticker in indices:
        return indices[ticker]
    return f"{ticker}.NS"
