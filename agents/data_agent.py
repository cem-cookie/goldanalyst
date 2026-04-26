import os
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from openai import OpenAI
import feedparser
import json
import yaml
from datetime import datetime, timedelta
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv

load_dotenv()



class DataAgent:
    def __init__(self, openai_api_key=None, config_path=None):
        """Data Agent: Fetch market data + news articles + call LLM for analysis with quality scoring"""
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

        # Load quality scoring configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "news_quality_scorer.yaml")
        self.config = self._load_config(config_path)


    def _to_utc_date(self, s: str):
        if not s:
            return None
        # Prefer ISO 8601
        try:
            s2 = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            pass
        # Fallback RFC822
        try:
            dt = parsedate_to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            return None
    def _parse_date_to_utc_date(self, s: str):
        """
        Parse varied date strings to UTC date (date only, no time).
        Returns None on failure.
        """
        if not s or not isinstance(s, str):
            return None

        # Common: RSS RFC822 (entry.published)
        try:
            dt = parsedate_to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            pass

        # Common plain dates
        for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(s.strip(), fmt)
                return dt.date()
            except Exception:
                continue

        # ISO fallback
        try:
            s2 = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            return None

    def _load_config(self, config_path):
        """Load YAML configuration for news quality scoring"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"[INFO] Loaded quality scoring config from {config_path}")
            return config
        except Exception as e:
            print(f"[WARN] Failed to load config: {e}. Using default settings.")
            return self._default_config()

    def _normalize_weight_factors(self, weight_factors_raw):
        """
        Normalize weight_factors to dict format for backward compatibility.
        Handles both list format: [{"key": val}, ...]
        and dict format: {"key": val, ...}
        """
        if weight_factors_raw is None:
            return {}
        if isinstance(weight_factors_raw, list):
            return {k: v for d in weight_factors_raw for k, v in d.items()}
        if isinstance(weight_factors_raw, dict):
            return weight_factors_raw
        return {}

    def _default_config(self):
        """Default configuration if YAML file not found"""
        return {
            "sources": {
                "YahooRSS": {"base_score": 75},
                "InvestingRSS": {"base_score": 80},
                "MetalsDaily": {"base_score": 70},
                "Unknown": {"base_score": 40}
            },
            "thresholds": {
                "minimum_score": 50,
                "high_quality": 75
            },
            "llm_selection": {
                "selected_bonus": 15,
                "not_selected_penalty": -10
            }
        }

    def _calculate_quality_score(self, news_item):
        """
        Calculate quality score for a news item based on:
        - Source credibility
        - Content length
        - Keyword analysis
        - LLM selection
        """
        source = news_item.get("source", "Unknown")
        source_config = self.config["sources"].get(source, self.config["sources"]["Unknown"])

        # Start with base score
        score = source_config["base_score"]

        # Full text bonus
        if news_item.get("full_text"):
            full_text_len = len(news_item["full_text"])
            weight_factors = self._normalize_weight_factors(source_config.get("weight_factors"))

            # Has full text bonus
            score += weight_factors.get("has_full_text", 10)

            # Length-based scoring
            min_length = weight_factors.get("article_length_min", 500)
            if full_text_len >= min_length:
                if full_text_len > 1000:
                    score += weight_factors.get("article_length_bonus", 5)

        # LLM selection bonus/penalty
        if news_item.get("selected"):
            score += self.config["llm_selection"]["selected_bonus"]
        else:
            score += self.config["llm_selection"]["not_selected_penalty"]

        # Keyword analysis
        text_to_analyze = (news_item.get("title", "") + " " +
                           news_item.get("summary", "") + " " +
                           (news_item.get("full_text", "") or "")[:500]).lower()

        # Positive keywords
        positive_keywords = self.config.get("content_quality", {}).get("keywords_positive", [])
        for keyword in positive_keywords:
            if keyword.lower() in text_to_analyze:
                score += self.config.get("content_quality", {}).get("positive_keyword_bonus", 3)

        # Negative keywords (clickbait indicators)
        negative_keywords = self.config.get("content_quality", {}).get("keywords_negative", [])
        for keyword in negative_keywords:
            if keyword.lower() in text_to_analyze:
                score += self.config.get("content_quality", {}).get("negative_keyword_penalty", -5)

        # Ensure score is within 0-100 range
        score = max(0, min(100, score))

        # Add quality label
        thresholds = self.config["thresholds"]
        if score >= thresholds["high_quality"]:
            quality_label = "high"
        elif score >= thresholds["minimum_score"]:
            quality_label = "medium"
        else:
            quality_label = "low"

        return {
            "score": round(score, 2),
            "quality": quality_label
        }

    # ======================================================
    # 1. Get gold price (Yahoo Finance)
    # ======================================================
    def get_yahoo_gold(self, days=30):
        """Get gold futures price from Yahoo Finance"""
        print("\n[INFO] Fetching Yahoo Finance gold price...")
        try:
            gold = yf.Ticker("GC=F")  # Gold Futures
            df = gold.history(period=f"{days}d")
            df = df.reset_index()[["Date", "Close"]]
            df["Source"] = "YahooFinance"
            print(df.tail())
            return df
        except Exception as e:
            print("[ERROR] Failed to fetch Yahoo Finance data:", e)
            return pd.DataFrame(columns=["Date", "Close", "Source"])

    # ======================================================
    # 2. Scrape Yahoo Finance gold news articles
    # ======================================================
    def get_yahoo_rss_news(self, limit=10):
        """
        Scrape Yahoo Finance gold-related summaries and return structured JSON
        """
        print("\n[INFO] Fetching Yahoo Finance RSS feed...")
        feed_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US"

        try:
            feed = feedparser.parse(feed_url)
            news = []
            for i, entry in enumerate(feed.entries[:limit]):
                # Get publication date
                pub_date = getattr(entry, "published", "")

                item = {
                    "id": i + 1,
                    "source": "YahooRSS",
                    "title": entry.title,
                    "summary": entry.summary,
                    "url": entry.link,
                    "date": pub_date,
                    "selected": False,
                    "full_text": None,
                    "quality_score": None,
                    "quality_label": None
                }
                news.append(item)

            print(f"[INFO] Parsed {len(news)} Yahoo RSS items.")
            return news

        except Exception as e:
            print("[ERROR] Failed to fetch RSS feed:", e)
            return []

    def filter_relevant_news(self, news_json):
        """
        Call LLM to determine which news items are highly relevant to gold markets.
        Returns JSON with selected flag.
        """
        if not news_json:
            print("[WARN] No news to filter.")
            return []

        summaries = "\n\n".join([f"{n['id']}. {n['title']}\n{n['summary']}" for n in news_json])

        prompt = f"""
You are a financial analyst specializing in gold markets.
Below are news headlines and summaries from various sources.
Select which ones are most relevant to **gold price movements, XAU/USD, gold mining, inflation, or Federal Reserve policy**.
Return only a JSON list of IDs, e.g. [1, 3, 5].

News:
{summaries}
"""

        print("\n[INFO] Filtering relevant news items via LLM...")

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are precise and concise when identifying relevant financial articles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        text = response.choices[0].message.content
        print(f"\n[LLM Selection Output]\n{text}\n")

        # Try to extract ID list from output
        try:
            ids = json.loads(re.search(r"\[.*\]", text).group(0))
        except Exception:
            ids = []

        for n in news_json:
            n["selected"] = n["id"] in ids

        print(f"[INFO] LLM selected {len(ids)} relevant items.")
        return news_json

    def fetch_full_articles(self, news_json):
        """
        Based on filtering results, fetch full text for news with selected=True.
        """
        print("\n[INFO] Fetching full text for selected news...")
        updated = []

        for n in news_json:
            if not n["selected"]:
                updated.append(n)
                continue

            try:
                print(f"   [INFO] Fetching article: {n['url']}")
                article = Article(n["url"])
                article.download()
                article.parse()
                n["full_text"] = article.text
                print(f"   ✅ {n['title']} ({len(article.text)} chars)")
            except Exception as e:
                print(f"[WARN] Failed to fetch full article for {n['title']}: {e}")
                n["full_text"] = None

            updated.append(n)

        return updated

    # ======================================================
    # 3. Scrape MetalsDaily news articles
    # ======================================================
    def get_metalsdaily_news(self, limit=20):
        """
        Scrape MetalsDaily website GOLD NEWS titles, links and dates
        """
        print("\n[INFO] Fetching MetalsDaily GOLD NEWS titles...")
        url = "https://www.metalsdaily.com/"
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            items = soup.select("div.topNews div.newsTable a.NewsItem")
            if not items:
                print("[WARN] No GOLD NEWS found. The selector may have changed.")
                return []

            news = []
            for i, a in enumerate(items[:limit], 1):
                title_span = a.select_one(".Title")
                date_span = a.select_one(".Date")

                title = title_span.get_text(strip=True) if title_span else a.get("title", "")
                link = a.get("href")
                date = date_span.get_text(strip=True) if date_span else ""

                item = {
                    "id": i,
                    "source": "MetalsDaily",
                    "title": title,
                    "summary": "",
                    "url": link,
                    "date": date,
                    "selected": False,
                    "full_text": None,
                    "quality_score": None,
                    "quality_label": None
                }
                news.append(item)
                print(f"🟧 [{i}] {title} ({date})")

            print(f"[INFO] Parsed {len(news)} GOLD NEWS headlines from MetalsDaily.")
            return news

        except Exception as e:
            print("[ERROR] Failed to fetch MetalsDaily:", e)
            return []

    # ======================================================
    # 4. Scrape Investing.com news articles
    # ======================================================
    def get_investing_news(self, limit=10):
        """
        Get gold news from Investing.com's official RSS feed.
        """
        print("\n[INFO] Fetching Investing.com RSS feed...")
        url = "https://www.investing.com/rss/news_301.rss"

        try:
            feed = feedparser.parse(url)
            news = []
            for i, entry in enumerate(feed.entries[:limit]):
                title = getattr(entry, "title", "No title")
                summary = getattr(entry, "summary", getattr(entry, "description", ""))
                link = getattr(entry, "link", "")
                date = getattr(entry, "published", "")

                item = {
                    "id": i + 1,
                    "source": "InvestingRSS",
                    "title": title,
                    "summary": summary,
                    "url": link,
                    "date": date,
                    "selected": False,
                    "full_text": None,
                    "quality_score": None,
                    "quality_label": None
                }
                news.append(item)
                print(f"🟩 [Investing #{i + 1}] {title}")

            print(f"[INFO] Parsed {len(news)} Investing RSS items.")
            return news

        except Exception as e:
            print("[ERROR] Failed to fetch Investing RSS:", e)
            return []

    def collect_all_news(self, limit=10, date_str=None):
        """
        Collect news from Yahoo / MetalsDaily / Investing
        Save to date-based folder structure: /data/news/YYYY-MM-DD/news.json
        """
        print("\n[STEP 1] Collecting all news sources...")

        yahoo = self.get_yahoo_rss_news(limit)
        metals = self.get_metalsdaily_news(limit)
        investing = self.get_investing_news(limit)

        all_news = yahoo + metals + investing
        for i, n in enumerate(all_news, 1):
            n["id"] = i  # Uniformly renumber

        # Use today's date if not specified
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Create date-based directory structure
        news_dir = f"data/news/{date_str}"
        os.makedirs(news_dir, exist_ok=True)
        save_path = f"{news_dir}/news.json"

        print(f"[INFO] Collected total {len(all_news)} news items.")

        # Add metadata
        output = {
            "collection_date": date_str,
            "collection_timestamp": datetime.now().isoformat(),
            "total_items": len(all_news),
            "news": all_news
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved to {save_path}")
        return all_news, save_path

    def filter_and_fetch_fulltext(self, json_path):
        """
        Filter once + fetch full text once + calculate quality scores + update and save
        """
        print("\n[STEP 2] Filtering & fetching full articles with quality scoring...")

        if not os.path.exists(json_path):
            print(f"[ERROR] {json_path} not found.")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both old format (list) and new format (dict with metadata)
        if isinstance(data, list):
            news = data
            metadata = {}
        else:
            news = data.get("news", [])
            metadata = {k: v for k, v in data.items() if k != "news"}

        # === Call LLM to filter ===
        news = self.filter_relevant_news(news)

        # === Fetch full text ===
        news = self.fetch_full_articles(news)

        # === Calculate quality scores ===
        print("\n[INFO] Calculating quality scores...")
        for n in news:
            quality_data = self._calculate_quality_score(n)
            n["quality_score"] = quality_data["score"]
            n["quality_label"] = quality_data["quality"]

            if n["selected"]:
                print(f"   [QS] [{n['source']}] {n['title'][:60]}... Score: {n['quality_score']} ({n['quality_label']})")

        # === Calculate aggregate statistics ===
        selected_news = [n for n in news if n["selected"]]
        if selected_news:
            avg_score = sum(n["quality_score"] for n in selected_news) / len(selected_news)
            high_quality_count = sum(1 for n in selected_news if n["quality_label"] == "high")
            low_quality_count = sum(1 for n in selected_news if n["quality_label"] == "low")

            metadata.update({
                "processed_timestamp": datetime.now().isoformat(),
                "selected_count": len(selected_news),
                "average_quality_score": round(avg_score, 2),
                "high_quality_count": high_quality_count,
                "medium_quality_count": len(selected_news) - high_quality_count - low_quality_count,
                "low_quality_count": low_quality_count
            })

            print(f"\n[SUMMARY] Average Quality Score: {avg_score:.2f}")
            print(f"[SUMMARY] High Quality: {high_quality_count}, Low Quality: {low_quality_count}")

        # === Write back with metadata ===
        output = {**metadata, "news": news}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Updated {json_path} with quality scores and selection marks.")
        return news

    def analyze_market_news(self, json_path, min_quality_score=None, max_chars_per_article=5000):
        """
        Read filtered JSON with full text and quality scores
        Call GPT to generate comprehensive analysis with quality-weighted consideration

        Args:
            json_path: Path to news JSON file
            min_quality_score: Minimum quality score threshold (default from config)
            max_chars_per_article: Maximum characters per article (default 5000, set to None for unlimited)
        """
        print("\n[STEP 3] Analyzing news with quality-weighted GPT analysis...")

        if not os.path.exists(json_path):
            print(f"[ERROR] {json_path} not found.")
            return "No data to analyze."

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_news = data.get("news", data) if isinstance(data, dict) else data

        if min_quality_score is None:
            min_quality_score = self.config["thresholds"]["minimum_score"]

        selected_news = [n for n in all_news
                         if n["selected"] and n["full_text"] and
                         (n.get("quality_score") or 0) >= min_quality_score]

        if not selected_news:
            return f"No relevant news with full text and quality score >= {min_quality_score} found."

        selected_news.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        # Prepare news with controlled text length
        news_entries = []
        for n in selected_news:
            quality_indicator = "⭐" * (
                1 if n["quality_label"] == "medium" else 2 if n["quality_label"] == "high" else 0)

            # Apply character limit if specified
            if max_chars_per_article and len(n['full_text']) > max_chars_per_article:
                content = n['full_text'][:max_chars_per_article] + f"\n[... truncated at {max_chars_per_article} chars]"
            else:
                content = n['full_text']

            entry = (f"[{n['source']}] {quality_indicator} (Score: {n['quality_score']})\n"
                     f"Title: {n['title']}\n"
                     f"Content: {content}\n")
            news_entries.append(entry)

        combined_text = "\n\n" + "=" * 80 + "\n\n".join(news_entries)
        avg_score = sum(n["quality_score"] for n in selected_news) / len(selected_news)

        print(f"[INFO] Input: {len(combined_text):,} chars (~{len(combined_text) // 4:,} tokens)")

        prompt = f"""You are a senior gold market analyst. 

        Analyze {len(selected_news)} articles (quality score: {avg_score:.1f}/100).
        ⭐⭐ = High quality (≥75), ⭐ = Medium quality (50-74)

        Create DETAILED analysis with these sections (use # headers only, NO equal signs or dashes):

        # Market Analysis Summary

        Sources Analyzed: {len(selected_news)} articles
        Average Quality Score: {avg_score:.2f}/100

        # Overall Market Sentiment

        Write 2-3 detailed paragraphs with:
        - Sentiment (bullish/bearish/neutral) with confidence (1-10)
        - Specific prices, percentages, changes (exact numbers)
        - Technical levels (support/resistance with exact prices)
        - Volume/open interest data if mentioned
        - Cross-asset correlations (USD, yields, equities)

        # Key Price Drivers and Catalysts

        For EACH driver, provide comprehensive detail:
        - **[Driver Name]**: Specific data (prices, %, dates), quotes, market probabilities, timeline

        Include: Fed policy, USD dynamics, inflation data, geopolitics, central bank purchases, mining supply, physical demand, ETF flows, technical factors.

        # Major Risks and Concerns

        For each risk: specific triggers, quantified impact, probability, timeline, mitigation factors

        # Source Consensus vs. Disagreement

        - Consensus areas (cite how many sources agree)
        - Disagreements (present both views with data)
        - Unique insights per source
        - Confidence rating (1-10)

        REQUIREMENTS:
        - Extract ALL specific numbers (prices, %, volumes, dates)
        - Use exact figures: "$2,045.50" not "~$2,045"
        - Quote officials/analysts verbatim
        - Cite sources: "(per Bloomberg)", "(3 sources confirm)"
        - Don't round numbers
        - Include all data points

        Note: This analysis was generated with quality-weighted filtering. Low quality sources (score < {min_quality_score}) were excluded.

        Articles:
        {combined_text}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a senior financial analyst. Extract ALL data points and be comprehensive. Cite sources for all claims."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096
        )

        summary = response.choices[0].message.content.strip()

        print(
            f"\n[INFO] Token usage: {response.usage.prompt_tokens:,} in + {response.usage.completion_tokens:,} out = {response.usage.total_tokens:,} total")
        print(f"[INFO] Output: {len(summary):,} chars (~{len(summary.split()):,} words)")

        analysis_path = json_path.replace("news.json", "market_analysis.txt")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"[INFO] Saved to {analysis_path}")

        return summary

    def save_gold_history_to_file(self, days=1095, filepath="data/gold_history.csv"):
        """
        Fetch GC=F daily close safely and save to CSV.
        Strategy:
          1) Try yfinance (GC=F, 1d).
          2) On failure, fall back to Stooq via pandas_datareader (XAUUSD for spot gold),
             then align column names to ['date','gold'].
        """
        import os
        import pandas as pd
        import datetime as dt

        print(f"\n[INFO] Fetching {days} days of gold price history...")

        # ---------- Helper: safe save ----------
        def _save(df: pd.DataFrame) -> pd.DataFrame:
            dirpath = os.path.dirname(filepath)
            os.makedirs(dirpath if dirpath else ".", exist_ok=True)
            df.to_csv(filepath, index=False)
            print(f"[INFO] Saved {len(df)} rows to {filepath}")
            return df

        # ---------- Path 1: yfinance ----------
        try:
            import yfinance as yf
            df = yf.download(
                tickers="GC=F",
                period=f"{int(days)}d",
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty and "Close" in df.columns:
                df = df[["Close"]].copy()
                # Convert timezone index to tz-naive
                try:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                except Exception:
                    df.index = pd.to_datetime(df.index)

                df = df.reset_index()
                # Handle Date / Datetime
                date_col = "Date" if "Date" in df.columns else (
                    "Datetime" if "Datetime" in df.columns else df.columns[0])
                df = df.rename(columns={date_col: "date", "Close": "gold"})
                return _save(df)
            else:
                raise RuntimeError("yfinance returned empty frame or missing Close column.")
        except Exception as e1:
            print(f"[WARN] yfinance failed: {e1}")

        # ---------- Path 2: Stooq fallback (spot gold XAUUSD) ----------
        try:
            from pandas_datareader import data as pdr
            end = dt.datetime.utcnow().date()
            start = end - dt.timedelta(days=int(days) + 5)  # Get extra days to avoid calendar gaps

            # Stooq uses 'XAUUSD' for spot gold (forex). Daily OHLC.
            df2 = pdr.DataReader("XAUUSD", "stooq", start, end)  # index descending sometimes
            if df2 is None or df2.empty:
                raise RuntimeError("Stooq returned empty frame for XAUUSD.")

            # Stooq returns columns ['Open','High','Low','Close','Volume']
            if "Close" not in df2.columns:
                raise KeyError("Column 'Close' not found in Stooq dataframe.")

            # Sort ascending
            df2 = df2.sort_index().copy()
            df2 = df2[["Close"]].rename(columns={"Close": "gold"}).reset_index()
            df2 = df2.rename(columns={"Date": "date"})  # DataReader column name is Date

            # Align with yfinance: keep only last N days
            if len(df2) > days:
                df2 = df2.iloc[-days:].copy()

            print("[INFO] Used Stooq fallback (XAUUSD).")
            return _save(df2)

        except Exception as e2:
            print(f"[ERROR] Fallback (Stooq) failed: {e2}")
            return None

    def get_gnews_articles(self, query="gold", start_date=None, end_date=None, limit=30):
        """
        Fetch gold-related news from GNews.io for a single UTC day (start_date).
        Uses pagination (page=1,2,...) to fill up to `limit` (GNews free: 10 per page).
        """
        api_key = os.getenv("GNEWS_API_KEY")
        if not api_key:
            raise ValueError("Please set GNEWS_API_KEY environment variable")

        if not start_date:
            raise ValueError("start_date (YYYY-MM-DD) is required")

        target_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        start_iso = f"{start_date}T00:00:00Z"
        end_iso = f"{start_date}T23:59:59Z"

        url = "https://gnews.io/api/v4/search"
        per_page = 10
        target_total = limit
        page = 1
        results = []
        seen_urls = set()

        while len(results) < target_total:
            params = {
                "q": query,
                "lang": "en",  # Remove or change for more languages
                "country": "us",  # Remove for broader coverage
                "from": start_iso,  # Start of day (UTC)
                "to": end_iso,  # End of day (UTC)
                "sortBy": "publishedAt",
                "max": per_page,  # Max 10 per page
                "page": page,  # Pagination
                "token": api_key
            }
            try:
                r = requests.get(url, params=params, timeout=12)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"[ERROR] GNews fetch failed (page {page}): {e}")
                break

            articles = data.get("articles", []) or []
            if not articles:
                break

            for a in articles:
                pub_date = self._to_utc_date(a.get("publishedAt"))
                if pub_date != target_date:
                    continue  # Strict filter: only same day

                url_ = a.get("url") or ""
                if url_ and url_ in seen_urls:
                    continue
                seen_urls.add(url_)

                results.append({
                    "id": len(results) + 1,
                    "source": (a.get("source") or {}).get("name", "GNews"),
                    "title": a.get("title"),
                    "summary": a.get("description"),
                    "url": url_,
                    "date": a.get("publishedAt"),
                    "selected": False,
                    "full_text": None,
                    "quality_score": None,
                    "quality_label": None
                })
                if len(results) >= target_total:
                    break

            page += 1

        print(f"[INFO] GNews {start_date}: collected {len(results)} items (pages tried: {page - 1})")
        return results

    # Collect and save by day
    def collect_past_news_with_gnews(self, date_str, query="gold", limit=30):
        news = self.get_gnews_articles(query=query, start_date=date_str, end_date=date_str, limit=limit)

        news_dir = f"data/news/{date_str}"
        os.makedirs(news_dir, exist_ok=True)
        save_path = f"{news_dir}/news.json"

        output = {
            "collection_date": date_str,
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_items": len(news),
            "news": news
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved {len(news)} GNews articles to {save_path}")
        return news, save_path
