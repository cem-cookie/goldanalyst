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
import glob
import re
from datetime import datetime, timedelta
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime


class DataAgent:
    def __init__(self, openai_api_key=None, config_path="config/news_quality_scorer.yaml"):
        """Data Agent: Fetch market data + news articles + call LLM for analysis with quality scoring"""
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.config = self._load_config(config_path)

    def _to_utc_date(self, s: str):
        if not s:
            return None
        try:
            s2 = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            pass
        try:
            dt = parsedate_to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            return None

    def _parse_date_to_utc_date(self, s: str):
        if not s or not isinstance(s, str):
            return None
        try:
            dt = parsedate_to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            pass
        for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(s.strip(), fmt)
                return dt.date()
            except Exception:
                continue
        try:
            s2 = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            return None

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"[INFO] Loaded quality scoring config from {config_path}")
            return config
        except Exception as e:
            print(f"[WARN] Failed to load config: {e}. Using default settings.")
            return self._default_config()

    def _default_config(self):
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
        source = news_item.get("source", "Unknown")
        source_config = self.config["sources"].get(source, self.config["sources"]["Unknown"])
        score = source_config["base_score"]

        if news_item.get("full_text"):
            full_text_len = len(news_item["full_text"])
            weight_factors = source_config.get("weight_factors", {})
            score += weight_factors.get("has_full_text", 10)
            min_length = weight_factors.get("article_length_min", 500)
            if full_text_len >= min_length:
                if full_text_len > 1000:
                    score += weight_factors.get("article_length_bonus", 5)

        if news_item.get("selected"):
            score += self.config["llm_selection"]["selected_bonus"]
        else:
            score += self.config["llm_selection"]["not_selected_penalty"]

        text_to_analyze = (news_item.get("title", "") + " " +
                           news_item.get("summary", "") + " " +
                           (news_item.get("full_text", "") or "")[:500]).lower()

        positive_keywords = self.config.get("content_quality", {}).get("keywords_positive", [])
        for keyword in positive_keywords:
            if keyword.lower() in text_to_analyze:
                score += self.config.get("content_quality", {}).get("positive_keyword_bonus", 3)

        negative_keywords = self.config.get("content_quality", {}).get("keywords_negative", [])
        for keyword in negative_keywords:
            if keyword.lower() in text_to_analyze:
                score += self.config.get("content_quality", {}).get("negative_keyword_penalty", -5)

        score = max(0, min(100, score))
        thresholds = self.config["thresholds"]
        if score >= thresholds["high_quality"]:
            quality_label = "high"
        elif score >= thresholds["minimum_score"]:
            quality_label = "medium"
        else:
            quality_label = "low"

        return {"score": round(score, 2), "quality": quality_label}

    def get_yahoo_gold(self, days=30):
        print("\n[INFO] Fetching Yahoo Finance gold price...")
        try:
            gold = yf.Ticker("GC=F")
            df = gold.history(period=f"{days}d")
            df = df.reset_index()[["Date", "Close"]]
            df["Source"] = "YahooFinance"
            print(df.tail())
            return df
        except Exception as e:
            print("[ERROR] Failed to fetch Yahoo Finance data:", e)
            return pd.DataFrame(columns=["Date", "Close", "Source"])

    def get_yahoo_rss_news(self, limit=10):
        print("\n[INFO] Fetching Yahoo Finance RSS feed...")
        feed_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US"
        try:
            feed = feedparser.parse(feed_url)
            news = []
            for i, entry in enumerate(feed.entries[:limit]):
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
        if not news_json:
            print("[WARN] No news to filter.")
            return []

        summaries = "\n\n".join([f"{n['id']}. {n['title']}\n{n['summary']}" for n in news_json])

        prompt = f"""You are a financial analyst specializing in gold markets.
Select which news items are most relevant to gold price movements, XAU/USD, gold mining, inflation, or Federal Reserve policy.
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

        try:
            ids = json.loads(re.search(r"\[.*\]", text).group(0))
        except Exception:
            ids = []

        for n in news_json:
            n["selected"] = n["id"] in ids

        print(f"[INFO] LLM selected {len(ids)} relevant items.")
        return news_json

    def fetch_full_articles(self, news_json):
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
                print(f"   OK {n['title'][:50]} ({len(article.text)} chars)")
            except Exception as e:
                print(f"[WARN] Failed to fetch full article for {n['title']}: {e}")
                n["full_text"] = None

            updated.append(n)

        return updated

    def get_metalsdaily_news(self, limit=20):
        print("\n[INFO] Fetching MetalsDaily GOLD NEWS titles...")
        url = "https://www.metalsdaily.com/"
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            items = soup.select("div.topNews div.newsTable a.NewsItem")
            if not items:
                print("[WARN] No GOLD NEWS found.")
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
                print(f"   [{i}] {title} ({date})")

            print(f"[INFO] Parsed {len(news)} MetalsDaily headlines.")
            return news
        except Exception as e:
            print("[ERROR] Failed to fetch MetalsDaily:", e)
            return []

    def get_investing_news(self, limit=10):
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
                print(f"   [#{i + 1}] {title}")

            print(f"[INFO] Parsed {len(news)} Investing RSS items.")
            return news
        except Exception as e:
            print("[ERROR] Failed to fetch Investing RSS:", e)
            return []

    def collect_all_news(self, limit=10, date_str=None):
        print("\n[STEP 1] Collecting all news sources...")

        yahoo = self.get_yahoo_rss_news(limit)
        metals = self.get_metalsdaily_news(limit)
        investing = self.get_investing_news(limit)

        all_news = yahoo + metals + investing
        for i, n in enumerate(all_news, 1):
            n["id"] = i

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        news_dir = f"data/news/{date_str}"
        os.makedirs(news_dir, exist_ok=True)
        save_path = f"{news_dir}/news.json"

        print(f"[INFO] Collected total {len(all_news)} news items.")

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
        print("\n[STEP 2] Filtering & fetching full articles with quality scoring...")

        if not os.path.exists(json_path):
            print(f"[ERROR] {json_path} not found.")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            news = data
            metadata = {}
        else:
            news = data.get("news", [])
            metadata = {k: v for k, v in data.items() if k != "news"}

        news = self.filter_relevant_news(news)
        news = self.fetch_full_articles(news)

        print("\n[INFO] Calculating quality scores...")
        for n in news:
            quality_data = self._calculate_quality_score(n)
            n["quality_score"] = quality_data["score"]
            n["quality_label"] = quality_data["quality"]

            if n["selected"]:
                print(f"   [{n['source']}] {n['title'][:60]}... Score: {n['quality_score']}")

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

        output = {**metadata, "news": news}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Updated {json_path} with quality scores.")
        return news

    def analyze_market_news(self, json_path, min_quality_score=None, max_chars_per_article=5000):
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
                         if n["selected"] and n.get("full_text") and
                         (n.get("quality_score") or 0) >= min_quality_score]

        if not selected_news:
            return f"No relevant news with quality score >= {min_quality_score} found."

        selected_news.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        news_entries = []
        for n in selected_news:
            quality_indicator = "**" * (
                1 if n["quality_label"] == "medium" else 2 if n["quality_label"] == "high" else 0)

            if max_chars_per_article and len(n['full_text']) > max_chars_per_article:
                content = n['full_text'][:max_chars_per_article] + f"\n[truncated at {max_chars_per_article} chars]"
            else:
                content = n['full_text']

            entry = (f"[{n['source']}] {quality_indicator} (Score: {n['quality_score']})\n"
                     f"Title: {n['title']}\n"
                     f"Content: {content}\n")
            news_entries.append(entry)

        combined_text = "\n\n" + "=" * 80 + "\n\n".join(news_entries)
        avg_score = sum(n["quality_score"] for n in selected_news) / len(selected_news)

        print(f"[INFO] Input: {len(combined_text):,} chars")

        prompt = f"""You are a senior gold market analyst. 
Analyze {len(selected_news)} articles (quality score: {avg_score:.1f}/100).

Create detailed analysis with sections: Market Sentiment, Key Drivers, Risks, Consensus.
Include specific prices, percentages, and dates. Cite sources.

Articles:
{combined_text}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a senior financial analyst. Extract all data points and cite sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096
        )

        summary = response.choices[0].message.content.strip()

        analysis_path = json_path.replace("news.json", "market_analysis.txt")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"[INFO] Saved to {analysis_path}")

        return summary

    def extract_price_range_summary(self, prices_dict: dict, start_date: str, end_date: str) -> dict:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except:
            return {}

        range_prices = {}
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            if date_str in prices_dict:
                range_prices[date_str] = prices_dict[date_str]
            current += timedelta(days=1)

        if not range_prices:
            return {}

        sorted_dates = sorted(range_prices.keys())
        prices_list = [range_prices[d] for d in sorted_dates]

        return {
            "start_date": sorted_dates[0],
            "end_date": sorted_dates[-1],
            "trading_days": len(prices_list),
            "open": round(prices_list[0], 2),
            "close": round(prices_list[-1], 2),
            "high": round(max(prices_list), 2),
            "low": round(min(prices_list), 2),
            "change_pct": round(((prices_list[-1] - prices_list[0]) / prices_list[0] * 100), 2),
            "prices_by_date": {d: round(prices_dict[d], 2) for d in sorted_dates}
        }

    def aggregate_weekly_data(self, news_dir: str, target_date: str, prices: dict, lookback_days: int = 7,
                              min_quality_score: float = 50) -> dict:
        try:
            target = datetime.strptime(target_date, "%Y-%m-%d")
        except:
            return {}

        start_date = (target - timedelta(days=lookback_days - 1)).strftime("%Y-%m-%d")
        end_date = target_date

        price_info = self.extract_price_range_summary(prices, start_date, end_date)

        if not price_info:
            return {}

        aggregated_news = []
        aggregated_analyses = []
        dates_with_data = set()

        for i in range(lookback_days):
            scan_date = (target - timedelta(days=lookback_days - 1 - i)).strftime("%Y-%m-%d")

            news_file = os.path.join(news_dir, scan_date, "news.json")
            analysis_file = os.path.join(news_dir, scan_date, "market_analysis.txt")

            if os.path.exists(news_file):
                try:
                    with open(news_file, "r", encoding="utf-8") as f:
                        news_data = json.load(f)
                    news_list = news_data.get("news", []) if isinstance(news_data, dict) else news_data

                    for n in news_list:
                        if (n.get("selected") and n.get("quality_score", 0) >= min_quality_score):
                            aggregated_news.append({
                                "date": scan_date,
                                "title": n.get("title", ""),
                                "source": n.get("source", "Unknown"),
                                "quality_score": n.get("quality_score", 0),
                                "quality_label": n.get("quality_label", "medium"),
                                "summary": n.get("summary", "")[:200]
                            })
                            dates_with_data.add(scan_date)
                except Exception:
                    pass

            if os.path.exists(analysis_file):
                try:
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        analysis_text = f.read().strip()
                    aggregated_analyses.append({
                        "date": scan_date,
                        "analysis": analysis_text[:1500]
                    })
                    dates_with_data.add(scan_date)
                except Exception:
                    pass

        aggregated_news.sort(key=lambda x: x["quality_score"], reverse=True)
        aggregated_news = aggregated_news[:10]

        combined_analysis = ""
        if aggregated_analyses:
            combined_analysis = "\n\n---\n\n".join([
                f"[{a['date']}]\n{a['analysis'][:800]}"
                for a in aggregated_analyses[-3:]
            ])

        return {
            "target_date": target_date,
            "date_range": f"{start_date} ~ {end_date}",
            "lookback_days": lookback_days,
            "price_info": price_info,
            "news_count": len(aggregated_news),
            "analysis_count": len(aggregated_analyses),
            "dates_with_data": len(dates_with_data),
            "news": aggregated_news,
            "combined_analysis": combined_analysis
        }

    def build_user_prompt(self, weekly_data: dict) -> str:
        if not weekly_data:
            return ""

        price_info = weekly_data["price_info"]
        news = weekly_data["news"]
        analysis = weekly_data["combined_analysis"]

        prices_table = "\n".join([
            f"{date}: ${price}"
            for date, price in price_info["prices_by_date"].items()
        ])

        news_summary = ""
        if news:
            news_lines = []
            for n in news[:8]:
                quality_stars = "**" * (2 if n["quality_label"] == "high" else 1)
                news_lines.append(
                    f"- [{n['date']}] [{n['source']}] {quality_stars}\n"
                    f"  {n['title'][:100]}"
                )
            news_summary = "\n".join(news_lines)

        prompt = f"""Date range: {weekly_data['date_range']} (7-day lookback)
Gold price history (daily close):
{prices_table}

Price summary:
- Open: ${price_info['open']}
- Close: ${price_info['close']}
- High: ${price_info['high']}
- Low: ${price_info['low']}
- Change: {price_info['change_pct']}% ({price_info['trading_days']} trading days)

High-quality news (past 7 days):
{news_summary}

Market Analysis (Past 7 days):
{analysis[:1000]}

Current position: FLAT
Question: At the end of {weekly_data['target_date']}, what are the best trading strategies?"""

        return prompt

    def generate_multi_strategies(self, user_prompt: str) -> dict:
        system_prompt = """You are a professional gold trading strategist. 
Based on price history, news, and market analysis, propose 3 alternative trading strategies:
1. Conservative (low risk, steady returns)
2. Balanced (medium risk/reward)
3. Aggressive (high risk, high reward)

For each strategy provide:
- name: Strategy name
- action: BUY, SELL, or HOLD
- confidence: 1-10
- reasoning: Brief explanation
- risk_level: low, medium, or high
- expected_return: low, medium, or high

Respond ONLY with valid JSON:
{
  "market_sentiment": "bullish/bearish/neutral",
  "overall_confidence": 1-10,
  "strategies": [
    {"name": "Conservative", "action": "BUY/SELL/HOLD", "confidence": 1-10, "reasoning": "...", "risk_level": "low", "expected_return": "low"},
    {"name": "Balanced", "action": "BUY/SELL/HOLD", "confidence": 1-10, "reasoning": "...", "risk_level": "medium", "expected_return": "medium"},
    {"name": "Aggressive", "action": "BUY/SELL/HOLD", "confidence": 1-10, "reasoning": "...", "risk_level": "high", "expected_return": "high"}
  ]
}"""

        print("  [LLM] Generating 3 alternative strategies...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            raw_response = response.choices[0].message.content.strip()

            try:
                clean_text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw_response.strip(), flags=re.MULTILINE)
                strategies_data = json.loads(clean_text)
                return strategies_data
            except json.JSONDecodeError:
                print(f"  [WARN] Failed to parse JSON response")
                return {
                    "market_sentiment": "neutral",
                    "overall_confidence": 5,
                    "strategies": [
                        {"name": "Conservative", "action": "HOLD", "confidence": 5, "reasoning": "Parse error",
                         "risk_level": "low", "expected_return": "low"},
                        {"name": "Balanced", "action": "HOLD", "confidence": 5, "reasoning": "Parse error",
                         "risk_level": "medium", "expected_return": "medium"},
                        {"name": "Aggressive", "action": "HOLD", "confidence": 5, "reasoning": "Parse error",
                         "risk_level": "high", "expected_return": "high"}
                    ]
                }

        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            return {
                "market_sentiment": "unknown",
                "overall_confidence": 1,
                "strategies": [
                    {"name": "Conservative", "action": "HOLD", "confidence": 1, "reasoning": f"Error: {str(e)}",
                     "risk_level": "low", "expected_return": "low"},
                    {"name": "Balanced", "action": "HOLD", "confidence": 1, "reasoning": f"Error: {str(e)}",
                     "risk_level": "medium", "expected_return": "medium"},
                    {"name": "Aggressive", "action": "HOLD", "confidence": 1, "reasoning": f"Error: {str(e)}",
                     "risk_level": "high", "expected_return": "high"}
                ]
            }