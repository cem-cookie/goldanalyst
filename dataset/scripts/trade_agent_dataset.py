#!/usr/bin/env python3
"""
Trading Agent Dataset Generator
--------------------------------

Use TradingAgent based on past 7 days of news + market analysis to generate high-quality training data:
- Summarize daily selected news and market_analysis.txt
- Calculate price features: 30/50-day moving averages, recent relative peaks, 3-year absolute peaks
- Call TradingAgent to output detailed multi-strategy recommendations
- Extend recommendations, additional target price / expected profit (can be used as training labels)

Usage example:
    python trade_agent_dataset.py \
        --output training_data_llm_improved.jsonl \
        --news-dir data/news \
        --price-file data/gold_history.csv
"""

import argparse
import glob
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from agents.trading_agent import TradingAgent
from prepare_price_data import load_gold_prices_from_csv


# ======================================================
# Price helper functions
# ======================================================

def calculate_moving_average(prices: Dict[str, float], target_date: str, window: int) -> Optional[float]:
    target_dt = parse_date(target_date)
    start_dt = target_dt - timedelta(days=window * 2)  # Reserve double the days to cover non-trading days

    window_prices: List[Tuple[str, float]] = [
        (d, p) for d, p in prices.items()
        if start_dt <= parse_date(d) <= target_dt
    ]
    window_prices.sort(key=lambda x: x[0])

    rolling_values = [price for _, price in window_prices if _ <= target_date]
    if len(rolling_values) < window:
        return None

    return round(sum(rolling_values[-window:]) / window, 2)


def find_relative_peak(prices: Dict[str, float], target_date: str, lookback_days: int = 90) -> Optional[Dict[str, float]]:
    target_dt = parse_date(target_date)
    start_dt = target_dt - timedelta(days=lookback_days)

    segment = [
        (d, prices[d]) for d in prices
        if start_dt <= parse_date(d) < target_dt
    ]
    if not segment:
        return None

    segment.sort(key=lambda x: x[0])
    best_date, best_price = max(segment, key=lambda x: x[1])
    return {"date": best_date, "price": round(best_price, 2)}


def find_absolute_peak(prices: Dict[str, float], target_date: str, years: int = 3) -> Optional[Dict[str, float]]:
    target_dt = parse_date(target_date)
    start_dt = target_dt - timedelta(days=365 * years)

    window_points = [
        (d, prices[d]) for d in prices
        if start_dt <= parse_date(d) <= target_dt
    ]
    if not window_points:
        return None

    window_points.sort(key=lambda x: x[0])
    best_date, best_price = max(window_points, key=lambda x: x[1])
    return {"date": best_date, "price": round(best_price, 2)}


def extract_price_summary(prices: Dict[str, float], start_date: str, end_date: str) -> Dict:
    try:
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
    except Exception:
        return {}

    slice_prices = {
        d: prices[d]
        for d in prices
        if start_dt <= parse_date(d) <= end_dt
    }

    if not slice_prices:
        return {}

    sorted_dates = sorted(slice_prices.keys())
    values = [slice_prices[d] for d in sorted_dates]

    return {
        "start_date": sorted_dates[0],
        "end_date": sorted_dates[-1],
        "trading_days": len(values),
        "open": round(values[0], 2),
        "close": round(values[-1], 2),
        "high": round(max(values), 2),
        "low": round(min(values), 2),
        "change_pct": round((values[-1] - values[0]) / values[0] * 100, 2),
        "prices_by_date": {d: round(slice_prices[d], 2) for d in sorted_dates}
    }


def compute_price_features(prices: Dict[str, float], target_date: str) -> Optional[Dict]:
    if target_date not in prices:
        return None

    ma_30 = calculate_moving_average(prices, target_date, 30)
    ma_50 = calculate_moving_average(prices, target_date, 50)
    relative_peak = find_relative_peak(prices, target_date)
    absolute_peak = find_absolute_peak(prices, target_date, years=3)

    return {
        "close": round(prices[target_date], 2),
        "ma_30": ma_30,
        "ma_50": ma_50,
        "recent_relative_peak": relative_peak,
        "absolute_peak_3y": absolute_peak
    }


# ======================================================
# News and Analysis Aggregation
# ======================================================

def load_news_for_date(news_dir: str, date_str: str) -> List[Dict]:
    news_file = os.path.join(news_dir, date_str, "news.json")
    if not os.path.exists(news_file):
        return []

    try:
        with open(news_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    news_list = payload.get("news", []) if isinstance(payload, dict) else payload
    selected = []
    for item in news_list:
        if item.get("selected"):
            full_text = item.get("full_text") or item.get("summary") or ""
            selected.append({
                "id": item.get("id"),
                "date": date_str,
                "title": item.get("title", ""),
                "source": item.get("source", "Unknown"),
                "quality_score": item.get("quality_score"),
                "quality_label": item.get("quality_label"),
                "summary": item.get("summary", ""),
                "full_text": full_text,
            })
    return selected


def load_market_analysis_for_date(news_dir: str, date_str: str) -> Optional[str]:
    analysis_file = os.path.join(news_dir, date_str, "market_analysis.txt")
    if not os.path.exists(analysis_file):
        return None
    try:
        with open(analysis_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def aggregate_week_context(news_dir: str,
                           target_date: str,
                           lookback_days: int,
                           min_quality_score: float = 50) -> Optional[Dict]:
    target_dt = parse_date(target_date)
    start_dt = target_dt - timedelta(days=lookback_days - 1)

    gathered_news: List[Dict] = []
    analysis_snippets: List[Tuple[str, str]] = []

    for i in range(lookback_days):
        scan_dt = start_dt + timedelta(days=i)
        scan_str = scan_dt.strftime("%Y-%m-%d")

        day_news = load_news_for_date(news_dir, scan_str)
        for item in day_news:
            if (item.get("quality_score") or 0) >= min_quality_score:
                gathered_news.append(item)

        analysis_text = load_market_analysis_for_date(news_dir, scan_str)
        if analysis_text:
            analysis_snippets.append((scan_str, analysis_text))

    if not gathered_news and not analysis_snippets:
        return None

    gathered_news.sort(key=lambda x: (x.get("quality_score", 0), x["date"]), reverse=True)
    top_news = gathered_news[:12]

    combined_analysis = ""
    if analysis_snippets:
        analysis_snippets.sort(key=lambda x: x[0])
        latest_snippets = analysis_snippets[-3:]
        combined_analysis = "\n\n---\n\n".join(
            f"[{date}] {text[:1200]}"
            for date, text in latest_snippets
        )

    return {
        "date_range": f"{start_dt.strftime('%Y-%m-%d')} ~ {target_date}",
        "news": top_news,
        "analysis": combined_analysis,
    }


# ======================================================
# TradingAgent Call & Dataset Construction
# ======================================================

def build_user_prompt(target_date: str,
                      price_summary: Dict,
                      price_features: Dict,
                      weekly_context: Dict) -> str:
    prices_table = "\n".join([
        f"{date}: ${price}"
        for date, price in price_summary.get("prices_by_date", {}).items()
    ])

    news_lines = []
    for item in weekly_context.get("news", []):
        stars = "⭐" * (2 if item.get("quality_label") == "high" else 1)
        news_lines.append(
            f"- [{item['date']}] [{item['source']}] {stars}\n"
            f"  {item['title'][:120]}"
        )
    news_block = "\n".join(news_lines)

    analysis_block = weekly_context.get("analysis", "")[:2000]

    relative_peak = price_features.get("recent_relative_peak")
    relative_peak_text = f"{relative_peak['date']} @ ${relative_peak['price']}" if relative_peak else "N/A"

    absolute_peak = price_features.get("absolute_peak_3y")
    absolute_peak_text = f"{absolute_peak['date']} @ ${absolute_peak['price']}" if absolute_peak else "N/A"

    prompt = f"""You are evaluating swing trades for gold (XAU/USD) as of {target_date}.

Date range (lookback {price_summary['trading_days']} trading days):
{price_summary['start_date']} → {price_summary['end_date']}

Daily closes:
{prices_table}

Price metrics:
- Close: ${price_features['close']}
- MA30: ${price_features.get('ma_30', 'N/A')}
- MA50: ${price_features.get('ma_50', 'N/A')}
- Recent relative peak (≈ last {relative_peak_text})
- 3y absolute peak: {absolute_peak_text}
- 7-day change: {price_summary['change_pct']}%

High-quality news (past week):
{news_block}

Market analysis excerpts:
{analysis_block}

Current position: FLAT
Task: produce detailed multi-strategy swing trading guidance, including preferred strategy."""

    return prompt


def enrich_decision_with_targets(decision: Dict, close_price: float) -> Dict:
    recommendation = decision.get("recommendation", {})
    action = recommendation.get("action", "HOLD")

    # Generate simple target price and expected profit (can be used as training labels)
    if action == "BUY":
        target_multiplier = random.uniform(1.012, 1.025)
        target_price = close_price * target_multiplier
        profit = target_price - close_price
    elif action == "SELL":
        target_multiplier = random.uniform(0.975, 0.988)
        target_price = close_price * target_multiplier
        profit = close_price - target_price
    else:
        target_price = close_price
        profit = 0.0

    recommendation["target_price"] = round(target_price, 2)
    recommendation["expected_profit"] = round(profit, 2)

    decision["recommendation"] = recommendation
    return decision


def build_dataset_entry(target_date: str,
                        weekly_context: Dict,
                        price_summary: Dict,
                        price_features: Dict,
                        trading_agent: TradingAgent) -> Optional[Dict]:
    if not weekly_context or not price_summary or not price_features:
        return None

    news_items = weekly_context["news"]
    if not news_items:
        return None

    analysis_data = trading_agent.analyze_market_strategies(news_items)
    decision = trading_agent.build_structured_decision(analysis_data)
    decision = enrich_decision_with_targets(decision, price_features["close"])

    prompt = build_user_prompt(target_date, price_summary, price_features, weekly_context)

    assistant_payload = {
        "market_summary": decision.get("market_summary"),
        "strategies": decision.get("strategies"),
        "recommendation": decision.get("recommendation"),
        "price_features": price_features,
        "weekly_context": {
            "date_range": weekly_context["date_range"],
            "news": [
                {
                    "date": n["date"],
                    "source": n["source"],
                    "title": n["title"],
                    "quality_score": n.get("quality_score"),
                    "quality_label": n.get("quality_label")
                }
                for n in news_items
            ],
            "analysis_excerpt": weekly_context.get("analysis", "")[:1000]
        }
    }

    example = {
        "date": target_date,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior gold trading strategist. Provide structured multi-strategy guidance in JSON. "
                    "Include detailed reasoning, confidence, risk assessment, and align with swing-trading horizons."
                )
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_payload, ensure_ascii=False)
            }
        ]
    }

    return example


# ======================================================
# Main Flow
# ======================================================

def iter_target_dates(news_dir: str) -> List[str]:
    pattern = os.path.join(news_dir, "????-??-??", "market_analysis.txt")
    files = sorted(glob.glob(pattern))
    return [os.path.basename(os.path.dirname(path)) for path in files]


def load_prices(price_file: str) -> Dict[str, float]:
    if not os.path.exists(price_file):
        raise FileNotFoundError(f"Price file not found: {price_file}")

    prices, _ = load_gold_prices_from_csv(price_file)
    if not prices:
        raise ValueError("No price records loaded from CSV.")
    return prices


def generate_dataset(output_path: str,
                     news_dir: str,
                     price_file: str,
                     lookback_days: int = 7,
                     min_quality_score: float = 55.0) -> Optional[str]:
    prices = load_prices(price_file)
    target_dates = iter_target_dates(news_dir)

    if not target_dates:
        print("[ERROR] No target dates found with market_analysis.txt")
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    trading_agent = TradingAgent(
        name="DatasetBuilder",
        use_deepseek=False,
        persist_outputs=False
    )

    examples: List[Dict] = []
    for idx, target_date in enumerate(target_dates, 1):
        print(f"\n[{idx}/{len(target_dates)}] Processing {target_date} ...")

        weekly_context = aggregate_week_context(
            news_dir=news_dir,
            target_date=target_date,
            lookback_days=lookback_days,
            min_quality_score=min_quality_score
        )
        if not weekly_context:
            print("  [SKIP] No sufficient weekly context.")
            continue

        price_summary = extract_price_summary(
            prices,
            weekly_context["date_range"].split(" ~ ")[0],
            target_date
        )
        price_features = compute_price_features(prices, target_date)

        if not price_summary or not price_features:
            print("  [SKIP] Missing price summary or features.")
            continue

        example = build_dataset_entry(
            target_date,
            weekly_context,
            price_summary,
            price_features,
            trading_agent
        )
        if example:
            examples.append(example)
            print(f"  ✅ Added example. Recommendation: {json.loads(example['messages'][2]['content'])['recommendation']['action']}")
        else:
            print("  [SKIP] Failed to build example.")

    if not examples:
        print("[ERROR] No dataset examples generated.")
        return None

    with open(output_path, "w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print(f"✅ Generated {len(examples)} examples → {output_path}")
    print("=" * 80)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset via TradingAgent.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--news-dir", default="data/news", help="Directory containing daily news folders.")
    parser.add_argument("--price-file", default="data/gold_history.csv", help="CSV file with historical gold prices.")
    parser.add_argument("--lookback", type=int, default=7, help="Lookback days for aggregation.")
    parser.add_argument("--min-quality", type=float, default=55.0, help="Minimum news quality score.")
    args = parser.parse_args()

    generate_dataset(
        output_path=args.output,
        news_dir=args.news_dir,
        price_file=args.price_file,
        lookback_days=args.lookback,
        min_quality_score=args.min_quality
    )


if __name__ == "__main__":
    main()

