#!/usr/bin/env python3
"""
Dataset Generator v4 INTEGRATED - Final Version

完整流程：
新闻(NEWS) + 历史数据(HISTORY) + MA30/50/200 + Profit/Target
  → 综合到LLM → 最终交易建议

Features:
✅ Load news from data/news/{date}/news.json
✅ Analyze market strategies from news using analyze_market_strategies()
✅ Calculate MA-30, MA-50, MA-200 from full history
✅ Find local peaks (relative + 3-year absolute)
✅ Simulate realistic profit targets based on peaks
✅ Combine ALL factors into single comprehensive prompt
✅ LLM synthesizes everything for final recommendation
"""

import os
import sys
import json
import re
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai not installed")
    sys.exit(1)


def load_prices_from_csv(csv_path="data/gold_history.csv"):
    """Load all historical prices"""
    print(f"[INFO] Loading prices from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        prices = {}
        for _, row in df.iterrows():
            prices[str(row['date']).strip()] = float(row['gold'])
        print(f"[INFO] ✅ Loaded {len(prices)} price records\n")
        return prices
    except Exception as e:
        print(f"[ERROR] {e}")
        return {}


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def calculate_moving_average(prices: Dict[str, float], target_date: str, window: int) -> Optional[float]:
    """Calculate N-day MA from full history"""
    try:
        target_dt = _parse_date(target_date)
        start_dt = target_dt - timedelta(days=window * 2)
        series = [prices[d] for d in sorted(prices.keys())
                  if start_dt <= _parse_date(d) <= target_dt]
        return round(sum(series[-window:]) / window, 2) if len(series) >= window else None
    except:
        return None


def find_local_peaks(prices: Dict[str, float]) -> List[Dict]:
    """Find all local peaks (mountain tops)"""
    sorted_dates = sorted(prices.keys())
    peaks = []
    for i in range(1, len(sorted_dates) - 1):
        p_prev = prices[sorted_dates[i - 1]]
        p_curr = prices[sorted_dates[i]]
        p_next = prices[sorted_dates[i + 1]]
        if p_curr > p_prev and p_curr > p_next:
            peaks.append({"date": sorted_dates[i], "price": round(p_curr, 2)})
    peaks.sort(key=lambda x: x["date"], reverse=True)
    return peaks


def find_recent_relative_peak(prices: Dict[str, float], target_date: str) -> Optional[Dict]:
    """Find the most recent local peak before target date"""
    try:
        target_dt = _parse_date(target_date)
        all_peaks = find_local_peaks(prices)
        for peak in all_peaks:
            if _parse_date(peak["date"]) < target_dt:
                days_ago = (target_dt - _parse_date(peak["date"])).days
                return {"date": peak["date"], "price": peak["price"], "days_ago": days_ago}
        return None
    except:
        return None


def find_absolute_peak(prices: Dict[str, float], target_date: str, years: int = 3) -> Optional[Dict]:
    """Find 3-year highest peak"""
    try:
        target_dt = _parse_date(target_date)
        start_dt = target_dt - timedelta(days=365 * years)
        candidates = [(d, prices[d]) for d in prices
                      if start_dt <= _parse_date(d) <= target_dt]
        if not candidates:
            return None
        peak_date, peak_price = max(candidates, key=lambda x: x[1])
        return {"date": peak_date, "price": round(peak_price, 2)}
    except:
        return None


def compute_price_features(prices: Dict[str, float], target_date: str) -> Optional[Dict]:
    """Compute all technical indicators"""
    if target_date not in prices:
        return None

    ma_30 = calculate_moving_average(prices, target_date, 30)
    ma_50 = calculate_moving_average(prices, target_date, 50)
    ma_200 = calculate_moving_average(prices, target_date, 200)
    rel_peak = find_recent_relative_peak(prices, target_date)
    abs_peak = find_absolute_peak(prices, target_date)

    return {
        "date": target_date,
        "close": round(prices[target_date], 2),
        "ma_30": ma_30,
        "ma_50": ma_50,
        "ma_200": ma_200,
        "recent_relative_peak": rel_peak,
        "absolute_peak_3y": abs_peak
    }


def load_news_for_date(target_date: str) -> List[Dict]:
    """Load news from data/news/{date}/news.json"""
    try:
        news_file = f"data/news/{target_date}/news.json"
        if not os.path.exists(news_file):
            return []
        with open(news_file, "r") as f:
            data = json.load(f)
        news_items = data.get("news", data) if isinstance(data, dict) else data
        selected = [n for n in news_items if n.get("selected") and n.get("full_text")]
        return selected[:10]
    except:
        return []


def analyze_market_strategies(client: OpenAI, news_items: List[Dict]) -> Optional[Dict]:
    """Generate strategies from news using TradingAgent method"""
    if not news_items:
        return None

    summaries = "\n\n".join(
        [f"[{n['source']}] {n['title']}\n{n['full_text'][:800]}" for n in news_items[:5]]
    )

    prompt = f"""You are a professional trading strategist specialized in gold.
Based on these news, propose 3 strategies (Conservative/Balanced/Aggressive).
For each: name, action (BUY/SELL/HOLD), rationale, confidence (0-5), risk, return.
Also provide market sentiment and trend.

Output ONLY valid JSON:
{{
  "sentiment": "bullish|bearish|neutral",
  "trend": "uptrend|downtrend|sideways",
  "strategies": [
    {{"name": "...", "action": "...", "rationale": "...", "confidence": 0, "expected_risk": "...", "expected_return": "..."}}
  ],
  "recommendation": "strategy name",
  "reasoning": "why optimal"
}}

NEWS:
{summaries}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text, flags=re.MULTILINE)
        return json.loads(text)
    except:
        return None


def build_comprehensive_prompt(
        price_features: Dict,
        price_range: Dict,
        news_analysis: Optional[Dict] = None
) -> str:
    """
    Build prompt: News + History + MA + Peaks + Targets → LLM → Recommendation

    Args:
        price_features: Dict with close, ma_30, ma_50, ma_200, recent_relative_peak, absolute_peak_3y
        price_range: Dict with dates as keys and prices as values (historical data)
        news_analysis: Optional dict with sentiment, trend, recommendation, reasoning

    Returns:
        Formatted prompt string for LLM analysis
    """

    # ==================== STEP 1: BUILD PRICES TABLE ====================
    prices_table = "\n".join([
        f"  {d}: ${p:,.2f}"
        for d, p in list(price_range.items())[-7:]
    ])

    # ==================== STEP 2: EXTRACT PRICE FEATURES ====================
    current = price_features["close"]
    ma_30 = price_features.get("ma_30")
    ma_50 = price_features.get("ma_50")
    ma_200 = price_features.get("ma_200")
    rel_peak = price_features.get("recent_relative_peak")
    abs_peak = price_features.get("absolute_peak_3y")

    # ==================== STEP 3: CALCULATE TARGETS ====================
    # Conservative & Moderate targets based on relative peak
    if rel_peak and rel_peak['price'] > current:
        dist = rel_peak['price'] - current
        t1 = current + (dist * 0.3)
        t2 = current + (dist * 0.55)
    else:
        t1 = current * 1.015
        t2 = current * 1.03

    # Aggressive target based on absolute peak
    if abs_peak and abs_peak['price'] > current:
        dist = abs_peak['price'] - current
        t3 = current + (dist * 0.75)
    else:
        t3 = current * 1.05

    # ==================== STEP 4: CALCULATE STOPS ====================
    stop_conservative = current * 0.99
    stop_moderate = current * 0.98
    stop_aggressive = current * 0.97

    # ==================== STEP 5: FORMAT MA LEVELS ====================
    ma_30_str = f"${ma_30:,.2f}" if ma_30 else "N/A"
    ma_50_str = f"${ma_50:,.2f}" if ma_50 else "N/A"
    ma_200_str = f"${ma_200:,.2f}" if ma_200 else "N/A"

    # ==================== STEP 6: FORMAT PEAK LEVELS ====================
    if rel_peak:
        rel_peak_str = f"{rel_peak['date']} @ ${rel_peak['price']:,.2f}"
    else:
        rel_peak_str = "N/A"

    if abs_peak:
        abs_peak_str = f"{abs_peak['date']} @ ${abs_peak['price']:,.2f}"
    else:
        abs_peak_str = "N/A"

    # ==================== STEP 7: FORMAT PROFIT PERCENTAGES ====================
    t1_pct = ((t1 - current) / current * 100)
    t2_pct = ((t2 - current) / current * 100)
    t3_pct = ((t3 - current) / current * 100)

    # ==================== STEP 8: BUILD NEWS SECTION ====================
    if news_analysis:
        news_text = f"""
NEWS-BASED ANALYSIS:
- Sentiment: {news_analysis.get('sentiment', 'N/A')}
- Trend: {news_analysis.get('trend', 'N/A')}
- Recommendation: {news_analysis.get('recommendation', 'N/A')}
- Reasoning: {news_analysis.get('reasoning', 'N/A')}
"""
    else:
        news_text = "(No news available)"

    # ==================== STEP 9: BUILD COMPLETE PROMPT ====================
    prompt = f"""Synthesize ALL the following data to provide a COHERENT trading recommendation:

📰 NEWS + SENTIMENT
{news_text}

📊 PRICE DATA (7-day window)
{prices_table}

📈 TECHNICAL INDICATORS (from full history)
- Current: ${current:,.2f}
- MA-30: {ma_30_str}
- MA-50: {ma_50_str}
- MA-200: {ma_200_str}

🏔️ PEAK LEVELS (Real data)
- Previous Peak: {rel_peak_str}
- 3-Year Peak: {abs_peak_str}

💰 PROFIT/TARGET LEVELS
- Conservative: ${t1:,.2f} ({t1_pct:+.2f}%)
- Moderate: ${t2:,.2f} ({t2_pct:+.2f}%)
- Aggressive: ${t3:,.2f} ({t3_pct:+.2f}%)

===== PROVIDE =====
1. Market assessment (trend + sentiment) citing BOTH news and technicals
2. Three strategies WITH specific targets/stops and WHY
3. Final recommendation synthesizing NEWS + TECHNICALS

Output ONLY valid JSON:
{{
  "market_assessment": {{
    "trend": "uptrend|downtrend|sideways",
    "sentiment": "bullish|bearish|neutral",
    "analysis": "detailed analysis citing both news and technical indicators"
  }},
  "strategies": [
    {{
      "name": "Conservative",
      "action": "BUY|SELL|HOLD",
      "target": {t1},
      "stop": {stop_conservative},
      "confidence": 8,
      "rationale": "explanation based on risk/reward and technicals"
    }},
    {{
      "name": "Moderate",
      "action": "BUY|SELL|HOLD",
      "target": {t2},
      "stop": {stop_moderate},
      "confidence": 7,
      "rationale": "explanation balancing aggression and safety"
    }},
    {{
      "name": "Aggressive",
      "action": "BUY|SELL|HOLD",
      "target": {t3},
      "stop": {stop_aggressive},
      "confidence": 6,
      "rationale": "explanation for high-risk, high-reward scenario"
    }}
  ],
  "recommendation": {{
    "action": "BUY|SELL|HOLD",
    "target": {t2},
    "stop": {stop_moderate},
    "reasoning": "Final synthesis aligning news sentiment + technical indicators + market structure"
  }}
}}
"""

    return prompt

# Simulate targets and profits for the dataset
def simulate_targets_and_profits(data):
    """Generate realistic target prices based on actual peaks"""
    import random
    random.seed(int(data * 100))

    strategies = []

    # Strategy 1: Conservative (30% toward recent peak)
    if rel_peak and rel_peak['price'] > current_price:
        peak_price = rel_peak['price']
        dist = peak_price - current_price
        target = current_price + (dist * 0.3)
        stop = current_price - (current_price * 0.01)
        profit = target - current_price
    else:
        target = current_price * 1.015
        stop = current_price * 0.99
        profit = target - current_price

    strategies.append({
        "name": "Conservative",
        "target": round(target, 2),
        "stop": round(stop, 2),
        "profit": round(profit, 2),
        "profit_pct": round((profit / current_price) * 100, 2)
    })

    # Strategy 2: Moderate (50-60% toward recent peak)
    if rel_peak and rel_peak['price'] > current_price:
        peak_price = rel_peak['price']
        dist = peak_price - current_price
        target = current_price + (dist * random.uniform(0.5, 0.6))
        stop = current_price - (current_price * 0.02)
        profit = target - current_price
    else:
        target = current_price * 1.03
        stop = current_price * 0.98
        profit = target - current_price

    strategies.append({
        "name": "Moderate",
        "target": round(target, 2),
        "stop": round(stop, 2),
        "profit": round(profit, 2),
        "profit_pct": round((profit / current_price) * 100, 2)
    })

    # Strategy 3: Aggressive (70-80% toward 3-year peak)
    if abs_peak and abs_peak['price'] > current_price:
        peak_price = abs_peak['price']
        dist = peak_price - current_price
        target = current_price + (dist * random.uniform(0.7, 0.8))
        stop = current_price - (current_price * 0.035)
        profit = target - current_price
    else:
        target = current_price * 1.05
        stop = current_price * 0.97
        profit = target - current_price

    strategies.append({
        "name": "Aggressive",
        "target": round(target, 2),
        "stop": round(stop, 2),
        "profit": round(profit, 2),
        "profit_pct": round((profit / current_price) * 100, 2)
    })

    return strategies


def call_llm(client: OpenAI, prompt: str) -> Optional[Dict]:
    """Call LLM with comprehensive prompt"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Synthesize news + technical data. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1500
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text, flags=re.MULTILINE)
        return json.loads(text)
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


def generate_dataset():
    """Main flow"""
    print("=" * 80)
    print(" DATASET GENERATOR v4 INTEGRATED")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY")
        return

    client = OpenAI(api_key=api_key)
    print("✅ OpenAI initialized")

    # Load data
    prices = load_prices_from_csv("data/gold_history.csv")
    if not prices:
        return

    sorted_dates = sorted(prices.keys())
    filtered = [d for d in sorted_dates if "2025-10-08" <= d <= "2025-10-18"]

    print(f"✅ Processing {len(filtered)} dates\n")

    examples = []
    for idx, target_date in enumerate(filtered, 1):
        print(f"[{idx}/{len(filtered)}] {target_date}", end=" ")

        # Get 7-day price range
        target_dt = _parse_date(target_date)
        start_dt = target_dt - timedelta(days=6)
        price_range = {}
        current_day = start_dt
        while current_day <= target_dt:
            day_str = current_day.strftime("%Y-%m-%d")
            if day_str in prices:
                price_range[day_str] = prices[day_str]
            current_day += timedelta(days=1)

        if not price_range:
            print("❌ (no price range)")
            continue

        price_features = compute_price_features(prices, target_date)
        if not price_features:
            print("❌ (no features)")
            continue

        # Get news strategies
        news_items = load_news_for_date(target_date)
        news_analysis = analyze_market_strategies(client, news_items) if news_items else None
        print(f"[news:{len(news_items)}]", end=" ")

        # Build and call LLM
        prompt = build_comprehensive_prompt(price_features, price_range, news_analysis)
        result = call_llm(client, prompt)

        if not result:
            print("❌ (LLM failed)")
            continue

        # Store example
        example = {
            "messages": [
                {"role": "system", "content": "You are a professional gold trading analyst."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)}
            ]
        }
        examples.append(example)
        print(f"✅")

    # Save
    if examples:
        with open("training_data_v4_integrated.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\n✅ Generated {len(examples)} examples")
        print(f"   Saved: training_data_v4_integrated.jsonl")


if __name__ == "__main__":
    generate_dataset()