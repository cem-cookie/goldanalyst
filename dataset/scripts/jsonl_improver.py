#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSONL Strategy Improver with OpenAI LLM Enhancement (v3 - Direct Update)
--------------------------------------------------------------------------
Core logic:
1. Load original strategy recommendation (e.g., SELL)
2. Backtest against future price data
3. If the ACTUAL market performance suggests a DIFFERENT action (e.g., BUY),
   directly UPDATE the strategy to the better action
4. Use LLM to generate NEW improved analysis for the updated strategy
5. Output enriched JSON with better action + improved reason

Key: REPLACE strategy if backtest suggests something better,
      don't preserve original - just improve it!
"""

from openai import OpenAI, APIError
import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class StrategyOutcome:
    name: str
    action: str
    target: Optional[float]
    stop: Optional[float]
    confidence: Optional[float]
    outcome: str
    exit_price: float
    exit_date: Optional[str]
    days_in_trade: Optional[int]
    realized_return: float
    realized_return_pct: float
    notes: Optional[str] = None


# ============================================================
# 1. Price utilities
# ============================================================

def load_price_history(csv_path: str) -> Dict[str, float]:
    """Load historical price data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Price CSV not found: {csv_path}")

    prices: Dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue
            date_str = row[0].strip()
            price_str = row[1].strip()
            try:
                price = float(price_str)
            except ValueError:
                continue
            prices[date_str] = price

    return prices


def future_price_path(prices: Dict[str, float], start_date: str, lookahead: int) -> List[Tuple[str, float]]:
    """Extract future price path starting from target date."""
    result: List[Tuple[str, float]] = []
    base = datetime.strptime(start_date, "%Y-%m-%d")
    for i in range(1, lookahead + 1):
        day = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        if day in prices:
            result.append((day, prices[day]))
    return result


# ============================================================
# 2. Parsing helpers
# ============================================================

def extract_user_prompt(record: Dict) -> Optional[str]:
    """Extract user message from record."""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def extract_assistant_payload(record: Dict) -> Optional[Dict]:
    """Extract assistant JSON payload from record."""
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        return None
    return None


def extract_target_date(user_content: str) -> Optional[str]:
    """Extract target date from user content (format: YYYY-MM-DD)."""
    matches = re.findall(r"\b(202\d-\d{2}-\d{2})\b", user_content)
    return matches[-1] if matches else None


def extract_current_price(user_content: str) -> Optional[float]:
    """Extract current price from user content."""
    patterns = [
        r"Current:\s*\$?([\d,]+\.?\d*)",
        r"price[^:]*:\s*\$?([\d,]+\.?\d*)",
        r"[Cc]lose[^:]*:\s*\$?([\d,]+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_content, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))

    return None


# ============================================================
# 3. Strategy evaluation - determine BEST action
# ============================================================

def evaluate_strategies(
        current_price: float,
        path: List[Tuple[str, float]],
) -> Dict[str, any]:
    """
    Evaluate BUY vs SELL vs HOLD against future price movements.
    Returns the BEST action and its performance.
    """

    if not path:
        return {
            "best_action": "HOLD",
            "best_return_pct": 0.0,
            "reasoning": "No future price data available",
            "details": {}
        }

    end_price = path[-1][1]
    max_price = max(p for _, p in path)
    min_price = min(p for _, p in path)

    # Evaluate BUY
    buy_return = (end_price - current_price) / current_price * 100 if current_price else 0
    buy_max_return = (max_price - current_price) / current_price * 100 if current_price else 0

    # Evaluate SELL
    sell_return = (current_price - end_price) / current_price * 100 if current_price else 0
    sell_max_return = (current_price - min_price) / current_price * 100 if current_price else 0

    # Evaluate HOLD
    hold_return = 0.0

    details = {
        "BUY": {
            "final_return": round(buy_return, 2),
            "max_return": round(buy_max_return, 2),
            "end_price": end_price,
        },
        "SELL": {
            "final_return": round(sell_return, 2),
            "max_return": round(sell_max_return, 2),
            "end_price": end_price,
        },
        "HOLD": {
            "return": 0.0,
            "end_price": end_price,
        }
    }

    # Determine best action based on ACTUAL price movement
    best_action = "HOLD"
    best_return_pct = 0.0

    if buy_return > 0 and buy_return > sell_return:
        best_action = "BUY"
        best_return_pct = buy_return
        reasoning = f"Price went UP to {end_price}, BUY would have made {buy_return:+.2f}%"
    elif sell_return > 0 and sell_return > buy_return:
        best_action = "SELL"
        best_return_pct = sell_return
        reasoning = f"Price went DOWN to {end_price}, SELL would have made {sell_return:+.2f}%"
    else:
        best_action = "HOLD"
        best_return_pct = 0.0
        price_change = ((end_price - current_price) / current_price * 100) if current_price else 0
        reasoning = f"Price went {('UP' if price_change > 0 else 'DOWN' if price_change < 0 else 'FLAT')} {price_change:+.2f}%, HOLD is safest"

    return {
        "best_action": best_action,
        "best_return_pct": best_return_pct,
        "reasoning": reasoning,
        "details": details,
        "price_path": {
            "start": current_price,
            "end": end_price,
            "min": min_price,
            "max": max_price,
            "days": len(path)
        }
    }


# ============================================================
# 4. OpenAI LLM Enhancement - Generate improved reason
# ============================================================

def call_openai_for_improved_reason(
        user_prompt: str,
        original_action: str,
        original_reason: str,
        original_style: str,
        updated_action: str,
        backtesting_result: Dict,
        current_price: float,
        client: OpenAI,
        model: str = "gpt-4o-mini",
) -> Dict:
    """
    Generate improved trading reason based on backtest results.
    """

    price_analysis = backtesting_result.get("price_path", {})

    prompt = f"""You are a professional trading analyst. A trading recommendation needs to be improved based on backtesting results.

【Original Recommendation】
Action: {original_action}
Style: {original_style}
Reason: {original_reason}

【Backtesting Results】
Best action based on actual price movements: {updated_action}
Analysis: {backtesting_result.get('reasoning')}

Price movement:
- Start: ${current_price}
- End: ${price_analysis.get('end', 'N/A')}
- Range: ${price_analysis.get('min', 'N/A')} - ${price_analysis.get('max', 'N/A')}
- Days: {price_analysis.get('days', 'N/A')}

Updated action return potential: {backtesting_result.get('best_return_pct', 0):.2f}%

Generate an improved trading recommendation in JSON:
{{
    "action": "{updated_action}",
    "style": "{original_style}",
    "confidence": 0.7,
    "reason": "Clear, concise reason why {updated_action} is the best action based on the price movement and analysis",
    "improvement_note": "Brief explanation of how this differs from or improves the original recommendation"
}}

Return ONLY valid JSON, no markdown."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional trading analyst. Generate improved trading recommendations.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.7,
            max_tokens=600,
        )

        response_text = response.choices[0].message.content

        try:
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                except json.JSONDecodeError:
                    result = {
                        "action": updated_action,
                        "style": original_style,
                        "confidence": 0.7,
                        "reason": backtesting_result.get("reasoning", "Based on backtesting analysis"),
                        "error": "Could not parse LLM response"
                    }
            else:
                result = {
                    "action": updated_action,
                    "style": original_style,
                    "confidence": 0.7,
                    "reason": backtesting_result.get("reasoning", "Based on backtesting analysis"),
                    "error": "Could not parse LLM response"
                }

        return result

    except APIError as e:
        return {
            "action": updated_action,
            "style": original_style,
            "confidence": 0.5,
            "reason": backtesting_result.get("reasoning", ""),
            "error": f"OpenAI API error: {str(e)}"
        }
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        raise RuntimeError("Failed to call OpenAI API.") from e


# ============================================================
# 5. Main pipeline
# ============================================================

def process_dataset(
        input_path: str,
        output_path: str,
        price_csv: str,
        lookahead_days: int,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
) -> None:
    """
    Process dataset with strategy improvement based on backtesting.

    Core logic:
    1. Load strategy recommendation
    2. Backtest against future prices
    3. If backtest suggests different action, UPDATE the strategy
    4. Generate improved reason with LLM
    5. Output improved JSONL
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    prices = load_price_history(price_csv)

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "❌ Missing OPENAI_API_KEY\n"
            "Please set: export OPENAI_API_KEY=sk-...\n"
            "Or provide --api-key parameter"
        )

    client = OpenAI(api_key=api_key)

    total = 0
    improved = 0
    errors = 0
    updated_count = 0  # Count how many strategies were actually changed

    with open(input_path, "r", encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            print(f"[{line_num}] Processing...", end=" ", flush=True)

            try:
                record = json.loads(line)
                user_prompt = extract_user_prompt(record)
                assistant_payload = extract_assistant_payload(record)

                if not user_prompt or not assistant_payload:
                    print("⊘ Missing fields")
                    continue

                target_date = extract_target_date(user_prompt)
                current_price = extract_current_price(user_prompt)

                if not target_date or current_price is None:
                    print(f"⊘ Missing date/price")
                    continue

                future_path = future_price_path(prices, target_date, lookahead_days)

                if not future_path:
                    print(f"⊘ No future price data")
                    continue

                # ============================================================
                # Key step: Evaluate what the BEST action should have been
                # ============================================================
                backtesting_result = evaluate_strategies(current_price, future_path)
                best_action = backtesting_result["best_action"]

                # Get original strategy
                original_action = assistant_payload.get("action", "HOLD").upper()
                original_reason = assistant_payload.get("reason", "")
                original_style = assistant_payload.get("style", "swing")

                # Check if strategy needs updating
                strategy_changed = (original_action != best_action)

                if strategy_changed:
                    print(f"🔄 {original_action}→{best_action}...", end=" ", flush=True)
                    updated_count += 1
                else:
                    print(f"✓ {best_action}...", end=" ", flush=True)

                # ============================================================
                # Generate improved reason with LLM
                # ============================================================
                print("🤖 LLM...", end=" ", flush=True)
                improved_recommendation = call_openai_for_improved_reason(
                    user_prompt,
                    original_action,
                    original_reason,
                    original_style,
                    best_action,
                    backtesting_result,
                    current_price,
                    client,
                    model,
                )

                # Check for errors
                if "error" in improved_recommendation and strategy_changed:
                    print(f"✗ LLM error: {improved_recommendation['error'][:40]}")
                    errors += 1
                    continue

                # ============================================================
                # Build improved record - preserve message structure
                # ============================================================
                improved_record = {
                    "messages": record.get("messages", [])
                }

                # Update only the assistant message
                for i, msg in enumerate(improved_record["messages"]):
                    if msg.get("role") == "assistant":
                        msg["content"] = json.dumps(improved_recommendation, ensure_ascii=False)
                        break

                f_out.write(json.dumps(improved_record, ensure_ascii=False) + "\n")
                improved += 1
                print("✓")

            except json.JSONDecodeError as e:
                print(f"✗ JSON error: {str(e)[:40]}")
                errors += 1
            except Exception as e:
                print(f"✗ {type(e).__name__}: {str(e)[:40]}")
                errors += 1

    print(f"\n{'=' * 60}")
    print(f"Total: {total} | Success: {improved} | Errors: {errors}")
    print(f"Strategies updated: {updated_count}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")


# ============================================================
# 6. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="JSONL strategy improver - Direct update based on backtesting (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy_improver_v3.py input.jsonl output.jsonl --prices prices.csv
  export OPENAI_API_KEY=sk-... && python strategy_improver_v3.py input.jsonl output.jsonl
  python strategy_improver_v3.py input.jsonl output.jsonl --api-key sk-... --lookahead 10
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument("--prices", default="data/gold_history.csv", help="Price CSV file")
    parser.add_argument("--lookahead", type=int, default=5, help="Number of future days to evaluate")
    parser.add_argument("--api-key", help="OpenAI API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")

    args = parser.parse_args()

    try:
        process_dataset(
            input_path=args.input,
            output_path=args.output,
            price_csv=args.prices,
            lookahead_days=args.lookahead,
            api_key=args.api_key,
            model=args.model,
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()