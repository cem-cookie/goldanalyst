#!/usr/bin/env python3
"""
Helper script to load price data from gold_history.csv.
"""

import csv
from datetime import datetime
from typing import Dict, Tuple


def load_gold_prices_from_csv(csv_file: str) -> Tuple[Dict[str, float], str]:
    """
    Load historical gold prices from CSV file.

    Expected CSV format:
    date,price
    2025-08-26,3000.00
    2025-08-27,3404.60
    ...

    Or other common formats, auto-detect delimiter and date/price columns.

    Returns:
        (prices_dict, format_description)
        prices_dict: {date_str: price_float}
        format_description: Data format description
    """

    prices = {}

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Try to auto-detect delimiter
            sample = f.read(1024)
            f.seek(0)

            # Try to detect delimiter
            delimiter = ','
            if '\t' in sample:
                delimiter = '\t'
            elif ';' in sample:
                delimiter = ';'

            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)

            if not header:
                print("CSV file is empty")
                return {}, "Empty file"

            print(f"Detected columns: {header}")

            # Find date and price columns
            date_col = None
            price_col = None

            for i, col in enumerate(header):
                col_lower = col.lower().strip()
                if any(x in col_lower for x in ['date', 'time', 'date']):
                    date_col = i
                if any(x in col_lower for x in ['price', 'close', 'price', 'close']):
                    price_col = i

            # If not found, use default indices
            if date_col is None:
                date_col = 0
            if price_col is None:
                price_col = 1

            print(f"Using column {date_col} for date, column {price_col} for price")

            # Read data
            for row_idx, row in enumerate(reader):
                if len(row) <= max(date_col, price_col):
                    continue

                try:
                    date_str = row[date_col].strip()
                    price_str = row[price_col].strip()

                    # Try to parse price
                    price = float(price_str)

                    # Normalize date format to YYYY-MM-DD
                    date_formatted = _parse_date(date_str)

                    if date_formatted:
                        prices[date_formatted] = price
                except (ValueError, IndexError) as e:
                    if row_idx < 5:  # Only print first few errors
                        print(f"  Row {row_idx + 2} parse error: {row}")

        print(f"\nSuccessfully loaded {len(prices)} price records")
        print(f"Date range: {min(prices.keys())} to {max(prices.keys())}")
        print(f"Price range: ${min(prices.values()):.2f} - ${max(prices.values()):.2f}")

        return prices, f"Loaded {len(prices)} records from CSV"

    except Exception as e:
        print(f"Load failed: {e}")
        return {}, str(e)


def _parse_date(date_str: str) -> str:
    """Convert various date formats to YYYY-MM-DD."""

    # Already correct format
    if isinstance(date_str, str) and len(date_str) == 10 and date_str[4] == '-':
        return date_str

    # Try multiple formats
    formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%m-%d-%Y',
        '%m/%d/%Y',
        '%d-%m-%Y',
        '%d/%m/%Y',
        '%Y%m%d',
        '%m-%d-%y',
        '%Y.%m.%d',
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime('%Y-%m-%d')
        except:
            continue

    return None


def validate_prices_for_improver(prices_dict: Dict[str, float]) -> bool:
    """Validate if price data is suitable for the improver."""

    if not prices_dict:
        print("Price dictionary is empty")
        return False

    # Check date format
    for date in list(prices_dict.keys())[:5]:
        if not isinstance(date, str) or len(date) != 10:
            print(f"Invalid date format: {date}")
            return False

    # Check price values
    for price in list(prices_dict.values())[:5]:
        if not isinstance(price, (int, float)) or price <= 0:
            print(f"Invalid price value: {price}")
            return False

    print("Price data validation passed")
    return True


def create_sample_jsonl_with_prices(prices_dict: Dict[str, float],
                                    output_file: str = "training_data_raw.jsonl"):
    """
    Create sample JSONL file with price data for testing the improver.
    """

    import json

    sample_records = [
        {
            "date": "2025-08-26",
            "prices": {
                "current": prices_dict.get("2025-08-26", 3000),
                "historical": {}
            },
            "strategy": {
                "recommendation": "BUY",
                "confidence": 7,
                "reasoning": ["Gold price at low level", "Economic uncertainty rising"]
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze gold trend for 2025-08-26"
                },
                {
                    "role": "assistant",
                    "content": "Recommend buy, expect upward movement in 7 days"
                }
            ]
        },
        {
            "date": "2025-08-27",
            "prices": {
                "current": prices_dict.get("2025-08-27", 3404),
                "historical": {}
            },
            "strategy": {
                "recommendation": "HOLD",
                "confidence": 5,
                "reasoning": ["Trend unclear"]
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Recommendation for 2025-08-27"
                }
            ]
        }
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nCreated sample JSONL: {output_file}")


# ============================================================
# Usage Example
# ============================================================

if __name__ == "__main__":
    import sys

    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/gold_history.csv"

    print(f"Loading {csv_file}...")
    prices, description = load_gold_prices_from_csv(csv_file)

    if prices:
        validate_prices_for_improver(prices)

        # Create sample JSONL
        create_sample_jsonl_with_prices(prices)

        print("\nNow you can run:")
        print("   python jsonl_improver.py")
        print("\nIf you need your own JSONL file, ensure each record contains:")
        print("  - 'date' field (YYYY-MM-DD format)")
        print("  - 'prices' field with 'current' price")
        print("  - 'strategy' field with recommendation and confidence")