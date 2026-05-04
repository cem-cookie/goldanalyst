"""
Historical News Data Collector
Retrieves news data for the past 30 days using DataAgent
Creates separate JSON files for each day in /data/news/YYYY-MM-DD/news.json
"""

import os
import sys
from datetime import datetime, timedelta
from agents.data_agent import DataAgent
import time


def collect_historical_news(days=30, openai_api_key=None, limit_per_source=10):
    """
    Collect news data for the past N days

    Args:
        days: Number of days to look back (default 30)
        openai_api_key: OpenAI API key (optional, will use env variable if not provided)
        limit_per_source: Number of articles to fetch per source
    """
    print("=" * 70)
    print(f"Historical News Data Collection - Past {days} Days")
    print("=" * 70)

    # Initialize DataAgent
    agent = DataAgent(openai_api_key=openai_api_key)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    print(f"\n[INFO] Collection Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"[INFO] Total days: {days}")
    print(f"[INFO] Articles per source: {limit_per_source}")

    successful_days = 0
    failed_days = 0

    # Iterate through each day
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")

        print("\n" + "=" * 70)
        print(f"Processing: {date_str}")
        print("=" * 70)

        try:
            use_gnews = bool(os.getenv("GNEWS_API_KEY"))

            try:
                if use_gnews:
                    news_list, json_path = agent.collect_past_news_with_gnews(
                        date_str=date_str,
                        query="gold",
                        limit=3 * limit_per_source  # About 30 total
                    )

                else:
                    print(f"[WARN] No GNews items for {date_str}. Skipping fulltext/analyze.")
                    successful_days += 1
                    current_date += timedelta(days=1)
                    continue

                print(f"\n[SUCCESS] Collected {len(news_list)} articles for {date_str}")
                print(f"[INFO] Saved to: {json_path}")

                agent.filter_and_fetch_fulltext(json_path)
                agent.analyze_market_news(json_path)
                successful_days += 1
                time.sleep(2)

            except Exception as e:
                print(f"\n[ERROR] Failed to process {date_str}: {e}")
                failed_days += 1

            print(f"\n[SUCCESS] Collected {len(news_list)} articles for {date_str}")
            print(f"[INFO] Saved to: {json_path}")

            # Filter and fetch full text with quality scoring
            agent.filter_and_fetch_fulltext(json_path)

            # Generate analysis
            agent.analyze_market_news(json_path)

            successful_days += 1

            # Small delay to avoid overwhelming servers
            time.sleep(2)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {date_str}: {e}")
            failed_days += 1

        # Move to next day
        current_date += timedelta(days=1)

    # Final summary
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Total days processed: {days}")
    print(f"Successful: {successful_days}")
    print(f"Failed: {failed_days}")
    print(f"Success rate: {successful_days / days * 100:.1f}%")
    print("\nData saved in: data/news/YYYY-MM-DD/")
    print("=" * 70)


def collect_today_news(openai_api_key=None, limit_per_source=10):
    """
    Convenience function to collect just today's news
    """
    print("=" * 70)
    print("Collecting Today's News")
    print("=" * 70)

    agent = DataAgent(openai_api_key=openai_api_key)

    # Collect today's news
    news_list, json_path = agent.collect_all_news(limit=limit_per_source)

    print(f"\n[SUCCESS] Collected {len(news_list)} articles for today")
    print(f"[INFO] Saved to: {json_path}")

    # Filter and fetch full text with quality scoring
    agent.filter_and_fetch_fulltext(json_path)

    # Generate analysis
    analysis = agent.analyze_market_news(json_path)

    return json_path, analysis


def analyze_existing_news(date_str, openai_api_key=None):
    """
    Analyze already collected news for a specific date

    Args:
        date_str: Date in YYYY-MM-DD format
        openai_api_key: OpenAI API key
    """
    json_path = f"data/news/{date_str}/news.json"

    if not os.path.exists(json_path):
        print(f"[ERROR] No data found for {date_str}")
        print(f"[INFO] Expected path: {json_path}")
        return None

    agent = DataAgent(openai_api_key=openai_api_key)

    print(f"\n[INFO] Analyzing existing news for {date_str}")
    analysis = agent.analyze_market_news(json_path)

    return analysis


if __name__ == "__main__":
    """
    Usage examples:

    # Collect past 30 days of news
    python collect_historical_news.py

    # Collect past 7 days
    python collect_historical_news.py --days 7

    # Collect today only
    python collect_historical_news.py --today

    # Analyze specific date
    python collect_historical_news.py --analyze 2025-11-01
    """

    import argparse

    parser = argparse.ArgumentParser(description='Collect and analyze historical gold market news')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to collect (default: 30)')
    parser.add_argument('--today', action='store_true',
                        help='Collect only today\'s news')
    parser.add_argument('--analyze', type=str,
                        help='Analyze existing news for specific date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=10,
                        help='Articles per source (default: 10)')
    parser.add_argument('--api-key', type=str,
                        help='OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)')

    args = parser.parse_args()

    try:
        if args.analyze:
            # Analyze existing data
            analyze_existing_news(args.analyze, args.api_key)
        elif args.today:
            # Collect today only
            collect_today_news(args.api_key, args.limit)
        else:
            # Collect historical data
            collect_historical_news(args.days, args.api_key, args.limit)

    except KeyboardInterrupt:
        print("\n\n[INFO] Collection interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)