"""
Demo script to showcase the improved news collection system with quality scoring
"""

import os
import json
from datetime import datetime
from agents.data_agent import DataAgent


def demo_quality_scoring():
    """
    Demonstrate the quality scoring system with sample news items
    """
    print("=" * 70)
    print("DEMO: Quality Scoring System")
    print("=" * 70)

    # Create sample news items
    sample_news = [
        {
            "id": 1,
            "source": "InvestingRSS",
            "title": "Federal Reserve Signals Potential Rate Cut Amid Inflation Concerns",
            "summary": "Central bank officials hint at policy shift as inflation data shows cooling...",
            "url": "https://example.com/article1",
            "date": "2025-11-05",
            "selected": True,
            "full_text": "Federal Reserve officials indicated today that interest rate cuts may be on the horizon as recent inflation data suggests cooling price pressures. The central bank's decision could have significant implications for gold markets, as lower interest rates typically boost XAU/USD. Market participants are closely watching Treasury yields..." * 3,
            "quality_score": None,
            "quality_label": None
        },
        {
            "id": 2,
            "source": "YahooRSS",
            "title": "Gold Prices Steady Ahead of CPI Release",
            "summary": "Investors await key inflation data...",
            "url": "https://example.com/article2",
            "date": "2025-11-05",
            "selected": True,
            "full_text": "Gold futures remained relatively stable in morning trading as market participants awaited the upcoming CPI data release. Analysts suggest that stronger-than-expected inflation could support gold prices." * 2,
            "quality_score": None,
            "quality_label": None
        },
        {
            "id": 3,
            "source": "Unknown",
            "title": "You Won't Believe What's Happening to Gold Prices!",
            "summary": "Shocking developments in the market...",
            "url": "https://example.com/article3",
            "date": "2025-11-05",
            "selected": False,
            "full_text": None,
            "quality_score": None,
            "quality_label": None
        },
        {
            "id": 4,
            "source": "MetalsDaily",
            "title": "Gold Mining Production Increases in Q3",
            "summary": "Major producers report output gains...",
            "url": "https://example.com/article4",
            "date": "2025-11-05",
            "selected": True,
            "full_text": "Leading gold mining companies reported increased production in the third quarter, with several operations exceeding targets. The rise in supply could put downward pressure on prices if demand remains stable.",
            "quality_score": None,
            "quality_label": None
        }
    ]

    # Initialize agent
    agent = DataAgent()

    print("\n📰 Sample News Items Before Scoring:")
    print("-" * 70)
    for news in sample_news:
        print(f"\n[{news['id']}] {news['source']}")
        print(f"    Title: {news['title']}")
        print(f"    Selected: {news['selected']}")
        print(f"    Has Full Text: {news['full_text'] is not None}")

    # Calculate quality scores
    print("\n\n🎯 Calculating Quality Scores...")
    print("-" * 70)

    for news in sample_news:
        score_data = agent._calculate_quality_score(news)
        news['quality_score'] = score_data['score']
        news['quality_label'] = score_data['quality']

        print(f"\n[{news['id']}] {news['source']}")
        print(f"    Title: {news['title'][:60]}...")
        print(f"    Score: {news['quality_score']}/100")
        print(f"    Quality: {news['quality_label'].upper()}")

        # Explain scoring
        base = agent.config['sources'].get(news['source'], agent.config['sources']['Unknown'])['base_score']
        print(f"    Base Score: {base}")

        if news['selected']:
            print(f"    + LLM Selected: +{agent.config['llm_selection']['selected_bonus']}")
        else:
            print(f"    - Not Selected: {agent.config['llm_selection']['not_selected_penalty']}")

        if news['full_text']:
            print(f"    + Has Full Text: +10")
            if len(news['full_text']) > 1000:
                print(f"    + Long Article: +5")

    # Show filtering results
    print("\n\n🔍 Filtering Results:")
    print("-" * 70)

    min_score = agent.config['thresholds']['minimum_score']
    high_quality = [n for n in sample_news if n['quality_score'] >= agent.config['thresholds']['high_quality']]
    medium_quality = [n for n in sample_news if
                      min_score <= n['quality_score'] < agent.config['thresholds']['high_quality']]
    low_quality = [n for n in sample_news if n['quality_score'] < min_score]

    print(f"\nMinimum Score Threshold: {min_score}")
    print(f"High Quality (≥75): {len(high_quality)} articles")
    print(f"Medium Quality (50-74): {len(medium_quality)} articles")
    print(f"Low Quality (<50): {len(low_quality)} articles")

    print("\n✅ Would be INCLUDED in analysis (score ≥ {}):".format(min_score))
    for news in sample_news:
        if news['quality_score'] >= min_score:
            print(f"   [{news['quality_score']:.1f}] {news['title'][:60]}...")

    print("\n❌ Would be EXCLUDED from analysis (score < {}):".format(min_score))
    for news in sample_news:
        if news['quality_score'] < min_score:
            print(f"   [{news['quality_score']:.1f}] {news['title'][:60]}...")

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


def demo_config_modification():
    """
    Show how to modify scoring configuration
    """
    print("\n" + "=" * 70)
    print("DEMO: Configuration Modification")
    print("=" * 70)

    config_path = "news_quality_scorer.yaml"

    print(f"\n📝 Current configuration loaded from: {config_path}")

    agent = DataAgent()

    print("\n🎯 Current Source Scoring:")
    print("-" * 70)
    for source, config in agent.config['sources'].items():
        print(f"{source:20s} Base Score: {config['base_score']}")

    print("\n🎯 Current Thresholds:")
    print("-" * 70)
    for threshold, value in agent.config['thresholds'].items():
        print(f"{threshold:20s} {value}")

    print("\n💡 To modify scoring:")
    print("1. Edit 'news_quality_scorer.yaml'")
    print("2. Adjust base_score values for sources")
    print("3. Modify thresholds for quality levels")
    print("4. Add new sources or keywords")


def demo_date_based_storage():
    """
    Show the date-based storage structure
    """
    print("\n" + "=" * 70)
    print("DEMO: Date-Based Storage Structure")
    print("=" * 70)

    print("\n📁 Expected Directory Structure:")
    print("-" * 70)
    print("""
data/
└── news/
    ├── 2025-10-06/
    │   ├── news.json              # News data with quality scores
    │   └── market_analysis.txt    # Generated analysis
    ├── 2025-10-07/
    │   ├── news.json
    │   └── market_analysis.txt
    ├── 2025-10-08/
    │   └── ...
    └── 2025-11-05/               # Today
        ├── news.json
        └── market_analysis.txt
    """)

    print("\n📋 Sample JSON Structure:")
    print("-" * 70)

    sample_structure = {
        "collection_date": "2025-11-05",
        "collection_timestamp": "2025-11-05T10:30:00",
        "total_items": 30,
        "selected_count": 12,
        "average_quality_score": 72.5,
        "high_quality_count": 5,
        "medium_quality_count": 6,
        "low_quality_count": 1,
        "news": [
            {
                "id": 1,
                "source": "InvestingRSS",
                "title": "Sample article title",
                "quality_score": 85.5,
                "quality_label": "high",
                "selected": True
            }
        ]
    }

    print(json.dumps(sample_structure, indent=2))


def demo_analysis_workflow():
    """
    Show the complete analysis workflow
    """
    print("\n" + "=" * 70)
    print("DEMO: Complete Analysis Workflow")
    print("=" * 70)

    print("""
STEP 1: Collect News
    ↓
    agent.collect_all_news(limit=10)
    - Fetches from YahooRSS
    - Fetches from MetalsDaily
    - Fetches from InvestingRSS
    - Saves to data/news/YYYY-MM-DD/news.json

STEP 2: Filter & Score
    ↓
    agent.filter_and_fetch_fulltext(json_path)
    - LLM filters relevant articles
    - Fetches full text for selected articles
    - Calculates quality scores
    - Updates JSON with scores

STEP 3: Generate Analysis
    ↓
    agent.analyze_market_news(json_path)
    - Filters by minimum quality score
    - Sorts by quality (highest first)
    - Sends to LLM with quality indicators
    - Generates quality-weighted analysis
    - Saves to market_analysis.txt

RESULT: High-quality, reliable market analysis
    """)


if __name__ == "__main__":
    """
    Run all demos
    """
    print("\n")
    print("=" * 70)
    print("🚀 IMPROVED NEWS COLLECTION SYSTEM DEMO")
    print("=" * 70)

    print("\nThis demo showcases the key features of the improved system:")
    print("1. Quality scoring based on source and content")
    print("2. Configuration management")
    print("3. Date-based storage structure")
    print("4. Complete analysis workflow")

    # Run demos
    demo_quality_scoring()
    demo_config_modification()
    demo_date_based_storage()
    demo_analysis_workflow()

    print("\n" + "=" * 70)
    print("✅ All demos completed!")
    print("=" * 70)

    print("\n📖 Next Steps:")
    print("1. Review 'README_IMPROVED_SYSTEM.md' for full documentation")
    print("2. Run 'python collect_historical_news.py --today' to collect real data")
    print("3. Adjust 'news_quality_scorer.yaml' to customize scoring")
    print("4. Use 'python collect_historical_news.py --days 30' for historical data")