from agents.data_agent import DataAgent

def main():
    agent = DataAgent()

    # Collect
    agent.collect_all_news(limit=10)

    # filter & fetch
    agent.filter_and_fetch_fulltext()

    # Comprehensive Analysis
    agent.analyze_market_news()


if __name__ == "__main__":
    main()
