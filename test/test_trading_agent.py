"""
test_trading_agent.py
Test the complete execution flow of TradingAgent:
1. Read news from gold_news.json
2. Use LLM to analyze market sentiment
3. Generate trading suggestions
4. Output to JSON file
"""

import os
import json
from agents.trading_agent import TradingAgent


def test_trading_agent():
    print("\n========== TEST: TradingAgent ==========")

    # 1. Path setup
    json_path = "data/gold_news.json"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(json_path):
        print(f"[ERROR] {json_path} not found. Please run DataAgent to generate this file.")
        return

    # 2. Initialize Agent
    agent = TradingAgent(json_path=json_path)

    # 3. Run analysis and decision-making
    decision = agent.run()

    # 4. Verify output files are generated
    analysis_path = "data/market_analysis.json"
    decision_path = "data/trading_decision.json"

    print("\n========== RESULT CHECK ==========")
    for path in [analysis_path, decision_path]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            print(f"✅ {os.path.basename(path)} 生成成功")
            print(json.dumps(content, indent=2, ensure_ascii=False)[:500], "...\n")
        else:
            print(f"❌ {path} 未生成")

    print("========== TEST FINISHED ==========\n")


if __name__ == "__main__":
    test_trading_agent()
