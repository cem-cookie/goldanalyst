import os
from agents.data_agent import DataAgent

if __name__ == "__main__":
    agent = DataAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))
    df = agent.save_gold_history_to_file(days=1095)
    if df is not None and not df.empty:
        print("✓ Gold price data updated successfully!")
    else:
        print("✗ Gold price data update failed.")
