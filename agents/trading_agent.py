import os
import re
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class TradingAgent:
    """Trading Agent: Manages account state and trading decisions"""

    def __init__(self, name="Agent", api_key=None, initial_cash=100_000.0, max_alloc=0.5,
                 fee_bps=10, json_path="data/gold_news.json",
                 persist_outputs=True, context=None, model_name="gpt-4o-mini"):
        self.model = model_name
        self.name = name
        self.api_key = api_key
        self.max_alloc = max_alloc
        self.fee_bps = fee_bps
        self.use_deepseek = False  # DeepSeek removed
        # Initialize LLM client – only OpenAI (ChatGPT) for now
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model_name


    def generate_trade_decision(self, user_prompt: str) -> dict:
        """Generate a single trading decision (BUY/SELL/HOLD) with fixed JSON structure."""
        system_prompt = """You are a professional gold trading analyst.
Analyse the provided market context and produce a single swing-trading decision.
Respond ONLY with valid JSON in this exact schema:
{
  "action": "BUY" | "SELL" | "HOLD",
  "reasoning": "Concise justification referencing price action and catalysts",
  "confidence": 1-10,
  "key_factors": ["factor1", "factor2"],
  "risk_level": "low" | "medium" | "high"
}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            raw_response = response.choices[0].message.content.strip()
            clean_text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw_response, flags=re.MULTILINE)
            strategy = json.loads(clean_text)

            # Ensure key fields exist
            strategy.setdefault("action", "HOLD")
            strategy.setdefault("reasoning", "No reasoning provided.")
            strategy.setdefault("confidence", 5)
            strategy.setdefault("key_factors", [])
            strategy.setdefault("risk_level", "medium")
            return strategy
        except Exception as exc:
            print(f"[ERROR] TradingAgent generate_trade_decision failed: {exc}")
            return {
                "action": "HOLD",
                "reasoning": f"Error: {exc}",
                "confidence": 1,
                "key_factors": [],
                "risk_level": "medium",
            }

    def decide(self, market_summary: str, gold_price: float) -> dict:
        """Method that subclasses must implement."""
        raise NotImplementedError("Subclass must implement decide()")
    # ======================================================
    # 1. Read News
    # ======================================================
    def load_news(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"{self.json_path} not found.")
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both formats: list or dict with "news" key
        if isinstance(data, dict):
            news = data.get("news", [])
        else:
            news = data
        
        # Ensure news is a list
        if not isinstance(news, list):
            print(f"[WARN] Unexpected news format: {type(news)}")
            return []
        
        # Filter for selected items with full text
        selected_news = []
        for n in news:
            if isinstance(n, dict) and n.get("selected") and n.get("full_text"):
                selected_news.append(n)
        
        return selected_news

    # ======================================================
    # 2. Call LLM to generate multi-strategy analysis
    # ======================================================
    def analyze_market_strategies(self, news_items):
        summaries = "\n\n".join(
            [f"[{n['source']}] {n['title']}\n{n['full_text'][:800]}" for n in news_items]
        )

        prompt = f"""
        You are a professional trading strategist specialized in gold (XAU/USD).
        Based on the following recent news, propose 3 alternative trading strategies:
        1. Conservative (low risk)
        2. Balanced / Neutral
        3. Aggressive (high risk, high reward)

        For each strategy, include:
        - name
        - action (BUY / SELL / HOLD)
        - rationale
        - confidence (0–5)
        - expected_risk (Low/Medium/High)
        - expected_return (Low/Medium/High)

        Also provide a high-level summary of the market (sentiment and trend)
        and recommend which strategy is optimal, with reasoning.

        Output ONLY valid JSON in this format:
        {{
          "sentiment": "...",
          "trend": "...",
          "strategies": [
            {{
              "name": "...",
              "action": "...",
              "rationale": "...",
              "confidence": ...,
              "expected_risk": "...",
              "expected_return": "..."
            }}
          ],
          "recommendation": "...",
          "reasoning": "..."
        }}

        News:
        {summaries}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert financial strategist who outputs valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        raw_output = response.choices[0].message.content.strip()
        print(f"\n[LLM RAW OUTPUT]\n{raw_output}\n")

        # Clean```json
        clean_text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw_output.strip(), flags=re.MULTILINE)

        try:
            analysis_data = json.loads(clean_text)
        except Exception as e:
            print(f"[WARN] Failed to parse JSON: {e}")
            analysis_data = {
                "sentiment": "neutral",
                "trend": "unknown",
                "strategies": [
                    {"name": "Fallback Hold", "action": "HOLD", "rationale": raw_output,
                     "confidence": 2, "expected_risk": "Medium", "expected_return": "Medium"}
                ],
                "recommendation": "Fallback Hold",
                "reasoning": "Parsing failed; fallback to HOLD."
            }

        analysis_data["timestamp"] = datetime.utcnow().isoformat()

        # Save Analysis File
        if self.persist_outputs:
            with open("data/market_analysis.json", "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            print("[INFO] Saved to data/market_analysis.json")

        return analysis_data

    # ======================================================
    # 3.organize Structure Output
    # ======================================================
    def build_structured_decision(self, analysis_data):
        strategies = analysis_data.get("strategies", [])
        if not strategies:
            print("[WARN] No strategies found, fallback to HOLD.")
            return {"recommended_action": "HOLD", "confidence": 2}

        # Best Strategy
        best = max(strategies, key=lambda s: float(s.get("confidence", 0)))
        best_id = strategies.index(best) + 1

        decision = {
            "timestamp": datetime.utcnow().isoformat(),
            "asset": "XAU/USD",
            "market_summary": {
                "sentiment": analysis_data.get("sentiment", "neutral"),
                "trend": analysis_data.get("trend", "unknown"),
                "volatility": "high" if any(
                    "volatility" in s.get("rationale", "").lower() for s in strategies) else "normal",
                "comment": analysis_data.get("reasoning", "")
            },
            "strategies": [
                {**s, "id": i + 1} for i, s in enumerate(strategies)
            ],
            "recommendation": {
                "strategy_id": best_id,
                "name": best.get("name"),
                "action": best.get("action"),
                "confidence": best.get("confidence"),
                "reason": analysis_data.get("reasoning", best.get("rationale", ""))
            }
        }

        if self.persist_outputs:
            with open("data/trading_decision.json", "w", encoding="utf-8") as f:
                json.dump(decision, f, ensure_ascii=False, indent=2)
            print("[INFO] Saved clean structured decision to data/trading_decision.json")

        return decision

    # ======================================================
    # 4. Main Flow
    # ======================================================
    def run(self):
        # Default decision in case anything fails
        default_decision = {
            "timestamp": datetime.utcnow().isoformat(),
            "asset": "XAU/USD",
            "market_summary": {
                "sentiment": "neutral",
                "trend": "unknown",
                "volatility": "normal",
                "comment": "Unable to analyze market due to errors"
            },
            "strategies": [
                {"id": 1, "name": "Wait and Watch", "action": "HOLD", "rationale": "Market analysis unavailable", "confidence": 1, "expected_risk": "Low", "expected_return": "Low"}
            ],
            "recommendation": {
                "strategy_id": 1,
                "name": "Wait and Watch",
                "action": "HOLD",
                "confidence": 1,
                "reason": "Unable to generate trading decision"
            }
        }

        try:
            print("\n[STEP 1] Loading news data...")
            news = self.load_news()
            if not news:
                print("[WARN] No selected news with full text found.")
                default_decision["market_summary"]["comment"] = "No selected news with full text found"
                default_decision["recommendation"]["reason"] = "Run News first to collect and analyze news"
                return default_decision

            print(f"[INFO] Loaded {len(news)} news items for strategy generation.")

            print("\n[STEP 2] Generating multi-strategy analysis via LLM...")
            analysis = self.analyze_market_strategies(news)

            print("\n[STEP 3] Structuring final decision output...")
            decision = self.build_structured_decision(analysis)

            print("\n=== FINAL TRADING DECISION ===")
            summary = decision["market_summary"]
            print(
                f"Market: {decision['asset']} | Sentiment: {summary['sentiment']} | Trend: {summary['trend']} | Volatility: {summary['volatility']}")
            print(
                f"-> Recommended: {decision['recommendation']['action']} (Confidence={decision['recommendation']['confidence']})")
            print(f"Reason: {decision['recommendation']['reason'][:300]}...\n")

            print("Alternative Strategies:")
            for s in decision["strategies"]:
                print(f"  [{s['id']}] {s['name']}: {s['action']} (conf={s['confidence']}, risk={s['expected_risk']})")

            return decision
            
        except Exception as e:
            print(f"[ERROR] Trading decision failed: {e}")
            default_decision["market_summary"]["comment"] = f"Error: {str(e)[:100]}"
            default_decision["recommendation"]["reason"] = f"Error during analysis: {str(e)[:100]}"
            return default_decision


if __name__ == "__main__":
    agent = TradingAgent()
    agent.run()
