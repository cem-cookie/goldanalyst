"""
agents/models/claude_haiku_agent.py
Claude 3.5 Haiku Trading Agent - Supports quantity-based decisions
"""
import os
import re
import json
from openai import OpenAI
from agents.trading_agent import TradingAgent


class ClaudeHaikuAgent(TradingAgent):
    """Use Claude 3.5 Haiku for trading decisions."""

    def __init__(self, name="Claude-Haiku", api_key=None, initial_cash=100_000.0,
                 max_alloc=0.5, fee_bps=10):
        super().__init__(name, api_key, initial_cash, max_alloc, fee_bps, use_deepseek=False)
        self.logs = []

    def decide(self, market_summary: str, gold_price: float) -> dict:
        """Decide buy/sell quantity based on market analysis."""

        max_buy_oz = (self.state.cash * self.max_alloc) / gold_price if gold_price > 0 else 0
        max_sell_oz = self.state.position_oz

        prompt = f"""You are a professional gold trader.

CURRENT STATE:
- Gold Price: ${gold_price:.2f}/oz
- Your Cash: ${self.state.cash:,.0f}
- Your Position: {self.state.position_oz:.2f} oz
- Max BUY: {max_buy_oz:.2f} oz
- Max SELL: {max_sell_oz:.2f} oz

MARKET ANALYSIS:
{market_summary}

Decide: BUY/SELL/HOLD and how much (in oz).
Respond with JSON only:
{{"action": "BUY", "amount_oz": 10, "confidence": 4, "reason": "..."}}"""

        try:
            client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
            rsp = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            raw = rsp.content[0].text.strip()
            clean = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE)
            analysis = json.loads(clean)

            action = analysis["action"].upper()
            amount_oz = float(analysis.get("amount_oz", 0))
            confidence = int(analysis.get("confidence", 2))
            reason = analysis.get("reason", "")

            # Validate quantity
            if action == "BUY":
                amount_oz = min(amount_oz, max_buy_oz)
            elif action == "SELL":
                amount_oz = min(amount_oz, max_sell_oz)
            else:
                amount_oz = 0

            return {
                "action": action,
                "amount_oz": max(0, amount_oz),
                "confidence": confidence,
                "reason": reason
            }

        except Exception as e:
            print(f"[WARN] Claude Haiku decide failed: {e}")
            return {"action": "HOLD", "amount_oz": 0, "confidence": 1, "reason": "Error"}