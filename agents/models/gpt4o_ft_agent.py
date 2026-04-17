from openai import OpenAI
import json


class GPT4oFTAgent:
    """Fine-tuned GPT-4o-mini trading agent"""

    def __init__(self, name, api_key, model_id, initial_cash=100_000, max_alloc=0.5, fee_bps=10):
        self.client = OpenAI(api_key=api_key)
        self.name = name
        self.model_id = model_id
        self.cash = initial_cash
        self.position_oz = 0.0
        self.max_alloc = max_alloc
        self.fee_bps = fee_bps
        self.logs = []

    def decide(self, market_summary, price):
        """Use fine-tuned model to make decision"""

        prompt = f"""
Below is a gold market analysis. Generate ONLY a JSON object.

{market_summary}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            raw = response.choices[0].message.content
            decision = json.loads(raw)

        except Exception as e:
            decision = {
                "action": "HOLD",
                "amount_oz": 0,
                "reason": f"FT model error: {str(e)}"
            }

        return decision

    def execute(self, decision, price, date_str):
        """Execute BUY/SELL/HOLD and record logs"""

        action = decision.get("action", "HOLD")
        amount = float(decision.get("amount_oz", 0))
        executed = False

        if action == "BUY":
            cost = price * amount
            if cost <= self.cash:
                fee = cost * self.fee_bps / 10000
                self.cash -= cost + fee
                self.position_oz += amount
                executed = True

        elif action == "SELL":
            if amount <= self.position_oz:
                revenue = price * amount
                fee = revenue * self.fee_bps / 10000
                self.cash += revenue - fee
                self.position_oz -= amount
                executed = True

        equity = self.cash + self.position_oz * price

        self.logs.append({
            "date": date_str,
            "action": action,
            "amount_oz": amount,
            "executed": executed,
            "price": price,
            "equity": equity,
            "position_oz": self.position_oz,
        })
