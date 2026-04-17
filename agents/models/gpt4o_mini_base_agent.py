from openai import OpenAI
import json

class GPT4oMiniBaseAgent:
    def __init__(self, name, api_key, initial_cash=100_000, max_alloc=0.5, fee_bps=10):
        self.client = OpenAI(api_key=api_key)
        self.name = name
        self.cash = initial_cash
        self.position_oz = 0.0
        self.max_alloc = max_alloc
        self.fee_bps = fee_bps
        self.logs = []

    def decide(self, market_summary, price):
        """调用基础 gpt-4o-mini 生成决策"""
        prompt = f"""
You are a gold trading assistant.
Given the following market summary, output ONLY a JSON object:

{market_summary}

Example:
{{
  "action": "BUY/SELL/HOLD",
  "amount_oz": 0,
  "reason": "..."
}}
"""
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        text = resp.choices[0].message.content

        try:
            decision = json.loads(text)
        except:
            decision = {"action": "HOLD", "amount_oz": 0, "reason": "Invalid JSON"}

        return decision

    def execute(self, decision, price, date_str):
        action = decision.get("action", "HOLD")
        amount = decision.get("amount_oz", 0)
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

        # 记录日志
        self.logs.append({
            "date": date_str,
            "action": action,
            "amount_oz": amount,
            "executed": executed,
            "price": price,
            "equity": equity,
            "position_oz": self.position_oz
        })
