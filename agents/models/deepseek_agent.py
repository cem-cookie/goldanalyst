"""
agents/models/deepseek_agent.py
DeepSeek 交易决策代理 - 支持数量决策
"""
import os
import re
import json
from openai import OpenAI
from agents.trading_agent import TradingAgent


class DeepSeekAgent(TradingAgent):
    """使用 DeepSeek 进行交易决策"""

    def __init__(self, name="DeepSeek", api_key=None, initial_cash=100_000.0, max_alloc=0.5, fee_bps=10):
        super().__init__(name, api_key, initial_cash, max_alloc, fee_bps)
        self.logs = []  # ✅ simulate_ds_comparison.py 依赖 agent.logs

    def decide(self, market_summary: str, gold_price: float) -> dict:
        """基于市场分析决定买卖数量（oz）"""
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
            api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY or pass api_key=...")

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

            rsp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            raw = rsp.choices[0].message.content.strip()
            clean = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
            analysis = json.loads(clean)

            action = str(analysis.get("action", "HOLD")).upper()
            amount_oz = float(analysis.get("amount_oz", 0))
            confidence = int(analysis.get("confidence", 2))
            reason = str(analysis.get("reason", ""))

            # 验证数量（保护仓位/现金）
            if action == "BUY":
                amount_oz = min(amount_oz, max_buy_oz)
            elif action == "SELL":
                amount_oz = min(amount_oz, max_sell_oz)
            else:
                action = "HOLD"
                amount_oz = 0

            return {
                "action": action,
                "amount_oz": max(0.0, amount_oz),
                "confidence": confidence,
                "reason": reason
            }

        except Exception as e:
            print(f"[WARN] DeepSeek decide failed: {e}")
            return {"action": "HOLD", "amount_oz": 0, "confidence": 1, "reason": "Error"}

    def execute(self, decision: dict, gold_price: float, date_str: str):
        """执行交易决策，更新账户状态，并写入 logs（simulate_ds_comparison.py 依赖）"""
        action = (decision.get("action") or "HOLD").upper()
        amount_oz = float(decision.get("amount_oz", 0))
        executed = False

        if action == "BUY":
            cost = amount_oz * gold_price
            fee = cost * self.fee_bps / 10000
            total_cost = cost + fee

            if total_cost <= self.state.cash and amount_oz > 0:
                self.state.cash -= total_cost
                self.state.position_oz += amount_oz
                executed = True
            else:
                executed = False

        elif action == "SELL":
            if 0 < amount_oz <= self.state.position_oz:
                revenue = amount_oz * gold_price
                fee = revenue * self.fee_bps / 10000
                net_revenue = revenue - fee

                self.state.cash += net_revenue
                self.state.position_oz -= amount_oz
                executed = True
            else:
                executed = False

        else:  # HOLD
            action = "HOLD"
            amount_oz = 0.0
            executed = True

        equity = self.state.cash + self.state.position_oz * gold_price

        self.logs.append({
            "date": date_str,
            "action": action,
            "amount_oz": amount_oz,
            "executed": executed,
            "price": gold_price,
            "equity": equity,
            "position_oz": self.state.position_oz,
            "cash": self.state.cash,
        })
