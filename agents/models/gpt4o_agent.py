"""
agents/models/gpt4o_agent.py
GPT-4o 交易决策代理 - 支持数量决策
"""
import os
import re
import json
from openai import OpenAI
from agents.trading_agent import TradingAgent


class GPT4oAgent(TradingAgent):
    """使用 GPT-4o 进行交易决策"""

    def __init__(self, name="GPT-4o", api_key=None, initial_cash=100_000.0,
                 max_alloc=0.5, fee_bps=10):
        super().__init__(name, api_key, initial_cash, max_alloc, fee_bps, use_deepseek=False)
        self.logs = []  # 添加这行

    def decide(self, market_summary: str, gold_price: float) -> dict:
        """基于市场分析决定买卖数量"""

        max_buy_oz = (self.state.cash * self.max_alloc) / gold_price if gold_price > 0 else 0
        max_sell_oz = self.state.position_oz

        prompt = f"""You are a professional gold trader. Make precise trading decisions.

    CURRENT STATE:
    - Gold Price: ${gold_price:.2f}/oz
    - Your Cash: ${self.state.cash:,.0f}
    - Your Position: {self.state.position_oz:.2f} oz
    - Max you can BUY: {max_buy_oz:.2f} oz
    - Max you can SELL: {max_sell_oz:.2f} oz

    MARKET ANALYSIS:
    {market_summary}

    RESPOND WITH VALID JSON ONLY. Example:
    {{"action": "BUY", "amount_oz": 10.5, "confidence": 4, "reason": "bullish"}}"""

        try:
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")

            if not api_key:
                print(f"[ERROR] No API key found!")
                return {"action": "HOLD", "amount_oz": 0, "confidence": 1, "reason": "No API key"}

            print(f"[DEBUG {self.name}] Calling GPT-4o-mini...")

            client = OpenAI(api_key=api_key)
            rsp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            raw = rsp.choices[0].message.content.strip()
            print(f"[DEBUG {self.name}] Raw: {raw[:150]}")

            # 清理 markdown
            clean = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
            print(f"[DEBUG {self.name}] Clean: {clean[:150]}")

            # 尝试解析 JSON
            try:
                analysis = json.loads(clean)
            except json.JSONDecodeError as e:
                print(f"[ERROR {self.name}] JSON decode failed: {e}")
                print(f"[ERROR {self.name}] Attempted to parse: {clean}")
                return {"action": "HOLD", "amount_oz": 0, "confidence": 1, "reason": "JSON parse error"}

            action = analysis.get("action", "HOLD").upper()
            amount_oz = float(analysis.get("amount_oz", 0))
            confidence = int(analysis.get("confidence", 2))
            reason = analysis.get("reason", "")

            # 验证数量
            if action == "BUY":
                amount_oz = min(amount_oz, max_buy_oz)
            elif action == "SELL":
                amount_oz = min(amount_oz, max_sell_oz)
            else:
                amount_oz = 0

            print(f"[DEBUG {self.name}] Decision: {action} {amount_oz:.2f}oz")

            return {
                "action": action,
                "amount_oz": max(0, amount_oz),
                "confidence": confidence,
                "reason": reason
            }

        except Exception as e:
            print(f"[ERROR {self.name}] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {"action": "HOLD", "amount_oz": 0, "confidence": 1, "reason": f"Error: {str(e)}"}

    def execute(self, decision: dict, gold_price: float, date_str: str):
        """执行交易决策，更新账户状态"""
        action = decision.get("action", "HOLD")
        amount_oz = float(decision.get("amount_oz", 0))
        confidence = decision.get("confidence", 0)
        reason = decision.get("reason", "")

        executed = False  # 标记是否成功执行

        if action == "BUY":
            cost = amount_oz * gold_price
            fee = cost * self.fee_bps / 10000
            total_cost = cost + fee

            if total_cost <= self.state.cash:
                self.state.cash -= total_cost
                self.state.position_oz += amount_oz
                executed = True
                print(f"[{date_str}] {self.name} BUY {amount_oz:.2f}oz @ ${gold_price:.2f}")
                print(f"  → Cash: ${self.state.cash:,.2f}, Position: {self.state.position_oz:.2f}oz")
            else:
                print(f"[{date_str}] {self.name} BUY REJECTED: 现金不足")

        elif action == "SELL":
            if amount_oz <= self.state.position_oz:
                revenue = amount_oz * gold_price
                fee = revenue * self.fee_bps / 10000
                net_revenue = revenue - fee
                self.state.cash += net_revenue
                self.state.position_oz -= amount_oz
                executed = True
                print(f"[{date_str}] {self.name} SELL {amount_oz:.2f}oz @ ${gold_price:.2f}")
                print(f"  → Cash: ${self.state.cash:,.2f}, Position: {self.state.position_oz:.2f}oz")
            else:
                print(f"[{date_str}] {self.name} SELL REJECTED: 持仓不足")

        elif action == "HOLD":
            executed = True
            print(f"[{date_str}] {self.name} HOLD")

        # 计算权益
        equity = self.state.cash + self.state.position_oz * gold_price

        # 记录到日志 ✅ 确保包含 executed 字段
        self.logs.append({
            "date": date_str,
            "action": action,
            "amount_oz": amount_oz,
            "executed": executed,  # ← 这个很重要
            "price": gold_price,
            "equity": equity,
            "position_oz": self.state.position_oz,
            "cash": self.state.cash
        })
    def get_portfolio_value(self, gold_price: float) -> float:
        """计算投资组合总价值 = 现金 + 持仓 × 当前价格"""
        return self.state.cash + self.state.position_oz * gold_price

    def get_return(self, gold_price: float) -> float:
        """计算收益率"""
        portfolio_value = self.get_portfolio_value(gold_price)
        return (portfolio_value - self.state.initial_cash) / self.state.initial_cash * 100