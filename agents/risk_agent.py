import os
import re
import json
from datetime import datetime, timezone
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RiskAgent:
    """
    Read TradingAgent Output (data/trading_decision.json),
    Conduct risk control assessments for each strategy based on user context,
    and save structured risk control reports into data/risk_report.json.
    """

    def __init__(self,
                 decision_path: str = "data/trading_decision.json",
                 out_path: str = "data/risk_report.json",
                 openai_api_key: str | None = None,
                 api_key: str | None = None,
                 context: Dict[str, Any] | None = None):
        """
        Args:
            decision_path: Path to trading decision JSON
            out_path: Path to output risk report JSON
            openai_api_key: OpenAI API key (preferred)
            api_key: OpenAI API key (alias, for compatibility)
            context: User context dict with:
                - strategy: "Scalping" / "Swing" / "Seasonal"
                - investment_level: "Passive" / "Active" / "Aggressive"
                - buy_price_threshold: float
                - sell_price_threshold: float
                - target_profit: float
        """
        self.decision_path = decision_path
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path) or "data", exist_ok=True)

        # Support both parameter names
        actual_api_key = openai_api_key or api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=actual_api_key)

        # User-selected parameters
        self.context = context or {}
        self.strategy_type = self.context.get("strategy", "Swing")
        self.investment_level = self.context.get("investment_level", "Active")
        buy_thresh = self.context.get("buy_price_threshold")
        sell_thresh = self.context.get("sell_price_threshold")
        self.buy_price_threshold = float(buy_thresh) if buy_thresh not in (None, 0) else 0
        self.sell_price_threshold = float(sell_thresh) if sell_thresh not in (None, 0) else 0
        self.target_profit = float(self.context.get("target_profit") or 0.1)

        print(f"\n[RISK AGENT INIT]")
        print(f"  Investment Level: {self.investment_level}")
        print(f"  Strategy Type: {self.strategy_type}")
        print(f"  Context: {self.context}")

    def _get_context_prompt(self) -> str:
        """
        Generate risk assessment guide based on user parameters.
        """
        # Get current gold price - CRITICAL for risk assessment
        current_price = self.context.get("latest_price", 0) if self.context else 0
        
        context_lines = [
            "\n=== USER PREFERENCES ===",
            f"Strategy Type: {self.strategy_type}",
            f"Investment Level: {self.investment_level}",
            f"CURRENT GOLD PRICE (XAU/USD): ${current_price:.2f}" if current_price else "CURRENT GOLD PRICE: Not available",
            f"Buy Price Threshold: {self.buy_price_threshold}%",
            f"Sell Price Threshold: {self.sell_price_threshold}%",
            f"Target Profit: {self.target_profit}%",
        ]

        # Risk tolerance and position size guide
        risk_profile = {
            "Passive": {
                "max_drawdown": 5,
                "position_size": "10-20%",
                "stop_loss": "strict",
                "leverage": "none",
                "description": "Conservative trading, risk-averse. Only trade on high certainty signals."
            },
            "Active": {
                "max_drawdown": 10,
                "position_size": "30-50%",
                "stop_loss": "moderate",
                "leverage": "limited",
                "description": "Balanced trading, risk/return balanced. Moderate trading frequency."
            },
            "Aggressive": {
                "max_drawdown": 20,
                "position_size": "60-80%",
                "stop_loss": "loose",
                "leverage": "allowed",
                "description": "Active trading, can accept higher risk. Seek high return opportunities."
            }
        }

        profile = risk_profile.get(self.investment_level, risk_profile["Active"])
        context_lines.append(f"Risk Profile: {profile['description']}")
        context_lines.append(f"Max Acceptable Drawdown: {profile['max_drawdown']}%")
        context_lines.append(f"Position Size Range: {profile['position_size']} of portfolio")
        context_lines.append(f"Stop Loss Policy: {profile['stop_loss']}")
        context_lines.append(f"Leverage: {profile['leverage']}")

        # Strategy-specific risk key points
        strategy_focus = {
            "Scalping": {
                "key_risks": ["Cumulative fees", "Slippage loss", "Execution risk", "Timing risk"],
                "focus": "Frequent trading risk: focus on costs and execution efficiency"
            },
            "Swing": {
                "key_risks": ["Overnight gap", "Time decay", "Market reversal", "News shock"],
                "focus": "Position risk: focus on market reversal and unexpected news during 2-5 day holds"
            },
            "Seasonal": {
                "key_risks": ["Long-term volatility", "Interest rate changes", "Holding costs", "Opportunity cost"],
                "focus": "Long-term position risk: focus on macro factors during weeks-months holds"
            }
        }

        strategy_info = strategy_focus.get(self.strategy_type, strategy_focus["Swing"])
        context_lines.append(f"Strategy Focus: {strategy_info['focus']}")
        context_lines.append(f"Key Risk Categories: {', '.join(strategy_info['key_risks'])}")
        context_lines.append("=" * 35)

        return "\n".join(context_lines)

    def _get_approval_threshold(self) -> int:
        """Get approval score threshold based on investment level."""
        threshold_map = {
            "Passive": 70,  # Passive investors require higher scores
            "Active": 60,  # Balanced investor standard threshold
            "Aggressive": 50  # Aggressive investors can accept lower scores
        }
        return threshold_map.get(self.investment_level, 60)

    def _get_max_drawdown(self) -> int:
        """Get max drawdown based on investment level."""
        drawdown_map = {
            "Passive": 5,
            "Active": 10,
            "Aggressive": 20
        }
        return drawdown_map.get(self.investment_level, 10)

    def _get_review_frequency(self) -> str:
        """Get review frequency based on strategy type."""
        freq_map = {
            "Scalping": "hourly",
            "Swing": "daily",
            "Seasonal": "weekly"
        }
        return freq_map.get(self.strategy_type, "daily")

    def load_decision(self) -> Dict[str, Any]:
        """Load trading decision JSON."""
        with open(self.decision_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _llm_assess(self, asset: str, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Conduct risk assessment using LLM with user parameters."""

        # Only pass necessary fields to avoid prompt bloat
        compact = [
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "action": s.get("action"),
                "confidence": s.get("confidence"),
                "expected_risk": s.get("expected_risk"),
                "expected_return": s.get("expected_return"),
                "rationale": s.get("rationale", "")[:800],
            }
            for s in strategies
        ]

        context_section = self._get_context_prompt()
        approval_threshold = self._get_approval_threshold()
        max_drawdown = self._get_max_drawdown()

        prompt = f"""
You are a strict risk officer conducting risk assessment for gold (XAU/USD) trading.

{context_section}

STRATEGIES TO ASSESS (you must assess ALL of these):
{json.dumps(compact, indent=2)}

Based on the user preferences above, assess risks for each strategy.

For every strategy, output an object with:
- id (int): Strategy identifier
- approval: one of "approved", "conditional", "rejected"
  * "approved" if approval_score >= {approval_threshold}
  * "rejected" if approval_score < {approval_threshold - 20} OR risk violates investor profile
  * "conditional" if between these bounds
- approval_score: integer 0-100
  * Score should reflect whether strategy fits the investor profile
  * For "{self.investment_level}" investor: adjust expectations accordingly
  * For "{self.strategy_type}" strategy: consider strategy-specific risks
  * Score >= {approval_threshold} = safe approval; < {approval_threshold - 20} = reject
- risk_level: one of "Low", "Medium", "High", "Very High"
- key_risks: concrete risks specific to this strategy (4-6 items)
- mitigations: specific mitigation actions (3-5 items)
- notes: brief analysis paragraph

CRITICAL CONSTRAINTS:
- Maximum acceptable drawdown for {self.investment_level} investor: {max_drawdown}%
- If strategy's expected_risk is "High" and investor is "Passive": strongly reject (score < 40)
- If strategy's expected_return is "Low" and investor is "Aggressive": consider rejection
- Position sizing should align with {self.investment_level} investor profile
- For {self.strategy_type} strategies, emphasize strategy-specific risks
- IMPORTANT: Respond in English only. Do not use Chinese or any other language. Do NOT use emojis in output - use plain text only.
- CRITICAL: You MUST assess ALL {len(compact)} strategies. The "items" array must contain one entry for EACH strategy (id: 1, 2, 3). Do NOT skip any strategy. The response must have {len(compact)} items in the "items" array.

Return ONLY valid JSON:
{{
  "asset": "{asset}",
  "evaluated_at": "ISO8601_timestamp",
  "items": [
    {{
      "id": 1,
      "approval": "approved|conditional|rejected",
      "approval_score": 0-100,
      "risk_level": "Low|Medium|High|Very High",
      "key_risks": ["risk1", "risk2", ...],
      "mitigations": ["action1", "action2", ...],
      "notes": "Analysis paragraph"
    }}
  ],
  "summary": {{
    "portfolio_risk": "Safe|Moderate|High|Excessive",
    "comment": "One-paragraph portfolio-level assessment"
  }}
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict risk officer. Output ONLY valid JSON without any other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content

            # Extract JSON from response
            content_encoded = content.encode("utf-8") if isinstance(content, str) else content
            match = re.search(r'\{[\s\S]*\}', content_encoded.decode("utf-8"))
            if match:
                risk_json = json.loads(match.group())
                items_count = len(risk_json.get("items", []))
                print(f"[DEBUG] LLM returned {items_count} strategy assessments")
                return risk_json
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            print(f"[ERROR] LLM risk assessment failed: {e}")
            raise

    def _heuristic_assess(self, asset: str, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Heuristic risk control: fallback when LLM fails.
        Adjust risk assessment based on investment level and strategy type.
        """
        items = []
        approval_threshold = self._get_approval_threshold()

        for s in strategies:
            action = s.get("action", "HOLD")
            conf = s.get("confidence", 3)
            exp_risk = s.get("expected_risk", "Medium")
            exp_ret = s.get("expected_return", "Medium")

            # Base score
            base_score = 50

            # Adjust by confidence
            base_score += min(conf * 8, 30)

            # Adjust by investment level
            if self.investment_level == "Passive":
                if exp_risk == "High":
                    base_score -= 20
                elif exp_risk == "Low":
                    base_score += 10
            elif self.investment_level == "Aggressive":
                if exp_ret == "High":
                    base_score += 10
                elif exp_ret == "Low":
                    base_score -= 10

            # Adjust by strategy type
            if self.strategy_type == "Scalping":
                if action == "SELL":
                    base_score -= 5
            elif self.strategy_type == "Seasonal":
                if action == "HOLD":
                    base_score += 5

            # Adjust risk level based on action
            if action == "HOLD":
                risk_level = "Low" if conf >= 3 else "Medium"
                key_risks = [
                    "Opportunity cost if market moves significantly",
                    "Trend misread or early signal",
                    "Market reversal during hold period"
                ]
                mitigations = [
                    "Define clear re-entry/exit triggers",
                    f"Set review cadence ({self._get_review_frequency()} reviews)",
                    "Monitor key support/resistance levels"
                ]
            elif action == "BUY":
                risk_level = "Medium" if conf >= 3 else "High"
                key_risks = [
                    "Further drawdown during correction phase",
                    "Macro shock affecting safe-haven flows",
                    "Position liquidation due to margin calls",
                    "Opportunity cost if trend continues without entry"
                ]
                mitigations = [
                    "Dollar-cost averaging (DCA) strategy",
                    "Tight stop-loss at technical support",
                    "Position sizing within portfolio limits",
                    "Hedge via put options if capital permits"
                ]
            else:  # SELL
                risk_level = "High" if conf <= 2 else "Medium"
                key_risks = [
                    "Short squeeze on unexpected rebound",
                    "Liquidity gap on breaking news",
                    "Forced cover due to margin pressure",
                    "Opportunity cost from early exit"
                ]
                mitigations = [
                    "Conservative position sizing",
                    "Hard stop-loss above recent highs",
                    "Avoid illiquid hours (e.g., US open, close)",
                    "Take profit at predetermined levels"
                ]

            # Calculate score
            score = max(0, min(100, int(base_score)))

            # Determine approval status
            if score >= approval_threshold:
                approval = "approved"
            elif score < approval_threshold - 20:
                approval = "rejected"
            else:
                approval = "conditional"

            items.append({
                "id": s.get("id"),
                "approval": approval,
                "approval_score": score,
                "risk_level": risk_level,
                "key_risks": key_risks,
                "mitigations": mitigations,
                "notes": (
                    f"Heuristic assessment for {self.investment_level} investor using {self.strategy_type} strategy. "
                    f"Action: {action}, Confidence: {conf}/5, Expected Risk: {exp_risk}, "
                    f"Expected Return: {exp_ret}."
                )
            })

        # Portfolio-level risk
        approved_count = sum(1 for i in items if i["approval"] == "approved")
        rejected_count = sum(1 for i in items if i["approval"] == "rejected")
        high_risk_count = sum(1 for i in items if i["risk_level"] in ["High", "Very High"])

        if rejected_count > len(items) * 0.5 or high_risk_count == len(items):
            overall = "Excessive"
        elif rejected_count > 0 or high_risk_count > len(items) * 0.5:
            overall = "High"
        elif approved_count == len(items):
            overall = "Safe"
        else:
            overall = "Moderate"

        return {
            "asset": asset,
            "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "items": items,
            "summary": {
                "portfolio_risk": overall,
                "comment": f"Portfolio risk is {overall.lower()} based on heuristic analysis."
            }
        }

    def build_and_save(self, risk_json: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance: Add user context and source information.
        """
        risk_json["source_decision"] = {
            "asset": decision.get("asset"),
            "timestamp": decision.get("timestamp"),
            "market_summary": decision.get("market_summary"),
        }

        risk_json["user_context"] = {
            "strategy_type": self.strategy_type,
            "investment_level": self.investment_level,
            "buy_price_threshold": self.buy_price_threshold,
            "sell_price_threshold": self.sell_price_threshold,
            "target_profit": self.target_profit,
        }

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(risk_json, f, ensure_ascii=False, indent=2)

        return risk_json

    # Main flow
    def run(self) -> Dict[str, Any] | None:
        """Main flow: Load decision -> Risk assessment -> Save report."""
        print("\n[STEP 1] Loading trading decision...")
        try:
            decision = self.load_decision()
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            raise

        asset = decision.get("asset", "XAU/USD")
        strategies = decision.get("strategies", [])

        if not strategies:
            print("[WARN] No strategies found in decision. Run Actions first.")
            return None

        print(f"[INFO] Loaded {len(strategies)} strategies for risk assessment.")
        print(f"[INFO] User context: Strategy={self.strategy_type}, Investment={self.investment_level}")

        print("[STEP 2] Assessing risks via LLM...")
        try:
            risk_json = self._llm_assess(asset, strategies)
        except Exception as e:
            print(f"[WARN] LLM risk assessment failed: {e}")
            print("[INFO] Falling back to heuristic assessment...")
            risk_json = self._heuristic_assess(asset, strategies)

        print("[STEP 3] Saving structured risk report...")
        result = self.build_and_save(risk_json, decision)

        # Print summary
        print("\n" + "=" * 60)
        print("=== RISK ASSESSMENT SUMMARY ===")
        print("=" * 60)

        summary = result["summary"]
        print(f"Portfolio Risk: {summary.get('portfolio_risk', 'Unknown')}")
        print(f"Comment: {summary.get('comment', 'N/A')}\n")

        print("Per-Strategy Assessment:")
        for item in result["items"]:
            print(f"  [Strategy {item['id']}] {item['approval'].upper()} "
                  f"(Score: {item['approval_score']}/100, Risk: {item['risk_level']})")

        print("=" * 60 + "\n")

        return result


# Test case: With user parameters
if __name__ == "__main__":
    import sys
    context = {
        "strategy": sys.argv[1] if len(sys.argv) > 1 else "Swing",
        "investment_level": sys.argv[2] if len(sys.argv) > 2 else "Active",
    }
    agent = RiskAgent(context=context)
    result = agent.run()
    print(result)