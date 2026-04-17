import os
import re
import json
from datetime import datetime
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
                 api_key: str | None = None,  # ✅ 添加别名支持
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

        # ✅ 支持两种参数名
        actual_api_key = openai_api_key or api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=actual_api_key)

        # 用户选择的参数
        self.context = context or {}
        self.strategy_type = self.context.get("strategy", "Swing")
        self.investment_level = self.context.get("investment_level", "Active")
        self.buy_price_threshold = float(self.context.get("buy_price_threshold", 0.5))
        self.sell_price_threshold = float(self.context.get("sell_price_threshold", 0.5))
        self.target_profit = float(self.context.get("target_profit", 0.1))

        print(f"\n[RISK AGENT INIT]")
        print(f"  Investment Level: {self.investment_level}")
        print(f"  Strategy Type: {self.strategy_type}")
        print(f"  Context: {self.context}")

    def _get_context_prompt(self) -> str:
        """
        生成基于用户参数的风险评估指南
        """
        context_lines = [
            "\n=== USER PREFERENCES ===",
            f"Strategy Type: {self.strategy_type}",
            f"Investment Level: {self.investment_level}",
            f"Buy Price Threshold: {self.buy_price_threshold}%",
            f"Sell Price Threshold: {self.sell_price_threshold}%",
            f"Target Profit: {self.target_profit}%",
        ]

        # 风险容忍度和头寸规模指南
        risk_profile = {
            "Passive": {
                "max_drawdown": 5,
                "position_size": "10-20%",
                "stop_loss": "strict",
                "leverage": "none",
                "description": "保守交易，风险厌恶。仅在高确定性信号时交易。"
            },
            "Active": {
                "max_drawdown": 10,
                "position_size": "30-50%",
                "stop_loss": "moderate",
                "leverage": "limited",
                "description": "平衡交易，风险/收益均衡。适度交易频率。"
            },
            "Aggressive": {
                "max_drawdown": 20,
                "position_size": "60-80%",
                "stop_loss": "loose",
                "leverage": "allowed",
                "description": "积极交易，可承受较高风险。寻求高回报机会。"
            }
        }

        profile = risk_profile.get(self.investment_level, risk_profile["Active"])
        context_lines.append(f"Risk Profile: {profile['description']}")
        context_lines.append(f"Max Acceptable Drawdown: {profile['max_drawdown']}%")
        context_lines.append(f"Position Size Range: {profile['position_size']} of portfolio")
        context_lines.append(f"Stop Loss Policy: {profile['stop_loss']}")
        context_lines.append(f"Leverage: {profile['leverage']}")

        # 根据策略类型的风险关键点
        strategy_focus = {
            "Scalping": {
                "key_risks": ["手续费累积", "滑点损失", "执行风险", "时间风险"],
                "focus": "频繁交易风险：在高频交易中关注成本和执行效率"
            },
            "Swing": {
                "key_risks": ["隔夜跳空", "时间衰减", "市场反向", "新闻冲击"],
                "focus": "持仓风险：在2-5天持仓期间关注市场反向和突发事件"
            },
            "Seasonal": {
                "key_risks": ["长期波动", "利率变化", "持仓成本", "机会成本"],
                "focus": "长期持仓风险：在数周至数月的持仓中关注宏观因素变化"
            }
        }

        strategy_info = strategy_focus.get(self.strategy_type, strategy_focus["Swing"])
        context_lines.append(f"Strategy Focus: {strategy_info['focus']}")
        context_lines.append(f"Key Risk Categories: {', '.join(strategy_info['key_risks'])}")
        context_lines.append("=" * 35)

        return "\n".join(context_lines)

    def _get_approval_threshold(self) -> int:
        """根据投资级别获取批准分数阈值"""
        threshold_map = {
            "Passive": 70,  # 被动投资者要求更高的分数
            "Active": 60,  # 平衡投资者的标准阈值
            "Aggressive": 50  # 激进投资者可以接受更低的分数
        }
        return threshold_map.get(self.investment_level, 60)

    def _get_max_drawdown(self) -> int:
        """根据投资级别返回最大回撤"""
        drawdown_map = {
            "Passive": 5,
            "Active": 10,
            "Aggressive": 20
        }
        return drawdown_map.get(self.investment_level, 10)

    def _get_review_frequency(self) -> str:
        """根据策略类型返回审查频率"""
        frequency_map = {
            "Scalping": "hourly",
            "Swing": "daily",
            "Seasonal": "weekly"
        }
        return frequency_map.get(self.strategy_type, "daily")

    # 1) Read Generated Actions
    def load_decision(self) -> Dict[str, Any]:
        """加载交易决策 JSON"""
        if not os.path.exists(self.decision_path):
            raise FileNotFoundError(f"{self.decision_path} not found. Run Actions first.")
        with open(self.decision_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2) LLM Risk Evaluation
    def _llm_assess(self, asset: str, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用 LLM 进行风险评估，融入用户参数"""

        # 仅传必要字段，避免 prompt 过长
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

Strategies to assess:
{json.dumps(compact, ensure_ascii=False, indent=2)}
"""

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a strict financial risk officer. Output valid JSON only, no additional text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        clean = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw, flags=re.MULTILINE)
        return json.loads(clean)

    # 3) Heuristic Risk Assessment (Fallback)
    def _heuristic_assess(self, asset: str, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        启发式风控：LLM 失败时的兜底方案
        根据投资级别和策略类型调整风险评估
        """
        items = []
        approval_threshold = self._get_approval_threshold()

        for s in strategies:
            action = (s.get("action") or "").upper()
            conf = int(s.get("confidence", 0))
            expected_risk = (s.get("expected_risk") or "").lower()
            expected_return = (s.get("expected_return") or "").lower()

            # 基础分数
            base_score = 50

            # 根据置信度调整
            base_score += min(conf * 8, 30)  # 置信度最多加30分

            # 根据投资级别调整
            if self.investment_level == "Passive":
                # 被动投资者：高风险被惩罚，低风险被奖励
                if expected_risk == "high":
                    base_score -= 30
                elif expected_risk == "low":
                    base_score += 10
                if expected_return == "high":
                    base_score -= 10  # 高收益期望不符合被动风格
            elif self.investment_level == "Aggressive":
                # 激进投资者：高风险被奖励，低收益被惩罚
                if expected_risk == "high":
                    base_score += 15
                if expected_return == "low":
                    base_score -= 15

            # 根据策略类型调整
            if self.strategy_type == "Scalping":
                if action == "SELL":
                    base_score -= 5  # Scalping 较少短卖
            elif self.strategy_type == "Seasonal":
                if action == "HOLD":
                    base_score += 5  # Seasonal 倾向持仓

            # 根据动作调整风险等级
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

            # 综合评分
            score = max(0, min(100, int(base_score)))

            # 确定批准状态
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
                    f"Action: {action}, Confidence: {conf}/5, Expected Risk: {expected_risk}, "
                    f"Expected Return: {expected_return}."
                )
            })

        # 投资组合整体风险
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
            "evaluated_at": datetime.utcnow().isoformat(),
            "items": items,
            "summary": {
                "portfolio_risk": overall,
                "comment": (
                    f"Portfolio-level risk assessment for {self.investment_level} investor using {self.strategy_type} strategy. "
                    f"{approved_count}/{len(items)} strategies approved, {rejected_count} rejected. "
                    f"{high_risk_count} strategies flagged as high-risk. "
                    f"Overall portfolio stance: {overall}."
                )
            }
        }

    # 4) Clean Up and Save
    def build_and_save(self, risk_json: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        增强：添加用户上下文和源信息
        """
        risk_json["user_context"] = {
            "strategy_type": self.strategy_type,
            "investment_level": self.investment_level,
            "buy_price_threshold": self.buy_price_threshold,
            "sell_price_threshold": self.sell_price_threshold,
            "target_profit": self.target_profit,
        }
        risk_json["source_decision"] = {
            "asset": decision.get("asset"),
            "timestamp": decision.get("timestamp"),
            "market_summary": decision.get("market_summary"),
        }

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(risk_json, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved risk report to {self.out_path}")
        return risk_json

    # 5) Main flow
    def run(self) -> Dict[str, Any] | None:
        """主流程：加载决策 → 风险评估 → 保存报告"""
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

        # 打印摘要
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


if __name__ == "__main__":
    # 测试用例：带有用户参数
    context = {
        "strategy": "Swing",
        "investment_level": "Active",
        "buy_price_threshold": 0.5,
        "sell_price_threshold": 0.5,
        "target_profit": 0.1,
    }

    agent = RiskAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        context=context
    )
    agent.run()