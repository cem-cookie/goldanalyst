import os
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

try:
    import yaml
except ImportError:
    yaml = None

try:
    from openai import OpenAI
except ImportError:
    # Minimal stub for environments without openai package (used only for attribute tests)
    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key
            self.base_url = base_url
        class chat:
            @staticmethod
            def completions():
                raise NotImplementedError("OpenAI client not available in this environment.")

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

load_dotenv()


try:
    from utils.llm_wrapper import LLM_CONFIG
except ImportError:
    LLM_CONFIG = {"llm_timeout_seconds": 15, "llm_max_retries": 3, "llm_base_delay_seconds": 1}


def _load_trading_config() -> dict:
    """Load trading configuration from config/trading.yaml with defaults."""
    config_path = Path(__file__).parent.parent / "config" / "trading.yaml"
    defaults = {
        "max_risk_percent": 0.02,
        "min_trade_size_oz": 0.01,
        "max_trade_size_oz": 100.0,
        "default_confidence_scale": 0.5,
        "fallback_position_size_oz": 1.0,
    }
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg:
                defaults.update(cfg)
        except Exception as e:
            print(f"[WARN] Failed to load trading config: {e}")
    return defaults


class TradingAgent:
    """Trading Agent: Manages account state and trading decisions"""

    def __init__(self, name="Agent", api_key=None, initial_cash=100_000.0, max_alloc=0.5,
                 fee_bps=10, json_path="gold_news.json",
                 persist_outputs=True, context=None, model_name="gpt-4o-mini"):
        self.model = model_name
        self.name = name
        self.api_key = api_key
        self.max_alloc = max_alloc
        self.fee_bps = fee_bps
        self.json_path = json_path
        self.persist_outputs = persist_outputs
        self._trading_config = _load_trading_config()
        self._context = context or {}
# Initialize OpenAI client (ChatGPT) – only used for LLM calls
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _calc_position_size(
        self,
        account_balance: float,
        confidence: float,
        user_risk_percent: float | None = None,
        gold_price: float | None = None,
    ) -> float:
        """
        Calculate position size in ounces based on account balance, confidence, and risk settings.

        Args:
            account_balance: Total account equity in USD
            confidence: Trading confidence score (0-10, will be normalized to 0-1)
            user_risk_percent: Optional user-specified risk percent (overrides config)
            gold_price: Current gold price per oz (if None, uses fallback)

        Returns:
            Position size in ounces
        """
        cfg = self._trading_config
        risk_percent = user_risk_percent if user_risk_percent is not None else cfg.get("max_risk_percent", 0.02)
        min_size = cfg.get("min_trade_size_oz", 0.01)
        max_size = cfg.get("max_trade_size_oz", 100.0)
        fallback_confidence_scale = cfg.get("default_confidence_scale", 0.5)
        fallback_size = cfg.get("fallback_position_size_oz", 1.0)

        # Validate inputs
        if account_balance <= 0 or confidence < 0:
            print("[WARN] Invalid balance or confidence, returning fallback size")
            return fallback_size
        if gold_price is None or gold_price <= 0:
            gold_price = 3500.0
            print(f"[WARN] No gold price provided, using fallback: {gold_price}")

        # Normalize confidence from 0-10 to 0-1
        normalized_conf = max(0.0, min(1.0, confidence / 10.0))

        # Base position size: balance * risk_percent / price
        base_size = (account_balance * risk_percent) / gold_price

        # Scale by confidence (use default_confidence_scale if missing)
        effective_confidence = normalized_conf if normalized_conf > 0 else fallback_confidence_scale
        scaled_size = base_size * effective_confidence

        # Apply min/max constraints
        final_size = max(min_size, min(max_size, scaled_size))

        # Round to 2 decimal places
        final_size = round(final_size, 2)

        if final_size < min_size:
            print(f"[WARN] Position size {final_size} below minimum {min_size}, returning 0")
            return 0.0

        return final_size

    # --------------------------------------------------
    # UI helper: wrap position size for use in Streamlit
    # --------------------------------------------------
    def calculate_position_size(
        self,
        account_balance: float,
        confidence: float = 5,
        user_risk_percent: float | None = None,
        gold_price: float | None = None,
    ) -> float:
        """Public API for position sizing (exposes functionality to UI)."""
        return self._calc_position_size(account_balance, confidence, user_risk_percent, gold_price)

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
            # Get timeout from config (defaults to 15s)
            timeout = LLM_CONFIG.get("llm_timeout_seconds", 15)
            
            def make_llm_call():
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
            
            # Apply timeout wrapper
            from utils.llm_wrapper import call_with_timeout
            response = call_with_timeout(
                make_llm_call,
                timeout_seconds=timeout,
                default_return=None,
            )
            
            if response is None:
                print("[WARN] LLM call timed out, returning HOLD")
                return {
                    "action": "HOLD",
                    "reasoning": "LLM request timed out",
                    "confidence": 1,
                    "key_factors": [],
                    "risk_level": "medium",
                }

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
        
        # Get current gold price from context - CRITICAL for strategy generation
        current_price = self._context.get("latest_price")
        if current_price is None:
            # Try to fetch current price if not in context
            try:
                import yfinance
                ticker = yfinance.Ticker("GC=F")
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    current_price = round(float(hist["Close"].iloc[-1]), 2)
            except:
                current_price = None
        
        price_info = f"CURRENT GOLD PRICE (XAU/USD): ${current_price:.2f}" if current_price else "CURRENT GOLD PRICE: Not available"
        
        prompt = f"""
You are a professional trading strategist specialized in gold (XAU/USD).

**CRITICAL: {price_info}**

All price calculations (entry_price, target_price) MUST be based on this current market price.
- For BUY strategies: entry_price should be AT or BELOW current price (e.g., current_price - 20)
- For SELL strategies: entry_price should be AT or ABOVE current price (e.g., current_price + 10)
- For target_price: calculate profit/loss target relative to current price

Based on the following recent news, propose 3 alternative trading strategies:
1. Conservative (low risk)
2. Balanced / Neutral
3. Aggressive (high risk, high reward)

For each strategy, include:
- name
- action (BUY / SELL / HOLD)
- rationale
- confidence (0-5)
- expected_risk (Low/Medium/High)
- expected_return (Low/Medium/High)
- entry_price (MUST be based on current price {current_price or 'N/A'})
- target_price (profit target relative to current price)

Also provide a high-level summary of the market (sentiment and trend)
and recommend which strategy is optimal, with reasoning.

Output ONLY valid JSON in this format. Do NOT use emojis:
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
      "expected_return": "...",
      "entry_price": ...,
      "target_price": ...
    }}
  ],
  "recommendation": "...",
  "reasoning": "..."
}}

News:
{summaries}
"""

        timeout = LLM_CONFIG.get("llm_timeout_seconds", 15)
        
        def make_llm_call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert financial strategist who outputs valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
        
        from utils.llm_wrapper import call_with_timeout
        response = call_with_timeout(make_llm_call, timeout_seconds=timeout, default_return=None)
        
        if response is None:
            print("[WARN] Strategy analysis LLM call timed out, using fallback")
            return {
                "sentiment": "neutral",
                "trend": "unknown",
                "strategies": [
                    {"name": "Fallback Hold", "action": "HOLD", "rationale": "Timeout",
                     "confidence": 2, "expected_risk": "Medium", "expected_return": "Medium"}
                ],
                "recommendation": "Fallback Hold",
                "reasoning": "LLM timeout, holding position"
            }

        raw_output = response.choices[0].message.content.strip()
        print(f"\n[LLM RAW OUTPUT]\n{raw_output}\n")

        # Handle encoding safely for non-ASCII characters
        try:
            raw_output.encode("utf-8")
        except UnicodeEncodeError:
            raw_output = raw_output.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")

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

        analysis_data["timestamp"] = datetime.now(timezone.utc).isoformat()

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

        # Best Strategy: Use LLM's explicit recommendation first, then fall back to highest confidence
        llm_recommended_name = analysis_data.get("recommendation", "")
        best = None
        
        if llm_recommended_name:
            # Try to find the LLM's recommended strategy by name
            for s in strategies:
                if s.get("name", "").lower() == llm_recommended_name.lower():
                    best = s
                    print(f"[DEBUG] Using LLM's recommendation: {llm_recommended_name}")
                    break
        
        if not best:
            # Fallback: select strategy with highest confidence
            best = max(strategies, key=lambda s: float(s.get("confidence", 0)))
            print(f"[DEBUG] Using highest confidence fallback: {best.get('name')} (conf={best.get('confidence')})")
        
        best_id = strategies.index(best) + 1

        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
