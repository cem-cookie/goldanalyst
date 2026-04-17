# services/risk_service.py
# -*- coding: utf-8 -*-
import os
import json
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from agents.risk_agent import RiskAgent

app = FastAPI(title="Risk Agent Service", version="1.0")

# ========= Request Schema =========
class RiskRequest(BaseModel):
    decision_path: str = "data/trading_decision.json"
    investment_level: str = "Active"     # Passive / Active / Aggressive
    strategy: str = "Swing"              # Scalping / Swing / Seasonal
    buy_price_threshold: float = 0.5
    sell_price_threshold: float = 0.5
    target_profit: float = 0.1


# ========= Root Endpoint =========
@app.get("/")
def root():
    return {"status": "ok", "message": "Risk Agent Service is running."}


# ========= Run Risk Assessment =========
@app.post("/evaluate_risk")
async def evaluate_risk(req: RiskRequest):
    """
    Perform risk assessment on the latest trading decision file.
    """
    try:
        if not os.path.exists(req.decision_path):
            return {"status": "error", "message": f"Decision file not found: {req.decision_path}"}

        # ✅ 初始化 RiskAgent
        context = {
            "strategy": req.strategy,
            "investment_level": req.investment_level,
            "buy_price_threshold": req.buy_price_threshold,
            "sell_price_threshold": req.sell_price_threshold,
            "target_profit": req.target_profit,
        }

        print(f"[INFO] Starting RiskAgent with context: {context}")
        agent = RiskAgent(
            decision_path=req.decision_path,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            context=context,
        )

        result = agent.run()

        if not result:
            return {"status": "error", "message": "Risk evaluation failed or returned no result."}

        # ✅ 构建精简返回结果（Postman 可读）
        summary = result.get("summary", {})
        items = result.get("items", [])

        top3 = [
            {
                "id": i["id"],
                "approval": i["approval"],
                "score": i["approval_score"],
                "risk": i["risk_level"],
            }
            for i in items[:3]
        ]

        return {
            "status": "success",
            "decision_source": req.decision_path,
            "portfolio_risk": summary.get("portfolio_risk", "Unknown"),
            "comment": summary.get("comment", "")[:300] + "...",
            "top_strategies": top3,
            "saved_file": "data/risk_report.json"
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ========= View Latest Risk Report =========
@app.get("/check_latest")
def check_latest():
    """
    Get the latest risk report summary.
    """
    try:
        path = "data/risk_report.json"
        if not os.path.exists(path):
            return {"status": "error", "message": "No risk_report.json found."}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        ctx = data.get("user_context", {})

        return {
            "status": "success",
            "portfolio_risk": summary.get("portfolio_risk", "Unknown"),
            "comment": summary.get("comment", "")[:300] + "...",
            "strategy": ctx.get("strategy_type", "Unknown"),
            "investment_level": ctx.get("investment_level", "Unknown"),
            "last_updated": data.get("evaluated_at", "Unknown")
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ========= Main Entry =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.risk_service:app", host="127.0.0.1", port=8083, reload=True)
