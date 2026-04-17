# services/trading_service.py
# -*- coding: utf-8 -*-
import os
import json
import traceback
import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
from agents.trading_agent import TradingAgent
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Trading Agent Service", version="1.1")

# ========= Request Schema =========
class TradingRequest(BaseModel):
    json_path: str = Field(..., regex=r"^.*\.json$", description="Path to a valid JSON file.")
    use_deepseek: bool = True


# ========= Root Endpoint =========
@app.get("/")
def root():
    return {"status": "ok", "message": "Trading Agent Service is running."}


# ========= Generate Strategy =========
@app.post("/generate_strategy")
async def generate_strategy(request: TradingRequest):
    """
    Run trading decision process based on analyzed gold news (.json)
    or analyzed report (.txt).
    """
    logging.info(f"Received request to generate strategy for {request.json_path}")
    try:
        path = request.json_path.strip()

        # Validate file existence
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail="File does not exist.")

        # ✅ 自动识别 txt 输入（包装成 JSON 格式）
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                return {"status": "error", "message": f"The file {path} is empty."}

            temp_json_path = "data/temp_analysis.json"
            temp_data = [{
                "id": 1,
                "source": "analyzed_text",
                "title": os.path.basename(path),
                "full_text": text,
                "selected": True
            }]

            with open(temp_json_path, "w", encoding="utf-8") as f:
                json.dump(temp_data, f, ensure_ascii=False, indent=2)

            print(f"[INFO] Wrapped TXT file into temporary JSON: {temp_json_path}")
            path = temp_json_path

        # ✅ 初始化并运行 TradingAgent
        print(f"[INFO] Starting TradingAgent with file: {path}")
        agent = TradingAgent(json_path=path, use_deepseek=request.use_deepseek)
        decision = agent.run()

        if not decision:
            return {"status": "error", "message": f"No valid trading decision generated from {path}."}

        # ✅ 结构化返回结果（便于 Postman 阅读）
        summary = decision.get("market_summary", {})
        recommendation = decision.get("recommendation", {})

        response = {
            "status": "success",
            "source_file": request.json_path,
            "summary": {
                "sentiment": summary.get("sentiment", "unknown"),
                "trend": summary.get("trend", "unknown"),
                "volatility": summary.get("volatility", "normal")
            },
            "recommendation": {
                "action": recommendation.get("action", "HOLD"),
                "confidence": recommendation.get("confidence", 0),
                "reason": recommendation.get("reason", "")[:300] + "..."
            },
            "num_strategies": len(decision.get("strategies", [])),
            "saved_files": {
                "market_analysis": "data/market_analysis.json",
                "trading_decision": "data/trading_decision.json"
            }
        }

        return response

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ========= Check Latest Decision =========
@app.get("/check_latest")
def check_latest():
    """
    Get a summary of the latest trading decision.
    """
    try:
        path = "data/trading_decision.json"
        if not os.path.exists(path):
            return {"status": "error", "message": "No trading_decision.json found."}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data.get("market_summary", {})
        recommendation = data.get("recommendation", {})

        return {
            "status": "success",
            "sentiment": summary.get("sentiment", "unknown"),
            "trend": summary.get("trend", "unknown"),
            "volatility": summary.get("volatility", "normal"),
            "recommended_action": recommendation.get("action", "HOLD"),
            "confidence": recommendation.get("confidence", 0),
            "reason": recommendation.get("reason", "")[:300] + "...",
            "strategies_count": len(data.get("strategies", []))
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ========= Main Entry =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.trading_service:app", host="127.0.0.1", port=8082, reload=True)
