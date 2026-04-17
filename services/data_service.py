# services/data_service.py
# -*- coding: utf-8 -*-
"""
FastAPI microservice wrapper for DataAgent.
You can start it independently with:
    uvicorn services.data_service:app --reload
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from agents.data_agent import DataAgent

# ---------- Initialize FastAPI ----------
app = FastAPI(
    title="DataAgent Microservice",
    description="Independent microservice for fetching and analyzing gold market news & price data",
    version="1.0.0"
)

# ---------- 模型 ----------
class CollectRequest(BaseModel):
    limit: int = 10
    date_str: Optional[str] = None
    source: Optional[str] = "all"  # all / gnews / yahoo / investing / metalsdaily
    query: Optional[str] = "gold"

class AnalyzeRequest(BaseModel):
    json_path: str
    min_quality_score: Optional[int] = None

# ---------- 初始化 Agent ----------
agent = DataAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))

# ---------- 路由 ----------
@app.get("/")
def root():
    return {
        "service": "DataAgent Microservice",
        "message": "Welcome! Available endpoints: /collect_news, /analyze_news, /gold_history"
    }

@app.post("/collect_news")
def collect_news(req: CollectRequest):
    """Collect latest news from multiple sources"""
    try:
        if req.source == "gnews":
            news, path = agent.collect_past_news_with_gnews(req.date_str or "2025-11-06", query=req.query, limit=req.limit)
        else:
            news, path = agent.collect_all_news(limit=req.limit, date_str=req.date_str)
        return {"status": "success", "saved_path": path, "total_items": len(news)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_news")
def analyze_news(req: AnalyzeRequest):
    """Analyze market news from an existing JSON file"""
    try:
        result = agent.analyze_market_news(req.json_path, min_quality_score=req.min_quality_score)
        return {"status": "success", "summary": result[:5000]}  # 截断长文本
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {req.json_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gold_history")
def gold_history(days: int = Query(365, description="Number of days of gold history to fetch")):
    """Fetch and save gold price history"""
    try:
        df = agent.save_gold_history_to_file(days=days)
        return {"status": "success", "rows": len(df) if df is not None else 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 启动 ----------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)
