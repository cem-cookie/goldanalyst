import streamlit as st
from cryptography.fernet import Fernet
import json
import os
import traceback
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import scheduler components
from dashboard.auto_scheduler import (
    get_scheduler, AutoScheduler, RateLimiter, ErrorTracker, LLMFallbackChain
)
# Import archiver
from dashboard.auto_archive import get_archiver, AutoArchiver


def _resolve_api_key_pipeline(user_provided_key: str = None) -> str | None:
    """
    Resolve API key with fallback chain: user input -> env var -> st.secrets.
    Priority: user_provided_key > os.getenv > st.secrets > None
    """
    if user_provided_key:
        return user_provided_key
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    try:
        if "openai_api_key" in st.secrets:
            return st.secrets["openai_api_key"]
    except Exception:
        pass
    return None


def _get_user_api_key_from_enc(encrypted_key: str | None = None, fernet_key: str | None = None) -> str | None:
    """
    Decrypt user-provided API key from encrypted session state.
    Returns None if no valid key can be decrypted.
    """
    if encrypted_key and fernet_key:
        try:
            f = Fernet(fernet_key.encode())
            return f.decrypt(encrypted_key.encode()).decode()
        except Exception:
            pass
    return None


class PipelineRunner:
    """Execute the full automated pipeline: Gold -> News -> Decision -> Risk"""
    
    def __init__(self):
        self.scheduler = get_scheduler()
        self.archiver = get_archiver()
        self.api_usage = 0
    
    def step_1_fetch_gold_price(self, silent: bool = True) -> Dict:
        """Step 1: Fetch latest gold price"""
        try:
            from agents.data_agent import DataAgent
            
            agent = DataAgent()
            df = agent.save_gold_history_to_file(days=1095)
            
            if df is not None and not df.empty:
                # Estimate API calls: yfinance ~5
                self.api_usage += 5
                self.scheduler.record_api_usage(5)
                
                return {
                    "success": True,
                    "message": f"Gold price updated: {len(df)} records",
                    "records": len(df)
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to fetch gold price"
                }
        except Exception as e:
            self.scheduler.add_error(str(e), "gold_price")
            return {
                "success": False,
                "message": f"Error fetching gold price: {e}"
            }
    
    def step_2_collect_news(self, sources: List[str] = None, limit: int = 10) -> Dict:
        """Step 2: Collect and analyze news"""
        if sources is None:
            sources = ["yahoo", "investing.com"]
        
        try:
            from agents.data_agent import DataAgent
            
            # Initialize with fallback chain
            openai_key = os.getenv("OPENAI_API_KEY")
            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            
            agent = DataAgent(openai_api_key=openai_key)
            
            # Collect news from all sources
            all_news = []
            if "yahoo" in sources:
                try:
                    all_news += agent.get_yahoo_rss_news(limit)
                except Exception as e:
                    self.scheduler.add_error(f"Yahoo RSS failed: {e}", "news_yahoo")
            
            if "investing.com" in sources:
                try:
                    all_news += agent.get_investing_news(limit)
                except Exception as e:
                    self.scheduler.add_error(f"Investing.com failed: {e}", "news_investing")
            
            if "MetalsDaily" in sources:
                try:
                    all_news += agent.get_metalsdaily_news(limit)
                except Exception as e:
                    self.scheduler.add_error(f"MetalsDaily failed: {e}", "news_metalsdaily")
            
            if not all_news:
                return {
                    "success": False,
                    "message": "No news collected from any source"
                }
            
            # Save to JSON
            os.makedirs("data", exist_ok=True)
            json_path = "data/gold_news.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_news, f, ensure_ascii=False, indent=2)
            
            # Estimate API calls: ~10 for news collection
            self.api_usage += 10
            self.scheduler.record_api_usage(10)
            
            # Filter relevant news via LLM
            try:
                filtered = agent.filter_and_fetch_fulltext(json_path)
                # LLM filtering adds ~50 API calls
                self.api_usage += 50
                self.scheduler.record_api_usage(50)
            except Exception as e:
                self.scheduler.add_error(f"News filtering failed: {e}", "news_filter")
            
            # Analyze with LLM
            try:
                summary = agent.analyze_market_news(json_path, min_quality_score=30)
                # Analysis adds ~5 API calls
                self.api_usage += 5
                self.scheduler.record_api_usage(5)
            except Exception as e:
                self.scheduler.add_error(f"News analysis failed: {e}", "news_analysis")
            
            return {
                "success": True,
                "message": f"Collected {len(all_news)} news items",
                "count": len(all_news)
            }
            
        except Exception as e:
            self.scheduler.add_error(str(e), "news_collection")
            return {
                "success": False,
                "message": f"Error collecting news: {e}"
            }
    
    def step_3_generate_decision(self, context: Dict = None, model_name: str = "gpt-4o-mini", api_key_enc: str | None = None) -> Dict:
        """Step 3: Generate trading decision"""
        # Map UI model names to API model IDs
        model_map = {
            "ChatGPT (OpenAI)": "gpt-4o-mini",
            "Claude (Anthropic)": "claude-sonnet-4-20250514",
        }
        api_model = model_map.get(model_name, model_name)
        
        if context is None:
            # Fetch latest gold price for fallback
            latest = None
            try:
                import yfinance
                ticker = yfinance.Ticker("GC=F")
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    latest = round(float(hist["Close"].iloc[-1]), 2)
            except:
                pass
            
            context = {
                "strategy": "Swing",
                "investment_level": "Active",
                "buy_price_threshold": latest,
                "sell_price_threshold": latest,
                "target_profit": 0.1,
                "latest_price": latest,
            }
        
        try:
            from agents.trading_agent import TradingAgent

            # Resolve API key with fallback chain
            user_key = _get_user_api_key_from_enc(api_key_enc, st.session_state.get('fernet_key'))
            api_key = _resolve_api_key_pipeline(user_key)
            if not api_key:
                return {
                    "success": False,
                    "message": "No API key available. Please enter your OpenAI API key in settings."
                }

            agent = TradingAgent(
                name="AutoTrader",
                api_key=api_key,
                json_path="gold_news.json",
                context=context,
                model_name=api_model
            )
            
            decision = agent.run()
            
            self.api_usage += 5
            self.scheduler.record_api_usage(5)
            
            if decision:
                return {
                    "success": True,
                    "message": f"Decision generated: {decision.get('recommendation', {}).get('action', 'UNKNOWN')}",
                    "decision": decision
                }
            else:
                return {"success": False, "message": "No decision generated"}
        
        except Exception as e:
            self.scheduler.add_error(str(e), "trading_decision")
            return {"success": False, "message": f"Error generating decision: {e}"}

    
    def step_4_risk_analysis(self, context: Dict = None) -> Dict:
        """Step 4: Run risk analysis"""
        if context is None:
            # Fetch latest gold price for fallback
            latest = None
            try:
                import yfinance
                ticker = yfinance.Ticker("GC=F")
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    latest = round(float(hist["Close"].iloc[-1]), 2)
            except:
                pass
            
            context = {
                "strategy": "Swing",
                "investment_level": "Active",
                "buy_price_threshold": latest,
                "sell_price_threshold": latest,
                "target_profit": 0.1,
                "latest_price": latest,
            }
        
        try:
            from agents.risk_agent import RiskAgent

            deepseek_key = os.getenv("DEEPSEEK_API_KEY")

            # Check if trading decision exists
            decision_path = "data/trading_decision.json"
            if not os.path.exists(decision_path):
                return {
                    "success": False,
                    "message": "No trading decision found"
                }

            # Resolve API key with fallback chain
            user_key = _get_user_api_key_from_enc(
                st.session_state.get('api_key_enc'),
                st.session_state.get('fernet_key')
            )
            openai_key = _resolve_api_key_pipeline(user_key)
            if not openai_key:
                return {
                    "success": False,
                    "message": "No API key available. Please enter your OpenAI API key in settings."
                }

            agent = RiskAgent(
                decision_path=decision_path,
                out_path="data/risk_report.json",
                openai_api_key=openai_key,
                context=context
            )
            
            try:
                report = agent.run()
            except Exception as primary_error:
                # Try DeepSeek
                try:
                    from openai import OpenAI
                    agent.client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
                    report = agent.run()
                except Exception as deepseek_error:
                    # Try Ollama
                    try:
                        import requests
                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": "llama3.2",
                                "prompt": "You are a risk analyst. Provide risk assessment.",
                                "stream": False
                            },
                            timeout=60
                        )
                        if response.status_code == 200:
                            report = {"summary": {"portfolio_risk": "medium", "comment": "Ollama fallback"}}
                        else:
                            raise Exception("Ollama unavailable")
                    except Exception:
                        raise Exception(f"Risk analysis failed: {primary_error}")
            
            # Record API usage (~5 for risk analysis)
            self.api_usage += 5
            self.scheduler.record_api_usage(5)
            
            if report:
                return {
                    "success": True,
                    "message": "Risk analysis completed",
                    "report": report
                }
            else:
                return {
                    "success": False,
                    "message": "No risk report generated"
                }
                
        except Exception as e:
            self.scheduler.add_error(str(e), "risk_analysis")
            return {
                "success": False,
                "message": f"Error in risk analysis: {e}"
            }
    
    def run_full_pipeline(self, context: Dict = None) -> Dict:
        """Execute complete automated pipeline"""
        print("\n" + "="*60)
        print("🚀 STARTING FULL AUTOMATED PIPELINE")
        print("="*60)
        
        results = {
            "started_at": datetime.now().isoformat(),
            "steps": {},
            "total_api_usage": 0,
            "success": True
        }
        
        # Check if we can run
        can_run, reason = self.scheduler.can_run()
        if not can_run:
            results["success"] = False
            results["blocked_reason"] = reason
            print(f"⏸ Pipeline blocked: {reason}")
            return results
        
        # Step 1: Gold Price
        print("\n[1/4] Fetching gold price...")
        step1 = self.step_1_fetch_gold_price()
        results["steps"]["gold_price"] = step1
        if not step1["success"]:
            print(f"  ⚠️ {step1['message']}")
        
        # Step 2: News Collection
        print("\n[2/4] Collecting and analyzing news...")
        step2 = self.step_2_collect_news()
        results["steps"]["news"] = step2
        if not step2["success"]:
            print(f"  ⚠️ {step2['message']}")
        
        # Step 3: Trading Decision
        print("\n[3/4] Generating trading decision...")
        step3 = self.step_3_generate_decision(context, model_name=st.session_state.get('selected_model','ChatGPT (OpenAI)'), api_key_enc=st.session_state.get('api_key_enc'))
        results["steps"]["decision"] = step3
        if not step3["success"]:
            print(f"  ⚠️ {step3['message']}")
        
        # Step 4: Risk Analysis
        print("\n[4/4] Running risk analysis...")
        step4 = self.step_4_risk_analysis(context)
        results["steps"]["risk"] = step4
        if not step4["success"]:
            print(f"  ⚠️ {step4['message']}")
        
        # Calculate success
        success_count = sum(1 for s in results["steps"].values() if s.get("success", False))
        results["steps_completed"] = success_count
        results["total_api_usage"] = self.api_usage
        
        # Record run
        self.scheduler.record_run(
            success=results["success"],
            steps_completed=success_count
        )
        
        print("\n" + "="*60)
        print(f"✅ PIPELINE COMPLETED: {success_count}/4 steps successful")
        print(f"📊 API Usage: {self.api_usage} requests")
        print("="*60 + "\n")
        
        return results
    
    def run_standalone_step(self, step_name: str, **kwargs) -> Dict:
        """Run a single step independently"""
        if step_name == "gold":
            return self.step_1_fetch_gold_price()
        elif step_name == "news":
            return self.step_2_collect_news(**kwargs)
        elif step_name == "decision":
            return self.step_3_generate_decision(**kwargs)
        elif step_name == "risk":
            return self.step_4_risk_analysis(**kwargs)
        else:
            return {"success": False, "message": f"Unknown step: {step_name}"}


# Global pipeline runner
_pipeline: Optional[PipelineRunner] = None


def get_pipeline() -> PipelineRunner:
    """Get or create global pipeline runner"""
    global _pipeline
    if _pipeline is None:
        _pipeline = PipelineRunner()
    return _pipeline


# Standalone function for easy imports
def run_automated_pipeline(context: Dict = None) -> Dict:
    """Quick function to run the full pipeline"""
    pipeline = get_pipeline()
    return pipeline.run_full_pipeline(context)