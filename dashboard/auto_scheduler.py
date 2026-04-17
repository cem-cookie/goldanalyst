import os
import json
import zipfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class RateLimiter:
    """Track and limit API usage"""
    
    DAILY_LIMIT = 1000
    
    def __init__(self, limit: int = None):
        self.daily_limit = limit or self.DAILY_LIMIT
        self.usage_file = "data/api_usage.json"
        self._load_usage()
    
    def _load_usage(self):
        """Load today's API usage from file"""
        today = datetime.now().strftime("%Y-%m-%d")
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("date") == today:
                        self.current_usage = data.get("count", 0)
                    else:
                        self.current_usage = 0
            except Exception:
                self.current_usage = 0
        else:
            self.current_usage = 0
    
    def _save_usage(self):
        """Save today's API usage to file"""
        os.makedirs("data", exist_ok=True)
        data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "count": self.current_usage
        }
        with open(self.usage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def can_make_request(self, count: int = 1) -> bool:
        """Check if we can make API request without exceeding limit"""
        return (self.current_usage + count) <= self.daily_limit
    
    def record_usage(self, count: int = 1):
        """Record API usage"""
        self.current_usage += count
        self._save_usage()
    
    def get_usage_info(self) -> dict:
        """Get current usage info"""
        return {
            "current": self.current_usage,
            "limit": self.daily_limit,
            "percentage": round((self.current_usage / self.daily_limit) * 100, 1),
            "remaining": max(0, self.daily_limit - self.current_usage)
        }
    
    def reset_if_needed(self):
        """Reset counter if it's a new day"""
        self._load_usage()


class ErrorTracker:
    """Track errors for alerting after multiple failures"""
    
    def __init__(self, alert_threshold: int = 3):
        self.alert_threshold = alert_threshold
        self.error_file = "data/error_tracker.json"
        self._load_errors()
    
    def _load_errors(self):
        """Load error history"""
        if os.path.exists(self.error_file):
            try:
                with open(self.error_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.errors = data.get("errors", [])
                    self.last_reset = data.get("last_reset", "")
            except Exception:
                self.errors = []
                self.last_reset = ""
        else:
            self.errors = []
            self.last_reset = ""
    
    def _save_errors(self):
        """Save error history"""
        os.makedirs("data", exist_ok=True)
        data = {
            "errors": self.errors[-10:],  # Keep last 10 errors
            "last_reset": self.last_reset
        }
        with open(self.error_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_error(self, error: str, step: str):
        """Add an error with timestamp"""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "error": str(error)[:200]  # Truncate long errors
        })
        self._save_errors()
    
    def should_alert(self) -> bool:
        """Check if we should alert user (3+ consecutive errors)"""
        recent_errors = [e for e in self.errors 
                        if (datetime.now() - datetime.fromisoformat(e["timestamp"])).total_seconds() < 3600]
        return len(recent_errors) >= self.alert_threshold
    
    def get_recent_errors(self) -> list:
        """Get recent errors"""
        return self.errors[-5:] if self.errors else []
    
    def reset(self):
        """Reset error tracker"""
        self.errors = []
        self.last_reset = datetime.now().isoformat()
        self._save_errors()


class LLMFallbackChain:
    """Try multiple LLM providers in sequence"""
    
    def __init__(self, openai_key: str = None, deepseek_key: str = None, ollama_url: str = "http://localhost:11434"):
        self.providers = []
        self.current_provider = None
        
        # Initialize providers in order of preference
        if openai_key:
            from openai import OpenAI
            self.providers.append({
                "name": "openai",
                "client": OpenAI(api_key=openai_key),
                "model": "gpt-4o-mini"
            })
        
        if deepseek_key:
            from openai import OpenAI
            self.providers.append({
                "name": "deepseek",
                "client": OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com"),
                "model": "deepseek-chat"
            })
        
        # Ollama fallback
        self.providers.append({
            "name": "ollama",
            "url": ollama_url,
            "model": "llama3.2"  # Default model
        })
    
    def complete(self, messages: list, temperature: float = 0.3, max_tokens: int = 500) -> Optional[dict]:
        """Try each provider until one works"""
        last_error = None
        
        for provider in self.providers:
            try:
                if provider["name"] == "ollama":
                    result = self._call_ollama(provider, messages, temperature, max_tokens)
                else:
                    result = self._call_api(provider, messages, temperature, max_tokens)
                
                self.current_provider = provider["name"]
                return result
            except Exception as e:
                last_error = e
                continue
        
        return {"error": f"All LLM providers failed. Last error: {last_error}"}
    
    def _call_api(self, provider: dict, messages: list, temperature: float, max_tokens: int) -> dict:
        """Call OpenAI-compatible API"""
        response = provider["client"].chat.completions.create(
            model=provider["model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return {
            "content": response.choices[0].message.content,
            "provider": provider["name"],
            "usage": dict(response.usage) if hasattr(response, 'usage') else {}
        }
    
    def _call_ollama(self, provider: dict, messages: list, temperature: float, max_tokens: int) -> dict:
        """Call Ollama local API"""
        import requests
        
        # Convert messages to Ollama format
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        
        payload = {
            "model": provider["model"],
            "prompt": f"System: {system_msg}\n\nUser: {user_msg}",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{provider['url']}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        return {
            "content": response.json().get("response", ""),
            "provider": "ollama",
            "usage": {}
        }


class SchedulerLogger:
    """Logger for scheduler events"""
    
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"scheduler_{datetime.now().strftime('%Y_%m')}.log"
    
    def log(self, level: str, message: str, step: str = None):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step_info = f" [{step}]" if step else ""
        log_line = f"[{timestamp}] [{level.upper()}]{step_info} {message}\n"
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)
    
    def info(self, message: str, step: str = None):
        self.log("INFO", message, step)
    
    def warn(self, message: str, step: str = None):
        self.log("WARN", message, step)
    
    def error(self, message: str, step: str = None):
        self.log("ERROR", message, step)


class AutoScheduler:
    """Main scheduler for automated data pipeline"""
    
    def __init__(self, 
                 interval_minutes: int = 60,
                 api_limit: int = 1000,
                 auto_mode: bool = True):
        self.interval_minutes = interval_minutes
        self.auto_mode = auto_mode
        self.is_paused = False
        self.last_run = None
        self.next_run = None
        
        # Initialize components
        self.rate_limiter = RateLimiter(api_limit)
        self.error_tracker = ErrorTracker()
        self.logger = SchedulerLogger()
        
        # Set next run time
        self._calculate_next_run()
    
    def _calculate_next_run(self):
        """Calculate next scheduled run time"""
        if self.last_run:
            self.next_run = self.last_run + timedelta(minutes=self.interval_minutes)
        else:
            self.next_run = datetime.now()
    
    def get_status(self) -> dict:
        """Get current scheduler status"""
        return {
            "auto_mode": self.auto_mode,
            "is_paused": self.is_paused,
            "interval_minutes": self.interval_minutes,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "api_usage": self.rate_limiter.get_usage_info(),
            "should_alert": self.error_tracker.should_alert(),
            "recent_errors": self.error_tracker.get_recent_errors()
        }
    
    def toggle_auto_mode(self):
        """Toggle auto mode on/off"""
        self.auto_mode = not self.auto_mode
        self.logger.info(f"Auto mode: {'enabled' if self.auto_mode else 'disabled'}")
        return self.auto_mode
    
    def toggle_pause(self):
        """Pause/resume scheduler"""
        self.is_paused = not self.is_paused
        self.logger.info(f"Scheduler: {'paused' if self.is_paused else 'resumed'}")
        return self.is_paused
    
    def set_interval(self, minutes: int):
        """Change interval"""
        self.interval_minutes = minutes
        self._calculate_next_run()
        self.logger.info(f"Interval changed to {minutes} minutes")
    
    def can_run(self) -> tuple[bool, str]:
        """Check if pipeline can run"""
        if not self.auto_mode:
            return False, "Auto mode is disabled"
        
        if self.is_paused:
            return False, "Scheduler is paused"
        
        if self.next_run and datetime.now() < self.next_run:
            remaining = (self.next_run - datetime.now()).total_seconds() / 60
            return False, f"Next run in {remaining:.0f} minutes"
        
        # Check rate limit (estimate ~80 requests per full pipeline)
        if not self.rate_limiter.can_make_request(80):
            return False, "API rate limit reached"
        
        return True, "Ready to run"
    
    def record_run(self, success: bool, steps_completed: int = 0):
        """Record pipeline run"""
        self.last_run = datetime.now()
        self._calculate_next_run()
        
        if success:
            self.error_tracker.reset()
            self.logger.info(f"Pipeline completed successfully ({steps_completed} steps)")
        else:
            self.logger.error(f"Pipeline failed after {steps_completed} steps")
    
    def record_api_usage(self, count: int):
        """Record API usage"""
        self.rate_limiter.record_usage(count)
    
    def add_error(self, error: str, step: str):
        """Record an error"""
        self.error_tracker.add_error(error, step)
        self.logger.error(f"Error in {step}: {error}")


# Global scheduler instance
_scheduler: Optional[AutoScheduler] = None


def get_scheduler(interval_minutes: int = 60, 
                  api_limit: int = 1000,
                  auto_mode: bool = True) -> AutoScheduler:
    """Get or create global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = AutoScheduler(
            interval_minutes=interval_minutes,
            api_limit=api_limit,
            auto_mode=auto_mode
        )
    return _scheduler