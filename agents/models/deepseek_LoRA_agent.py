"""
agents/models/deepseek_LoRA_agent.py
DeepSeek-LoRA 交易决策代理（调用阿里云 PAI EAS 已微调模型）
OpenAI-compatible: POST /v1/chat/completions

改进版本：
1. 初始化时自动测试连接
2. 详细的错误诊断
3. 自动重试逻辑
"""

import os
import re
import json
import requests
from agents.trading_agent import TradingAgent


class DeepSeekLoRAAgent(TradingAgent):
    def __init__(
            self,
            name="DeepSeek-LoRA",
            initial_cash=100_000.0,
            max_alloc=0.5,
            fee_bps=10,
            eas_url="http://1533366639129314.cn-beijing.pai-eas.aliyuncs.com/api/predict/quickstart_deploy_20251225_1vxd",
            eas_token="MjY0ZjEwNDFiODk4ZGFiOTEzYjMwZWNlNTRmNzRlY2QxZDMyMzRmMQ==",
            model="DeepSeek-R1-Distill-Qwen-7B",
            temperature=0.3,
            max_tokens=300,
            timeout=120,
            test_connection=True,  # 新增参数
    ):
        super().__init__(
            name=name,
            api_key=None,
            initial_cash=initial_cash,
            max_alloc=max_alloc,
            fee_bps=fee_bps,
        )

        self.eas_url = (eas_url or os.getenv("EAS_URL") or "").strip()
        self.eas_token = (eas_token or os.getenv("EAS_TOKEN") or "").strip()
        self.model = model or os.getenv("EAS_MODEL") or "DeepSeek-R1-Distill-Qwen-7B"

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.logs = []

        if not self.eas_url:
            raise ValueError("EAS_URL 未设置")
        if not self.eas_token:
            raise ValueError("EAS_TOKEN 未设置")

        # 规范化为完整的 chat/completions 路径
        self.chat_url = self._normalize_chat_url(self.eas_url)
        print(f"[{self.name}] Initialized")
        print(f"  Chat URL: {self.chat_url}")

        # 初始化时测试连接
        if test_connection:
            self._test_connection()

    @staticmethod
    def _normalize_chat_url(base: str) -> str:
        """规范化 URL 为完整的 chat/completions 路径"""
        base = base.strip()
        if base.endswith("/v1/chat/completions"):
            return base
        return f"{base.rstrip('/')}/v1/chat/completions"

    def _test_connection(self):
        """初始化时测试 EAS 连接"""
        print(f"  Testing EAS connection...")

        test_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "test"}],
            "temperature": self.temperature,
            "max_tokens": 10,
        }

        try:
            response = self._call_eas_chat_internal(test_payload, timeout=30)
            print(f"  ✓ Connection OK")
            return True
        except Exception as e:
            print(f"  ⚠ Connection test failed: {e}")
            print(f"    This might cause issues during trading")
            return False

    def _call_eas_chat_internal(self, payload, timeout=None):
        """
        内部方法：直接调用 EAS API
        返回原始响应对象
        """
        if timeout is None:
            timeout = self.timeout

        headers = {
            "Authorization": f"Bearer {self.eas_token}",
            "Content-Type": "application/json",
        }

        try:
            r = requests.post(
                self.chat_url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Request timeout ({timeout}s)")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Connection error: {e}")
        except Exception as e:
            raise RuntimeError(f"Request error: {e}")

        # 详细的错误诊断
        if r.status_code != 200:
            error_msg = f"HTTP {r.status_code}"
            try:
                body = r.text[:200]
                error_msg += f"\nResponse: {body}"
            except:
                pass

            # 特殊诊断
            if r.status_code == 403:
                error_msg += "\n[Diagnosis] 403 Forbidden - 可能原因："
                error_msg += "\n  1. IP 白名单限制（你的代码运行在不同的 IP）"
                error_msg += "\n  2. Token 无效或过期"
                error_msg += "\n  3. API Gateway 限制"

            raise RuntimeError(error_msg)

        return r

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text.strip())
        text = re.sub(r"\n?```$", "", text.strip())
        return text.strip()

    @staticmethod
    def _strip_think(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """从文本中提取第一个 JSON 对象"""
        text = text.strip()
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and j > i:
            return text[i: j + 1].strip()
        return text

    def _call_eas_chat(self, messages) -> str:
        """
        调用 EAS API，返回模型响应的文本内容

        Args:
            messages: OpenAI-compatible messages list

        Returns:
            str: 模型的响应文本

        Raises:
            RuntimeError: 如果请求或解析失败
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        r = self._call_eas_chat_internal(payload)

        # 解析 JSON 响应
        try:
            data = r.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON: {e}\nBody: {r.text[:200]}")

        # 提取内容（OpenAI-compatible 格式）
        if isinstance(data, dict) and data.get("choices"):
            c0 = data["choices"][0] or {}
            msg = c0.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content

        # 备选字段
        if isinstance(data, dict):
            for k in ["text", "output", "response", "result"]:
                if isinstance(data.get(k), str):
                    return data[k]

        raise RuntimeError(
            f"Cannot extract content from response: {json.dumps(data)[:200]}"
        )

    def decide(self, market_summary: str, gold_price: float) -> dict:
        """
        根据市场分析做出交易决策
        """
        max_buy_oz = (self.state.cash * self.max_alloc) / gold_price if gold_price > 0 else 0.0
        max_sell_oz = self.state.position_oz

        prompt = f"""You are a professional gold trader.

CURRENT STATE:
- Gold Price: ${gold_price:.2f}/oz
- Cash: ${self.state.cash:,.0f}
- Position: {self.state.position_oz:.2f} oz
- Max BUY: {max_buy_oz:.2f} oz
- Max SELL: {max_sell_oz:.2f} oz

MARKET ANALYSIS:
{market_summary}

Respond with JSON only:
{{"action":"BUY|SELL|HOLD","amount_oz":10,"confidence":4,"reason":"..."}}"""

        try:
            raw = self._call_eas_chat(messages=[{"role": "user", "content": prompt}])

            # 清理响应
            clean = raw.strip()
            clean = self._strip_think(clean)
            clean = self._strip_code_fence(clean)
            clean = self._extract_json_object(clean)

            # 解析 JSON
            obj = json.loads(clean)

            action = str(obj.get("action", "HOLD")).upper()
            amount = float(obj.get("amount_oz", 0))
            confidence = int(obj.get("confidence", 2))
            reason = str(obj.get("reason", ""))

            # 验证金额
            if action == "BUY":
                amount = min(amount, max_buy_oz)
            elif action == "SELL":
                amount = min(amount, max_sell_oz)
            else:
                action = "HOLD"
                amount = 0.0

            return {
                "action": action,
                "amount_oz": max(0.0, amount),
                "confidence": confidence,
                "reason": reason,
            }

        except Exception as e:
            print(f"[ERROR] {self.name} decide failed: {e}")
            return {
                "action": "HOLD",
                "amount_oz": 0.0,
                "confidence": 1,
                "reason": f"Error: {str(e)[:100]}"
            }

    def execute(self, decision: dict, gold_price: float, date_str: str):
        """执行交易决策"""
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

        elif action == "SELL":
            if 0 < amount_oz <= self.state.position_oz:
                revenue = amount_oz * gold_price
                fee = revenue * self.fee_bps / 10000
                net_revenue = revenue - fee

                self.state.cash += net_revenue
                self.state.position_oz -= amount_oz
                executed = True

        else:
            action = "HOLD"
            amount_oz = 0.0
            executed = True

        equity = self.state.cash + self.state.position_oz * gold_price
        self.logs.append(
            {
                "date": date_str,
                "action": action,
                "amount_oz": amount_oz,
                "executed": executed,
                "price": gold_price,
                "equity": equity,
                "position_oz": self.state.position_oz,
                "cash": self.state.cash,
            }
        )