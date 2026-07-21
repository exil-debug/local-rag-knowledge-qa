"""DeepSeek-V4 对话接口封装模块

基于 OpenAI 兼容接口调用 DeepSeek-V4 大模型。
API Key 统一从环境变量读取，禁止硬编码。
"""

import os
from typing import Optional, List, Dict

import requests
from dotenv import load_dotenv

load_dotenv()


class DeepSeekClient:
    """DeepSeek-V4 API 客户端，封装对话生成接口。"""

    def __init__(self, api_key=None, api_base=None, model=None):
        """初始化客户端。参数优先使用传入值，否则从环境变量读取。"""
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").strip()
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未配置！请检查 .env 文件。")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def chat(self, messages, temperature=0.3, max_tokens=2048):
        """发送对话请求并返回模型回复文本。"""
        url = f"{self.api_base.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = self._session.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"DeepSeek API 调用失败: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"解析 API 响应失败: {e}") from e

    def chat_with_context(self, system_prompt, user_question, context, temperature=0.3):
        """基于检索到的文档上下文进行问答。"""
        filled_prompt = system_prompt.replace("{context}", context).replace("{question}", user_question)
        messages = [
            {"role": "system", "content": filled_prompt},
            {"role": "user", "content": user_question},
        ]
        return self.chat(messages, temperature=temperature)


if __name__ == "__main__":
    client = DeepSeekClient()
    reply = client.chat([{"role": "user", "content": "你好"}])
    print(f"模型回复: {reply}")