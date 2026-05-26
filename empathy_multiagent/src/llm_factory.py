# llm_factory.py

import os
import json
import asyncio
from openai import AsyncOpenAI
from .config import MODEL_REGISTRY


class LLMFactory:
    """Универсальная фабрика для работы с локальным vLLM-сервером (OpenAI-compatible API)."""

    def __init__(self, model_key: str, config: dict = None):
        self.cfg = (config or MODEL_REGISTRY)[model_key]
        self.model_key = model_key

        api_key = os.environ.get(self.cfg["api_key_env"], "")
        if not api_key:
            raise ValueError(
                f"API key not found. Set environment variable:\n"
                f"  export {self.cfg['api_key_env']}=EMPTY\n"
                f"Then start the server: python serve_local.py --model {model_key}"
            )

        self._client = AsyncOpenAI(base_url=self.cfg["base_url"], api_key=api_key)
        self.model = self.cfg["model"]
        self.max_rpm = self.cfg.get("max_rpm", 120)
        self._last_call_time = 0.0
        self._call_count = 0
        self._rate_lock = asyncio.Lock()

    async def _rate_limit(self):
        """Простой rate limiter: не больше max_rpm запросов в минуту."""
        async with self._rate_lock:
            min_interval = 60.0 / self.max_rpm
            elapsed = asyncio.get_event_loop().time() - self._last_call_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_call_time = asyncio.get_event_loop().time()
            self._call_count += 1

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        retries: int = 3,
    ) -> str:
        """Один вызов LLM. Возвращает текст ответа."""
        disable_thinking = self.cfg.get("disable_thinking", False)
        if disable_thinking:
            system_prompt = (
                "IMPORTANT: Do NOT use reasoning or thinking mode. "
                "Do NOT output <think> tags. Respond directly and concisely.\n\n"
                + system_prompt
            )

        for attempt in range(retries):
            try:
                await self._rate_limit()
                effective_max_tokens = max(max_tokens, self.cfg.get("min_max_tokens", 0))
                create_kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                )
                if disable_thinking:
                    # vLLM + Qwen3: отключаем thinking через chat_template_kwargs
                    create_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                response = await self._client.chat.completions.create(**create_kwargs)
                text = response.choices[0].message.content.strip()
                # Убираем блоки <think>...</think> (Qwen-3 и др.)
                if "<think>" in text and "</think>" in text:
                    text = text[text.rfind("</think>") + len("</think>"):].strip()
                return text
            except Exception as e:
                wait = 2 ** attempt
                print(f"  [Retry {attempt + 1}/{retries}] {type(e).__name__}: {e}")
                print(f"  Waiting {wait:.1f}s...")
                await asyncio.sleep(wait)

        raise RuntimeError(f"Failed after {retries} retries")

    async def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        retries: int = 3,
    ) -> dict:
        """Вызов LLM с ожиданием JSON-ответа. Автоматически парсит JSON, при ошибке — retry."""
        sys_prompt = system_prompt
        for attempt in range(retries):
            text = await self.generate(
                system_prompt=sys_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=3,
            )
            try:
                clean = text.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                if clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                return json.loads(clean.strip())
            except json.JSONDecodeError:
                if attempt < retries - 1:
                    sys_prompt = (
                        sys_prompt
                        + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                        "Respond with ONLY a JSON object, no markdown, no explanation."
                    )
                    continue

        return {"_raw": text, "_parse_error": True}

    @property
    def info(self) -> str:
        return f"{self.model_key} ({self.cfg['size']}) via {self.cfg['provider']}"
