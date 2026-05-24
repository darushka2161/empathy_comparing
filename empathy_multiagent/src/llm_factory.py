# llm_factory.py

import os
import re
import time
import json
import asyncio
from openai import AsyncOpenAI
from .config import MODEL_REGISTRY


class LLMFactory:
    """Универсальная фабрика для работы с любой OpenAI-совместимой моделью."""

    def __init__(self, model_key: str, config: dict = None):
        """
        model_key: ключ из MODEL_REGISTRY (например "llama-3.1-8b")
        config: можно передать свой dict вместо MODEL_REGISTRY
        """
        self.cfg = (config or MODEL_REGISTRY)[model_key]
        self.model_key = model_key

        # Поддержка нескольких ключей через запятую: KEY=key1,key2,key3
        raw_keys = os.environ.get(self.cfg["api_key_env"], "")
        keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not keys:
            raise ValueError(
                f"API key not found. Set environment variable:\n"
                f"  export {self.cfg['api_key_env']}=your_key_here\n"
                f"Get it from: {self._get_signup_url()}"
            )

        self._clients = [
            AsyncOpenAI(base_url=self.cfg["base_url"], api_key=k)
            for k in keys
        ]
        self._key_idx = 0
        self._rotation_streak = 0  # сколько ротаций подряд без успешного вызова
        self.model = self.cfg["model"]
        self.max_rpm = self.cfg.get("max_rpm", 30)
        self._last_call_time = 0.0
        self._call_count = 0
        self._rate_lock = asyncio.Lock()

    def _get_signup_url(self) -> str:
        urls = {
            "groq": "https://console.groq.com",
            "together": "https://api.together.ai",
            "mistral": "https://console.mistral.ai",
            "github-models": "https://github.com/settings/tokens",
            "openrouter": "https://openrouter.ai/settings/keys",
            "local-vllm": "запусти сервер: python serve_local.py --model <key>",
        }
        return urls.get(self.cfg["provider"], "check provider docs")

    def _rotate_key(self) -> bool:
        """Переключает на следующий ключ. Возвращает True если прошли полный круг."""
        self._key_idx = (self._key_idx + 1) % len(self._clients)
        self._last_call_time = 0.0
        self._rotation_streak += 1
        full_cycle = (self._rotation_streak % len(self._clients) == 0)
        print(f"  [Key rotation] -> key {self._key_idx + 1}/{len(self._clients)}")
        return full_cycle

    async def _rate_limit(self):
        """Простой rate limiter: не больше max_rpm запросов в минуту.
        Lock гарантирует, что параллельные вызовы (asyncio.gather) выстраиваются
        в очередь, а не стреляют одновременно."""
        async with self._rate_lock:
            min_interval = 60.0 / self.max_rpm
            elapsed = time.time() - self._last_call_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_call_time = time.time()
            self._call_count += 1

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        retries: int = 3,
    ) -> str:
        """
        Один вызов LLM. Возвращает текст ответа.
        Автоматический retry с exponential backoff.
        """
        disable_thinking = self.cfg.get("disable_thinking", False)
        if disable_thinking:
            system_prompt = (
                "IMPORTANT: Do NOT use reasoning or thinking mode. "
                "Do NOT output <think> tags. Respond directly and concisely.\n\n"
                + system_prompt
            )

        # С несколькими ключами нужно больше попыток: N ключей * 3 цикла + запас
        effective_retries = max(retries, len(self._clients) * 3) if len(self._clients) > 1 else retries

        for attempt in range(effective_retries):
            try:
                await self._rate_limit()
                # Если модель требует больше токенов (напр. для завершения <think>-блока)
                effective_max_tokens = max(
                    max_tokens, self.cfg.get("min_max_tokens", 0)
                )
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
                    provider = self.cfg.get("provider", "")
                    if provider == "groq":
                        # Groq-специфичный параметр: скрывает <think>-блок на стороне сервера
                        create_kwargs["extra_body"] = {"reasoning_format": "hidden"}
                    elif provider == "local-vllm":
                        # vLLM + Qwen3: отключаем thinking через chat_template_kwargs
                        create_kwargs["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": False}
                        }
                response = await self._clients[self._key_idx].chat.completions.create(**create_kwargs)
                text = response.choices[0].message.content.strip()
                # Убираем блоки <think>...</think> (Qwen-3, DeepSeek-R1 и др.)
                if "<think>" in text and "</think>" in text:
                    text = text[text.rfind("</think>") + len("</think>"):].strip()
                self._rotation_streak = 0  # успешный вызов — сбрасываем счётчик
                return text
            except Exception as e:
                wait = 2 ** attempt
                # Парсим "Please try again in X.XXs" из ответа rate-limit
                err_str = str(e)
                if "Please try again in" in err_str:
                    try:
                        raw = err_str.split("Please try again in")[1].strip()
                        m = re.match(r'(\d+)m([\d.]+)s', raw)
                        if m:
                            after = int(m.group(1)) * 60 + float(m.group(2))
                        else:
                            m = re.match(r'([\d.]+)s', raw)
                            after = float(m.group(1)) if m else 0.0
                        if after > 5.0 and len(self._clients) > 1:
                            full_cycle = self._rotate_key()
                            if full_cycle:
                                print(f"  [All {len(self._clients)} keys exhausted] waiting 3 minutes...")
                                wait = 180.0
                            else:
                                wait = 1.0
                        else:
                            wait = max(wait, after + 1.0)
                    except (ValueError, AttributeError):
                        pass
                elif "timed out" in err_str.lower() and len(self._clients) > 1:
                    self._rotate_key()
                    wait = 1.0
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
        """
        Вызов LLM с ожиданием JSON-ответа.
        Автоматически парсит JSON, при ошибке — retry с подсказкой.
        """
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


async def quick_test():
    """Проверка что API работает."""
    from dotenv import load_dotenv
    load_dotenv()
    llm = LLMFactory("llama-3.1-8b")
    print(f"Testing: {llm.info}")
    response = await llm.generate(
        system_prompt="You are a helpful assistant.",
        user_message="Say hello in one sentence.",
    )
    print(f"Response: {response}")
    print(f"Total calls: {llm._call_count}")


if __name__ == "__main__":
    asyncio.run(quick_test())
