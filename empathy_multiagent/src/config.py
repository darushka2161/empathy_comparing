# config.py

MODEL_REGISTRY = {
    # === LOCAL VLLM (собственный сервер) ===
    # Требует: pip install vllm
    # Запуск сервера: python serve_local.py --model <key>
    # Переменная окружения: LOCAL_API_KEY=EMPTY (любая строка)
    "mistral-small-3.2": {
        "base_url": "http://localhost:8000/v1",
        "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "api_key_env": "LOCAL_API_KEY",
        "provider": "local-vllm",
        "size": "24B",
        "max_rpm": 120,
        "max_rpd": 999999,
        "notes": "Локальный vLLM. Запуск: python serve_local.py --model mistral-small-3.2",
    },
    "qwen3-32b-local": {
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen3-32B",
        "api_key_env": "LOCAL_API_KEY",
        "provider": "local-vllm",
        "size": "32B",
        "max_rpm": 120,
        "max_rpd": 999999,
        "disable_thinking": True,
        "min_max_tokens": 2048,
        "notes": "Локальный vLLM. Запуск: python serve_local.py --model qwen3-32b-local",
    },
    "llama-3.1-8b-local": {
        "base_url": "http://localhost:8000/v1",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "api_key_env": "LOCAL_API_KEY",
        "provider": "local-vllm",
        "size": "8B",
        "max_rpm": 120,
        "max_rpd": 999999,
        "notes": "Локальный vLLM. Запуск: python serve_local.py --model llama-3.1-8b-local",
    }
}
