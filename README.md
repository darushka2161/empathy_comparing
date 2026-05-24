# Empathy Multiagent

Сравнение мультиагентных архитектур для генерации эмпатичных ответов на датасете **EmpatheticDialogues** (test split, ~2 547 диалогов).

---

## Структура репозитория

```
v2_vkr/
├── experiment.ipynb            # Ноутбук: запуск всех экспериментов и сводные таблицы
│
├── empathy_multiagent/
│   ├── run_experiment.py       # Главный скрипт запуска эксперимента
│   ├── serve_local.py          # Запуск локального vLLM-сервера
│   ├── build_index.py          # Построение FAISS-индекса (нужен для RAG/MAS-C/TRACE)
│   ├── requirements.txt
│   ├── .env.example            # Шаблон переменных окружения
│   │
│   ├── src/
│   │   ├── config.py           # MODEL_REGISTRY — все доступные модели и их параметры
│   │   ├── llm_factory.py      # Универсальный async LLM-клиент (OpenAI-compatible API)
│   │   ├── load_dataset.py     # Загрузка EmpatheticDialogues (HuggingFace)
│   │   ├── metrics.py          # BLEU, ROUGE, BERTScore, Distinct, Accuracy, AvgLen
│   │   └── fixed_few_shot.py   # Выборка few-shot примеров из train-сплита
│   │
│   ├── architectures/
│   │   ├── empathy_zero_shot.py   # 1 LLM-вызов
│   │   ├── empathy_few_shot.py    # 1 LLM-вызов + few-shot примеры
│   │   ├── empathy_ektc.py        # TOOL-ED/EKTC + COMET
│   │   ├── empathy_trace.py       # TRACE (Liu et al., 2025): 4 вызова + RAG
│   │   ├── empathy_insideout.py   # InsideOut (ACL 2024): 4 агента-эмоции + агрегатор: 6 вызовов
│   │   ├── empathy_chain.py       # Каскадная цепочка: 4 вызова
│   │   ├── empathy_debate.py      # 3 агента + арбитр: 5 вызовов
│   │   ├── empathy_loop.py        # Итеративная рефинация: 5–11 вызовов
│   │   ├── empathy_rag.py         # RAG + FAISS: 3 вызова
│   │   └── empathy_mas_c.py       # RAG + Planner + 2×Gen + Selector: 5 вызовов
│   │
│   ├── analysis/
│   │   ├── compare_results.py     # Сводная таблица по всем outputs/
│   │   └── recompute_metrics.py   # Пересчёт метрик без повторного инференса
│   │
│   ├── outputs/                   # JSON-результаты экспериментов (создаётся автоматически)
│   └── retriever_cache/           # FAISS-индекс (создаётся через build_index.py)
│
├── EKTC/                       # Оригинальный репозиторий TOOL-ED (референс)
└── TRACE/                      # Оригинальный репозиторий TRACE (референс)
```

---

## Установка

```bash
cd empathy_multiagent
pip install -r requirements.txt
```

Для локального инференса через vLLM:
```bash
pip install vllm
```

---

## Настройка окружения

```bash
cp empathy_multiagent/.env.example empathy_multiagent/.env
```

Заполни нужные ключи в `.env`. Для облачных моделей достаточно одного из провайдеров (Groq — бесплатно).

Для локальных моделей добавь:
```
LOCAL_API_KEY=EMPTY
```

---

## Запуск экспериментов

### Через ноутбук

Открой `experiment.ipynb` — там настроен автоматический прогон всех архитектур для каждой модели.

### Вручную

```bash
cd empathy_multiagent
python run_experiment.py --model <MODEL> --arch <ARCH> [--limit N] [--no-bertscore]
```

**Аргументы:**

| Аргумент | Описание |
|---|---|
| `--model` | Ключ модели из `src/config.py` |
| `--arch` | Архитектура (`empathy_zero_shot`, `empathy_chain`, ...) |
| `--limit N` | Сколько диалогов прогнать (по умолчанию все ~2547) |
| `--no-bertscore` | Пропустить BERTScore (быстрее, не нужен GPU) |

**Пример:**
```bash
python run_experiment.py --model llama-3.1-8b --arch empathy_chain --limit 50
```

Результат сохраняется в `outputs/<model>_<arch>.json`.

### Для архитектур с RAG

Перед первым запуском `empathy_rag`, `empathy_mas_c` или `empathy_trace` нужно построить FAISS-индекс (~2 мин):

```bash
cd empathy_multiagent
python build_index.py
```

---

## Локальный инференс (vLLM)

```bash
# Терминал 1: запустить сервер
cd empathy_multiagent
python serve_local.py --model mistral-small-3.2

# Терминал 2: запустить эксперимент как обычно
python run_experiment.py --model mistral-small-3.2 --arch empathy_chain --limit 50
```

Доступные флаги `serve_local.py`: `--port`, `--gpu-memory-utilization`, `--tensor-parallel-size`, `--max-model-len`, `--dtype`.

---

## Доступные модели

### API-провайдеры

| Ключ | Провайдер | Размер | Лимиты | Ключ окружения |
|---|---|---|---|---|
| `llama-3.1-8b` | Groq | 8B | 30 RPM / 14 400 RPD | `GROQ_API_KEY` |
| `qwen-3-32b` | Groq | 32B | 30 RPM / 14 400 RPD | `GROQ_API_KEY` |
| `llama-3.3-70b` | Groq | 70B | 30 RPM / 14 400 RPD | `GROQ_API_KEY` |
| `mistral-small` | Mistral API | 24B | 2 RPM / 1B tok·мес | `MISTRAL_API_KEY` |

### Локальные (vLLM)

| Ключ | Модель | Размер |
|---|---|---|
| `mistral-small-3.2` | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | 24B |
| `qwen3-32b-local` | Qwen/Qwen3-32B | 32B |
| `llama-3.1-8b-local` | meta-llama/Llama-3.1-8B-Instruct | 8B |

Добавить новую модель — в `src/config.py` в `MODEL_REGISTRY`.

---

## Анализ результатов

```bash
cd empathy_multiagent

# Сводная таблица и CSV по всем outputs/
python analysis/compare_results.py

# Пересчёт метрик без повторного инференса
python analysis/recompute_metrics.py [--no-bertscore]

# Графики
cd ..
python make_figures.py
```
