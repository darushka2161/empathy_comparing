# Empathy Multiagent

Comparison of multi-agent architectures for generating empathetic responses on the **EmpatheticDialogues** dataset (test split, ~2,547 dialogues).

---

## Repository Structure

```
v2_vkr/
├── experiment.ipynb            # Notebook: runs all experiments and builds summary tables
│
└── empathy_multiagent/
    ├── run_experiment.py       # Main experiment runner
    ├── serve_local.py          # Launch local vLLM server
    ├── build_index.py          # Build FAISS index (required for RAG/MAS-C/TRACE)
    ├── requirements.txt
    ├── .env.example            # Environment variables template
    │
    ├── src/
    │   ├── config.py           # MODEL_REGISTRY — available models and parameters
    │   ├── llm_factory.py      # Async LLM client (OpenAI-compatible API)
    │   ├── load_dataset.py     # Load EmpatheticDialogues from HuggingFace
    │   ├── metrics.py          # BLEU, ROUGE, BERTScore, Accuracy, AvgLen
    │   └── fixed_few_shot.py   # Few-shot example selection from the train split
    │
    ├── architectures/
    │   ├── empathy_zero_shot.py      # 1 LLM call, no examples
    │   ├── empathy_zero_shot_short.py# 1 LLM call, response ≤ 15 words
    │   ├── empathy_few_shot.py       # 1 LLM call + 5 fixed examples
    │   ├── empathy_ektc.py           # emotion → knowledge → reflection → response (4 calls)
    │   ├── empathy_chain.py          # emotion → cause → strategy → response (4 calls)
    │   ├── empathy_debate.py         # 3 debater agents + arbiter (5 calls)
    │   ├── empathy_loop.py           # iterative refinement (5–11 calls)
    │   ├── empathy_rag.py            # RAG + FAISS (3 calls)
    │   ├── empathy_mas_c.py          # multi-agent selector: planner + 2×generator (5 calls)
    │   ├── empathy_trace.py          # TRACE (Liu et al., 2025): 4 calls + RAG
    │   ├── empathy_insideout.py      # InsideOut (ACL 2024): 4 emotion agents + aggregator (6 calls)
    │   └── chain_of_empathy.py       # CBT-style reasoning before response (1 call)
    │
    ├── analysis/
    │   ├── compare_results.py        # Summary table across all outputs/
    │   ├── recompute_metrics.py      # Recompute metrics without rerunning inference
    │   ├── plot_architectures.py     # Bar charts per architecture
    │   └── experiment_k_rag.py       # Experiment: effect of RAG k on quality
    │
    ├── outputs/                      # JSON experiment results (auto-generated)
    └── retriever_cache/              # FAISS index (generated via build_index.py)
```

---

## Installation

```bash
cd empathy_multiagent
pip install -r requirements.txt
```

For local inference with vLLM (requires GPU):

```bash
pip install vllm
```

---

## Environment Setup

```bash
cp empathy_multiagent/.env.example empathy_multiagent/.env
```

`.env.example` already contains `LOCAL_API_KEY=EMPTY` — no changes needed for local vLLM.

---

## Running Experiments

### Using the Notebook

Open `experiment.ipynb` — it contains an automated pipeline for running all architectures on each model.

### Manual Run

```bash
cd empathy_multiagent
python run_experiment.py --model <MODEL> --arch <ARCH> [--limit N] [--no-bertscore]
```

| Argument | Description |
|---|---|
| `--model` | Model key from `src/config.py` |
| `--arch` | Architecture name (see list below) |
| `--limit N` | Number of dialogues to process (default: all ~2,547) |
| `--no-bertscore` | Skip BERTScore computation (faster, no GPU required) |

**Example:**

```bash
python run_experiment.py --model mistral-small-3.2 --arch empathy_chain --limit 50
```

Results are saved to `outputs/<model>_<arch>.json`. If the run is interrupted, it resumes automatically — completed examples are not reprocessed.

---

## Architectures with RAG

Before running `empathy_rag`, `empathy_mas_c`, or `empathy_trace` for the first time, build the FAISS index (~2 min):

```bash
cd empathy_multiagent
python build_index.py
```

---

## Local Inference (vLLM)

```bash
# Terminal 1: launch the server
cd empathy_multiagent
python serve_local.py --model mistral-small-3.2

# Terminal 2: run the experiment
python run_experiment.py --model mistral-small-3.2 --arch empathy_chain --limit 50
```

Available `serve_local.py` flags: `--port`, `--gpu-memory-utilization`, `--tensor-parallel-size`, `--max-model-len`, `--dtype`.

---

## Available Models

| Key | Model | Size |
|---|---|---|
| `mistral-small-3.2` | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | 24B |
| `qwen3-32b-local` | Qwen/Qwen3-32B | 32B |
| `llama-3.1-8b-local` | meta-llama/Llama-3.1-8B-Instruct | 8B |

To add a new model, add an entry to `MODEL_REGISTRY` in `src/config.py`.

---

## Analysing Results

```bash
cd empathy_multiagent

# Summary table and CSV across all outputs/
python analysis/compare_results.py

# Recompute metrics without rerunning inference
python analysis/recompute_metrics.py [--no-bertscore]
```