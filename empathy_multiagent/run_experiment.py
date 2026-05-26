# run_experiment.py

import os
os.environ["OMP_NUM_THREADS"] = "1"

import asyncio
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.llm_factory import LLMFactory
from src.load_dataset import prepare_examples
from architectures.empathy_zero_shot import empathy_zero_shot
from architectures.empathy_few_shot import empathy_few_shot
from architectures.empathy_ektc import empathy_ektc
from architectures.empathy_chain import empathy_chain
from architectures.empathy_debate import empathy_debate
from architectures.empathy_loop import empathy_loop
from architectures.empathy_rag import EmpathyRetriever, empathy_rag
from architectures.empathy_mas_c import empathy_mas_c
from architectures.empathy_trace import empathy_trace
from architectures.empathy_insideout import empathy_insideout
from architectures.chain_of_empathy import chain_of_empathy
from architectures.empathy_zero_shot_short import empathy_zero_shot_short
from src.metrics import compute_all_metrics, compute_rag_metrics
import numpy as np

load_dotenv()

# --- Retriever (строится один раз, кэшируется на диск) ---

_retriever = None

def get_retriever(cache_dir: str = "retriever_cache") -> EmpathyRetriever:
    global _retriever
    if _retriever is not None:
        return _retriever

    cache_path = Path(cache_dir)
    if (cache_path / "examples.pkl").exists():
        print("Loading cached retriever...")
        _retriever = EmpathyRetriever.load(str(cache_path))
    else:
        print("Building retriever from ED train set (~2 min on first run)...")
        train_examples = prepare_examples("train")
        _retriever = EmpathyRetriever(train_examples)
        _retriever.save(str(cache_path))

    return _retriever


def compute_mas_c_metrics(results: list) -> dict:
    """Специфичные метрики для EmpathyMAS-C v2."""
    comfort_wins, explore_wins = 0, 0
    checker_empathy, checker_relevance, checker_naturalness = [], [], []

    for r in results:
        interm = r.get("intermediate", {})

        sel_gen = interm.get("selected_generator", "")
        if sel_gen == "comfort":
            comfort_wins += 1
        elif sel_gen == "explore":
            explore_wins += 1

        checker = interm.get("checker", {})
        if isinstance(checker, dict) and "empathy" in checker:
            checker_empathy.append(checker.get("empathy", 0))
            checker_relevance.append(checker.get("relevance", 0))
            checker_naturalness.append(checker.get("naturalness", 0))

    total_sel = (comfort_wins + explore_wins) or 1
    metrics = {}
    metrics["Comfort win rate (%)"] = round(comfort_wins / total_sel * 100, 1)
    metrics["Explore win rate (%)"] = round(explore_wins / total_sel * 100, 1)
    if checker_empathy:
        metrics["Avg checker empathy"] = round(float(np.mean(checker_empathy)), 2)
        metrics["Avg checker relevance"] = round(float(np.mean(checker_relevance)), 2)
        metrics["Avg checker naturalness"] = round(float(np.mean(checker_naturalness)), 2)
    return metrics


async def empathy_rag_wrapper(dialogue_context: str, llm) -> dict:
    return await empathy_rag(dialogue_context, llm, get_retriever(), top_k=2)


async def empathy_mas_c_wrapper(dialogue_context: str, llm) -> dict:
    return await empathy_mas_c(dialogue_context, llm, get_retriever())


async def empathy_trace_wrapper(dialogue_context: str, llm) -> dict:
    return await empathy_trace(dialogue_context, llm, get_retriever())


ARCHITECTURES = {
    "empathy_zero_shot": empathy_zero_shot,
    "empathy_few_shot": empathy_few_shot,
    "empathy_ektc": empathy_ektc,
    "empathy_chain": empathy_chain,
    "empathy_debate": empathy_debate,
    "empathy_loop": empathy_loop,
    "empathy_rag": empathy_rag_wrapper,
    "empathy_mas_c": empathy_mas_c_wrapper,
    "empathy_trace": empathy_trace_wrapper,
    "empathy_insideout": empathy_insideout,
    "chain_of_empathy": chain_of_empathy,
    "empathy_zero_shot_short": empathy_zero_shot_short,
}




async def run_single(example: dict, arch_fn, llm: LLMFactory):
    """Запуск одного примера."""
    start = time.time()
    result = await arch_fn(example["context"], llm)
    elapsed = (time.time() - start) * 1000

    return {
        "conv_id": example["conv_id"],
        "emotion": example["emotion"],
        "gold_response": example["gold_response"],
        "generated_response": result["response"],
        "llm_calls": result.get("llm_calls", 1),
        "intermediate": {k: v for k, v in result.items() if k not in ("response", "llm_calls")},
        "latency_ms": round(elapsed),
    }


async def run_experiment(
    model_key: str,
    arch_name: str,
    limit: int = None,
    skip_bertscore: bool = False,
):
    """Полный эксперимент: модель × архитектура."""
    llm = LLMFactory(model_key)
    arch_fn = ARCHITECTURES[arch_name]
    examples = prepare_examples("test", limit)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{model_key}_{arch_name}.json"

    # Загружаем уже обработанные примеры (resume после обрыва)
    results = []
    done_ids: set = set()
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("partial"):
                all_prev = existing.get("results", [])
                # Берём только успешные — ошибки будут ретраиться
                results = [r for r in all_prev if not r.get("error")]
                done_ids = {r["conv_id"] for r in results}
                errors_prev = len(all_prev) - len(results)
                print(f"Resuming: {len(done_ids)} done, {errors_prev} errors will be retried, from {out_path.name}")
        except Exception:
            pass

    print(f"\n{'=' * 60}")
    print(f"Model:        {llm.info}")
    print(f"Architecture: {arch_name}")
    print(f"Examples:     {len(examples)}")
    if skip_bertscore:
        print("BERTScore:    SKIPPED (--no-bertscore flag)")
    print(f"{'=' * 60}\n")

    for ex in tqdm(examples, desc=f"{model_key}/{arch_name}"):
        if ex["conv_id"] in done_ids:
            continue
        try:
            r = await run_single(ex, arch_fn, llm)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {ex['conv_id']}: {e}")
            results.append({
                "conv_id": ex["conv_id"],
                "emotion": ex["emotion"],
                "gold_response": ex["gold_response"],
                "generated_response": "",
                "error": str(e),
                "latency_ms": 0,
            })
        # Сохраняем после каждого примера
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_key, "architecture": arch_name,
                       "partial": True, "results": results}, f, ensure_ascii=False)

    # Метрики
    hypotheses = [r["generated_response"] for r in results if r.get("generated_response")]
    references = [r["gold_response"] for r in results if r.get("generated_response")]

    if not hypotheses:
        errors = sum(1 for r in results if "error" in r)
        print(f"\nERROR: все {errors}/{len(results)} примеров упали с ошибкой. Метрики не считаются.")
        print("Скорее всего превышен rate limit. Попробуй запустить снова или уменьши --limit.")
        return {"model": model_key, "architecture": arch_name, "errors": errors, "metrics": {}}

    predicted_emotions = []
    gold_emotions = []
    for r in results:
        intermediate = r.get("intermediate", {})
        # empathy_chain / empathy_loop / empathy_debate: emotion в intermediate
        em = intermediate.get("emotion", {})
        if isinstance(em, dict) and "emotion" in em:
            predicted_emotions.append(em["emotion"])
            gold_emotions.append(r["emotion"])
        # empathy_mas_c v2: emotion на верхнем уровне intermediate
        elif isinstance(intermediate.get("emotion"), dict) and "emotion" in intermediate.get("emotion", {}):
            predicted_emotions.append(intermediate["emotion"]["emotion"])
            gold_emotions.append(r["emotion"])

    metrics = compute_all_metrics(
        hypotheses,
        references,
        predicted_emotions if predicted_emotions else None,
        gold_emotions if gold_emotions else None,
        results,
        skip_bertscore=skip_bertscore,
    )

    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
    metrics["Avg latency ms"] = round(sum(latencies) / len(latencies)) if latencies else 0
    metrics["Errors"] = sum(1 for r in results if "error" in r)

    if arch_name == "empathy_rag":
        metrics.update(compute_rag_metrics(results))

    if arch_name == "empathy_mas_c":
        metrics.update(compute_rag_metrics(results))
        metrics.update(compute_mas_c_metrics(results))

    output = {
        "model": model_key,
        "model_info": llm.info,
        "architecture": arch_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_examples": len(examples),
        "metrics": metrics,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to: {out_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run empathetic dialogue experiment on EmpatheticDialogues test set."
    )
    parser.add_argument(
        "--model",
        default="mistral-small-3.2",
        help=(
            "Model key from config.py MODEL_REGISTRY. "
            "Available: mistral-small-3.2, qwen3-32b-local, llama-3.1-8b-local"
        ),
    )
    parser.add_argument(
        "--arch",
        default="empathy_zero_shot",
        choices=list(ARCHITECTURES.keys()),
        help="Architecture to run. Choices: " + ", ".join(ARCHITECTURES.keys()),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Number of dialogues to process from the test split. "
            "Omit to run all ~2547. Example: --limit 50 for a quick test."
        ),
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        default=False,
        help="Skip BERTScore computation (useful if no GPU or for fast debug runs).",
    )
    args = parser.parse_args()

    asyncio.run(
        run_experiment(
            model_key=args.model,
            arch_name=args.arch,
            limit=args.limit,
            skip_bertscore=args.no_bertscore,
        )
    )
