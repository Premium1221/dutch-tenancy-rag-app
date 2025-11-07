from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path when invoked as a script (eval/compare_chunking.py)
import sys as _sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from rag_app.config import load_config, AppConfig
from rag_app.ingestion import Ingestor
from rag_app.rag import RAGPipeline
from rag_app.eval import run_retrieval_eval


def _set_strategy(cfg: AppConfig, strategy: str, size: int, overlap: int) -> None:
    cfg.chunk_strategy = strategy
    if strategy == "tokens":
        cfg.token_chunk_size = size
        cfg.token_overlap = overlap
    else:
        cfg.chunk_size = size
        cfg.chunk_overlap = overlap


def compare(eval_path: str, data_dir: str, topk: int | None) -> Dict[str, Any]:
    cfg = load_config()

    runs = [
        ("recursive", 1000, 150),
        ("tokens", 384, 64),
        ("sentences", 1200, 150),
        ("markdown", 1600, 200),
    ]

    out: List[Dict[str, Any]] = []
    for strategy, size, overlap in runs:
        # Configure
        cfg = load_config()
        _set_strategy(cfg, strategy, size, overlap)

        # Chunk preview (count + stats)
        ing = Ingestor(cfg)
        docs = ing.load_dir(data_dir)
        stats = ing.chunk_stats(docs)
        chunk_count = stats.get("chunks", 0)

        # Build index (measure time)
        rag = RAGPipeline(cfg)
        t0 = time.time()
        built = rag.ingest_and_index(data_dir)
        build_time = time.time() - t0

        # Retrieval eval (no rebuild here, we just built it)
        res = run_retrieval_eval(cfg, eval_path, data_dir=data_dir, rebuild=False, top_k=topk)
        summary = res["summary"]

        out.append({
            "strategy": strategy,
            "size": size,
            "overlap": overlap,
            "chunks": chunk_count,
            "build_time_s": round(build_time, 2),
            **summary,
        })

    return {"results": out}


def main():
    p = argparse.ArgumentParser(description="Compare chunking strategies on a fixed eval set")
    p.add_argument("--eval", required=True, help="Path to eval JSON (questions + must patterns)")
    p.add_argument("--data", default="data", help="Data directory to index")
    p.add_argument("--topk", type=int, default=None, help="Top-k for retrieval")
    p.add_argument("--out", default=None, help="Optional path to save JSON results")
    args = p.parse_args()

    res = compare(args.eval, args.data, args.topk)
    rows = res["results"]

    # Pretty print summary
    print("\nstrategy | size/overlap | chunks | build_s | hit@1 | hit@k | mrr | avg_lat_s")
    for r in rows:
        print(
            f"{r['strategy']:9} | {r['size']}/{r['overlap']:7} | {r['chunks']:6} | {r['build_time_s']:7} | "
            f"{r['hit@1']:5} | {r['hit@k']:5} | {r['mrr']:4} | {r['avg_latency_s']:9}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
