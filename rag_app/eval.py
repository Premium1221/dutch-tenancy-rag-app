from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .config import AppConfig
from .rag import RAGPipeline


@dataclass
class EvalItem:
    q: str
    must: List[str]
    k: Optional[int] = None


def _match(doc: Document, patterns: List[str]) -> bool:
    meta = doc.metadata or {}
    src = (meta.get("source_rel") or meta.get("source") or "").lower()
    text = (doc.page_content or "").lower()
    for p in patterns:
        p = p.lower()
        if p in src or p in text:
            return True
    return False


def _reciprocal_rank(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (rank + 1)


def run_retrieval_eval(
    cfg: AppConfig,
    eval_path: str | Path,
    data_dir: Optional[str | Path] = None,
    rebuild: bool = True,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Build index (optionally) then run retrieval hit@k and MRR on questions.

    eval_path JSON format:
        [
          {"q": "question text", "must": ["substring1", "substring2"], "k": 4},
          ...
        ]
    The 'must' array is checked against source_rel/source and page_content.
    """
    rag = RAGPipeline(cfg)

    # Ingest and index for a clean build with the current chunking settings
    if rebuild:
        rag.ingest_and_index(str(data_dir) if data_dir else None)

    items_data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    items: List[EvalItem] = [EvalItem(**it) for it in items_data]

    k_used = top_k or cfg.k
    results: List[Dict[str, Any]] = []

    t0 = time.time()
    for it in items:
        kq = it.k or k_used
        t_q0 = time.time()
        docs = rag.store.similarity_search(it.q, k=kq)
        latency = time.time() - t_q0
        rank: Optional[int] = None
        for i, d in enumerate(docs):
            if _match(d, it.must):
                rank = i
                break
        results.append({
            "q": it.q,
            "k": kq,
            "latency_s": round(latency, 3),
            "hit": rank is not None,
            "rank": rank,
            "rr": round(_reciprocal_rank(rank), 4),
            "first_source": (docs[0].metadata or {}).get("source_rel") if docs else None,
        })

    total = len(results)
    hit1 = sum(1 for r in results if r["rank"] == 0)
    hitk = sum(1 for r in results if r["hit"])  # within provided k
    mrr = sum(r["rr"] for r in results) / total if total else 0.0
    t_all = time.time() - t0

    summary = {
        "items": total,
        "hit@1": round(hit1 / total, 3) if total else 0.0,
        "hit@k": round(hitk / total, 3) if total else 0.0,
        "mrr": round(mrr, 3),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / total, 3) if total else 0.0,
        "wall_time_s": round(t_all, 2),
    }
    return {"summary": summary, "details": results}

