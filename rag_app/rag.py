# rag_app/rag.py
from typing import Tuple
from textwrap import dedent
from langchain_core.documents import Document

from .config import AppConfig
from .ingestion import Ingestor
from .embeddings import EmbeddingModel
from .store import VectorStore
from .llm import LLM


SYSTEM_PROMPT = dedent("""
You are a careful assistant that answers with grounded, concise explanations.
Use only the provided context. If something is missing, say what's missing.
Answer in the language of the question (e.g., English or Dutch).
At the end, include short source attributions like [source: <file>, p.<page>].
""").strip()


def _format_sources(docs: list[Document]) -> str:
    lines = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page")
        if page is not None:
            lines.append(f"- {src} p.{int(page) + 1}")
        else:
            lines.append(f"- {src}")
    return "\n".join(lines)


def _build_prompt(question: str, docs: list[Document]) -> str:
    context = "\n\n".join(d.page_content for d in docs)
    sources = _format_sources(docs)
    return dedent(f"""
        {SYSTEM_PROMPT}

        Context:
        {context}

        Question: {question}

        Answer using only the context above. Then list the sources as bullets.
        Sources:
        {sources}
    """).strip()


class RAGPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.ingestor = Ingestor(cfg)
        self.embed = EmbeddingModel(cfg)
        self.store = VectorStore(cfg, self.embed)
        self.llm = LLM(cfg)

    # --- indexing ---
    def ingest_and_index(self, data_dir: str | None = None) -> int:
        docs = self.ingestor.load_dir(data_dir)
        chunks = self.ingestor.chunk(docs)
        count = self.store.build(chunks)
        return count

    # --- QA ---
    def ask(self, question: str) -> Tuple[str, list[Document]]:
        # Heuristic: if the question looks like a statutory citation (e.g., 7:244 BW),
        # prefer retrieving from the laws category and then blend with general results.
        import re

        def _is_law_query(q: str) -> bool:
            ql = q.lower()
            has_num = re.search(r"\b7:\d{1,3}\b", ql) is not None
            mentions_bw = any(w in ql for w in [" bw", "burgerlijk", "civil code", "boek 7", "book 7"]) or "bw" in ql
            mentions_art = any(w in ql for w in ["art.", "artikel", "article", "чл.", "член"])  # bg/nl/en
            return has_num or (mentions_bw and mentions_art)

        def _extract_article(q: str) -> str | None:
            # Accept forms like '7:244', 'artikel 244', 'чл. 244' and map to '7:244' if book 7 is implied
            ql = q.lower()
            m = re.search(r"\b(7:\d{1,3}[a-z]?)\b", ql)
            if m:
                return m.group(1)
            m2 = re.search(r"\b(?:art\.|artikel|article|чл\.|член)\s*(\d{1,3}[a-z]?)\b", ql)
            if m2:
                return f"7:{m2.group(1)}"  # assume Book 7 for tenancy context
            return None

        k = self.cfg.k
        article_id = _extract_article(question)
        if _is_law_query(question):
            law_k = max(2, k // 2)
            where = {"category": "laws"}
            if article_id:
                where = {"article": article_id}
            law_hits = self.store.similarity_search(question, k=law_k, where=where)
            gen_hits = self.store.similarity_search(question, k=k)
            # Merge, keeping order and uniqueness by (source,page,text)
            seen = set()
            merged: list[Document] = []
            def _add(seq: list[Document]):
                for d in seq:
                    m = d.metadata or {}
                    key = (m.get("source"), m.get("page"), d.page_content[:64])
                    if key not in seen:
                        seen.add(key)
                        merged.append(d)
                        if len(merged) >= k:
                            return
            _add(law_hits)
            if len(merged) < k:
                _add(gen_hits)
            retrieved = merged
        else:
            retrieved = self.store.similarity_search(question, k=k)
        prompt = _build_prompt(question, retrieved)
        answer = self.llm.generate(prompt)
        return answer, retrieved
