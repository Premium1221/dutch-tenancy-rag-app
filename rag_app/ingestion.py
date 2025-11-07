# rag_app/ingestion.py
from pathlib import Path
from typing import Iterable, List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
try:
    from langchain_community.document_loaders import PyMuPDFLoader  # better PDF parsing
except ImportError:  # pragma: no cover
    PyMuPDFLoader = None  # fallback handled at runtime
# LangChain splitters moved to a separate package in newer versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError as e:
        raise ImportError(
            "Missing RecursiveCharacterTextSplitter. Install 'langchain-text-splitters' or use a compatible LangChain version."
        ) from e
from langchain_core.documents import Document
import re
from .config import AppConfig

class Ingestor:
    """Loads PDF/TXT files and splits them using one chunking strategy (recursive)."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        # Build splitter based on configured strategy
        self.splitter = self._build_splitter()

    def _build_splitter(self):
        strategy = (self.cfg.chunk_strategy or "recursive").lower()
        # Lazy imports for optional splitters
        if strategy == "tokens":
            try:
                from langchain_text_splitters import SentenceTransformersTokenTextSplitter
                return SentenceTransformersTokenTextSplitter(
                    tokens_per_chunk=self.cfg.token_chunk_size,
                    chunk_overlap=self.cfg.token_overlap,
                    model_name=self.cfg.embedding_model_name,
                )
            except Exception:
                # Fallback to recursive if token splitter unavailable
                strategy = "recursive"
        if strategy == "sentences":
            try:
                from langchain_text_splitters import NLTKTextSplitter
                return NLTKTextSplitter(
                    chunk_size=self.cfg.chunk_size,
                    chunk_overlap=self.cfg.chunk_overlap,
                    separator="\n\n",
                    language="english",  # works reasonably for NL as well
                )
            except Exception:
                strategy = "recursive"
        if strategy == "markdown":
            try:
                from langchain_text_splitters import MarkdownTextSplitter
                return MarkdownTextSplitter(
                    chunk_size=self.cfg.chunk_size,
                    chunk_overlap=self.cfg.chunk_overlap,
                )
            except Exception:
                strategy = "recursive"

        # Default recursive character splitter (paragraph/sentence fallbacks)
        return RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "],
        )

    def load_paths(self, paths: Iterable[Path]) -> List[Document]:
        docs: List[Document] = []
        for p in paths:
            before = len(docs)
            if p.suffix.lower() == ".pdf":
                # Prefer PyMuPDF; fallback to PyPDFLoader per-file
                if PyMuPDFLoader is not None:
                    try:
                        docs.extend(PyMuPDFLoader(str(p)).load())
                        continue
                    except Exception:
                        pass
                docs.extend(PyPDFLoader(str(p)).load())
            elif p.suffix.lower() in {".txt", ".md"}:
                # Use chardet auto-detection if available; otherwise fall back to UTF-8 with safe error handling
                try:
                    import chardet  # type: ignore  # noqa: F401
                    docs.extend(TextLoader(str(p), autodetect_encoding=True).load())
                except Exception:
                    try:
                        docs.extend(
                            TextLoader(
                                str(p),
                                encoding="utf-8",
                                autodetect_encoding=False,
                                errors="ignore",  # type: ignore[arg-type]
                            ).load()
                        )
                    except TypeError:
                        # Older versions may not support 'errors'
                        docs.extend(
                            TextLoader(
                                str(p),
                                encoding="utf-8",
                                autodetect_encoding=False,
                            ).load()
                        )
            # annotate category/source_rel for the newly added docs from this path
            after = len(docs)
            if after > before:
                try:
                    rel = p.relative_to(self.cfg.data_dir)
                except Exception:
                    rel = p
                parts = list(rel.parts)
                category = parts[0] if len(parts) > 1 else "root"
                source_rel = rel.as_posix()
                for i in range(before, after):
                    md = dict(docs[i].metadata or {})
                    md.setdefault("category", category)
                    md.setdefault("source_rel", source_rel)
                    docs[i].metadata = md
        return docs

    def load_dir(self, data_dir: Path | None = None) -> List[Document]:
        base = Path(data_dir) if data_dir else self.cfg.data_dir
        files = [*base.rglob("*.pdf"), *base.rglob("*.txt"), *base.rglob("*.md")]
        return self.load_paths(files)

    def _split_law_document(self, d: Document) -> List[Document]:
        text = d.page_content or ""
        meta = dict(d.metadata or {})
        # Try to infer book number from source path (e.g., 'Boek7')
        src_rel = (meta.get("source_rel") or meta.get("source") or "")
        mbook = re.search(r"Boek(\d+)", src_rel, re.IGNORECASE)
        book = mbook.group(1) if mbook else None

        # Find Artikel headings and slice per article
        # Matches: 'Artikel 244', 'Artikel 244a', 'Artikel 244 b'
        pattern = re.compile(r"(?mi)^(\s*Artikel\s+(\d+[a-z]?))\b.*$")
        idxs: list[tuple[int, int, str]] = []  # (start_index, heading_end_index, article_num)
        for m in pattern.finditer(text):
            full = m.group(1)
            num = m.group(2)
            idxs.append((m.start(), m.end(), num))

        if not idxs:
            return [d]

        docs_out: list[Document] = []
        for i, (s, e, num) in enumerate(idxs):
            s2 = e
            e2 = idxs[i + 1][0] if i + 1 < len(idxs) else len(text)
            chunk_txt = text[s:e2].strip()
            md = dict(meta)
            md["article_num"] = num
            if book:
                md["article"] = f"{book}:{num}"
                md["book"] = book
            else:
                md["article"] = num
            docs_out.append(Document(page_content=chunk_txt, metadata=md))
        return docs_out

    def chunk(self, docs: List[Document]) -> List[Document]:
        # Special handling: split laws by 'Artikel <num>' to enable precise retrieval
        law_docs: list[Document] = []
        other_docs: list[Document] = []
        for d in docs:
            cat = (d.metadata or {}).get("category")
            if cat == "laws":
                law_docs.extend(self._split_law_document(d))
            else:
                other_docs.append(d)

        # Apply the configured splitter to both sets (keeps article boundaries but caps long ones)
        out: list[Document] = []
        if law_docs:
            out.extend(self.splitter.split_documents(law_docs))
        if other_docs:
            out.extend(self.splitter.split_documents(other_docs))
        return out

    # Utility to preview chunk counts without indexing
    def chunk_stats(self, docs: List[Document]) -> dict:
        chunks = self.chunk(docs)
        sizes = [len(c.page_content) for c in chunks]
        total = len(chunks)
        if not sizes:
            return {"chunks": 0, "avg": 0, "p95": 0, "max": 0}
        sizes_sorted = sorted(sizes)
        p95 = sizes_sorted[int(0.95 * (len(sizes_sorted)-1))]
        return {
            "chunks": total,
            "avg": int(sum(sizes)/total),
            "p95": p95,
            "max": max(sizes),
        }
