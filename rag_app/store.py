# rag_app/store.py
from pathlib import Path
from typing import Iterable, List
import chromadb
from langchain_core.documents import Document
from .config import AppConfig
from .embeddings import EmbeddingModel


class VectorStore:
    """ChromaDB-backed vector store with persistent storage and similarity search.

    Mirrors the notebook's approach: use Sentence-Transformers to generate embeddings
    and store/query them via ChromaDB's PersistentClient.
    """

    def __init__(self, cfg: AppConfig, embed: EmbeddingModel):
        self.cfg = cfg
        self.embed = embed
        self._client: chromadb.PersistentClient | None = None
        self._coll = None

    @property
    def path(self) -> Path:
        return self.cfg.index_dir

    @property
    def collection_name(self) -> str:
        return self.cfg.chroma_collection_name

    def _connect(self) -> None:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.path))
        if self._coll is None:
            self._coll = self._client.get_or_create_collection(self.collection_name)

    # --- E5 support helpers ---
    def _use_e5_prefix(self) -> bool:
        try:
            name = (self.embed.cfg.embedding_model_name or "").lower()
        except Exception:
            name = ""
        return "e5" in name  # applies to intfloat/multilingual-e5-* models

    def build(self, docs: Iterable[Document]) -> int:
        # Always rebuild the collection (matches FAISS overwrite behavior)
        docs = list(docs)
        # Connect and reset collection
        self._connect()
        assert self._client is not None
        # Drop existing collection if present to avoid duplicates
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._coll = self._client.get_or_create_collection(self.collection_name)

        # Prepare payloads
        texts = [d.page_content for d in docs]
        metadatas = [dict(d.metadata or {}) for d in docs]
        ids = [f"doc-{i}" for i in range(len(docs))]

        # Compute embeddings via Sentence-Transformers wrapper
        if self._use_e5_prefix():
            to_embed = [f"passage: {t}" for t in texts]
        else:
            to_embed = texts
        vectors = self.embed.inner.embed_documents(to_embed)

        # Persist to ChromaDB
        self._coll.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=vectors)
        return len(docs)

    def load(self) -> None:
        # For Chroma, loading is connecting to the persistent client and collection
        self._connect()

    def ensure_loaded(self) -> None:
        if self._coll is None:
            self.load()

    def similarity_search(self, query: str, k: int, where: dict | None = None) -> List[Document]:
        self.ensure_loaded()
        # Embed the query and run vector search
        if self._use_e5_prefix():
            q = f"query: {query}"
        else:
            q = query
        qv = self.embed.inner.embed_query(q)
        query_kwargs = dict(
            query_embeddings=[qv],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        # Only pass 'where' if a filter was provided; some Chroma versions error on empty dict
        if where is not None:
            query_kwargs["where"] = where
        res = self._coll.query(**query_kwargs)

        docs_out: List[Document] = []
        if res and res.get("documents"):
            docs_list = res["documents"][0]
            metas_list = res.get("metadatas", [[{}]])[0]
            for txt, meta in zip(docs_list, metas_list):
                docs_out.append(Document(page_content=txt, metadata=meta))
        return docs_out
