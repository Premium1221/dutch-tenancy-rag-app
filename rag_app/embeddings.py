"""Embedding model wrapper."""

from .config import AppConfig

# Prefer new package; fall back to community for compatibility
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install 'langchain-huggingface' or 'langchain-community' to use HuggingFaceEmbeddings."
        ) from e


class EmbeddingModel:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._model = HuggingFaceEmbeddings(model_name=cfg.embedding_model_name)

    @property
    def inner(self):
        return self._model
