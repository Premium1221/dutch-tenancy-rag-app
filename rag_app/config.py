from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional          
from dotenv import load_dotenv

@dataclass
class AppConfig:
    # Paths
    data_dir: Path = Path("data")
    index_dir: Path = Path("index")

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # Chunking strategy: 'recursive' | 'tokens' | 'sentences' | 'markdown'
    chunk_strategy: str = "recursive"
    # Token-based splitter (only used when strategy == 'tokens')
    token_chunk_size: int = 350
    token_overlap: int = 60

    # Embeddings
    # Use a stronger multilingual model (E5) for cross-lingual retrieval (EN ↔ NL ↔ BG)
    embedding_model_name: str = "intfloat/multilingual-e5-base"

    # LLM selection: "groq" or "openai"
    llm_provider: str = "groq"

    # Groq settings
    groq_model: str = "mixtral-8x7b-32768"
    groq_api_key: str | None = None

    # OpenAI settings
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    # Retrieval
    k: int = 4

    # ChromaDB
    chroma_collection_name: str = "rag_collection"


def load_config() -> AppConfig:
    """Load environment variables and initialize the AppConfig."""
    load_dotenv(override=False)
    cfg = AppConfig()

    # Allow environment overrides
    cfg.data_dir = Path(os.getenv("RAG_DATA_DIR", cfg.data_dir))
    cfg.index_dir = Path(os.getenv("RAG_INDEX_DIR", cfg.index_dir))
    cfg.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", cfg.chunk_size))
    cfg.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", cfg.chunk_overlap))
    cfg.chunk_strategy = os.getenv("RAG_CHUNK_STRATEGY", cfg.chunk_strategy)
    cfg.token_chunk_size = int(os.getenv("RAG_TOKEN_CHUNK", cfg.token_chunk_size))
    cfg.token_overlap = int(os.getenv("RAG_TOKEN_OVERLAP", cfg.token_overlap))
    cfg.embedding_model_name = os.getenv("RAG_EMBED_MODEL", cfg.embedding_model_name)

    # LLM provider + models/keys
    cfg.llm_provider = os.getenv("RAG_LLM_PROVIDER", cfg.llm_provider)

    cfg.groq_model = os.getenv("GROQ_MODEL", cfg.groq_model)
    cfg.groq_api_key = os.getenv("GROQ_API_KEY")

    cfg.openai_model = os.getenv("OPENAI_MODEL", cfg.openai_model)
    cfg.openai_api_key = os.getenv("OPENAI_API_KEY")

    cfg.k = int(os.getenv("RAG_TOP_K", cfg.k))

    # Chroma
    cfg.chroma_collection_name = os.getenv("CHROMA_COLLECTION", cfg.chroma_collection_name)

    # Ensure dirs exist
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.index_dir.mkdir(parents=True, exist_ok=True)

    return cfg
