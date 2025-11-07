# rag_app/llm.py
from .config import AppConfig

# Try to import both; only one will actually be used.
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


class LLM:
    """Handles whichever LLM provider is selected in config."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        if cfg.llm_provider == "groq":
            if ChatGroq is None:
                raise RuntimeError("langchain-groq not installed.")
            if not cfg.groq_api_key:
                raise RuntimeError("Missing GROQ_API_KEY in .env")
            self._llm = ChatGroq(
                model=cfg.groq_model,
                temperature=0.1,
                groq_api_key=cfg.groq_api_key,
            )

        elif cfg.llm_provider == "openai":
            if ChatOpenAI is None:
                raise RuntimeError("langchain-openai not installed.")
            if not cfg.openai_api_key:
                raise RuntimeError("Missing OPENAI_API_KEY in .env")
            self._llm = ChatOpenAI(
                model=cfg.openai_model,
                temperature=0.1,
                openai_api_key=cfg.openai_api_key,
            )

        else:
            raise ValueError(f"Unknown LLM provider: {cfg.llm_provider}")

    def generate(self, prompt: str) -> str:
        """Return the LLM's text output for a given prompt."""
        resp = self._llm.invoke(prompt)
        return getattr(resp, "content", str(resp))
