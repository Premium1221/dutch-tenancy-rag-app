import gradio as gr
from pathlib import Path
from typing import Dict, Tuple, List

from rag_app.config import load_config, AppConfig
from rag_app.rag import RAGPipeline


_PIPELINES: Dict[Tuple[str, str], RAGPipeline] = {}


def get_pipeline(provider: str | None = None) -> RAGPipeline:
    cfg = load_config()
    if provider:
        cfg.llm_provider = provider
    key = (cfg.llm_provider, cfg.embedding_model_name)
    if key not in _PIPELINES:
        _PIPELINES[key] = RAGPipeline(cfg)
    return _PIPELINES[key]


def ui_ask(question: str, top_k: int, provider: str) -> Tuple[str, List[dict]]:
    if not question or not question.strip():
        return "Please enter a question.", []
    try:
        rag = get_pipeline(provider)
        # Override k for this request
        rag.cfg.k = int(top_k)
        answer, docs = rag.ask(question)
        rows = []
        for d in docs:
            meta = d.metadata or {}
            rows.append({"source": meta.get("source"), "page": meta.get("page")})
        return answer, rows
    except Exception as e:
        return f"Error: {e}", []


def ui_rebuild(data_dir: str, provider: str) -> str:
    try:
        rag = get_pipeline(provider)
        path = data_dir.strip() if data_dir else None
        count = rag.ingest_and_index(path or None)
        return f"Indexed {count} chunks from {path or rag.cfg.data_dir} -> {rag.cfg.index_dir}"
    except Exception as e:
        return f"Error during indexing: {e}"


def build_ui() -> gr.Blocks:
    cfg = load_config()
    with gr.Blocks(title="Dutch Tenancy RAG") as demo:
        gr.Markdown("# Dutch Tenancy RAG\nAsk questions grounded in your indexed documents.")

        with gr.Tab("Ask"):
            with gr.Row():
                provider = gr.Dropdown(
                    choices=["openai", "groq"],
                    value=cfg.llm_provider,
                    label="LLM Provider",
                    interactive=True,
                )
                top_k = gr.Slider(1, 10, value=cfg.k, step=1, label="Top-K")
            question = gr.Textbox(label="Question", placeholder="Type your question...", lines=3)
            ask_btn = gr.Button("Ask")
            answer = gr.Markdown()
            sources = gr.Dataframe(headers=["source", "page"], datatype=["str", "number"], wrap=True)

            ask_btn.click(ui_ask, inputs=[question, top_k, provider], outputs=[answer, sources])

        with gr.Tab("Index"):
            gr.Markdown("Rebuild the vector index from a local folder of PDFs/TXT/MD.")
            data_dir = gr.Textbox(label="Data Folder", value=str(cfg.data_dir))
            provider_idx = gr.Dropdown(
                choices=["openai", "groq"], value=cfg.llm_provider, label="LLM Provider"
            )
            rebuild_btn = gr.Button("Rebuild Index")
            status = gr.Textbox(label="Status", interactive=False)

            rebuild_btn.click(ui_rebuild, inputs=[data_dir, provider_idx], outputs=[status])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()

