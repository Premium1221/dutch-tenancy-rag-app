# main.py
import argparse
from pathlib import Path
from rag_app.config import load_config
from rag_app.rag import RAGPipeline
from rag_app.crawl import CrawlOptions, crawl_and_save
from rag_app.eval import run_retrieval_eval


def cli():
    p = argparse.ArgumentParser(description="RAG app: index PDFs/TXT/MD, crawl sites, and answer questions")
    p.add_argument("--ingest", metavar="DIR", help="Folder with PDFs/TXT/MD to (re)index", default=None)
    p.add_argument("--rebuild", action="store_true", help="Force rebuild the Chroma index")
    p.add_argument("--ask", metavar="QUESTION", help="Ask a question against the index", nargs="+", default=None)
    # Crawl options
    p.add_argument("--crawl", metavar="URL", help="Crawl a URL (same domain) and save pages into data/...", default=None)
    p.add_argument("--depth", type=int, default=1, help="Crawl depth (links from the start page)")
    p.add_argument("--max-pages", type=int, default=200, help="Max pages to fetch during crawl")
    p.add_argument("--out", metavar="DIR", default="data/government_portal", help="Output folder root for crawled pages")
    p.add_argument("--prefix", action="append", default=None, help="Allowed path prefix to keep crawl scoped (can be given multiple times)")
    p.add_argument("--include-pdfs", action="store_true", help="Also download linked PDFs")
    # Chunking preview options
    p.add_argument("--chunk-stats", metavar="DIR", help="Preview chunk counts for a folder using current (or overridden) strategy", default=None)
    p.add_argument("--strategy", choices=["recursive", "tokens", "sentences", "markdown"], default=None, help="Override chunking strategy for this run")
    p.add_argument("--size", type=int, default=None, help="Override chunk size (chars or tokens)")
    p.add_argument("--overlap", type=int, default=None, help="Override chunk overlap (chars or tokens)")
    # Retrieval evaluation
    p.add_argument("--eval", metavar="FILE", default=None, help="Run retrieval eval with a JSON file of questions + expected patterns. Rebuilds index with current strategy.")
    p.add_argument("--topk", type=int, default=None, help="Top-k to use for retrieval in eval (defaults to config.k)")
    args = p.parse_args()

    cfg = load_config()
    # Apply runtime overrides for chunking if provided
    if args.strategy:
        cfg.chunk_strategy = args.strategy
    if args.size:
        if cfg.chunk_strategy == "tokens":
            cfg.token_chunk_size = args.size
        else:
            cfg.chunk_size = args.size
    if args.overlap:
        if cfg.chunk_strategy == "tokens":
            cfg.token_overlap = args.overlap
        else:
            cfg.chunk_overlap = args.overlap
    rag = RAGPipeline(cfg)

    if args.crawl:
        out_paths = crawl_and_save(CrawlOptions(
            base_url=args.crawl,
            depth=int(args.depth),
            max_pages=int(args.max_pages),
            out_dir=Path(args.out),
            allowed_path_prefixes=tuple(args.prefix) if args.prefix else None,
            include_pdfs=bool(args.include_pdfs),
        ))
        print(f"Saved {len(out_paths)} pages under {args.out}/")

    if args.ingest:
        if args.rebuild and cfg.index_dir.exists():
            # Robustly remove entire index dir (Windows-safe)
            import shutil, os, stat
            def _on_rm_error(func, path, exc_info):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass
            shutil.rmtree(cfg.index_dir, ignore_errors=False, onerror=_on_rm_error)
            cfg.index_dir.mkdir(parents=True, exist_ok=True)
        count = rag.ingest_and_index(args.ingest)
        print(f"Indexed {count} chunks from {args.ingest} -> {cfg.index_dir}")

    if args.ask:
        question = " ".join(args.ask) if isinstance(args.ask, list) else str(args.ask)
        ans, docs = rag.ask(question)
        print("\n=== Answer ===\n")
        print(ans)
        print("\n=== Retrieved sources ===")
        for d in docs:
            meta = d.metadata or {}
            print({k: meta.get(k) for k in ("source", "page")})

    if args.eval:
        # Ensure a clean rebuild so strategies are comparable
        if cfg.index_dir.exists():
            import shutil, os, stat
            def _on_rm_error(func, path, exc_info):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass
            shutil.rmtree(cfg.index_dir, ignore_errors=False, onerror=_on_rm_error)
            cfg.index_dir.mkdir(parents=True, exist_ok=True)
        res = run_retrieval_eval(cfg, args.eval, data_dir=args.ingest or str(cfg.data_dir), rebuild=True, top_k=args.topk)
        print("\n=== Retrieval Eval Summary ===")
        for k, v in res["summary"].items():
            print(f"{k}: {v}")
        print("\nDetails:")
        for row in res["details"]:
            print(row)


    # Chunk preview (does not persist an index)
    if args.chunk_stats:
        docs = rag.ingestor.load_dir(args.chunk_stats)
        stats = rag.ingestor.chunk_stats(docs)
        print(f"Strategy={cfg.chunk_strategy} size={cfg.token_chunk_size if cfg.chunk_strategy=='tokens' else cfg.chunk_size} overlap={cfg.token_overlap if cfg.chunk_strategy=='tokens' else cfg.chunk_overlap}")
        print(f"Files: {len(docs)} chunks: {stats['chunks']} avg_chars: {stats['avg']} p95: {stats['p95']} max: {stats['max']}")
        # Exit early if only previewing
        if not args.ingest and not args.ask:
            return


if __name__ == "__main__":
    cli()
