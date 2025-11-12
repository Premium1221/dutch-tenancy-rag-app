# Dutch Tenancy RAG App

Retrieval-Augmented Generation (RAG) pipeline focused on Dutch tenancy law. The CLI lets you crawl housing portals, ingest PDF/TXT/Markdown sources, build a ChromaDB vector store with multilingual E5 embeddings, and query the corpus through Groq or OpenAI models. Extra tools help compare chunking strategies and run retrieval quality checks.

## Features
- CLI entry point (`main.py`) for ingest, ask, crawl, chunk preview, and retrieval evaluation workflows
- Configurable chunking strategies (recursive, token, sentence, markdown) with specialized handling for statutory text
- Persistent ChromaDB store powered by multilingual E5 embeddings for Dutch/English cross-lingual questions
- Pluggable LLM layer with Groq Mixtral or OpenAI GPT-4o-mini via LangChain wrappers
- Lightweight crawler that mirrors public pages (plus optional PDFs) into `data/` for later ingestion
- Eval helpers (`rag_app/eval.py`, `eval/compare_chunking.py`) to measure hit@k and MRR across chunking strategies

## Project structure
```
dutch-tenancy-rag-app/
├─ rag_app/
│  ├─ config.py          # env + runtime settings
│  ├─ ingestion.py       # loaders + chunkers
│  ├─ embeddings.py      # HuggingFace E5 wrapper
│  ├─ store.py           # ChromaDB persistence + search
│  ├─ llm.py             # Groq/OpenAI chat backends
│  ├─ rag.py             # end-to-end RAG pipeline
│  ├─ crawl.py           # domain-restricted crawler
│  └─ eval.py            # retrieval metrics
├─ eval/compare_chunking.py
├─ main.py               # CLI
├─ requirements.txt
├─ .env.example          # template for secrets
└─ data/, index/         # ignored locally (see below)
```

## Prerequisites
- Python 3.10+ (tested on 3.12)
- Git (for versioning and GitHub)
- Groq/OpenAI API keys (depending on provider)

## Local setup
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

copy .env.example .env  # or `cp` on macOS/Linux
# Fill in GROQ_API_KEY / OPENAI_API_KEY and adjust overrides as needed
```


## Data & index locations
- Place source documents under `data/` (e.g., `data/laws/boek7/...pdf`, `data/government_portal/*.md`). This folder is git-ignored so you can keep proprietary material locally.
- ChromaDB persists under `index/` (also ignored). Delete this folder (or run `--rebuild`) when you need a clean rebuild.

## Using the CLI
Ingest documents (optionally rebuilding the vector store first):
```bash
python main.py --ingest data/laws --rebuild
```

Ask questions once an index exists:
```bash
python main.py --ask "Wat zegt artikel 7:244 BW over de huurprijs?"
```

Crawl a site and save markdown/PDF snapshots into `data/government_portal`:
```bash
python main.py --crawl https://www.rijksoverheid.nl/onderwerpen/huurwoning --depth 1 --include-pdfs
```

Preview chunk statistics for a folder using the current (or overridden) strategy:
```bash
python main.py --chunk-stats data/laws --strategy tokens --size 384 --overlap 64
```

Run retrieval evaluation against a JSON file (`eval/sample_questions.json`) after rebuilding:
```bash
python main.py --eval eval/sample_questions.json --ingest data/laws --topk 6
```

## Comparing chunking strategies
The helper script rebuilds the index for several chunkers and summarizes chunk counts, build time, and retrieval metrics:
```bash
python eval/compare_chunking.py --eval eval/sample_questions.json --data data/laws --topk 6 --out eval/results.json
```

## Development tips
- `.gitignore` already excludes virtual environments, secrets, generated data, and editor folders. Add more entries if you introduce new build artifacts.
- Keep notebooks (`document.ipynb`) out of long-running scripts; you can convert key steps into CLI flags as shown in `main.py`.
- Before pushing, run a quick ingestion/ask cycle to ensure new dependencies or config changes work end-to-end.

## Preparing for GitHub
1. Verify `git status` is clean (only files you intend to commit should appear).
2. Remove or sanitize any accidental secrets from tracked history (`.env` must remain untracked).
3. Commit with a clear message such as `git commit -am "Add README and project scaffolding"`.
4. Add your GitHub remote (only once): `git remote add origin https://github.com/<user>/dutch-tenancy-rag-app.git`.
5. Push the branch: `git push -u origin main`.

Happy building!
