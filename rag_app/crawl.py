from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set
from urllib.parse import urljoin, urlparse, urldefrag
import urllib.robotparser as robotparser

import requests
from bs4 import BeautifulSoup


DEFAULT_UA = "RAGCrawler/1.0 (+https://example.com)"


@dataclass
class CrawlOptions:
    base_url: str
    depth: int = 1
    max_pages: int = 200
    delay_s: float = 0.5
    out_dir: Path = Path("data/government_portal")
    allowed_path_prefixes: tuple[str, ...] | None = None
    include_pdfs: bool = False


def _normalize_url(url: str) -> str:
    # Remove fragment, keep query
    url, _frag = urldefrag(url)
    # Strip trailing slash (except root)
    if url.endswith("/") and len(url) > len("https://a.b/"):
        url = url[:-1]
    return url


def _allowed(url: str, base_netloc: str, prefixes: tuple[str, ...] | None) -> bool:
    p = urlparse(url)
    if p.netloc and p.netloc.lower() != base_netloc.lower():
        return False
    if prefixes:
        path = p.path or "/"
        return any(path.startswith(pref) for pref in prefixes)
    return True


def _extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    # Remove noise
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n") if main else soup.get_text("\n")
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\x0b\r]+", " ", text)
    text = text.strip()
    return title, text


def _safe_slug(path: str) -> Path:
    # Mirror the path under out_dir; fallback to index.md
    path = path or "/"
    if path.endswith("/"):
        path += "index"
    # Replace unsafe chars
    sanitized = re.sub(r"[^A-Za-z0-9_\-/]", "-", path)
    sanitized = re.sub(r"/{2,}", "/", sanitized)
    if not sanitized.startswith("/"):
        sanitized = "/" + sanitized
    return Path(sanitized.lstrip("/"))


def crawl_and_save(opts: CrawlOptions) -> list[Path]:
    """Crawl starting from base_url within same domain and optional path prefixes.

    Saves pages as markdown (.md) files under opts.out_dir/<domain>/...
    Returns list of written file paths.
    """
    start = _normalize_url(opts.base_url)
    start_p = urlparse(start)
    domain = start_p.netloc
    base = f"{start_p.scheme}://{start_p.netloc}"
    prefixes = opts.allowed_path_prefixes
    if prefixes is None:
        # Default to the start path prefix
        prefixes = (start_p.path or "/",)

    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(base, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        # If robots cannot be fetched, assume allowed
        pass

    session = requests.Session()
    session.headers.update({
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    })

    out_root = opts.out_dir / domain
    out_root.mkdir(parents=True, exist_ok=True)

    visited: Set[str] = set()
    written: list[Path] = []

    frontier: list[tuple[str, int]] = [(start, 0)]

    while frontier and len(visited) < opts.max_pages:
        url, depth = frontier.pop(0)
        url = _normalize_url(url)
        if url in visited:
            continue
        visited.add(url)

        if not _allowed(url, domain, prefixes):
            continue
        if hasattr(rp, "can_fetch") and not rp.can_fetch(DEFAULT_UA, url):
            continue

        try:
            resp = session.get(url, timeout=20)
            if resp.status_code != 200:
                continue
        except Exception:
            continue

        ctype = resp.headers.get("content-type", "").lower()
        if "text/html" in ctype:
            title, text = _extract_main_text(resp.text)
            if not text:
                continue
            rel_path = _safe_slug(urlparse(url).path).with_suffix(".md")
            out_file = out_root / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            body = f"# {title}\n\nSource: {url}\n\n" + text + "\n"
            out_file.write_text(body, encoding="utf-8")
            written.append(out_file)
        elif opts.include_pdfs and "application/pdf" in ctype:
            # Save PDFs as-is so the PDF loader can parse them during ingestion
            upath = urlparse(url).path or "/index.pdf"
            rel_path = _safe_slug(upath)
            # Ensure .pdf extension
            if rel_path.suffix.lower() != ".pdf":
                rel_path = rel_path.with_suffix(".pdf")
            out_file = out_root / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                out_file.write_bytes(resp.content)
                written.append(out_file)
            except Exception:
                pass
        else:
            # Skip other content types
            continue

        # Expand links if depth allows
        if depth < opts.depth:
            try:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    nxt = urljoin(url, a["href"])
                    nxt = _normalize_url(nxt)
                    if nxt not in visited and _allowed(nxt, domain, prefixes):
                        frontier.append((nxt, depth + 1))
            except Exception:
                pass

        time.sleep(opts.delay_s)

    return written
