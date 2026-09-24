"""Opt-in web research pipeline for minimemory.

Research produces evidence, not automatic truth. The module never writes to the
memory database unless the caller explicitly calls ``store_supported`` and the
configured policy accepts the evidence.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .budget import ResourceBudget


@dataclass
class Source:
    title: str
    url: str
    snippet: str = ""
    text: str = ""
    ok: bool = False
    bytes_read: int = 0

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.casefold()


@dataclass
class Evidence:
    status: str
    score: float
    source_count: int
    domain_count: int
    sources: List[str]
    note: str


class WebResearcher:
    def __init__(self, *, max_results: int = 5, max_page_chars: int = 6000,
                 timeout: float = 15.0, delay: float = 0.5,
                 budget: Optional[ResourceBudget] = None) -> None:
        if max_results < 1 or max_page_chars < 100 or timeout <= 0 or delay < 0:
            raise ValueError("invalid research configuration")
        self.max_results = int(max_results)
        self.max_page_chars = int(max_page_chars)
        self.timeout = float(timeout)
        self.delay = float(delay)
        self.budget = budget

    def search_web(self, query: str) -> List[Dict[str, str]]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError as exc:
                raise ImportError("Install web research support with: pip install 'minimemory[research]'") from exc
        results: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=self.max_results):
                url = item.get("href") or item.get("url")
                if url:
                    results.append({"title": str(item.get("title") or "").strip(), "url": str(url).strip(), "snippet": str(item.get("body") or "").strip()})
        return results

    def fetch_page(self, url: str) -> Source:
        req = Request(url, headers={"User-Agent": "minimemory/0.8 research client", "Accept-Language": "en-US,en;q=0.8"})
        try:
            with urlopen(req, timeout=self.timeout) as response:
                data = response.read(self.max_page_chars * 4)
                content_type = response.headers.get("Content-Type", "").lower()
                if self.budget and not self.budget.allow_network(len(data)):
                    return Source(url=url, title=url, ok=False, text="", bytes_read=0, snippet="network budget exceeded")
                if self.budget:
                    self.budget.consume_network(len(data))
                if "html" not in content_type and "text" not in content_type:
                    return Source(url=url, title=url, ok=False, bytes_read=len(data), snippet="non-text response")
                html = data.decode("utf-8", errors="replace")
                text = self.extract_text(html)
                return Source(title=url, url=url, text=text, ok=bool(text), bytes_read=len(data))
        except Exception as exc:
            return Source(title=url, url=url, ok=False, snippet=str(exc)[:300])

    def extract_text(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError("Install web research support with: pip install 'minimemory[research]'") from exc
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside", "svg", "iframe", "button"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        lines = [line.strip() for line in main.get_text(separator="\n").splitlines()]
        lines = [line for line in lines if len(line) > 40]
        cleaned: List[str] = []
        for line in lines:
            if not cleaned or cleaned[-1] != line:
                cleaned.append(line)
        return "\n".join(cleaned)[: self.max_page_chars]

    def evidence(self, sources: List[Source]) -> Evidence:
        usable = [s for s in sources if s.ok and s.text]
        domains = {s.domain for s in usable if s.domain}
        if not usable:
            return Evidence("unverified", 0.0, 0, 0, [], "No readable sources were retrieved.")
        score = min(1.0, 0.45 + 0.12 * min(len(usable), 4) + 0.08 * min(len(domains), 3))
        if len(domains) >= 2 and len(usable) >= 2:
            status = "supported"
            note = "Multiple readable sources from more than one domain support further review; this is not a guarantee of truth."
        else:
            status = "single-source"
            note = "Only one source/domain was available; treat the result as unverified until checked."
        return Evidence(status, round(score, 4), len(usable), len(domains), [s.url for s in usable], note)

    def research(self, question: str, *, max_results: Optional[int] = None) -> Dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        old_max = self.max_results
        if max_results is not None:
            if max_results < 1:
                raise ValueError("max_results must be >= 1")
            self.max_results = int(max_results)
        try:
            hits = self.search_web(question)
            sources: List[Source] = []
            seen = set()
            for hit in hits:
                url = hit["url"]
                if url in seen:
                    continue
                seen.add(url)
                source = self.fetch_page(url)
                source.title = hit["title"] or url
                source.snippet = source.snippet or hit["snippet"]
                sources.append(source)
                if self.delay:
                    time.sleep(self.delay)
            ev = self.evidence(sources)
            return {"question": question, "sources": [asdict(s) for s in sources], "evidence": asdict(ev)}
        finally:
            self.max_results = old_max

    @staticmethod
    def candidate(question: str, answer: str, result: Dict[str, Any]) -> Dict[str, Any]:
        evidence = result.get("evidence", {})
        return {"question": question, "answer": answer, "source": "web-research", "confidence": float(evidence.get("score", 0.0)), "metadata": {"evidence": evidence}}

    @staticmethod
    def validator_accepts(candidate: Dict[str, Any], validator: Optional[Callable[[Dict[str, Any]], bool]]) -> bool:
        if validator is None:
            return False
        return bool(validator(candidate))
