# nodes/crawler.py
#
# CONCEPT: What a node is
# A node is just a function. It takes the full state, reads what it needs,
# does its work, and returns a dict of ONLY the keys it changed.
# LangGraph merges that dict back into state automatically.
#
# This node has NO LLM. It's pure Python.
# Rule of thumb: if a task doesn't need reasoning, don't use an LLM for it.

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List
from state import BlogAgentState


def crawl_and_extract(state: BlogAgentState) -> dict:
    """
    Crawls the given URL and up to 10 internal pages.
    Returns raw page data into state.

    What this node reads from state: url
    What this node writes to state:  raw_pages, crawl_error
    """

    print(f"\n[Crawler] Starting crawl: {state['url']}")

    try:
        pages = _crawl_site(state["url"], max_pages=10)
        print(f"[Crawler] Done. Collected {len(pages)} pages.")

        # IMPORTANT: return only what changed.
        # LangGraph merges this dict into the existing state.
        return {
            "raw_pages": pages,
            "crawl_error": None
        }

    except Exception as e:
        print(f"[Crawler] Failed: {e}")
        return {
            "raw_pages": [],
            "crawl_error": str(e)
        }


# ─── Helpers (not nodes, just internal functions) ────────────────────────────

def _crawl_site(start_url: str, max_pages: int = 10) -> List[dict]:
    """
    Crawls pages within the same domain. Returns a list of page dicts.
    """
    base_domain = urlparse(start_url).netloc
    visited = set()
    queue = [start_url]
    pages = []

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BlogAgent/1.0)"
    }

    while queue and len(pages) < max_pages:
        url = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        try:
            response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)

            # Skip non-HTML responses (PDFs, images, etc.)
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            page_data = {
                "url": url,
                "title": _extract_title(soup),
                "headings": _extract_headings(soup),
                "body": _extract_body_text(soup),
                "links": _extract_internal_links(soup, url, base_domain),
                "ctas": _extract_ctas(soup),
            }
            pages.append(page_data)
            print(f"  [Crawler] OK {url}")

            # Add new internal links to the queue
            for link in page_data["links"]:
                if link not in visited:
                    queue.append(link)

        except Exception as e:
            print(f"  [Crawler] FAIL {url} - {e}")
            continue

    return pages


def _extract_title(soup: BeautifulSoup) -> str:
    tag = soup.find("title")
    return tag.get_text(strip=True) if tag else ""


def _extract_headings(soup: BeautifulSoup) -> List[str]:
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if text:
            headings.append(text)
    return headings[:20]  # cap at 20 so state stays lean


def _extract_body_text(soup: BeautifulSoup) -> str:
    # Remove noise: nav, footer, scripts, ads
    for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Collapse whitespace
    import re
    text = re.sub(r"\s+", " ", text)

    # Cap at 3000 chars per page — enough context, not too heavy for state
    return text[:3000]


def _extract_internal_links(soup: BeautifulSoup, base_url: str, base_domain: str) -> List[str]:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Only follow links within the same domain, no fragments
        if parsed.netloc == base_domain and not parsed.fragment:
            clean = parsed._replace(query="", fragment="").geturl()
            if clean not in links:
                links.append(clean)

    return links[:20]


def _extract_ctas(soup: BeautifulSoup) -> List[str]:
    """Extract call-to-action text (buttons, prominent links)."""
    ctas = []
    for tag in soup.find_all(["button", "a"]):
        text = tag.get_text(strip=True)
        # Short text on interactive elements = likely a CTA
        if text and len(text) < 60:
            ctas.append(text)
    return list(set(ctas))[:10]