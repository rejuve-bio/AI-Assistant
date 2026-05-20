"""
Literature tools: arXiv, Google Scholar, DOI supplementary data extraction,
URL and PDF content extraction.
"""

import logging
import requests

logger = logging.getLogger(__name__)


def search_arxiv(query: str, max_results: int = 10, category: str = "q-bio") -> dict:
    """
    Search arXiv for scientific preprints.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 10)
        category: arXiv category filter (default 'q-bio', use '' for all categories)

    Returns:
        dict with keys: papers (list of {title, authors, abstract, arxiv_id, published, url})
    """
    try:
        import feedparser
        cat_filter = f"+AND+cat:{category}" if category else ""
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{requests.utils.quote(query)}{cat_filter}"
            f"&max_results={max_results}&sortBy=relevance"
        )
        feed = feedparser.parse(url)
        papers = []
        for entry in feed.entries:
            papers.append({
                "title": entry.get("title", "").replace("\n", " "),
                "authors": [a.get("name", "") for a in entry.get("authors", [])],
                "abstract": entry.get("summary", "")[:500],
                "arxiv_id": entry.get("id", "").split("/abs/")[-1],
                "published": entry.get("published", ""),
                "url": entry.get("link", ""),
            })
        return {"query": query, "total": len(papers), "papers": papers}
    except ImportError:
        return {"error": "feedparser not installed. Run: pip install feedparser"}
    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return {"error": str(e)}


def search_scholar(query: str, max_results: int = 10) -> dict:
    """
    Search Google Scholar for scientific papers.

    Args:
        query: Search query string (supports operators like author:, source:)
        max_results: Maximum number of results (default 10)

    Returns:
        dict with keys: papers (list of {title, authors, year, citations, url, snippet})
    """
    try:
        from scholarly import scholarly
        search_results = scholarly.search_pubs(query)
        papers = []
        for i, result in enumerate(search_results):
            if i >= max_results:
                break
            bib = result.get("bib", {})
            papers.append({
                "title": bib.get("title", ""),
                "authors": bib.get("author", []),
                "year": bib.get("pub_year", ""),
                "abstract": bib.get("abstract", "")[:400],
                "citations": result.get("num_citations", 0),
                "url": result.get("pub_url", ""),
            })
        return {"query": query, "total": len(papers), "papers": papers}
    except ImportError:
        return {"error": "scholarly not installed. Run: pip install scholarly"}
    except Exception as e:
        logger.error(f"Scholar search failed: {e}")
        return {"error": str(e)}


def get_doi_supplementary(doi: str) -> dict:
    """
    Fetch metadata and supplementary data links for a paper given its DOI.

    Args:
        doi: Digital Object Identifier (e.g. '10.1038/s41586-021-03305-5')

    Returns:
        dict with keys: title, abstract, authors, journal, supplementary_links
    """
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {"fields": "title,abstract,authors,year,venue,externalIds,openAccessPdf"}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return {"doi": doi, "error": f"Semantic Scholar returned {resp.status_code}"}
        data = resp.json()
        return {
            "doi": doi,
            "title": data.get("title", ""),
            "abstract": (data.get("abstract") or "")[:600],
            "authors": [a.get("name", "") for a in data.get("authors", [])],
            "year": data.get("year"),
            "journal": data.get("venue", ""),
            "open_access_pdf": (data.get("openAccessPdf") or {}).get("url"),
            "source": "Semantic Scholar",
        }
    except Exception as e:
        logger.error(f"DOI fetch failed: {e}")
        return {"doi": doi, "error": str(e)}


def extract_url_content(url: str, max_chars: int = 5000) -> dict:
    """
    Extract and clean text content from a URL (paper page, database entry, etc.).

    Args:
        url: URL to fetch
        max_chars: Maximum characters to return from the page

    Returns:
        dict with keys: url, title, content (cleaned text)
    """
    try:
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
        resp = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        title = soup.title.string if soup.title else ""
        content = " ".join(soup.get_text(separator=" ").split())[:max_chars]
        return {"url": url, "title": title, "content": content}
    except ImportError:
        return {"error": "beautifulsoup4 not installed. Run: pip install beautifulsoup4"}
    except Exception as e:
        logger.error(f"URL extraction failed: {e}")
        return {"url": url, "error": str(e)}