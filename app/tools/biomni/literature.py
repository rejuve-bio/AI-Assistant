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


# ─────────────────────────────────────────────────────────────────────────────
# PubMed search
# ─────────────────────────────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int = 10, min_year: int = None) -> dict:
    """
    Search PubMed for scientific papers and return abstracts with citations.

    Args:
        query: Search terms (e.g. 'BRCA1 breast cancer GWAS', 'mTOR aging longevity')
        max_results: Number of papers to return (default 10)
        min_year: Filter to papers from this year onward (e.g. 2020)

    Returns:
        dict with keys: query, papers (list of {pmid, title, authors, year, abstract, url})
    """
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_params = {
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json", "sort": "relevance",
            "tool": "rejuve-ai-assistant", "email": "assistant@rejuve.bio",
        }
        if min_year:
            search_params["mindate"] = f"{min_year}/01/01"
            search_params["datetype"] = "pdat"

        search = requests.get(f"{base}/esearch.fcgi", params=search_params, timeout=15)
        search.raise_for_status()
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"query": query, "papers": [], "count": 0, "source": "PubMed"}

        fetch = requests.get(f"{base}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml",
            "tool": "rejuve-ai-assistant", "email": "assistant@rejuve.bio",
        }, timeout=15)
        fetch.raise_for_status()

        import xml.etree.ElementTree as ET
        root = ET.fromstring(fetch.content)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", "")
            title = article.findtext(".//ArticleTitle", "")
            abstract = " ".join(
                t.text or "" for t in article.findall(".//AbstractText")
            )[:600]
            year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate", "")[:4]
            authors = [
                f"{a.findtext('LastName', '')} {a.findtext('Initials', '')}".strip()
                for a in article.findall(".//Author")[:3]
            ]
            papers.append({
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return {"query": query, "papers": papers, "count": len(papers), "source": "PubMed"}
    except Exception as e:
        logger.error(f"search_pubmed error: {e}")
        return {"query": query, "error": str(e), "source": "PubMed"}


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrials.gov search
# ─────────────────────────────────────────────────────────────────────────────

def search_clinical_trials(query: str, status: str = "RECRUITING", max_results: int = 10) -> dict:
    """
    Search ClinicalTrials.gov for clinical studies.

    Args:
        query: Search terms (e.g. 'FOXO3 aging', 'rapamycin longevity', 'Alzheimer APOE')
        status: Trial status filter: 'RECRUITING', 'COMPLETED', 'ACTIVE_NOT_RECRUITING', or '' for all
        max_results: Number of trials to return

    Returns:
        dict with keys: query, trials (list of {nct_id, title, phase, status, conditions, interventions, url})
    """
    try:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.term": query,
            "pageSize": max_results,
            "format": "json",
            "fields": "NCTId,BriefTitle,Phase,OverallStatus,Condition,InterventionName,StartDate,CompletionDate",
        }
        if status:
            params["filter.overallStatus"] = status

        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        studies = data.get("studies", [])
        trials = []
        for s in studies:
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            conditions = proto.get("conditionsModule", {}).get("conditions", [])
            interventions = [
                i.get("interventionName", "")
                for i in proto.get("armsInterventionsModule", {}).get("interventions", [])
            ]
            nct_id = ident.get("nctId", "")
            trials.append({
                "nct_id": nct_id,
                "title": ident.get("briefTitle", ""),
                "phase": design.get("phases", []),
                "status": status_mod.get("overallStatus", ""),
                "conditions": conditions[:3],
                "interventions": interventions[:3],
                "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
            })
        return {"query": query, "trials": trials, "count": len(trials), "source": "ClinicalTrials.gov"}
    except Exception as e:
        logger.error(f"search_clinical_trials error: {e}")
        return {"query": query, "error": str(e), "source": "ClinicalTrials.gov"}