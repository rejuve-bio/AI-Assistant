"""
Literature search tools: PubMed and ClinicalTrials.gov
"""

import logging
import os
import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def search_pubmed(query: str, max_results: int = 10, min_year: int = None) -> dict:
    """
    Search PubMed for scientific papers and return abstracts with citations.

    Args:
        query: Search terms (e.g. 'BRCA1 breast cancer GWAS', 'mTOR aging longevity')
        max_results: Number of papers to return (default 10)
        min_year: Filter to papers from this year onward (e.g. 2020)

    Returns:
        dict with keys: query, papers (list of {pmid, title, authors, year, abstract, url}), count, source
    """
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "tool": "rejuve-ai-assistant",
            "email": "assistant@rejuve.bio",
        }
        if min_year:
            search_params["mindate"] = f"{min_year}/01/01"
            search_params["datetype"] = "pdat"

        search = requests.get(f"{base}/esearch.fcgi", params=search_params, timeout=15)
        search.raise_for_status()
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"query": query, "papers": [], "count": 0, "source": "PubMed"}

        fetch = requests.get(
            f"{base}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "tool": "rejuve-ai-assistant",
                "email": "assistant@rejuve.bio",
            },
            timeout=15,
        )
        fetch.raise_for_status()

        root = ET.fromstring(fetch.content)

        def _element_full_text(el) -> str:
            """Extract all text including text inside child tags (e.g. <i>, <sup>)."""
            return "".join(el.itertext()).strip() if el is not None else ""

        papers = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", "")
            title_el = article.find(".//ArticleTitle")
            title = _element_full_text(title_el)
            if not title or len(title) < 5:
                continue  # skip articles with missing/broken titles
            abstract_parts = [_element_full_text(t) for t in article.findall(".//AbstractText")]
            abstract = " ".join(p for p in abstract_parts if p)[:600]
            year = article.findtext(".//PubDate/Year") or article.findtext(
                ".//PubDate/MedlineDate", ""
            )[:4]
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
        return {"query": query, "error": str(e), "papers": [], "count": 0, "source": "PubMed"}


def search_clinical_trials(
    query: str, status: str = "RECRUITING", max_results: int = 10
) -> dict:
    """
    Search ClinicalTrials.gov for clinical studies.

    Args:
        query: Search terms (e.g. 'FOXO3 aging', 'rapamycin longevity', 'Alzheimer APOE')
        status: Trial status filter: 'RECRUITING', 'COMPLETED', 'ACTIVE_NOT_RECRUITING', or '' for all
        max_results: Number of trials to return

    Returns:
        dict with keys: query, trials (list of {nct_id, title, phase, status, conditions, interventions, url}), count, source
    """
    try:
        api_url = os.getenv("CLINICAL_TRIALS_API_URL", "https://clinicaltrials.gov/api/v2/studies")
        params = {
            "query.cond": query,   # condition/disease search — more targeted than query.term
            "query.term": query,   # also search full-text as fallback coverage
            "pageSize": max_results,
            "format": "json",
        }
        if status:
            params["filter.overallStatus"] = status

        r = requests.get(api_url, params=params, timeout=15)
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
                i.get("name", "")
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
        return {
            "query": query,
            "trials": trials,
            "count": len(trials),
            "source": "ClinicalTrials.gov",
        }

    except Exception as e:
        logger.error(f"search_clinical_trials error: {e}")
        return {
            "query": query,
            "error": str(e),
            "trials": [],
            "count": 0,
            "source": "ClinicalTrials.gov",
        }
