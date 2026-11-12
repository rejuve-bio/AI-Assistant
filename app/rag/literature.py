"""
Literature search tools: PubMed and ClinicalTrials.gov
"""

import logging
import os
import re
import requests

logger = logging.getLogger(__name__)

_GENERIC_QUERY_TERMS = {
    "gene", "genes", "human", "disease", "diseases", "function", "functions",
    "associated", "association", "associations", "protein", "proteins",
    "and", "the", "for", "with", "its", "it", "what", "does", "mean", "of",
    "in", "on", "to", "a", "an", "is", "are", "variant", "variants",
}


def _matches_actual_query(haystack_text: str, query: str) -> bool:
    terms = [t for t in re.findall(r"[A-Za-z0-9\-]+", query) if len(t) > 2 and t.lower() not in _GENERIC_QUERY_TERMS]
    if not terms:
        return True
    haystack = haystack_text.lower()
    return any(t.lower() in haystack for t in terms)


def search_pubmed(query: str, max_results: int = 10, min_year: int = None) -> dict:

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
            from datetime import datetime
            search_params["mindate"] = f"{min_year}/01/01"
            search_params["maxdate"] = datetime.now().strftime("%Y/%m/%d")
            search_params["datetype"] = "pdat"

        search = requests.get(f"{base}/esearch.fcgi", params=search_params, timeout=15)
        search.raise_for_status()
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        logger.info(f"PubMed esearch -- term={query!r} -- status={search.status_code} -- {len(ids)} id(s): {ids}")
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
        logger.info(f"PubMed efetch -- ids={ids} -- status={fetch.status_code} -- {len(fetch.content)} bytes")

        import xml.etree.ElementTree as ET

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
        relevant_papers = [
            p for p in papers
            if _matches_actual_query(f"{p.get('title', '')} {p.get('abstract', '')}", query)
        ]
        if papers and not relevant_papers:
            logger.warning(
                f"PubMed: esearch returned {len(papers)} result(s) for {query!r} but none actually "
                "mention its specific terms -- treating as no relevant results found."
            )
        logger.info(f"PubMed result -- term={query!r} -- {len(relevant_papers)} relevant paper(s): {[p['pmid'] for p in relevant_papers]}")
        return {"query": query, "papers": relevant_papers, "count": len(relevant_papers), "source": "PubMed"}

    except Exception as e:
        logger.error(f"search_pubmed error: {e}")
        return {"query": query, "error": str(e), "papers": [], "count": 0, "source": "PubMed"}


def search_clinical_trials(
    query: str, status: str = "RECRUITING", max_results: int = 10
) -> dict:
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
        logger.info(f"ClinicalTrials.gov -- query={query!r} status_filter={status!r} -- http_status={r.status_code} -- {len(studies)} stud(y/ies)")
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
        relevant_trials = [
            t for t in trials
            if _matches_actual_query(
                f"{t['title']} {' '.join(t['conditions'])} {' '.join(t['interventions'])}", query
            )
        ]
        if trials and not relevant_trials:
            logger.warning(
                f"ClinicalTrials.gov returned {len(trials)} result(s) for {query!r} but none actually "
                "mention its specific terms -- treating as no relevant results found."
            )
        return {
            "query": query,
            "trials": relevant_trials,
            "count": len(relevant_trials),
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
