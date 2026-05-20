import requests
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CT_API      = "https://clinicaltrials.gov/api/v2/studies"
PUBMED_TOOL = "ai-assistant"
PUBMED_EMAIL = "assistant@rejuve.bio"


class WebSearch:
    def __init__(self, llm):
        self.llm = llm

    def search(self, query: str, sub_type: str = "general") -> str:
        logger.info(f"WebSearch.search: sub_type={sub_type}, query={query[:80]}")
        if sub_type == "pubmed":
            return self._pubmed(query)
        elif sub_type == "clinical_trials":
            return self._clinical_trials(query)
        else:
            return self._general(query)

    # ── PubMed ────────────────────────────────────────────────────────────────

    def _pubmed(self, query: str, max_results: int = 5) -> str:
        try:
            search_resp = requests.get(ESEARCH_URL, params={
                "db": "pubmed", "term": query, "retmax": max_results,
                "retmode": "json", "tool": PUBMED_TOOL, "email": PUBMED_EMAIL,
            }, timeout=10)
            search_resp.raise_for_status()
            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return f"No PubMed results found for: {query}"

            fetch_resp = requests.get(EFETCH_URL, params={
                "db": "pubmed", "id": ",".join(ids), "rettype": "abstract",
                "retmode": "xml", "tool": PUBMED_TOOL, "email": PUBMED_EMAIL,
            }, timeout=15)
            fetch_resp.raise_for_status()

            papers = self._parse_pubmed_xml(fetch_resp.text)
            if not papers:
                return f"Found {len(ids)} PubMed papers but could not parse abstracts."

            raw = "\n\n---\n\n".join(p["text"] for p in papers)
            synthesis = self.llm.generate(
                f"Summarize the key findings from the following PubMed abstracts "
                f"relevant to the query: '{query}'\n\n{raw[:4000]}\n\n"
                f"Write 3-5 sentences. Cite inline as (Author et al., Year)."
            )
            references = "\n".join(
                f"- {p['title']} ({p['year']}) — https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/"
                for p in papers if p["pmid"]
            )
            return f"{synthesis}\n\nReferences:\n{references}"

        except Exception as e:
            logger.error(f"PubMed search error: {e}", exc_info=True)
            return f"PubMed search failed: {e}"

    def _parse_pubmed_xml(self, xml_text: str) -> list:
        papers = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                title_el    = article.find(".//ArticleTitle")
                abstract_el = article.find(".//AbstractText")
                pmid_el     = article.find(".//PMID")
                year_el     = article.find(".//PubDate/Year")
                papers.append({
                    "pmid":  pmid_el.text  if pmid_el  is not None else "",
                    "year":  year_el.text  if year_el  is not None else "",
                    "title": title_el.text if title_el is not None else "No title",
                    "text":  f"PMID:{pmid_el.text if pmid_el is not None else ''} "
                             f"({year_el.text if year_el is not None else ''})\n"
                             f"{title_el.text if title_el is not None else ''}\n"
                             f"{abstract_el.text if abstract_el is not None else 'No abstract'}",
                })
        except ET.ParseError as e:
            logger.error(f"PubMed XML parse error: {e}")
        return papers

    # ── ClinicalTrials ────────────────────────────────────────────────────────

    def _clinical_trials(self, query: str, max_results: int = 5) -> str:
        try:
            resp = requests.get(CT_API, params={
                "query.term": query,
                "pageSize": max_results,
                "format": "json",
                "fields": "NCTId,BriefTitle,OverallStatus,Phase,BriefSummary,Condition,InterventionName",
            }, timeout=15)
            resp.raise_for_status()
            studies = resp.json().get("studies", [])
            if not studies:
                return f"No ClinicalTrials.gov results for: {query}"

            lines = []
            for s in studies:
                proto    = s.get("protocolSection", {})
                id_mod   = proto.get("identificationModule", {})
                stat_mod = proto.get("statusModule", {})
                desc_mod = proto.get("descriptionModule", {})
                des_mod  = proto.get("designModule", {})
                cond_mod = proto.get("conditionsModule", {})
                arms_mod = proto.get("armsInterventionsModule", {})
                lines.append(
                    f"NCT:{id_mod.get('nctId','')} | {stat_mod.get('overallStatus','')} | "
                    f"Phase:{', '.join(des_mod.get('phases', []))}\n"
                    f"Title: {id_mod.get('briefTitle','')}\n"
                    f"Conditions: {', '.join(cond_mod.get('conditions', []))}\n"
                    f"Interventions: {', '.join(i.get('interventionName','') for i in arms_mod.get('interventions',[]))}\n"
                    f"Summary: {desc_mod.get('briefSummary','')[:300]}"
                )

            raw = "\n\n---\n\n".join(lines)
            return self.llm.generate(
                f"Summarize the following ClinicalTrials.gov results for '{query}'.\n\n"
                f"{raw[:4000]}\n\n"
                f"Provide a 3-5 sentence synthesis of what is being tested and current trial status."
            )

        except Exception as e:
            logger.error(f"ClinicalTrials search error: {e}", exc_info=True)
            return f"ClinicalTrials search failed: {e}"

    # ── General web search ────────────────────────────────────────────────────

    def _general(self, query: str, max_results: int = 5) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"No web results found for: {query}"
            lines = [
                f"[{r['title']}]({r['href']})\n{r['body']}"
                for r in results
            ]
            return "\n\n".join(lines)
        except Exception as e:
            logger.error(f"General web search error: {e}")
            return f"Web search failed: {e}"
