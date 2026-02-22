"""
PubMed Tool — direct NCBI E-utilities REST API (zero metapub dependency).

Pipeline:
  1. ESearch (retmode=json) → query → list of PMIDs
  2. EFetch  (retmode=xml)  → PMID  → article XML (title, abstract, journal, year, authors)

Rate limits respected via _throttle() (10 req/s with API key, 3 req/s without).
"""

import time
import logging
import requests
import xml.etree.ElementTree as ET

from src.evidence import Evidence

log = logging.getLogger(__name__)

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ELINK   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"


class PubMedTool:
    name = "pubmed"
    description = "Search biomedical literature via NCBI PubMed"
    capabilities = ["literature_search", "biomedical"]

    def __init__(self, api_key: str = None, enabled: bool = True, **kwargs):
        # Strip any stray Unicode quotes / whitespace from the key
        raw = (api_key or "").strip().strip("\u00a8").strip('"').strip("'").strip()
        self.api_key = raw if raw and raw != "xxxx" else None
        self.enabled = enabled
        self.config = kwargs
        log.info(
            f"[PubMedTool] Initialized | api_key={'set' if self.api_key else 'none'} "
            f"| enabled={self.enabled}"
        )

    # ------------------------------------------------------------------
    # Rate-limiting
    # ------------------------------------------------------------------

    def _throttle(self):
        """
        Respect NCBI rate limits:
        - 10 req/sec with an API key (0.11 s)
        -  3 req/sec without          (0.34 s)
        """
        time.sleep(0.11 if self.api_key else 0.34)

    # ------------------------------------------------------------------
    # Low-level E-utilities helpers
    # ------------------------------------------------------------------

    def _api_params(self) -> dict:
        """Base params shared by all E-utility calls (just api_key)."""
        params: dict = {}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _search_pmids(self, query: str, retmax: int = 5) -> list[str]:
        """
        ESearch: free-text query → list of PMIDs.
        Uses retmode=json (ESearch does NOT reliably support retmode=xml).
        """
        params = {
            **self._api_params(),
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
        }
        self._throttle()
        resp = requests.get(_ESEARCH, params=params, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _fetch_article(self, pmid: str) -> dict:
        """
        EFetch: single PMID → dict with title, abstract, journal, year, authors.
        Uses retmode=xml (EFetch requires XML for full article records).
        """
        params = {
            **self._api_params(),
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }
        self._throttle()
        resp = requests.get(_EFETCH, params=params, timeout=15)
        log.debug(f"[PubMedTool] EFetch raw ({resp.status_code}): {resp.text[:500]}")
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        art = root.find(".//MedlineCitation/Article")

        # Handle missing or empty <Article> (book chapters, errata, editorials)
        if art is None or len(art) == 0:
            # Try BookDocument as fallback
            book = root.find(".//BookDocument")
            if book is not None:
                title = book.findtext("ArticleTitle") or book.findtext("BookTitle/BookTitle") or "No Title"
                abstract = self._extract_abstract(book)
                return {
                    "title": title.strip(),
                    "abstract": abstract,
                    "journal": "Book Chapter",
                    "year": "Unknown",
                    "authors": [],
                    "doi": None,
                    "pmcid": None,
                }
            return {
                "title": "No Title",
                "abstract": "No Abstract",
                "journal": "Unknown",
                "year": "Unknown",
                "authors": [],
                "doi": None,
                "pmcid": None,
            }

        title    = (art.findtext("ArticleTitle") or "No Title").strip()
        abstract = self._extract_abstract(art)
        journal  = art.findtext("Journal/Title") or "Unknown"
        year     = (
            art.findtext("Journal/JournalIssue/PubDate/Year")
            or art.findtext("Journal/JournalIssue/PubDate/MedlineDate")
            or "Unknown"
        )
        authors = self._extract_authors(art)
        doi = root.findtext(".//ArticleId[@IdType='doi']")
        pmcid = root.findtext(".//ArticleId[@IdType='pmc']")

        return {
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": authors,
            "doi": doi,
            "pmcid": pmcid,
        }

    @staticmethod
    def _extract_abstract(art_el) -> str:
        abs_el = art_el.find("Abstract")
        if abs_el is None:
            return "No Abstract"
        parts = []
        for txt in abs_el.findall("AbstractText"):
            label = txt.get("Label", "")
            body  = "".join(txt.itertext()).strip()
            if label:
                parts.append(f"{label}: {body}")
            else:
                parts.append(body)
        return "\n".join(parts) or "No Abstract"

    @staticmethod
    def _extract_authors(art_el) -> list[str]:
        authors = []
        for au in art_el.findall("AuthorList/Author"):
            last  = au.findtext("LastName") or ""
            fore  = au.findtext("ForeName") or au.findtext("Initials") or ""
            name  = f"{last} {fore}".strip()
            if name:
                authors.append(name)
        return authors

    # ------------------------------------------------------------------
    # Public API — used by retrieval_node (query search)
    # ------------------------------------------------------------------

    def run(self, query: str) -> list:
        """
        Main entry point for free-text search.  Returns List[Evidence].
        Pipeline: ESearch (query→PMIDs) → EFetch (PMIDs→articles)
        """
        if not self.enabled:
            log.warning("[PubMedTool] Disabled.")
            return []

        log.info(f"[PubMedTool] Searching: '{query}'")
        start = time.time()
        results = []

        try:
            pmids = self._search_pmids(query, retmax=5)
            log.info(f"[PubMedTool] ESearch returned {len(pmids)} PMIDs: {pmids}")

            for pmid in pmids:
                try:
                    info = self._fetch_article(pmid)
                    results.append(Evidence(
                        source="pubmed",
                        id=str(pmid),
                        title=info["title"],
                        text=info["abstract"],
                        relevance=0.8,
                        meta={
                            "journal": info["journal"],
                            "year":    info["year"],
                            "authors": info["authors"],
                            "doi":     info.get("doi"),
                            "pmcid":   info.get("pmcid"),
                        },
                    ))
                except Exception as e:
                    log.error(f"[PubMedTool] PMID {pmid}: {e}")
        except Exception as e:
            log.error(f"[PubMedTool] Search failed: {e}")

        log.info(f"[PubMedTool] Done in {time.time()-start:.2f}s | {len(results)} papers.")
        return results

    # ------------------------------------------------------------------
    # Public API — used by research_node (fetch by PMID)
    # ------------------------------------------------------------------

    def fetch_by_pmid(self, pmid: str) -> dict:
        """Fetch a single article by PMID.  Returns raw dict."""
        return self._fetch_article(str(pmid))

    def pmid_to_pmcid(self, pmid: str) -> str:
        """Resolve a PMID to a PMCID using the NCBI ELink API. Returns the PMCID or None."""
        params = {
            **self._api_params(),
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "xml"
        }
        try:
            self._throttle()
            resp = requests.get(_ELINK, params=params, timeout=15)
            resp.raise_for_status()

            root = ET.fromstring(resp.text)
            pmcid = root.findtext(".//LinkSetDb/Link/Id")
            return pmcid
        except Exception as e:
            log.error(f"[PubMedTool] ELink failed for PMID {pmid}: {e}")
            return None

    def download_pdf(self, pmcid: str, download_path: str = "./papers") -> str:
        """Download the full-text PDF from PubMed Central and save it locally."""
        import os
        os.makedirs(download_path, exist_ok=True)
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            ctype = resp.headers.get("Content-Type", "")
            if "pdf" not in ctype.lower():
                log.error(f"[PubMedTool] PMC{pmcid} did not return PDF (Content-Type={ctype})")
                return None

            path = os.path.join(download_path, f"PMC{pmcid}.pdf")
            with open(path, "wb") as f:
                f.write(resp.content)
            return path
        except Exception as e:
            log.error(f"[PubMedTool] PDF download failed for PMC{pmcid}: {e}")
            return None

    def search_and_summarize(self, query: str) -> list:
        return self.run(query)
