import time
import logging

try:
    from metapub import PubMedFetcher, FindIt
except ImportError:
    PubMedFetcher = None
    FindIt = None

from src.evidence import Evidence

log = logging.getLogger(__name__)

class PubMedTool:
    name = "pubmed"
    description = "Search biomedical literature"
    capabilities = ["literature_search", "biomedical"]

    def __init__(self, api_key: str = None, enabled: bool = True, **kwargs):
        self.api_key = api_key if api_key and api_key != "xxxx" else None
        self.enabled = enabled
        self.config = kwargs
        
        if PubMedFetcher is None:
            log.warning("[PubMedTool] metapub not installed. Tool disabled.")
            self.enabled = False
            self.fetch = None
        elif self.api_key:
            self.fetch = PubMedFetcher(api_key=self.api_key)
        else:
            self.fetch = PubMedFetcher()

    def _throttle(self):
        """
        Respect NCBI rate limits:
        - 10 req/sec with an API key (0.11s)
        - 3 req/sec without (0.34s)
        """
        if self.api_key:
            time.sleep(0.11)
        else:
            time.sleep(0.34)

    def run(self, query: str) -> list:
        """
        Main entry point. Returns List[Evidence].
        """
        if not self.enabled:
            log.warning("[PubMedTool] Disabled.")
            return []

        log.info(f"[PubMedTool] Executing search for query: '{query}'")
        start_time = time.time()
        
        results = []
        try:
            self._throttle()
            pmids = self.fetch.pmids_for_query(query, retmax=5)
            
            for pmid in pmids:
                try:
                    self._throttle()
                    article = self.fetch.article_by_pmid(pmid)
                    
                    self._throttle()
                    pdf_src = FindIt(pmid) if FindIt else None
                    
                    results.append(Evidence(
                        source="pubmed",
                        id=str(pmid),
                        title=getattr(article, "title", "No Title"),
                        text=getattr(article, "abstract", "No Abstract") or "No Abstract",
                        relevance=0.8,  # PubMed relevance is query-matched by NCBI
                        meta={
                            "journal": getattr(article, "journal", "Unknown"),
                            "year": getattr(article, "year", "Unknown"),
                            "pdf_url": pdf_src.url if pdf_src and pdf_src.url else "N/A",
                        }
                    ))
                except Exception as e:
                    log.error(f"[PubMedTool] Error processing PMID {pmid}: {str(e)}")
                    continue
                    
        except Exception as e:
            log.error(f"[PubMedTool] Search failed: {str(e)}")

        latency = time.time() - start_time
        log.info(f"[PubMedTool] Completed in {latency:.2f}s | {len(results)} paper(s).")
        
        return results

    def search_and_summarize(self, query: str) -> list:
        return self.run(query)
