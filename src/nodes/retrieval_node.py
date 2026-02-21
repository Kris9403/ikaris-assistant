import logging
from src.evidence import Evidence

log = logging.getLogger(__name__)

# Biomedical intent keywords for capability routing
_BIOMEDICAL_TERMS = [
    "gene", "protein", "cell", "clinical", "patient", "drug", "disease",
    "therapy", "mutation", "biomarker", "cancer", "neural", "synapse",
    "receptor", "pathway", "immune", "antibody", "trial", "diagnosis",
    "treatment", "toxicity", "dosage", "pharmacology", "epidemiology",
    "pubmed", "biomedical", "medical",
]

def _infer_biomedical_intent(query: str) -> bool:
    """Heuristic check: does this query contain biomedical terms?"""
    q_lower = query.lower()
    return any(term in q_lower for term in _BIOMEDICAL_TERMS)

def retrieval_node(state, tools):
    """
    Hybrid retrieval: queries FAISS (local PDFs) and PubMed (biomedical),
    normalizes scores, deduplicates, and returns interleaved top-k Evidence.
    """
    questions = state.get("open_questions", [])
    if not questions:
        questions = [state.get("goal")]
        
    paper_tool = next((t for t in tools if type(t).__name__ == "PaperTool"), None)
    pubmed_tool = next((t for t in tools if type(t).__name__ == "PubMedTool"), None)
        
    all_evidence: list[Evidence] = []

    for q in questions[:2]:
        # --- Source 1: FAISS (local PDFs) ---
        if paper_tool:
            faiss_results = paper_tool.query(q)
            all_evidence.extend(faiss_results)
            log.info(f"[Retrieval] FAISS returned {len(faiss_results)} chunks for: '{q[:50]}...'")

        # --- Source 2: PubMed (biomedical, capability-routed) ---
        if pubmed_tool and _infer_biomedical_intent(q):
            pubmed_results = pubmed_tool.run(q)
            all_evidence.extend(pubmed_results)
            log.info(f"[Retrieval] PubMed returned {len(pubmed_results)} papers for: '{q[:50]}...'")

    # --- Merge: combine old + new evidence ---
    current_evidence = state.get("evidence", [])
    
    # Convert any legacy dicts into Evidence objects
    for item in current_evidence:
        if isinstance(item, dict):
            all_evidence.append(Evidence.from_dict(item))
        elif isinstance(item, Evidence):
            all_evidence.append(item)
    
    # --- Deduplicate by (source, id) ---
    seen = set()
    unique = []
    for ev in all_evidence:
        key = (ev.source, ev.id)
        if key not in seen and ev.text.strip():
            seen.add(key)
            unique.append(ev)
    
    # --- Rank: sort by relevance descending, take top 10 ---
    unique.sort(key=lambda e: e.relevance, reverse=True)
    top_k = unique[:10]
    
    log.info(f"[Retrieval] Merged {len(top_k)} unique evidence items (from {len(all_evidence)} raw).")
    
    # Serialize back to dicts for LangGraph state compatibility
    return {"evidence": [e.to_dict() for e in top_k]}
