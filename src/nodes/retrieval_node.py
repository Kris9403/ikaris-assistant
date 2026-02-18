from src.tools.paper_tool import query_papers

def retrieval_node(state):
    """
    Agentic Worker: Fetches evidence for open questions.
    """
    questions = state.get("open_questions", [])
    if not questions:
        # If no questions but we are here, fall back to goal
        questions = [state.get("goal")]
        
    new_evidence = []
    # Limit to top 2 questions to save tokens/time
    for q in questions[:2]: 
        packets = query_papers(q)
        new_evidence.extend(packets)
        
    # FIX: Properly combine old + new evidence
    current_evidence = state.get("evidence", [])
    combined_raw = current_evidence + new_evidence
    
    # --- DEDUPLICATION LOGIC ---
    unique_evidence = []
    seen_hashes = set()
    
    for item in combined_raw:
        # Create a unique signature based on content
        content = item.get("content", "").strip()
        if not content:
            continue
            
        content_sig = hash(content)
        
        if content_sig not in seen_hashes:
            seen_hashes.add(content_sig)
            unique_evidence.append(item)
    
    return {"evidence": unique_evidence}
