from langchain_core.messages import SystemMessage
import json

def reasoning_node(state, llm):
    """
    Agentic Brain: Evaluates if we have enough evidence to satisfy the goal.
    Handles unified Evidence dicts from hybrid retrieval.
    """
    goal = state.get("goal", "")
    evidence = state.get("evidence", [])
    loop_count = state.get("loop_count", 0) + 1

    # Format evidence — supports both Evidence dicts and legacy dicts
    evidence_text = ""
    for i, pkt in enumerate(evidence):
        # Unified Evidence format
        title = pkt.get("title", "")
        text = pkt.get("text", pkt.get("content", ""))
        source = pkt.get("source", "unknown")
        relevance = pkt.get("relevance", "N/A")
        meta = pkt.get("meta", pkt.get("metadata", {}))
        
        # Extract anchors if available
        anchors = []
        if "hierarchy" in meta:
            h = meta["hierarchy"]
            anchors.append(f"Sec {h.get('parent_section', '?')}")
        if "sections" in meta and not anchors:
            anchors.append(f"Sec {meta['sections'][0]}")
        if "equations" in meta:
            anchors.append(f"Eq {', '.join(meta['equations'])}")
        
        anchor_str = f" (Anchors: {', '.join(anchors)})" if anchors else ""
        evidence_text += (
            f"[Item {i+1}] [{source.upper()}] (Rel: {relevance}){anchor_str}\n"
            f"  Title: {title}\n"
            f"  {text[:200]}...\n\n"
        )

    system_prompt = (
        "You are the Research Director. Your job is to decide if we have enough evidence to answer the user's goal.\n"
        f'GOAL: "{goal}"\n\n'
        f"CURRENT EVIDENCE:\n{evidence_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the evidence vs the goal.\n"
        "2. If sufficient, set confidence = 1.0.\n"
        "3. If insufficient, list specific 'open_questions' that need to be researched next.\n"
        '4. Respond in JSON format ONLY: {"confidence": float, "open_questions": [str], "reasoning": str}'
    )
    
    response = llm.invoke([SystemMessage(content=system_prompt)])
    
    try:
        content = response.content.strip().replace("```json", "").replace("```", "")
        result = json.loads(content)
        return {
            "confidence": result.get("confidence", 0.0),
            "open_questions": result.get("open_questions", []),
            "loop_count": loop_count
        }
    except Exception:
        return {"confidence": 0.0, "open_questions": [goal], "loop_count": loop_count}
