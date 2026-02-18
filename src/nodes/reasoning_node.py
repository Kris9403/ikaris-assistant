from langchain_core.messages import SystemMessage
from src.utils.llm_client import llm_instance
import json

def reasoning_node(state):
    """
    Agentic Brain: Evaluates if we have enough evidence to satisfy the goal.
    """
    goal = state.get("goal", "")
    evidence = state.get("evidence", [])
    loop_count = state.get("loop_count", 0) + 1  # Increment loop counter
    
    # Format evidence for the "Research Manager" output
    evidence_text = ""
    for i, pkt in enumerate(evidence):
        meta = pkt.get('metadata', {})
        anchors = []
        if 'hierarchy' in meta:
            h = meta['hierarchy']
            anchors.append(f"Sec {h.get('parent_section', '?')}")
        
        # Fallback to flat anchors
        if 'sections' in meta and not anchors: anchors.append(f"Sec {meta['sections'][0]}")
        if 'equations' in meta: anchors.append(f"Eq {', '.join(meta['equations'])}")
        
        relevance = pkt.get('relevance', 'N/A')
        evidence_text += f"[Item {i+1}] (Rel: {relevance}) {pkt['content'][:200]}... (Anchors: {', '.join(anchors)})\n"

    system_prompt = (
        "You are the Research Director. Your job is to decide if we have enough evidence to answer the user's goal.\n"
        f"GOAL: \"{goal}\"\n\n"
        f"CURRENT EVIDENCE:\n{evidence_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the evidence vs the goal.\n"
        "2. If sufficient, set confidence = 1.0.\n"
        "3. If insufficient, list specific 'open_questions' that need to be researched next.\n"
        "4. Respond in JSON format ONLY: {\"confidence\": float, \"open_questions\": [str], \"reasoning\": str}"
    )
    
    response = llm_instance.invoke([SystemMessage(content=system_prompt)])
    
    try:
        # Simple JSON parsing (add robustness in prod)
        content = response.content.strip().replace("```json", "").replace("```", "")
        
        result = json.loads(content)
        return {
            "confidence": result.get("confidence", 0.0),
            "open_questions": result.get("open_questions", []),
            "loop_count": loop_count # Pass the incremented count back to state
        }
    except Exception as e:
        # Fallback strategy if JSON fails
        return {"confidence": 0.0, "open_questions": [goal], "loop_count": loop_count}
