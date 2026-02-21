import logging
from langchain_core.messages import AIMessage, SystemMessage

log = logging.getLogger(__name__)

def synthesis_node(state, llm):
    """
    Analyzes multiple paper abstracts to find consensus or conflict.
    Takes evidence from state and produces a comparative synthesis.
    """
    evidence = state.get("evidence", [])
    goal = state.get("goal", "")
    
    if not evidence or len(evidence) < 2:
        return {"messages": [AIMessage(
            content="I couldn't find enough papers to synthesize a comparison. "
                    "Try broadening your research query."
        )]}

    # Format evidence into a context block
    context_block = ""
    for i, ev in enumerate(evidence, 1):
        # Handle both Evidence dicts and raw dicts
        title = ev.get("title", "Untitled")
        text = ev.get("text", ev.get("content", ""))
        source = ev.get("source", "unknown")
        meta = ev.get("meta", ev.get("metadata", {}))
        
        journal = meta.get("journal", "")
        year = meta.get("year", "")
        cite_info = f" ({journal}, {year})" if journal and year else ""
        
        context_block += (
            f"--- Paper [{i}] [{source.upper()}]{cite_info} ---\n"
            f"Title: {title}\n"
            f"{text[:1500]}\n\n"
        )

    prompt = (
        f"You are Ikaris, a research synthesis engine.\n\n"
        f"GOAL: {goal}\n\n"
        f"Below are {len(evidence)} retrieved papers/chunks from multiple sources "
        f"(local PDFs and PubMed):\n\n"
        f"{context_block}\n\n"
        "Provide a **Comparative Synthesis**:\n"
        "1. **Consensus**: What do these studies agree on?\n"
        "2. **Conflicts**: Are there any contradictory findings?\n"
        "3. **Key Methods**: What are the most notable methodologies?\n"
        "4. **Research Gaps**: What questions remain unanswered?\n\n"
        "Cite papers by their [Paper N] tag inline. Be precise and scientific."
    )

    log.info(f"[SynthesisNode] Synthesizing {len(evidence)} evidence items.")
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    return {"messages": [response]}
