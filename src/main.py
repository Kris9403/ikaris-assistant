from typing import Literal
import sqlite3
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import IkarisState
from src.tools.hardware import get_system_stats
from src.tools.paper_tool import query_papers
from src.tools.logseq_tool import add_logseq_note

from src.utils.llm_client import llm_instance
from src.utils.summarizer import summarize_history
from src.nodes.llm_node import llm_node
from src.nodes.research_node import research_node
from src.nodes.retrieval_node import retrieval_node
from src.nodes.reasoning_node import reasoning_node

# --- 1. Define the Nodes ---

def agent_planning_node(state: IkarisState):
    """Sets up the initial agent state from the user query."""
    user_query = state["messages"][-1].content
    return {
        "goal": user_query,
        "open_questions": [user_query], # Start with the main goal as the first question
        "evidence": [],
        "confidence": 0.0,
        "loop_count": 0
    }

def generate_answer_node(state: IkarisState):
    """Final Answer Step: Synthesizes evidence into a response."""
    evidence = state.get("evidence", [])
    goal = state.get("goal", "")
    
    # Format evidence
    context = ""
    for i, pkt in enumerate(evidence, 1):
        content = pkt.get("content", "")
        meta = pkt.get("metadata", {})
        
        # Extract Anchors for Citation
        anchors = []
        if 'hierarchy' in meta:
            h = meta['hierarchy']
            anchors.append(f"Sec {h.get('parent_section', '?')}")
        if 'equations' in meta: anchors.append(f"Eq {', '.join(meta['equations'])}")
        
        anchor_str = f"| Anchors: {', '.join(anchors)}" if anchors else ""
        context += f"--- Evidence {i} {anchor_str} ---\n{content}\n\n"
        
    prompt = (
        f"You are Ikaris. Answer the goal ONLY using the provided evidence.\n\n"
        f"GOAL: {goal}\n\n"
        f"EVIDENCE:\n{context}\n\n"
        "Cite anchors (e.g., [Eq. 3], [Sec 4.1]) inline. Be precise and scientific."
    )
    
    response = llm_instance.invoke(prompt)
    
    # Optional: Tagging logic (Logseq) could reside here or in a separate node
    # For now, let's keep it simple and just return the answer
    return {"messages": [response]}

def summarize_node(state: IkarisState):
    """Compresses conversation history when it grows too large."""
    messages = state["messages"]
    existing_summary = state.get("summary", "")
    
    new_summary, trimmed = summarize_history(messages, llm_instance, existing_summary)
    
    if new_summary != existing_summary:
        # Prepend summary context so the LLM always has history awareness
        summary_msg = SystemMessage(content=f"[Conversation Summary]: {new_summary}")
        return {"messages": [summary_msg] + trimmed, "summary": new_summary}
    
    return {"messages": [], "summary": existing_summary}

def router_logic(state: IkarisState) -> Literal["hardware_node", "agent_planning_node", "logseq_node", "research_node", "llm_node"]:
    """Decides where to send the user's request with improved intelligence."""
    user_msg = state["messages"][-1].content.lower()
    
    # 1. Hardware stats take priority
    if any(word in user_msg for word in ["battery", "cpu", "stats", "hardware"]):
        return "hardware_node"
    
    # 2. DOWNLOADER logic - link, "download" command, or multiple ArXiv IDs
    arxiv_ids = re.findall(r'\d{4}\.\d{4,5}', user_msg)
    if any(word in user_msg for word in ["arxiv.org", "download", "fetch"]) or len(arxiv_ids) > 0:
        if len(arxiv_ids) > 1 or not any(word in user_msg for word in ["what", "how", "why", "explain"]):
            return "research_node"
    
    # 3. AGENTIC RESEARCH logic - for questions about existing papers
    if any(word in user_msg for word in ["paper", "research", "study", "according to"]):
        return "agent_planning_node"
    
    # 4. Personal Logseq notes
    if any(word in user_msg for word in ["note", "notes", "logseq", "journal", "diary"]):
        return "logseq_node"
        
    return "llm_node"

def reasoning_router(state: IkarisState) -> Literal["retrieval_node", "generate_answer_node"]:
    """Determines if we need more evidence or if we are ready to answer."""
    confidence = state.get("confidence", 0.0)
    loop_count = state.get("loop_count", 0)
    
    # SAFETY VALVE: If we've researched 3 times, force an answer
    if confidence >= 0.8 or loop_count >= 3:
        return "generate_answer_node"
    else:
        return "retrieval_node"


# --- Missing Node Definitions (Added during refactor) ---

def hardware_node(state: IkarisState):
    """Fetches system stats."""
    stats = get_system_stats()
    return {"messages": [SystemMessage(content=f"System Stats: {stats}")]}

def logseq_node(state: IkarisState):
    """Adds a note to Logseq."""
    last_msg = state["messages"][-1].content
    # Simple extraction: treat the whole message as the note
    result = add_logseq_note(last_msg)
    return {"messages": [SystemMessage(content=f"Logseq: {result}")]}

# --- 2. Build the Graph with Conditional Edges (Factory Pattern) ---
# MOVED TO src/agent.py to resolve circular dependencies.
# def build_graph(): ...
# ikaris_app = ... 

