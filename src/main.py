from typing import Literal
import sqlite3
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import IkarisState
from src.tools.hardware import get_system_stats

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

def generate_answer_node(state: IkarisState, llm):
    """Final Answer Step: Synthesizes evidence into a response."""
    evidence = state.get("evidence", [])
    goal = state.get("goal", "")
    
    # Format evidence — handles unified Evidence dicts and legacy dicts
    context = ""
    for i, pkt in enumerate(evidence, 1):
        text = pkt.get("text", pkt.get("content", ""))
        title = pkt.get("title", "")
        source = pkt.get("source", "unknown")
        meta = pkt.get("meta", pkt.get("metadata", {}))
        
        # Extract Anchors for Citation
        anchors = []
        if 'hierarchy' in meta:
            h = meta['hierarchy']
            anchors.append(f"Sec {h.get('parent_section', '?')}")
        if 'equations' in meta: anchors.append(f"Eq {', '.join(meta['equations'])}")
        
        anchor_str = f" | Anchors: {', '.join(anchors)}" if anchors else ""
        source_tag = f"[{source.upper()}]" if source != "unknown" else ""
        title_tag = f" — {title}" if title else ""
        context += f"--- Evidence {i} {source_tag}{title_tag}{anchor_str} ---\n{text}\n\n"
        
    prompt = (
        f"You are Ikaris. Answer the goal ONLY using the provided evidence.\n\n"
        f"GOAL: {goal}\n\n"
        f"EVIDENCE:\n{context}\n\n"
        "Cite evidence by [Evidence N] tags inline. Be precise and scientific."
    )
    
    response = llm.invoke(prompt)
    return {"messages": [response]}

def summarize_node(state: IkarisState, llm):
    """Compresses conversation history when it grows too large."""
    messages = state["messages"]
    existing_summary = state.get("summary", "")
    
    new_summary, trimmed = summarize_history(messages, llm, existing_summary)
    
    if new_summary != existing_summary:
        # Prepend summary context so the LLM always has history awareness
        summary_msg = SystemMessage(content=f"[Conversation Summary]: {new_summary}")
        return {"messages": [summary_msg] + trimmed, "summary": new_summary}
    
    return {"messages": [], "summary": existing_summary}

def router_logic(state: IkarisState) -> Literal["hardware_node", "agent_planning_node", "logseq_node", "research_node", "llm_node"]:
    """Decides where to send the user's request with improved intelligence."""
    from langchain_core.messages import HumanMessage
    import logging
    log = logging.getLogger(__name__)
    
    # Find the LAST HumanMessage (not just messages[-1], which may be
    # an AI response after checkpoint merge / summarization)
    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content.lower()
            break
    
    if not user_msg:
        log.warning(f"[Router] No HumanMessage found! messages[-1] = {state['messages'][-1].content[:80]}")
        return "llm_node"
    
    log.info(f"[Router] Routing: '{user_msg[:80]}'")
    
    # 1. Hardware stats take priority
    if any(word in user_msg for word in ["battery", "cpu", "stats", "hardware"]):
        log.info("[Router] → hardware_node")
        return "hardware_node"
    
    # 2. DOWNLOADER logic - ArXiv IDs, PubMed PMIDs, or explicit download commands
    arxiv_ids = re.findall(r'\d{4}\.\d{4,5}', user_msg)
    pmids = re.findall(r'\b\d{6,10}\b', user_msg)  # PMIDs are 6-10 digits
    is_pubmed = any(w in user_msg for w in ["pubmed", "pmid"])
    is_download = any(w in user_msg for w in ["arxiv.org", "download", "fetch"])
    
    # PubMed keyword search OR PMID fetch
    if is_pubmed:
        log.info(f"[Router] → research_node (PubMed search)")
        return "research_node"
    
    # ArXiv download
    if is_download or len(arxiv_ids) > 0:
        if len(arxiv_ids) > 1 or not any(word in user_msg for word in ["what", "how", "why", "explain"]):
            log.info(f"[Router] → research_node (ArXiv: {arxiv_ids})")
            return "research_node"
    
    # 3. AGENTIC RESEARCH logic - for questions about existing papers or general searches
    if any(word in user_msg for word in ["paper", "research", "study", "according to", "search"]):
        log.info("[Router] → agent_planning_node")
        return "agent_planning_node"
    
    # 4. Personal Logseq notes
    if any(word in user_msg for word in ["note", "notes", "logseq", "journal", "diary"]):
        log.info("[Router] → logseq_node")
        return "logseq_node"
    
    log.info("[Router] → llm_node (default)")
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

def logseq_node(state: IkarisState, tools):
    """Adds a note to Logseq."""
    last_msg = state["messages"][-1].content
    logseq_tool = next((t for t in tools if type(t).__name__ == "LogseqTool"), None)
    
    if logseq_tool:
        result = logseq_tool.add_note(last_msg)
    else:
        result = "LogseqTool is disabled or not configured."
        
    return {"messages": [SystemMessage(content=f"Logseq: {result}")]}

# --- 2. Build the Graph with Conditional Edges (Factory Pattern) ---
# MOVED TO src/agent.py to resolve circular dependencies.
# def build_graph(): ...
# ikaris_app = ... 

def start_agent_loop(cfg, agent):
    """Entry point from run.py after Hydra initialization."""
    import os
    import sys
    
    # --- 1. Linux GUI Fix (Prevents "Wayland" warnings & crashes) ---
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    # --- 2. Hugging Face Optimization (The "Reuse Forever" Fix) ---
    # tells HF to NEVER check the internet for models (forces local cache)
    os.environ["HF_HUB_OFFLINE"] = "1" 
    # Silences the "Unauthenticated" warning
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    # Silences the "Loading weights" progress bars and info logs
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
    
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    # No need to overwrite global LLM client anymore!
    # Global state is completely removed
    
    from PyQt5.QtWidgets import QApplication
    from src.ui.main_window import IkarisMainWindow
    from src.ui.styles import DARK_THEME

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    app.setApplicationName("Ikaris Assistant")
    app.setStyleSheet(DARK_THEME)

    window = IkarisMainWindow(agent)
    window.show()

    sys.exit(app.exec_())

