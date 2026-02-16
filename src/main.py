from typing import Literal
import sqlite3
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import IkarisState
from src.tools.hardware import get_system_stats
from src.tools.paper_tool import query_papers
from src.tools.logseq_tool import add_logseq_note

from src.utils.llm_client import llm_instance
from src.nodes.llm_node import llm_node
from src.nodes.research_node import research_node

# --- 1. Define the Nodes ---

def router_logic(state: IkarisState) -> Literal["hardware_node", "paper_node", "logseq_node", "research_node", "llm_node"]:
    """Decides where to send the user's request with improved intelligence."""
    user_msg = state["messages"][-1].content.lower()
    
    # 1. Hardware stats take priority
    if any(word in user_msg for word in ["battery", "cpu", "stats", "hardware"]):
        return "hardware_node"
    
    # 2. DOWNLOADER logic - link, "download" command, or multiple ArXiv IDs
    arxiv_ids = re.findall(r'\d{4}\.\d{4,5}', user_msg)
    if any(word in user_msg for word in ["arxiv.org", "download", "fetch"]) or len(arxiv_ids) > 0:
        # Extra check: if they are asking a question ABOUT a paper, don't download
        # UNLESS they have multiple IDs, which implies a batch request
        if len(arxiv_ids) > 1 or not any(word in user_msg for word in ["what", "how", "why", "explain"]):
            return "research_node"
    
    # 3. RESEARCH/RAG logic - for questions about existing papers
    if any(word in user_msg for word in ["paper", "research", "study", "according to"]):
        return "paper_node"
    
    # 4. Personal Logseq notes
    if any(word in user_msg for word in ["note", "notes", "logseq", "journal", "diary"]):
        return "logseq_node"
        
    return "llm_node"

def hardware_node(state: IkarisState):
    """Execution node for system stats."""
    stats = get_system_stats()
    # Use AIMessage instead of a tuple
    return {"messages": [AIMessage(content=f"System Status: {stats}")], "hardware_info": stats}

def paper_node(state: IkarisState):
    """Node to handle research paper queries with smart Logseq tagging."""
    user_query = state["messages"][-1].content
    
    # 1. Get info from the papers
    context = query_papers(user_query)
    
    # 2. Ask LM Studio to summarize that info
    prompt = (
        f"You are Ikaris, an advanced assistant running on this ROG Strix. "
        f"I found this in your papers: {context}\n\n"
        f"User asked: {user_query}\n"
        "Give a concise, expert answer. Don't say 'according to the text'—just give me the facts."
    )
    response = llm_instance.invoke(prompt)
    
    # 3. Smart "Memory Sync" Tagging
    # Scan history/query for important tags
    tags = []
    history_text = " ".join([m.content for m in state["messages"]]).lower()
    if "mahesh" in history_text:
        tags.append("# [[Mahesh]]")
    if any(k in history_text for k in ["msc", "m.sc", "project", "assignment"]):
        tags.append("# [[MSc Project]]")
    if "attention" in history_text or "transformer" in history_text:
        tags.append("# [[LLM Research]]")
    
    tag_str = " ".join(tags)
    
    # 4. Automatically log this insight with tags
    log_msg = f"Researched: {user_query} | Insight: {response.content[:100]}..."
    add_logseq_note(log_msg, tags=tag_str)
    
    return {"messages": [response]}

def logseq_node(state: IkarisState):
    """Node to handle personal Logseq note queries."""
    from src.tools.logseq_tool import search_logseq_notes
    user_query = state["messages"][-1].content
    
    # 1. Search Logseq pages
    notes_content = search_logseq_notes(user_query)
    
    # 2. Ask LM Studio to respond based on personal notes
    prompt = (
        f"You are Ikaris. I found these personal notes in your Logseq graph:\n\n{notes_content}\n\n"
        f"Based on these notes, answer the user's question: {user_query}"
    )
    response = llm_instance.invoke(prompt)
    return {"messages": [response]}

# --- 2. Build the Graph with Conditional Edges ---

builder = StateGraph(IkarisState)

# Add our nodes
builder.add_node("llm_node", llm_node)
builder.add_node("hardware_node", hardware_node)
builder.add_node("paper_node", paper_node)
builder.add_node("logseq_node", logseq_node)
builder.add_node("research_node", research_node)

# Add the starting logic: START -> (Route Decision)
builder.add_conditional_edges(
    START, 
    router_logic,
    {
        "hardware_node": "hardware_node",
        "paper_node": "paper_node",
        "logseq_node": "logseq_node",
        "research_node": "research_node",
        "llm_node": "llm_node"
    }
)

# All paths lead to the END
builder.add_edge("hardware_node", END)
builder.add_edge("paper_node", END)
builder.add_edge("logseq_node", END)
builder.add_edge("research_node", END)
builder.add_edge("llm_node", END)

# Persistent Memory Setup
conn = sqlite3.connect("ikaris_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

ikaris_app = builder.compile(checkpointer=memory)
