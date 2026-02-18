
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.state import IkarisState
from src.main import (
    summarize_node, llm_node, hardware_node, logseq_node, research_node,
    agent_planning_node, retrieval_node, reasoning_node, generate_answer_node,
    router_logic, reasoning_router
)

# --- Build the Graph with Conditional Edges (Factory Pattern) ---

def build_graph():
    """Constructs and compiles the StateGraph. Call this at runtime."""
    builder = StateGraph(IkarisState)

    # Add nodes
    builder.add_node("summarize_node", summarize_node)
    builder.add_node("llm_node", llm_node)
    builder.add_node("hardware_node", hardware_node)
    builder.add_node("logseq_node", logseq_node)
    builder.add_node("research_node", research_node)

    # Agentic Loop Nodes
    builder.add_node("agent_planning_node", agent_planning_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("reasoning_node", reasoning_node)
    builder.add_node("generate_answer_node", generate_answer_node)

    # Flow: START -> summarize -> router -> node -> END
    builder.add_edge(START, "summarize_node")
    builder.add_conditional_edges(
        "summarize_node", 
        router_logic,
        {
            "hardware_node": "hardware_node",
            "agent_planning_node": "agent_planning_node",
            "logseq_node": "logseq_node",
            "research_node": "research_node",
            "llm_node": "llm_node"
        }
    )

    # Agentic Loop Edges
    builder.add_edge("agent_planning_node", "retrieval_node")
    builder.add_edge("retrieval_node", "reasoning_node")
    builder.add_conditional_edges(
        "reasoning_node",
        reasoning_router,
        {
            "retrieval_node": "retrieval_node",
            "generate_answer_node": "generate_answer_node"
        }
    )
    builder.add_edge("generate_answer_node", END)

    # All paths lead to the END
    builder.add_edge("hardware_node", END)
    # builder.add_edge("paper_node", END) # Removed legacy node
    builder.add_edge("logseq_node", END)
    builder.add_edge("research_node", END)
    builder.add_edge("llm_node", END)

    # Persistent Memory Setup
    # Note: connect() check_same_thread=False is needed for multi-threaded GUI
    conn = sqlite3.connect("ikaris_memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    return builder.compile(checkpointer=memory)

_ikaris_app = None

def get_ikaris_app():
    global _ikaris_app
    if _ikaris_app is None:
        _ikaris_app = build_graph()
    return _ikaris_app

