
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.state import IkarisState
from src.main import (
    summarize_node, llm_node, hardware_node, logseq_node, research_node,
    agent_planning_node, retrieval_node, reasoning_node, generate_answer_node,
    router_logic, reasoning_router
)
from src.nodes.synthesis_node import synthesis_node

# --- Build the Graph with Conditional Edges (Factory Pattern) ---

class Agent:
    def __init__(self, llm, tools, ui=None, audio=None):
        self.llm = llm
        self.tools = tools
        self.ui = ui
        self.audio = audio
        self.app = self.build_graph()

    def build_graph(self):
        """Constructs and compiles the StateGraph."""
        builder = StateGraph(IkarisState)

        # Add nodes with injected dependencies
        builder.add_node("summarize_node", lambda state: summarize_node(state, self.llm))
        builder.add_node("llm_node", lambda state: llm_node(state, self.llm))
        builder.add_node("hardware_node", hardware_node)
        builder.add_node("logseq_node", lambda state: logseq_node(state, self.tools))
        builder.add_node("research_node", lambda state: research_node(state, self.tools))

        # Agentic Loop Nodes
        builder.add_node("agent_planning_node", agent_planning_node)
        builder.add_node("retrieval_node", lambda state: retrieval_node(state, self.tools))
        builder.add_node("reasoning_node", lambda state: reasoning_node(state, self.llm))
        builder.add_node("generate_answer_node", lambda state: generate_answer_node(state, self.llm))
        
        # Synthesis Node: comparative analysis across multi-source evidence
        builder.add_node("synthesis_node", lambda state: synthesis_node(state, self.llm))

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
        
        # Research -> Synthesis -> END (comparative analysis path)
        builder.add_edge("research_node", "synthesis_node")
        builder.add_edge("synthesis_node", END)

        # All paths lead to the END
        builder.add_edge("hardware_node", END)
        builder.add_edge("logseq_node", END)
        builder.add_edge("llm_node", END)

        # Persistent Memory Setup
        conn = sqlite3.connect("ikaris_memory.db", check_same_thread=False)
        memory = SqliteSaver(conn)

        return builder.compile(checkpointer=memory)

    def run(self):
        # For the GUI, we pass this agent object to GUI bootstrap
        pass
