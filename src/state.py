from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class IkarisState(TypedDict):
    # Conversation history
    messages: Annotated[list, add_messages]
    # Hardware info from system stats
    hardware_info: str
    # Rolling conversation summary (for SQLite bloat prevention)
    # Rolling conversation summary (for SQLite bloat prevention)
    summary: str
    
    # --- Agentic Control Fields ---
    goal: str                # Current research goal
    open_questions: list[str]# Questions to investigate
    evidence: list[dict]     # Accumulated structured evidence packets
    confidence: float        # 0.0 to 1.0 score of answer readiness
    loop_count: int          # Safety Guard
