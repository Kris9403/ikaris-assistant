from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class IkarisState(TypedDict):
    # This keeps track of the conversation history
    messages: Annotated[list, add_messages]
    # You can add custom fields here, like user mood or system status
    hardware_info: str
