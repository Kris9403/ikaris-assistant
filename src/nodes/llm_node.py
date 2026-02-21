import platform
from src.utils.llm_client import call_lm_studio
from langchain_core.messages import AIMessage

def llm_node(state, llm):
    """
    Technical LLM processing node with the Ikaris persona.
    """
    messages = state["messages"]
    
    # DYNAMIC SYSTEM INFO: Get real OS and Python version
    os_info = f"{platform.system()} {platform.release()}"
    machine = platform.machine()
    
    # Define the Ikaris System Persona (Updated)
    system_prompt = (
        "You are Ikaris, a highly technical research assistant for a Computer Science Master's student. "
        f"You are running locally on a ROG Strix G16 (RTX 5070 Ti, 32GB RAM) hosted on {os_info} ({machine}). "
        "Your tone is professional, expert, yet grounded and slightly witty. "
        "Focus on delivering clear, actionable research insights and system stats analysis."
    )
    
    # Generate response
    response = call_lm_studio(llm, messages, system_prompt)
    
    return {"messages": [AIMessage(content=response)]}
