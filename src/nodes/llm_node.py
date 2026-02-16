from src.utils.llm_client import call_lm_studio

def llm_node(state):
    """
    Technical LLM processing node with the Ikaris persona.
    """
    messages = state["messages"]
    
    # Define the Ikaris System Persona
    system_prompt = (
        "You are Ikaris, a highly technical research assistant for a Computer Science Master's student. "
        "You are running locally on a powerful ROG Strix G16 (RTX 5070 Ti, 32GB RAM). "
        "Your tone is professional, expert, yet grounded and slightly witty. "
        "Focus on delivering clear, actionable research insights and system stats analysis."
    )
    
    # Generate response
    response = call_lm_studio(messages, system_prompt)
    
    return {"messages": [("ai", response)]}
