from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Initialize the shared client pointing to LM Studio local server
# You can change the base_url or model name here in the future
llm_instance = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def call_lm_studio(messages, system_prompt):
    """
    Calls the local LM Studio server with a system prompt and conversation history.
    """
    # Prepare the messages for the model
    formatted_messages = [SystemMessage(content=system_prompt)] + messages
    
    # Invoke the model
    response = llm_instance.invoke(formatted_messages)
    
    return response.content
