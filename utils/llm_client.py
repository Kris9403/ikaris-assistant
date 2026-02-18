from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Initialize the shared client pointing to LM Studio local server
llm_instance = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def call_lm_studio(messages, system_prompt):
    """
    Calls the local LM Studio server with a system prompt and conversation history.
    """
    formatted_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm_instance.invoke(formatted_messages)
    return response.content

def stream_lm_studio(messages, system_prompt):
    """
    Streams tokens from LM Studio. Yields each token chunk as a string.
    """
    formatted_messages = [SystemMessage(content=system_prompt)] + messages
    for chunk in llm_instance.stream(formatted_messages):
        if chunk.content:
            yield chunk.content
