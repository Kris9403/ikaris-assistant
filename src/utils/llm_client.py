from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

def LMStudioClient(base_url: str, model_name: str, temperature: float):
    return ChatOpenAI(base_url=base_url, model=model_name, temperature=temperature, api_key="lm-studio")

def OllamaClient(base_url: str, model_name: str, temperature: float):
    return ChatOpenAI(base_url=base_url, model=model_name, temperature=temperature, api_key="ollama")

def call_lm_studio(llm, messages, system_prompt):
    """
    Calls the local LM Studio server with a system prompt and conversation history.
    """
    formatted_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(formatted_messages)
    return response.content

def stream_lm_studio(llm, messages, system_prompt):
    """
    Streams tokens from LM Studio. Yields each token chunk as a string.
    """
    formatted_messages = [SystemMessage(content=system_prompt)] + messages
    for chunk in llm.stream(formatted_messages):
        if chunk.content:
            yield chunk.content
