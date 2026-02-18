from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

MAX_MESSAGES = 20
KEEP_RECENT = 6

def summarize_history(messages, llm, existing_summary=""):
    """
    When conversation history exceeds MAX_MESSAGES, compress older messages
    into a rolling summary while keeping the most recent KEEP_RECENT messages.
    
    Returns (summary_str, trimmed_messages).
    """
    if len(messages) <= MAX_MESSAGES:
        return existing_summary, messages

    # Split: older messages to summarize, recent to keep
    older = messages[:-KEEP_RECENT]
    recent = messages[-KEEP_RECENT:]

    # Build the conversation text for summarization
    convo_lines = []
    for msg in older:
        role = "User" if isinstance(msg, HumanMessage) else "Ikaris"
        content = msg.content if hasattr(msg, "content") else str(msg)
        convo_lines.append(f"{role}: {content}")

    conversation_text = "\n".join(convo_lines)

    # Include previous summary for continuity
    prev = f"Previous summary:\n{existing_summary}\n\n" if existing_summary else ""

    prompt = (
        f"{prev}"
        f"Conversation to summarize:\n{conversation_text}\n\n"
        "Provide a concise but comprehensive summary of the above conversation. "
        "Capture key topics discussed, decisions made, and any important context. "
        "Keep it under 200 words."
    )

    summary_messages = [
        SystemMessage(content="You are a conversation summarizer. Be concise and factual."),
        HumanMessage(content=prompt),
    ]
    
    response = llm.invoke(summary_messages)
    new_summary = response.content

    return new_summary, recent
