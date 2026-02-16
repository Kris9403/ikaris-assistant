from src.tools.research_tool import fetch_multi_papers
from src.tools.logseq_tool import add_logseq_note
from src.tools.paper_tool import ingest_papers
from langchain_core.messages import AIMessage

def research_node(state):
    """Batch paper download and indexing node."""
    user_msg = state["messages"][-1].content
    
    # Use the tool to find and download all IDs in the text
    batch_results = fetch_multi_papers(user_msg)
    
    if isinstance(batch_results, str):
        return {"messages": [AIMessage(content=batch_results)]}

    new_papers_count = 0
    skipped_count = 0
    errors = []

    for item in batch_results:
        if isinstance(item, dict): # Successfully downloaded new paper
            # Professional Logseq Block Template
            logseq_entry = (
                f"## [[{item['title']}]]\n"
                f"  - **Source**: #arxiv\n"
                f"  - **Status**: #[[To Read]]\n"
                f"  - **Summary**: {item['summary'][:300]}...\n"
                f"  - **Local Path**: `{item['path']}`"
            )
            add_logseq_note(logseq_entry)
            new_papers_count += 1
        elif "Skipped" in str(item):
            skipped_count += 1
        else:
            errors.append(str(item))

    # ONE single re-index for the whole batch to save compute
    if new_papers_count > 0:
        ingest_papers() 

    summary = f"Batch process complete.\n- New papers: {new_papers_count}\n- Skipped: {skipped_count}"
    if errors:
        summary += f"\n- Errors: {len(errors)}"
    
    summary += "\n\nLogseq journal updated and FAISS index refreshed."
    return {"messages": [AIMessage(content=summary)]}
