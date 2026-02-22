import re
from langchain_core.messages import AIMessage, HumanMessage

def research_node(state, tools):
    """Batch paper download and indexing node. Supports ArXiv + PubMed."""
    # Find the last HumanMessage (messages[-1] may be AI after checkpoint merge)
    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break
    user_lower = user_msg.lower()
    
    # Extract tools
    research_tool = next((t for t in tools if type(t).__name__ == "ResearchTool"), None)
    logseq_tool = next((t for t in tools if type(t).__name__ == "LogseqTool"), None)
    paper_tool = next((t for t in tools if type(t).__name__ == "PaperTool"), None)
    pubmed_tool = next((t for t in tools if type(t).__name__ == "PubMedTool"), None)
    
    # --- Detect PubMed intent ---
    is_pubmed = any(w in user_lower for w in ["pubmed", "pmid"])
    pmids = re.findall(r'\b(\d{6,10})\b', user_msg)  # Modern PMIDs are 6-10 digits
    
    if is_pubmed and pmids:
        if pubmed_tool and pubmed_tool.enabled:
            return _handle_pubmed(pmids, pubmed_tool, logseq_tool, paper_tool)
        else:
            return {"messages": [AIMessage(
                content="❌ PubMed tool is disabled in config. "
                        "Check `configs/tools/pubmed.yaml` and ensure `enabled: true`."
            )]}
    
    # --- Default: ArXiv path ---
    if not research_tool:
        return {"messages": [AIMessage(content="ResearchTool is disabled.")]}
    
    # Use the tool to find and download all IDs in the text
    batch_results = research_tool.fetch_multi(user_msg)
    
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
            if logseq_tool: logseq_tool.add_note(logseq_entry)
            new_papers_count += 1
        elif "Skipped" in str(item):
            skipped_count += 1
        else:
            errors.append(str(item))

    # ONE single re-index for the whole batch to save compute
    if new_papers_count > 0 and paper_tool:
        paper_tool.ingest() 

    summary = f"Batch process complete.\n- New papers: {new_papers_count}\n- Skipped: {skipped_count}"
    if errors:
        summary += f"\n- Errors: {len(errors)}"
    
    summary += "\n\nLogseq journal updated and FAISS index refreshed."
    return {"messages": [AIMessage(content=summary)]}


def _handle_pubmed(pmids, pubmed_tool, logseq_tool, paper_tool=None):
    """Fetch PubMed articles by PMID and return formatted results."""
    results = []
    errors = []
    new_pdfs_count = 0
    
    for pmid in pmids:
        try:
            info = pubmed_tool.fetch_by_pmid(pmid)
            
            title    = info.get("title", "No Title")
            abstract = info.get("abstract", "No Abstract")
            journal  = info.get("journal", "Unknown")
            year     = info.get("year", "Unknown")
            authors  = info.get("authors", [])
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            
            entry = (
                f"### 📄 PMID {pmid}: {title}\n"
                f"**Journal**: {journal} ({year})\n"
                f"**Authors**: {author_str}\n\n"
                f"**Abstract**:\n{abstract}\n"
            )
            
            pmcid = pubmed_tool.pmid_to_pmcid(pmid)
            pdf_path = None
            if pmcid:
                pdf_path = pubmed_tool.download_pdf(pmcid)
                if pdf_path:
                    entry += f"\n📥 PDF saved to: `{pdf_path}`\n"
                    new_pdfs_count += 1
            
            results.append(entry)
            
            # Log to Logseq
            if logseq_tool:
                logseq_entry = (
                    f"## [[{title}]]\n"
                    f"  - **Source**: #pubmed PMID:{pmid}\n"
                    f"  - **Journal**: {journal} ({year})\n"
                    f"  - **Status**: #[[To Read]]\n"
                    f"  - **Abstract**: {abstract[:300]}...\n"
                )
                if pdf_path:
                    logseq_entry += f"  - **Local Path**: `{pdf_path}`\n"
                logseq_tool.add_note(logseq_entry)
                
        except Exception as e:
            errors.append(f"PMID {pmid}: {str(e)}")
            
    if new_pdfs_count > 0 and paper_tool:
        paper_tool.ingest()
    
    if not results and errors:
        return {"messages": [AIMessage(content=f"❌ PubMed fetch failed:\n" + "\n".join(errors))]}
    
    output = f"🔬 **PubMed Results** ({len(results)} paper(s))\n\n" + "\n---\n".join(results)
    if errors:
        output += f"\n\n⚠️ Errors: {', '.join(errors)}"
    
    return {"messages": [AIMessage(content=output)]}


