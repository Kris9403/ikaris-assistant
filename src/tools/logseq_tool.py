import datetime
import os
from typing import List

LOGSEQ_GRAPH_PATH = "/home/krishna/Desktop/LogSeq/Ikaris_Graph/journals"
LOGSEQ_PAGES_PATH = "/home/krishna/Desktop/LogSeq/Ikaris_Graph/pages"

def add_logseq_note(content: str, tags: str = ""):
    """Appends a note as a new block in today's Logseq journal."""
    # Logseq journal files are usually YYYY_MM_DD.md
    today = datetime.datetime.now().strftime("%Y_%m_%d")
    file_path = os.path.join(LOGSEQ_GRAPH_PATH, f"{today}.md")
    
    # Ensure the directory exists
    os.makedirs(LOGSEQ_GRAPH_PATH, exist_ok=True)
    
    # Logseq blocks start with a dash '-'
    tag_str = f" {tags}" if tags else ""
    formatted_note = f"\n- # [[Ikaris AI]]{tag_str} {datetime.datetime.now().strftime('%H:%M')}: {content}"
    
    with open(file_path, "a") as f:
        f.write(formatted_note)
        
    return f"Note added to Logseq journal for {today}."

def search_logseq_notes(query: str) -> str:
    """Searches the Logseq 'pages' folder for specific keywords."""
    if not os.path.exists(LOGSEQ_PAGES_PATH):
        return f"Error: Logseq pages directory not found at {LOGSEQ_PAGES_PATH}"
    
    query_terms = query.lower().split()
    results = []
    
    # Iterate through all markdown files in the pages directory
    for filename in os.listdir(LOGSEQ_PAGES_PATH):
        if filename.endswith(".md"):
            file_path = os.path.join(LOGSEQ_PAGES_PATH, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Simple scoring: count how many query terms are found in the file
                    score = sum(1 for term in query_terms if term in content.lower())
                    
                    if score > 0:
                        results.append((score, filename, content))
            except Exception as e:
                continue
    
    if not results:
        return "I checked your Logseq pages, but couldn't find anything relevant."
    
    # Sort by score and take top 2 relative results
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:2]
    
    formatted_results = []
    for score, name, content in top_results:
        # Take a snippet of the content
        snippet = content[:500] + ("..." if len(content) > 500 else "")
        formatted_results.append(f"--- From Page: {name} ---\n{snippet}")
        
    return "\n\n".join(formatted_results)

class LogseqTool:
    def __init__(self, **kwargs):
        self.config = kwargs
        # Optional: could update LOGSEQ_GRAPH_PATH to self.config.get('path') if passed
    
    def add_note(self, content: str, tags: str = ""):
        return add_logseq_note(content, tags)
    
    def search(self, query: str):
        return search_logseq_notes(query)
