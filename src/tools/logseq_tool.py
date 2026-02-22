import datetime
import os
from typing import List

from src.workspaces.workspace_manager import WorkspaceManager

def add_logseq_note(content: str, tags: str = ""):
    """Appends a note as a new block in today's workspace journal."""
    wm = WorkspaceManager()
    notes_dir = wm.get_logseq_dir()
    
    today = datetime.datetime.now().strftime("%Y_%m_%d")
    file_path = os.path.join(notes_dir, f"{today}.md")
    
    os.makedirs(notes_dir, exist_ok=True)
    
    # Logseq blocks start with a dash '-'
    tag_str = f" {tags}" if tags else ""
    formatted_note = f"\n- # [[Ikaris AI]]{tag_str} {datetime.datetime.now().strftime('%H:%M')}: {content}"
    
    with open(file_path, "a") as f:
        f.write(formatted_note)
        
    return f"Note added to {wm.get_active_workspace()} workspace journal for {today}."

def search_logseq_notes(query: str) -> str:
    """Searches the active workspace notes folder for specific keywords."""
    wm = WorkspaceManager()
    notes_dir = wm.get_logseq_dir()
    
    if not os.path.exists(notes_dir):
        return f"Error: Notes directory not found at {notes_dir}"
    
    query_terms = query.lower().split()
    results = []
    
    # Iterate through all markdown files in the notes directory
    for filename in os.listdir(notes_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(notes_dir, filename)
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
