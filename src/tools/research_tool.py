import re
import arxiv
import os
from langchain_community.document_loaders import ArxivLoader

def fetch_multi_papers(input_text: str):
    """Finds all arXiv IDs in text and downloads them."""
    # Regex to find ArXiv IDs (e.g., 1706.03762 or 2307.09288)
    arxiv_id_pattern = r'\d{4}\.\d{4,5}'
    paper_ids = re.findall(arxiv_id_pattern, input_text)
    
    if not paper_ids:
        return "I couldn't find any ArXiv IDs in that message."

    results = []
    download_path = "./papers"
    os.makedirs(download_path, exist_ok=True)
    client = arxiv.Client()

    for p_id in paper_ids:
        try:
            # Metadata lookup
            search = arxiv.Search(id_list=[p_id])
            paper_meta = next(client.results(search))
            
            # Clean title
            clean_title = "".join([c if c.isalnum() else "_" for c in paper_meta.title])
            file_path = os.path.join(download_path, f"{clean_title}.pdf")

            if os.path.exists(file_path):
                results.append(f"Skipped (Exists): {clean_title}")
                continue

            # Download physical PDF
            paper_meta.download_pdf(dirpath=download_path, filename=f"{clean_title}.pdf")
            results.append({
                "title": clean_title,
                "summary": paper_meta.summary,
                "path": file_path
            })
        except Exception as e:
            results.append(f"Error downloading {p_id}: {str(e)}")
            
    return results

def fetch_and_save_paper(query: str):
    """Legacy single paper fetch (fallback/simple queries)."""
    # 1. Load metadata via LangChain
    loader = ArxivLoader(query=query, load_max_docs=1)
    docs = loader.load()
    
    if not docs:
        return "I couldn't find that paper on arXiv."

    paper = docs[0]
    # Use a safe get for the title
    raw_title = paper.metadata.get('Title', 'Untitled_Paper')
    clean_title = "".join([c if c.isalnum() else "_" for c in raw_title])
    
    # 2. Extract ID safely for the downloader
    paper_id = query.split('/')[-1] if '/' in query else query

    try:
        # Use the arxiv client directly for the download
        search = arxiv.Search(id_list=[paper_id])
        client = arxiv.Client()
        p = next(client.results(search))
        
        download_path = "./papers"
        os.makedirs(download_path, exist_ok=True)
        
        filename = f"{clean_title}.pdf"
        p.download_pdf(dirpath=download_path, filename=filename)
        
        return {
            "title": clean_title,
            "summary": paper.metadata.get('Summary', 'No summary available.'),
            "path": os.path.join(download_path, filename)
        }
    except Exception as e:
        return f"Arxiv Download Error: {str(e)}"
