import os
import re
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.evidence import Evidence

log = logging.getLogger(__name__)

# Lazy-load embeddings to prevent import-time side effects
_embeddings_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cpu'}
        )
    return _embeddings_instance

VECTOR_DB_PATH = "faiss_ikaris_index"

def extract_metadata_anchors(text: str) -> dict:
    """Extracts immutable anchors and infers hierarchy for Layer 3 (Graph)."""
    flat_anchors = {
        "sections": re.findall(r'(?:Section|Sec\.?)\s*(\d+(?:\.\d+)*)', text, re.IGNORECASE),
        "equations": re.findall(r'(?:Equation|Eq\.?)\s*\(?(\d+)\)?', text, re.IGNORECASE),
        "figures": re.findall(r'(?:Figure|Fig\.?)\s*(\d+)', text, re.IGNORECASE),
        "tables": re.findall(r'(?:Table|Tab\.?)\s*(\d+)', text, re.IGNORECASE),
    }
    result = {k: sorted(list(set(v))) for k, v in flat_anchors.items() if v}
    
    if "sections" in result:
        parent_sec = result["sections"][0]
        hierarchy = {
            "parent_section": parent_sec,
            "contains_equations": result.get("equations", []),
            "contains_figures": result.get("figures", []),
            "contains_tables": result.get("tables", [])
        }
        result["hierarchy"] = {k: v for k, v in hierarchy.items() if v}
        
    return result

def ingest_papers():
    """Chunks PDFs and creates a local vector store."""
    papers_path = "./papers"
    if not os.path.exists(papers_path):
        os.makedirs(papers_path)
        return "Created 'papers' folder. Please add PDFs and try again."

    loader = DirectoryLoader(papers_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        return "No PDFs found in the 'papers' folder."
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    for doc in texts:
        anchors = extract_metadata_anchors(doc.page_content)
        doc.metadata.update(anchors)
    
    vectorstore = FAISS.from_documents(texts, get_embeddings())
    vectorstore.save_local(VECTOR_DB_PATH)
    return f"Successfully indexed {len(texts)} chunks with Layer 3 metadata anchors."

def query_papers(question: str) -> list:
    """Retrieves relevant text as Evidence objects (Layer 3.5)."""
    if not os.path.exists(VECTOR_DB_PATH):
        ingest_result = ingest_papers()
        if "Successfully indexed" not in ingest_result:
            return [Evidence(source="faiss", id="error", title="Ingest Error", text=ingest_result)]
        
    db = FAISS.load_local(VECTOR_DB_PATH, get_embeddings(), allow_dangerous_deserialization=True)
    docs_and_scores = db.similarity_search_with_score(question, k=5)
    
    evidence = []
    for doc, score in docs_and_scores:
        relevance = 1.0 / (1.0 + score)
        chunk_id = str(hash(doc.page_content))
        
        evidence.append(Evidence(
            source="faiss",
            id=chunk_id,
            title=doc.metadata.get("source", "Unknown PDF"),
            text=doc.page_content,
            relevance=round(relevance, 2),
            meta=doc.metadata,
        ))
    
    log.info(f"[PaperTool] FAISS returned {len(evidence)} evidence chunks.")
    return evidence

class PaperTool:
    name = "paper"
    description = "Search local PDF papers via FAISS"
    capabilities = ["literature_search", "local_papers"]

    def __init__(self, **kwargs):
        self.config = kwargs

    def query(self, question: str) -> list:
        return query_papers(question)

    def ingest(self):
        return ingest_papers()
