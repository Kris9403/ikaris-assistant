import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize local embeddings (Explicitly on CPU to save GPU VRAM for LLM)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cpu'}
)
VECTOR_DB_PATH = "faiss_ikaris_index"

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
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    return f"Successfully indexed {len(texts)} chunks from your research papers."

def query_papers(question: str) -> str:
    """Retrieves relevant text from the PDFs."""
    if not os.path.exists(VECTOR_DB_PATH):
        # Automatically ingest if index doesn't exist
        ingest_result = ingest_papers()
        if "Successfully indexed" not in ingest_result:
            return ingest_result
        
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question, k=3)
    
    return "\n\n".join([doc.page_content for doc in docs])
