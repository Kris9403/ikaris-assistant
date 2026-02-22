
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies before import
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_community.document_loaders'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['langchain_community.vectorstores'] = MagicMock()

from src.tools.paper_tool import extract_metadata_anchors, query_papers

def test_layer_3_extraction():
    print("Testing Layer 3: Metadata Anchor Extraction...")
    
    sample_text = """
    In Section 4.1 we discuss the transformer architecture.
    The loss function is defined in Eq. 3 below:
    L = ...
    As shown in Figure 2, the attention mechanism matches Table 5 results.
    """
    
    anchors = extract_metadata_anchors(sample_text)
    print(f"Extracted Anchors: {anchors}")
    
    expected = {
        'sections': ['4.1'],
        'equations': ['3'],
        'figures': ['2'],
        'tables': ['5']
    }
    
    for k, v in expected.items():
        if anchors.get(k) != v:
            print(f"❌ Failed to extract {k}: Expected {v}, got {anchors.get(k)}")
            return False
            
    print("✅ Layer 3 Extraction Logic Passed!")
    return True

@patch('src.tools.paper_tool.FAISS')
@patch('src.tools.paper_tool.os.path.exists')
def test_layer_3_5_retrieval(mock_exists, mock_faiss):
    print("\nTesting Layer 3.5: Structured Evidence Retrieval...")
    
    # Mock index existence
    mock_exists.return_value = True
    
    # Mock DB and docs
    mock_db = MagicMock()
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Content chunk 1"
    mock_doc1.metadata = {"page": 1, "sections": ["2"]}
    
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Content chunk 2"
    mock_doc2.metadata = {"page": 5, "equations": ["4"]}
    
    mock_db.similarity_search.return_value = [mock_doc1, mock_doc2]
    mock_faiss.load_local.return_value = mock_db
    
    # Run query
    results = query_papers("test query")
    
    if not isinstance(results, list):
        print(f"❌ query_papers did NOT return a list. Type: {type(results)}")
        return False
        
    if len(results) != 2:
        print(f"❌ Expected 2 evidence packets, got {len(results)}")
        return False
        
    if results[0]['metadata']['sections'] != ['2']:
        print("❌ Metadata not preserved in packet 1")
        return False
        
    print("✅ Layer 3.5 Structured Retrieval Passed!")
    return True

if __name__ == "__main__":
    if test_layer_3_extraction() and test_layer_3_5_retrieval():
        sys.exit(0)
    sys.exit(1)
