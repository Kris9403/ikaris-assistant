import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock

# Mock Dependencies
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_community.document_loaders'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['langchain_community.vectorstores'] = MagicMock()

class TestLayer2Embeddings(unittest.TestCase):
    """Layer 2: Semantic Index Tests"""

    @patch('src.tools.paper_tool.FAISS.load_local')
    @patch('src.tools.paper_tool.get_embeddings')
    def test_vector_search_returns_original_text(self, mock_get_embeddings, mock_load_local):
        """Verify that searching the vector store returns the correct document objects."""
        # Mock FAISS Index
        mock_db = MagicMock()
        mock_load_local.return_value = mock_db
        
        # Mock Search Results
        mock_doc = MagicMock()
        mock_doc.page_content = "Transformers use self-attention mechanisms."
        mock_doc.metadata = {"page": 1}
        
        # Setup similarity_search_with_score returns (doc, score) tuple
        # Score 0.1 means very close
        mock_db.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
        
        from src.tools.paper_tool import query_papers
        
        # We need to ensure we don't trigger ingest
        with patch('os.path.exists', return_value=True):
             results = query_papers("attention")
             
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertIn("Transformers use self-attention", results[0]['content'])
        self.assertIn("relevance", results[0])
        # 1 / (1 + 0.1) = 0.91
        self.assertGreater(results[0]['relevance'], 0.90)

if __name__ == '__main__':
    unittest.main()
