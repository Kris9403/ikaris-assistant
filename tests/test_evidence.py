import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock

# Mock Dependencies
# We use patch instead of global sys.modules hacks


class TestLayer3_5Evidence(unittest.TestCase):
    """Layer 3.5: Evidence Assembly Tests"""

    def test_assemble_evidence_structure(self):
        """Test that retrieval returns formatted evidence packets."""
        mock_db = MagicMock()
        
        # Mock Doc
        mock_doc = MagicMock()
        mock_doc.page_content = "Raw extracted text."
        mock_doc.metadata = {"sections": ["2.1"], "equations": ["4"]}
        
        # Return doc with score 0.0 (perfect match)
        mock_db.similarity_search_with_score.return_value = [(mock_doc, 0.0)]
        
        with patch('src.tools.paper_tool.FAISS.load_local', return_value=mock_db), \
             patch('src.tools.paper_tool.get_embeddings'), \
             patch('os.path.exists', return_value=True):
            from src.tools.paper_tool import query_papers
            evidence = query_papers("query")
            
        pkt = evidence[0]
        
        # Assert Structure
        self.assertIn("content", pkt)
        self.assertIn("metadata", pkt)
        self.assertIn("relevance", pkt)
        
        # Assert Logic
        self.assertEqual(pkt['content'], "Raw extracted text.")
        self.assertEqual(pkt['metadata']['sections'], ["2.1"])
        self.assertEqual(pkt['relevance'], 1.0) # 1 / (1+0)

    def test_relevance_normalization(self):
        """Test relevance score calculation."""
        # Using the same mock setup
        mock_db = MagicMock()
        
        mock_doc = MagicMock()
        mock_doc.page_content = "X"
        mock_doc.metadata = {}
        
        # Score 1.0 (Distance = 1.0) -> Relevance = 1/(1+1) = 0.5
        mock_db.similarity_search_with_score.return_value = [(mock_doc, 1.0)]
        
        with patch('src.tools.paper_tool.FAISS.load_local', return_value=mock_db), \
             patch('src.tools.paper_tool.get_embeddings'), \
             patch('os.path.exists', return_value=True):
            from src.tools.paper_tool import query_papers
            evidence = query_papers("query")
                    
        self.assertEqual(evidence[0]['relevance'], 0.5)

if __name__ == '__main__':
    unittest.main()
