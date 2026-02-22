import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock

# We verify purely the regex logic, so we rely on real imports.
from src.tools.paper_tool import extract_metadata_anchors

class TestLayer3Anchors(unittest.TestCase):
    """Layer 3: Graph Pointers Tests"""

    def test_anchor_extraction_sections(self):
        text = "This is described in Section 4.5 and Sec. 10.2 later."
        result = extract_metadata_anchors(text)
        self.assertEqual(result['sections'], ['10.2', '4.5'])

    def test_anchor_extraction_equations(self):
        text = "As seen in Equation 3 and Eq. (5)."
        result = extract_metadata_anchors(text)
        self.assertEqual(result['equations'], ['3', '5'])

    def test_anchor_extraction_figures(self):
        text = "Figure 1 shows the architecture. Fig. 2 details the loop."
        result = extract_metadata_anchors(text)
        self.assertEqual(result['figures'], ['1', '2'])
        
    def test_anchor_extraction_tables(self):
        text = "Table 4 lists results."
        result = extract_metadata_anchors(text)
        self.assertEqual(result['tables'], ['4'])

    def test_hierarchy_inference(self):
        """Test that equations are linked to the parent section in the chunk."""
        text = "Section 3. Methodology. We define Equation 5 here."
        result = extract_metadata_anchors(text)
        
        self.assertIn('hierarchy', result)
        self.assertEqual(result['hierarchy']['parent_section'], '3')
        self.assertEqual(result['hierarchy']['contains_equations'], ['5'])

    def test_multiple_anchors_same_chunk(self):
        text = "Section 2. Eq 1. Fig 3. Table 9."
        result = extract_metadata_anchors(text)
        self.assertEqual(result['sections'], ['2'])
        self.assertEqual(result['equations'], ['1'])
        self.assertEqual(result['figures'], ['3'])
        self.assertEqual(result['tables'], ['9'])

if __name__ == '__main__':
    unittest.main()
