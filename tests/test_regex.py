import unittest
import re
import sys

# Since this test copies logic for verification, we can just keep it as a standalone unit test
# or import the real function if we want to test the source. 
# Given the user instruction, we should test REAL logic.
# So let's import the real regex function if possible, or keep this as a "golden logic" test?
# The user said "Fix regex test to not return bool". 
# The file content shows it IS a copy.
# Let's just fix the return bool for now to satisfy pytest discovery if it runs this file.

# A better approach: This file duplicates logic from paper_tool.py.
# We already have test_anchors.py testing the real logic.
# This file seems redundant or a legacy verification script.
# I will convert it to a proper test that imports the real function, 
# effectively merging it with test_anchors.py's purpose, but since it exists, let's make it pass.

from src.tools.paper_tool import extract_metadata_anchors

class TestRegexLogic(unittest.TestCase):
    def test_layer_3_extraction(self):
        sample_text = """
        In Section 4.1 we discuss the transformer architecture.
        The loss function is defined in Eq. 3 below:
        L = ...
        As shown in Figure 2, the attention mechanism matches Table 5 results.
        """
        
        anchors = extract_metadata_anchors(sample_text)
        
        expected = {
            'sections': ['4.1'],
            'equations': ['3'],
            'figures': ['2'],
            'tables': ['5']
        }
        
        for k, v in expected.items():
            self.assertEqual(anchors.get(k), v, f"Failed to extract {k}")

if __name__ == "__main__":
    unittest.main()
