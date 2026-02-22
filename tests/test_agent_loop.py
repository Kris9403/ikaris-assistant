import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module under test safely
# We use patch to mock external dependencies if they are heavy or require credentials
# But since src.main is safe to import, we can just import the functions we need.
# However, if we want to isolate reasoning_router from its dependencies? 
# reasoning_router only depends on 'state' dict. It has NO external dependencies.
# So we can just import it.

from src.main import reasoning_router

class TestAgentLoop(unittest.TestCase):
    """Agent Loop Behavior Tests"""

    def test_agent_loops_on_low_confidence(self):
        """Test that low confidence routes back to retrieval."""
        state = {"confidence": 0.5, "messages": []}
        next_node = reasoning_router(state)
        self.assertEqual(next_node, "retrieval_node")

    def test_agent_answers_on_high_confidence(self):
        """Test that high confidence routes to answer."""
        state = {"confidence": 0.9, "messages": []}
        next_node = reasoning_router(state)
        self.assertEqual(next_node, "generate_answer_node")

    def test_reasoning_node_output_parsing(self):
        """Test that reasoning node correctly parses LLM JSON."""
        # Mock state
        state = {"goal": "test", "evidence": []}
        
        # Mock LLM response
        mock_response = MagicMock()
        # The code expects `response.content`
        mock_response.content = '{"confidence": 0.7, "open_questions": ["Why?"]}'
        
        # We patch the `llm_instance` imported inside `reasoning_node.py`.
        # This replaces the Pydantic model instance with a MagicMock entirely.
        with patch('src.nodes.reasoning_node.llm_instance') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            
            from src.nodes.reasoning_node import reasoning_node
            
            result = reasoning_node(state)
            
            self.assertEqual(result['confidence'], 0.7)
            self.assertEqual(result['open_questions'], ["Why?"])

if __name__ == '__main__':
    unittest.main()
