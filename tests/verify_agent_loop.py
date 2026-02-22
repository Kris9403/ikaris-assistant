
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_community.document_loaders'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['langchain_community.vectorstores'] = MagicMock()

# Mock retrieval_tool to avoid disk I/O
with patch("src.nodes.retrieval_node.query_papers") as mock_query:
    mock_query.return_value = [{"content": "Evidence 1", "metadata": {"sections": ["1"]}}]
    
    # Mock LLM for reasoning node
    from src.utils.llm_client import llm_instance
    mock_llm_response = MagicMock()
    # First response: Low confidence -> Loop
    # Second response: High confidence -> Answer
    mock_llm_response.content = '{"confidence": 0.5, "open_questions": ["More info?"]}'
    
    llm_instance.invoke = MagicMock(side_effect=[
        MagicMock(content='{"confidence": 0.5, "open_questions": ["Q2"]}'), # Reasoning 1
        MagicMock(content="Final Answer"), # Answer Node (if called, but our loop might verify confidence state first)
        MagicMock(content="Final Answer")
    ])

    from src.main import reasoning_router
    from src.state import IkarisState

    def test_reasoning_router():
        print("Testing Reasoning Router Logic...")
        
        # Case 1: Low Confidence -> Loop
        state_low = {"confidence": 0.5}
        next_node = reasoning_router(state_low)
        print(f"Confidence 0.5 -> Next Node: {next_node}")
        if next_node != "retrieval_node":
            print("❌ Failed: Should route to retrieval_node")
            return False
            
        # Case 2: High Confidence -> Answer
        state_high = {"confidence": 0.9}
        next_node = reasoning_router(state_high)
        print(f"Confidence 0.9 -> Next Node: {next_node}")
        if next_node != "generate_answer_node":
            print("❌ Failed: Should route to generate_answer_node")
            return False
            
        print("✅ Reasoning Router Logic Passed!")
        return True

    if __name__ == "__main__":
        if test_reasoning_router():
            sys.exit(0)
        sys.exit(1)
