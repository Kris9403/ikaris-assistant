
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock key components before importing main
with patch('src.utils.llm_client.llm_instance') as mock_llm:
    # Setup mock returns
    mock_response = MagicMock()
    mock_response.content = "I am Ikaris. System is operational."
    mock_llm.invoke.return_value = mock_response
    
    # Mock stream to yield a chunk
    mock_chunk = MagicMock()
    mock_chunk.content = "I am Ikaris."
    mock_llm.stream.return_value = [mock_chunk]

    # Import the app after mocking
    from src.agent import get_ikaris_app
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

    def test_run_cli_logic():
        ikaris_app = get_ikaris_app()
        print("Testing CLI/Worker Input Logic...")
        
        user_msg = "Hello Ikaris"
        
        # This matches the FIX in run_cli.py and workers.py
        inputs = {
            "messages": [HumanMessage(content=user_msg)],
            "hardware_info": "",
            "summary": ""
        }
        
        print(f"Input State: {inputs}")
        
        # Run the graph
        try:
            for event in ikaris_app.stream(inputs, config={"configurable": {"thread_id": "test_thread"}}):
                for node_name, value in event.items():
                    print(f"Node '{node_name}' returned: {value.keys()}")
                    
                    # Verify 'messages' is always returned if it's the summarizer or any other node expected to
                    if "messages" in value:
                        messages = value["messages"]
                        for m in messages:
                            if not isinstance(m, BaseMessage):
                                print(f"❌ ERROR: Node '{node_name}' returned non-BaseMessage: {type(m)} -> {m}")
                                return False
                            print(f"✅ Node '{node_name}' returned valid BaseMessage: {type(m).__name__}")
        except Exception as e:
            print(f"❌ CRITICAL ERROR during graph execution: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        print("\n✅ CLI/Worker Graph Execution Test Passed!")
        return True

    if __name__ == "__main__":
        success = test_run_cli_logic()
        if not success:
            sys.exit(1)
        sys.exit(0)
