from src.agent import get_ikaris_app
from src.utils.voice import get_voice_input
from transformers import logging

# Silencing Transformers logs
logging.set_verbosity_error()

from langchain_core.messages import HumanMessage
from src.utils.helpers import get_system_health

def start_ikaris():
    ikaris_app = get_ikaris_app()
    health = get_system_health()
    print(f"--- Ikaris OS Initialized [{health['status']}] ---")
    print(f"Stats: VRAM {health['vram_used']}MB/{health['vram_total']}MB | RAM {health['ram_percent']}%")
    print("Tip: Press 'T' to Type, 'V' for Voice command, or 'exit' to quit.")
    
    config = {"configurable": {"thread_id": "krishna_research_session"}}
    
    while True:
        mode = input("\nUser (Mode T/V): ").lower()
        
        if mode == 'exit':
            print("Shutting down Ikaris...")
            break
            
        if mode == 'v':
            user_msg = get_voice_input()
            print(f"You said: {user_msg}")
        elif mode == 't':
            user_msg = input("Type your message: ")
        else:
            # Fallback to direct text if not T/V
            user_msg = mode
            
        if user_msg:
            # FIX: Use HumanMessage and initialize FULL state
            inputs = {
                "messages": [HumanMessage(content=user_msg)],
                "hardware_info": "",
                "summary": "",
                "loop_count": 0
            }
            
            for event in ikaris_app.stream(inputs, config=config):
                for value in event.values():
                    # Handle both single message or list of messages if returned
                    messages_val = value.get('messages', [])
                    if isinstance(messages_val, list) and messages_val:
                        last_msg = messages_val[-1]
                    elif isinstance(messages_val, list):
                        continue # Empty list
                    else:
                        last_msg = messages_val
                    
                    # Handling the message object safety
                    if hasattr(last_msg, 'content'):
                        content = last_msg.content
                    else:
                        # Fallback for any legacy tuple returns (though we are fixing them)
                        content = str(last_msg)
                        
                    print(f"Ikaris: {content}")
                    if 'hardware_info' in value:
                        print(f"[System Log: {value['hardware_info']}]")

if __name__ == "__main__":
    start_ikaris()
