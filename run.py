from src.main import ikaris_app
from src.utils.voice import get_voice_input
from transformers import logging

# Silencing Transformers logs
logging.set_verbosity_error()

from src.utils.helpers import get_system_health

def start_ikaris():
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
            inputs = {"messages": [("user", user_msg)]}
            for event in ikaris_app.stream(inputs, config=config):
                for value in event.values():
                    last_msg = value['messages'][-1]
                    
                    # Handling the message object safety
                    if hasattr(last_msg, 'content'):
                        content = last_msg.content
                    else:
                        content = last_msg[1] # If it's a tuple ("role", "content")
                        
                    print(f"Ikaris: {content}")
                    if 'hardware_info' in value:
                        print(f"[System Log: {value['hardware_info']}]")

if __name__ == "__main__":
    start_ikaris()
