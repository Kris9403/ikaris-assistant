import sqlite3
import json

def list_checkpoints(db_path="ikaris_memory.db", thread_id="krishna_research_session"):
    """Lists all saved states for a specific thread."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the checkpoints table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
    if not cursor.fetchone():
        print("No checkpoints found yet.")
        return
        
    cursor.execute(
        "SELECT checkpoint_id, thread_id, checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC", 
        (thread_id,)
    )
    rows = cursor.fetchall()
    
    if not rows:
        print(f"No history found for thread: {thread_id}")
        return

    print(f"\n--- History for {thread_id} ---")
    print(f"{'ID':<10} | {'Last Message Preview'}")
    print("-" * 50)
    
    for row in rows:
        checkpoint_id = row[0]
        # Checkpoint blob is often serialized
        try:
            # This is a simplified preview logic for LangGraph checkpoints
            print(f"{checkpoint_id:<10} | [State Saved]")
        except:
            print(f"{checkpoint_id:<10} | [Binary Data]")
            
    conn.close()

if __name__ == "__main__":
    list_checkpoints()
