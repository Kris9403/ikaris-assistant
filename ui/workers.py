import os
from PyQt5.QtCore import QThread, pyqtSignal
from src.utils.llm_client import llm_instance, stream_lm_studio
from src.tools.paper_tool import ingest_papers
from langchain_core.messages import SystemMessage, HumanMessage


class LLMWorker(QThread):
    """
    Runs LLM streaming in a background thread.
    Emits tokens one at a time for real-time display.
    """
    token_received = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, messages, system_prompt, parent=None):
        super().__init__(parent)
        self.messages = messages
        self.system_prompt = system_prompt

    def run(self):
        try:
            for token in stream_lm_studio(self.messages, self.system_prompt):
                self.token_received.emit(token)
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))


class GraphWorker(QThread):
    """
    Runs the full LangGraph pipeline in a background thread.
    Used for non-streaming nodes (hardware, research, paper, logseq).
    """
    result_ready = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, ikaris_app, user_msg, config, parent=None):
        super().__init__(parent)
        self.ikaris_app = ikaris_app
        self.user_msg = user_msg
        self.config = config

    def run(self):
        try:
            inputs = {
                "messages": [HumanMessage(content=self.user_msg)],
                "loop_count": 0, # Initialize safety counter
            }
            
            for event in self.ikaris_app.stream(inputs, config=self.config):
                for node_name, value in event.items():
                    # FILTER: Only show output from the final answer node 
                    # OR specific tool nodes (like hardware/research)
                    if node_name in ["generate_answer_node", "hardware_node", "research_node", "logseq_node"]:
                        
                        messages = value.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                            self.result_ready.emit(content)
                            
        except Exception as e:
            self.error_signal.emit(str(e))


class IndexWorker(QThread):
    """
    Runs PDF ingestion in the background.
    Emits progress updates and completion signal.
    """
    progress = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def run(self):
        try:
            self.progress.emit("Scanning papers directory...")
            papers_path = "./papers"
            
            if not os.path.exists(papers_path):
                os.makedirs(papers_path)
                self.finished_signal.emit("Created 'papers' folder. Add PDFs and try again.")
                return

            pdf_files = [f for f in os.listdir(papers_path) if f.endswith('.pdf')]
            if not pdf_files:
                self.finished_signal.emit("No PDFs found in the 'papers' folder.")
                return

            self.progress.emit(f"Found {len(pdf_files)} PDFs. Indexing...")
            result = ingest_papers()
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(f"Indexing error: {str(e)}")
