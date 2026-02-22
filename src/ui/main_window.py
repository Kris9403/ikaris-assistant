import os
import sys
os.environ["QT_QPA_PLATFORM"] = "xcb"

import re
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QApplication, QMessageBox
)
from PyQt5.QtCore import Qt
from src.ui.chat_widget import ChatWidget
from src.ui.sidebar_widget import SidebarWidget
from src.ui.status_bar import StatusBarWidget
from src.ui.workers import GraphWorker, LLMWorker, IndexWorker, VoiceWorker
from src.main import router_logic


class IkarisMainWindow(QMainWindow):
    """
    Main application window for Ikaris Assistant.
    Dark-themed researcher UI with chat, paper sidebar, and system status.
    """

    def __init__(self, agent=None):
        super().__init__()
        self.setWindowTitle("Ikaris Assistant 🦾")
        self.setMinimumSize(1000, 650)
        self.resize(1200, 750)

        self.agent = agent
        self.config = {"configurable": {"thread_id": "krishna_research_session"}}
        
        # Keep explicit references to workers to prevent "QThread destroyed" crashes
        self._current_worker = None
        self._index_worker = None 
        self._voice_worker = None

        self.ikaris_app = agent.app if agent else None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter: sidebar | chat
        splitter = QSplitter(Qt.Horizontal)

        self.sidebar = SidebarWidget()
        splitter.addWidget(self.sidebar)

        self.chat = ChatWidget()
        splitter.addWidget(self.chat)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Status bar at bottom
        self.status_bar = StatusBarWidget()
        main_layout.addWidget(self.status_bar)

    def _connect_signals(self):
        self.chat.message_sent.connect(self._on_message)
        self.chat.voice_requested.connect(self._on_voice)
        self.sidebar.index_btn.clicked.connect(self._on_index)

    # ── Message Handling ──

    def _on_message(self, text):
        """Handle user message: route to streaming LLM or graph worker."""
        # SAFETY CHECK: Don't start if a worker is already running
        if self._current_worker and self._current_worker.isRunning():
            self.chat.add_system_message("⚠️ Please wait for the current task to finish.")
            return

        self.chat.add_user_message(text)
        self.chat.set_input_enabled(False)

        # Check which route the message would take
        msg_lower = text.lower()
        
        # Hardware, research, paper, logseq, pubmed → use full graph (non-streaming)
        needs_graph = (
            any(w in msg_lower for w in ["battery", "cpu", "stats", "hardware"]) or
            any(w in msg_lower for w in ["arxiv.org", "download", "fetch"]) or
            any(w in msg_lower for w in ["pubmed", "pmid"]) or
            bool(re.findall(r'\d{4}\.\d{4,5}', msg_lower)) or
            any(w in msg_lower for w in ["paper", "research", "study", "according to"]) or
            any(w in msg_lower for w in ["note", "notes", "logseq", "journal", "diary"])
        )

        if needs_graph:
            self._run_graph(text)
        else:
            self._run_streaming(text)

    def _run_streaming(self, text):
        """Stream LLM tokens for general chat."""
        self.chat.start_ai_message()

        # Import platform to get real OS details
        import platform
        os_info = f"{platform.system()} {platform.release()}"

        system_prompt = (
            "You are Ikaris, a highly technical research assistant for a Computer Science Master's student. "
            f"You are running locally on a ROG Strix G16 (RTX 5070 Ti, 32GB RAM) hosted on {os_info}. "
            "Your tone is professional, expert, yet grounded and slightly witty. "
            "Focus on delivering clear, actionable research insights and system stats analysis."
        )

        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=text)]

        self._current_worker = LLMWorker(self.agent.llm, messages, system_prompt)
        self._current_worker.token_received.connect(self.chat.append_token)
        self._current_worker.finished_signal.connect(self._on_stream_done)
        self._current_worker.error_signal.connect(self._on_error)
        self._current_worker.start()

    def _run_graph(self, text):
        """Run the full LangGraph pipeline for specialized nodes."""
        self.chat.add_system_message("⚡ Processing...")

        self._current_worker = GraphWorker(self.ikaris_app, text, self.config)
        self._current_worker.result_ready.connect(self._on_graph_result)
        self._current_worker.error_signal.connect(self._on_error)
        self._current_worker.start()

    def _on_stream_done(self):
        self.chat.finish_ai_message()
        self.chat.set_input_enabled(True)
        # We don't set self._current_worker = None here immediately to avoid race conditions,
        # but relying on isRunning() check is safer.

    def _on_graph_result(self, content):
        self.chat.add_ai_message(content)
        self.chat.set_input_enabled(True)

    def _on_error(self, error_msg):
        self.chat.add_system_message(f"❌ Error: {error_msg}")
        self.chat.set_input_enabled(True)

    # ── Voice Input ──

    def _on_voice(self):
        if self._current_worker and self._current_worker.isRunning():
            self.chat.add_system_message("⚠️ I'm busy thinking. Please wait.")
            return
        if self._voice_worker and self._voice_worker.isRunning():
            self.chat.add_system_message("⚠️ Already listening. Please wait.")
            return

        # Check if audio stack supports speech input
        audio = self.agent.audio if self.agent else None
        if audio is None or not getattr(audio, 'has_stt', False):
            self.chat.add_system_message("⚠️ Audio is disabled. Use text input (run with audio=npu or audio=cpu).")
            return

        self.chat.set_input_enabled(False)
        vad_tag = ' (VAD-gated)' if getattr(audio, 'has_vad', False) else ''
        self.chat.add_system_message(f"🎤 Listening{vad_tag}...")

        self._voice_worker = VoiceWorker(audio)
        self._voice_worker.partial_text.connect(self._on_voice_partial)
        self._voice_worker.finished_signal.connect(self._on_voice_done)
        self._voice_worker.error_signal.connect(self._on_voice_error)
        self._voice_worker.start()

    def _on_voice_partial(self, partial_text):
        """Update the chat with live partial transcription (feels magical)."""
        self.chat.add_system_message(f'🎤 ... {partial_text}')

    def _on_voice_done(self, text, confidence, provider):
        """Handle completed voice transcription with confidence scoring."""
        # Confidence badge
        if confidence >= 0.7:
            badge = f'🟢 {confidence:.0%}'
        elif confidence >= 0.4:
            badge = f'🟡 {confidence:.0%}'
        else:
            badge = f'🔴 {confidence:.0%}'

        # Provider indicator (shows auto-switch if fallback was used)
        provider_tag = f' [{provider}]' if provider != self.agent.audio.provider else ''
        if provider_tag:
            provider_tag = f' ⚡ [auto-switched → {provider}]'

        self.chat.add_system_message(f'Heard ({badge}{provider_tag}): "{text}"')
        self._on_message(text)

    def _on_voice_error(self, error_msg):
        """Handle voice input errors."""
        self.chat.add_system_message(f'⚠️ {error_msg}')
        self.chat.set_input_enabled(True)

    # ── PDF Indexing ──

    def _on_index(self):
        if self._current_worker and self._current_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Please wait for the current chat to finish.")
            return

        # LOCK THE UI: Cannot chat while indexing or the DB will crash
        self.chat.set_input_enabled(False)
        self.sidebar.set_indexing(True)
        self.chat.add_system_message("📚 Background PDF indexing started... (Chat Locked)")

        self._index_worker = IndexWorker()
        self._index_worker.progress.connect(self._on_index_progress)
        self._index_worker.finished_signal.connect(self._on_index_done)
        self._index_worker.error_signal.connect(self._on_index_error)
        self._index_worker.start()

    def _on_index_progress(self, msg):
        self.sidebar.set_status(msg)

    def _on_index_done(self, result):
        self.sidebar.set_indexing(False)
        self.sidebar.set_status(result)
        self.sidebar.refresh_papers()
        self.chat.add_system_message(f"✅ {result}")
        # UNLOCK UI
        self.chat.set_input_enabled(True)

    def _on_index_error(self, error):
        self.sidebar.set_indexing(False)
        self.sidebar.set_status(f"Error: {error}")
        self.chat.add_system_message(f"❌ Indexing error: {error}")
        # UNLOCK UI
        self.chat.set_input_enabled(True)
