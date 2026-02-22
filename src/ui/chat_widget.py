from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QLineEdit, QPushButton, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
import datetime


class ChatWidget(QWidget):
    """
    Chat panel with message display and input bar.
    Supports streaming token display.
    """
    message_sent = pyqtSignal(str)
    voice_requested = pyqtSignal()
    voice_stopped = pyqtSignal()
    citation_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header = QLabel("💬 Ikaris Chat")
        header.setObjectName("sidebarTitle")
        layout.addWidget(header)

        # Chat display
        self.chat_display = QTextBrowser()
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setOpenExternalLinks(False)
        self.chat_display.anchorClicked.connect(self._on_anchor_clicked)
        self.chat_display.setFont(QFont("Inter", 13))
        layout.addWidget(self.chat_display)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.voice_btn = QPushButton("🎤 Hold to Talk")
        self.voice_btn.setObjectName("voiceBtn")
        self.voice_btn.setToolTip("Push and hold to record voice")
        
        # NotebookLM style push-to-talk UX
        self.voice_btn.pressed.connect(self.voice_requested.emit)
        self.voice_btn.released.connect(self.voice_stopped.emit)
        
        input_row.addWidget(self.voice_btn)
        
        # Add the text input and send button after the mic button
        self.input_field = QLineEdit()
        self.input_field.setObjectName("chatInput")
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.returnPressed.connect(self._on_send)
        input_row.addWidget(self.input_field)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setFixedWidth(80)
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.send_btn)

        layout.addLayout(input_row)

    def _on_send(self):
        text = self.input_field.text().strip()
        if text:
            self.input_field.clear()
            self.message_sent.emit(text)

    def _on_anchor_clicked(self, url):
        """Intercept clicked links natively inside the chat display."""
        if url.scheme() == "evidence":
            try:
                idx = int(url.path())
                self.citation_clicked.emit(idx)
            except ValueError:
                pass

    def add_user_message(self, text):
        """Append a user message bubble."""
        timestamp = datetime.datetime.now().strftime("%H:%M")
        self.chat_display.append(
            f'<p style="color:#7aa2f7; margin:4px 0 0 0;">'
            f'<b>You</b> <span style="color:#565f89; font-size:11px;">{timestamp}</span></p>'
            f'<p style="color:#c0caf5; margin:0 0 8px 12px;">{text}</p>'
        )
        self._scroll_to_bottom()

    def start_ai_message(self):
        """Start a new AI message block for streaming."""
        timestamp = datetime.datetime.now().strftime("%H:%M")
        self.chat_display.append(
            f'<p style="color:#9ece6a; margin:4px 0 0 0;">'
            f'<b>Ikaris</b> <span style="color:#565f89; font-size:11px;">{timestamp}</span></p>'
            f'<p style="color:#c0caf5; margin:0 0 0 12px;" id="streaming">'
        )
        self._scroll_to_bottom()

    def append_token(self, token):
        """Append a single token to the current streaming message."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.chat_display.setTextCursor(cursor)
        self._scroll_to_bottom()

    def finish_ai_message(self):
        """Close the streaming message block."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n")
        self.chat_display.setTextCursor(cursor)
        self._scroll_to_bottom()

    def add_ai_message(self, text):
        """Add a complete AI message (non-streaming) with interactive links."""
        import re
        timestamp = datetime.datetime.now().strftime("%H:%M")
        
        # Parse inline [Evidence X] to HTML links
        html_text = text.replace('\n', '<br>')
        html_text = re.sub(
            r'\[(?:Evidence\s*)?(\d+)\]',
            r'<a href="evidence:\1" style="color:#7aa2f7; text-decoration:none;"><b>[\1]</b></a>',
            html_text,
            flags=re.IGNORECASE
        )
        
        self.chat_display.append(
            f'<p style="color:#9ece6a; margin:4px 0 0 0;">'
            f'<b>Ikaris</b> <span style="color:#565f89; font-size:11px;">{timestamp}</span></p>'
            f'<p style="color:#c0caf5; margin:0 0 8px 12px;">{html_text}</p>'
        )
        self._scroll_to_bottom()

    def add_system_message(self, text):
        """Add a system/status message."""
        self.chat_display.append(
            f'<p style="color:#565f89; font-style:italic; margin:2px 0;">{text}</p>'
        )
        self._scroll_to_bottom()

    def remove_message(self, text):
        """Finds and removes exactly matching text blocks (used for clearing spinners)."""
        document = self.chat_display.document()
        cursor = document.find(text)
        if not cursor.isNull():
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()

    def set_input_enabled(self, enabled):
        """Enable/disable input during processing."""
        self.input_field.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.voice_btn.setEnabled(enabled)
        if enabled:
            self.input_field.setFocus()

    def _scroll_to_bottom(self):
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
