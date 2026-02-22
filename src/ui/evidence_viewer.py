from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser, QPushButton
from PyQt5.QtCore import Qt

class EvidenceViewer(QDialog):
    """
    Popup dialog that displays the raw source text and metadata 
    of a citation clicked within the Chat UI.
    """
    def __init__(self, evidence_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Citation Evidence")
        self.setMinimumSize(600, 450)
        self.setStyleSheet("background-color: #1a1b26; color: #c0caf5;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Determine Title
        title_text = evidence_data.title if hasattr(evidence_data, "title") else evidence_data.get("title", "Unknown Source")
        source = evidence_data.source if hasattr(evidence_data, "source") else evidence_data.get("source", "unknown")
        
        header = QLabel(f"<b>{title_text}</b>")
        header.setWordWrap(True)
        header.setStyleSheet("color: #7aa2f7; font-size: 16px;")
        layout.addWidget(header)
        
        meta_label = QLabel(f"Source: {str(source).upper()}")
        meta_label.setStyleSheet("color: #565f89; font-size: 13px; font-weight: bold;")
        layout.addWidget(meta_label)
        
        # Meta dictionary
        meta = evidence_data.meta if hasattr(evidence_data, "meta") else evidence_data.get("meta", {})
        if meta:
            meta_str = " | ".join(f"{k}: {v}" for k,v in meta.items() if v)
            if meta_str:
                meta_details = QLabel(meta_str)
                meta_details.setWordWrap(True)
                meta_details.setStyleSheet("color: #565f89; font-size: 11px;")
                layout.addWidget(meta_details)
        
        # Content body
        content = QTextBrowser()
        content.setStyleSheet("background-color: #24283b; padding: 12px; border-radius: 6px; font-family: Inter; font-size: 13px;")
        
        raw_text = evidence_data.text if hasattr(evidence_data, "text") else evidence_data.get("text", "")
        content.setPlainText(raw_text)
        layout.addWidget(content)
        
        # Footer
        footer = QHBoxLayout()
        footer.addStretch()
        
        close_btn = QPushButton("Close Viewer")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #414868; 
                padding: 8px 16px; 
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #565f89;
            }
        """)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        
        layout.addLayout(footer)
