import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QProgressBar
)
from PyQt5.QtCore import Qt


class SidebarWidget(QWidget):
    """
    Sidebar showing paper list, indexing controls, and status.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(260)
        self._setup_ui()
        self.refresh_papers()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(4)

        # Title
        title = QLabel("📚 Research Papers")
        title.setObjectName("sidebarTitle")
        layout.addWidget(title)

        # Paper list
        self.paper_list = QListWidget()
        layout.addWidget(self.paper_list)

        # Index section
        section = QLabel("INDEX CONTROL")
        section.setObjectName("sectionHeader")
        layout.addWidget(section)

        self.index_btn = QPushButton("⚡ Re-Index Papers")
        self.index_btn.setObjectName("indexBtn")
        layout.addWidget(self.index_btn)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def refresh_papers(self):
        """Scan the papers directory and populate the list."""
        self.paper_list.clear()
        papers_path = "./papers"
        
        if not os.path.exists(papers_path):
            os.makedirs(papers_path, exist_ok=True)

        pdf_files = sorted([f for f in os.listdir(papers_path) if f.endswith('.pdf')])
        
        if not pdf_files:
            item = QListWidgetItem("No papers yet. Add PDFs to ./papers/")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.paper_list.addItem(item)
        else:
            for pdf in pdf_files:
                # Clean display name
                display = pdf.replace("_", " ").replace(".pdf", "")
                if len(display) > 35:
                    display = display[:32] + "..."
                item = QListWidgetItem(f"📄 {display}")
                item.setToolTip(pdf)
                self.paper_list.addItem(item)

    def set_indexing(self, active):
        """Show/hide indexing progress."""
        if active:
            self.progress_bar.show()
            self.index_btn.setEnabled(False)
            self.status_label.setText("Indexing in progress...")
        else:
            self.progress_bar.hide()
            self.index_btn.setEnabled(True)

    def set_status(self, text):
        """Update the sidebar status label."""
        self.status_label.setText(text)
