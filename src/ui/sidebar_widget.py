import os
import shutil
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QProgressBar, QComboBox, QInputDialog, QFileDialog
)
from PyQt5.QtCore import Qt
from src.workspaces.workspace_manager import WorkspaceManager


class SidebarWidget(QWidget):
    """
    Sidebar showing paper list, indexing controls, and status.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(260)
        self.setAcceptDrops(True)
        self.wm = WorkspaceManager()
        self._setup_ui()
        self.refresh_workspaces()
        self.refresh_papers()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(4)

        # Title
        title = QLabel("📂 Workspaces")
        title.setObjectName("sidebarTitle")
        layout.addWidget(title)

        # Workspace controls
        ws_layout = QHBoxLayout()
        self.workspace_combo = QComboBox()
        self.workspace_combo.setObjectName("workspaceCombo")
        self.workspace_combo.currentTextChanged.connect(self._on_workspace_changed)
        ws_layout.addWidget(self.workspace_combo)
        
        self.new_ws_btn = QPushButton("➕")
        self.new_ws_btn.setFixedWidth(40)
        self.new_ws_btn.clicked.connect(self._on_new_workspace)
        ws_layout.addWidget(self.new_ws_btn)
        layout.addLayout(ws_layout)

        # Paper list
        section = QLabel("SOURCES")
        section.setObjectName("sectionHeader")
        layout.addWidget(section)

        self.paper_list = QListWidget()
        layout.addWidget(self.paper_list)

        # Index section
        section = QLabel("INDEX CONTROL")
        section.setObjectName("sectionHeader")
        layout.addWidget(section)

        self.add_pdfs_btn = QPushButton("📁 Add PDFs")
        self.add_pdfs_btn.setObjectName("addPdfsBtn")
        self.add_pdfs_btn.clicked.connect(self._on_add_pdfs)
        layout.addWidget(self.add_pdfs_btn)

        self.index_btn = QPushButton("⚡ Index Workspace")
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

    def refresh_workspaces(self):
        """Populate the combo box with available workspaces."""
        self.workspace_combo.blockSignals(True)
        self.workspace_combo.clear()
        workspaces = self.wm.get_workspaces()
        if not workspaces:
            self.wm.set_workspace("default")
            workspaces = ["default"]
            
        self.workspace_combo.addItems(workspaces)
        self.workspace_combo.setCurrentText(self.wm.get_active_workspace())
        self.workspace_combo.blockSignals(False)

    def _on_workspace_changed(self, name):
        if name:
            self.wm.set_workspace(name)
            self.refresh_papers()

    def _on_new_workspace(self):
        text, ok = QInputDialog.getText(self, "New Workspace", "Workspace Name:")
        if ok and text:
            self.wm.set_workspace(text)
            self.refresh_workspaces()
            self.refresh_papers()

    def refresh_papers(self):
        """Scan the active workspace papers directory and populate the list."""
        self.paper_list.clear()
        papers_path = self.wm.get_papers_dir()
        
        if not os.path.exists(papers_path):
            os.makedirs(papers_path, exist_ok=True)

        pdf_files = sorted([f for f in os.listdir(papers_path) if f.endswith('.pdf')])
        
        if not pdf_files:
            item = QListWidgetItem("No papers yet. Click 'Add PDFs'.")
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

    def _on_add_pdfs(self):
        """Open file dialog to add PDFs to the active workspace."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF Papers", "", "PDF Files (*.pdf)"
        )
        if files:
            self._import_pdfs(files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile().lower().endswith('.pdf')]
        if files:
            self._import_pdfs(files)

    def _import_pdfs(self, files):
        papers_path = self.wm.get_papers_dir()
        os.makedirs(papers_path, exist_ok=True)
        added_count = 0
        for f in files:
            dest = os.path.join(papers_path, os.path.basename(f))
            if not os.path.exists(dest):
                shutil.copy(f, dest)
                added_count += 1
        
        if added_count > 0:
            self.set_status(f"Added {added_count} new PDF(s).")
            self.refresh_papers()
        else:
            self.set_status("No new PDFs were added (skipping duplicates).")

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
