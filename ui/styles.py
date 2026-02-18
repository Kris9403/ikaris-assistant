"""
Ikaris Dark Theme — QSS Stylesheet
Inspired by research tools like Zotero Dark, Obsidian, and VS Code.
"""

DARK_THEME = """
QMainWindow {
    background-color: #1a1b26;
}

QWidget {
    background-color: #1a1b26;
    color: #c0caf5;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 13px;
}

/* ── Chat Area ── */
QTextEdit#chatDisplay {
    background-color: #16161e;
    color: #c0caf5;
    border: 1px solid #292e42;
    border-radius: 8px;
    padding: 12px;
    font-size: 14px;
    line-height: 1.6;
    selection-background-color: #33467c;
}

QLineEdit#chatInput {
    background-color: #24283b;
    color: #c0caf5;
    border: 1px solid #3b4261;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
}

QLineEdit#chatInput:focus {
    border: 1px solid #7aa2f7;
}

/* ── Buttons ── */
QPushButton {
    background-color: #3b4261;
    color: #c0caf5;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #414868;
}

QPushButton#sendBtn {
    background-color: #7aa2f7;
    color: #1a1b26;
}

QPushButton#sendBtn:hover {
    background-color: #89b4fa;
}

QPushButton#voiceBtn {
    background-color: #f7768e;
    color: #1a1b26;
}

QPushButton#voiceBtn:hover {
    background-color: #ff9e64;
}

QPushButton#indexBtn {
    background-color: #9ece6a;
    color: #1a1b26;
}

QPushButton#indexBtn:hover {
    background-color: #73daca;
}

/* ── Sidebar ── */
QWidget#sidebar {
    background-color: #1f2335;
    border-right: 1px solid #292e42;
}

QLabel#sidebarTitle {
    color: #7aa2f7;
    font-size: 15px;
    font-weight: bold;
    padding: 8px;
}

QListWidget {
    background-color: #1f2335;
    color: #a9b1d6;
    border: none;
    padding: 4px;
    font-size: 13px;
}

QListWidget::item {
    padding: 8px 10px;
    border-radius: 4px;
    margin: 2px 4px;
}

QListWidget::item:hover {
    background-color: #292e42;
}

QListWidget::item:selected {
    background-color: #33467c;
    color: #c0caf5;
}

/* ── Status Bar ── */
QWidget#statusBar {
    background-color: #16161e;
    border-top: 1px solid #292e42;
    padding: 4px 12px;
}

QLabel#statusLabel {
    color: #565f89;
    font-size: 12px;
}

QLabel#statusLabelGood {
    color: #9ece6a;
    font-size: 12px;
}

QLabel#statusLabelWarn {
    color: #e0af68;
    font-size: 12px;
}

/* ── Progress Bar ── */
QProgressBar {
    background-color: #292e42;
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #7aa2f7;
    border-radius: 4px;
}

/* ── Scrollbar ── */
QScrollBar:vertical {
    background-color: #1a1b26;
    width: 8px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #3b4261;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #565f89;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* ── Splitter ── */
QSplitter::handle {
    background-color: #292e42;
    width: 2px;
}

/* ── Group Labels ── */
QLabel#sectionHeader {
    color: #565f89;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    padding: 6px 8px 2px 8px;
    letter-spacing: 1px;
}
"""
