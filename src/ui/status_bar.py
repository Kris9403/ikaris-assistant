from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer
from src.utils.helpers import get_system_health


class StatusBarWidget(QWidget):
    """
    Bottom status bar showing CPU, VRAM, RAM, and connection status.
    Auto-refreshes every 5 seconds.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusBar")
        self.setFixedHeight(32)
        self._setup_ui()
        self._start_polling()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(20)

        self.status_label = QLabel("Initializing...")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.vram_label = QLabel("")
        self.vram_label.setObjectName("statusLabel")
        layout.addWidget(self.vram_label)

        self.ram_label = QLabel("")
        self.ram_label.setObjectName("statusLabel")
        layout.addWidget(self.ram_label)

        self.gpu_label = QLabel("")
        self.gpu_label.setObjectName("statusLabel")
        layout.addWidget(self.gpu_label)

    def _start_polling(self):
        self._update_stats()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_stats)
        self.timer.start(5000)

    def _update_stats(self):
        try:
            health = get_system_health()
            status_text = health.get("status", "Unknown")
            self.status_label.setText(f"Status: {status_text}")
            
            vram_used = health.get("vram_used", 0)
            vram_total = health.get("vram_total", 0)
            self.vram_label.setText(f"VRAM: {vram_used}/{vram_total} MB")
            
            ram_pct = health.get("ram_percent", 0)
            self.ram_label.setText(f"RAM: {ram_pct}%")
            
            gpu_util = health.get("gpu_util", "N/A")
            self.gpu_label.setText(f"GPU: {gpu_util}")
            
            # Color coding
            if vram_used > 10000:
                self.vram_label.setObjectName("statusLabelWarn")
            else:
                self.vram_label.setObjectName("statusLabelGood")
            self.vram_label.style().unpolish(self.vram_label)
            self.vram_label.style().polish(self.vram_label)
        except Exception:
            self.status_label.setText("Status: ❌ Monitor Error")
