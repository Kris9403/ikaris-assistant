import os
import json

class WorkspaceManager:
    """Singleton to manage dynamic workspace paths for Ikaris plugins/tools."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WorkspaceManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.root_dir = os.path.abspath("workspaces")
        self.active_workspace = "default"
        
        # Don't create directories immediately, only when setting/using workspace
        self._load_state()

    def _load_state(self):
        os.makedirs(self.root_dir, exist_ok=True)
        state_file = os.path.join(self.root_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    self.active_workspace = data.get("active_workspace", "default")
            except Exception:
                pass
        self.set_workspace(self.active_workspace)

    def _save_state(self):
        state_file = os.path.join(self.root_dir, "state.json")
        with open(state_file, "w") as f:
            json.dump({"active_workspace": self.active_workspace}, f)

    def set_workspace(self, name: str):
        # Sanitize name
        name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        if not name:
            name = "default"
            
        self.active_workspace = name
        ws_dir = os.path.join(self.root_dir, name)
        
        # Ensure workspace isolated directories exist
        os.makedirs(os.path.join(ws_dir, "sources"), exist_ok=True)
        os.makedirs(os.path.join(ws_dir, "notes"), exist_ok=True)
        self._save_state()

    def get_active_workspace(self) -> str:
        return self.active_workspace
        
    def get_workspaces(self) -> list:
        if not os.path.exists(self.root_dir):
            return []
        return sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
    def get_papers_dir(self) -> str:
        """Isolated PDF ingest directory."""
        return os.path.join(self.root_dir, self.active_workspace, "sources")
        
    def get_faiss_index_dir(self) -> str:
        """Isolated FAISS DB."""
        return os.path.join(self.root_dir, self.active_workspace, "index.faiss")
        
    def get_logseq_dir(self) -> str:
        """Isolated Notes directory."""
        return os.path.join(self.root_dir, self.active_workspace, "notes")
