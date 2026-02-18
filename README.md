# Ikaris Assistant 🦾 (v1.1.0)

A hyper-personalized local AI assistant running on Linux (ROG Strix), powered by LangChain and LangGraph.

> [!IMPORTANT]
> **Release Version**: v1.1.0
> This version includes critical stability fixes for Wayland and offline model caching.

## 🚀 Key Features

- **Local Brain**: Powered by LM Studio (OpenAI-compatible server).
- **Intelligent Routing**: Uses a graph-based router to decide between hardware stats, research analysis, or general chat.
- **Hardware Monitoring**: Real-time monitoring of CPU usage and Battery status (optimized for ROG Strix).
- **RAG Research Analysis**: Local PDF analysis using FAISS vector store and HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **Persistent Memory**: Remembers your conversations across restarts using a local SQLite database.
- **Logseq Sync**: Automatically logs insights to your Logseq journal and retrieves handwritten notes to answer questions.
- **Local Voice Input**: Speak to Ikaris using **Faster-Whisper** locally on your GPU.
- **Offline First**: Optimized for offline usage with aggressive model caching.

## 🛠 Project Structure

```text
ikaris_assistant/
├── .env                # API keys and Config
├── .gitignore          # Environment and cache ignores
├── environment.yml     # Conda environment definition
├── README.md           # Project documentation
├── run.py              # Main entry point (GUI)
├── papers/             # Drop your research PDFs here
│
└── src/
    ├── main.py         # Graph compilation and routing logic
    ├── state.py        # LangGraph State definitions
    ├── nodes/          # Logic for LLM and Paper analysis
    ├── tools/          # Hardware, Paper, and Logseq tools
    └── utils/          # Voice STT and Helper functions
```

## ⚙️ Setup

1.  **Environment**:
    ```bash
    conda env create -f environment.yml
    conda activate ikaris_env
    pip install -r requirements.txt
    ```
2.  **Local Server**:
    Start LM Studio and ensure the local server is running at `http://localhost:1234/v1`.
3.  **Research Papers**:
    Place PDFs in `ikaris_assistant/papers/`. They will be automatically indexed upon first query.

## 🎤 Usage

Run the assistant GUI:
```bash
python run.py
```

-   **Chat**: Just type or speak.
-   **Hardware**: Ask about "battery" or "cpu".
-   **Research**: Ask questions about your papers (e.g., "What is Scaled Dot-Product Attention?").
-   **Logseq**: Every research insight is automatically logged to your Logseq journal.

---
*Built for the ROG Strix G16.*
