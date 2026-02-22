# Ikaris Assistant 🦾 (v1.2.1)

A hyper-personalized local AI research assistant powered by **Hydra**, **LangGraph**, and **Sherpa-ONNX** — a full local multimodal research agent with hybrid RAG, comparative synthesis, and NPU-accelerated voice.

> [!IMPORTANT]
> **Release Version**: v1.2.1
> This version introduces Audio v2: Silero VAD speech gating, live partial hypothesis display, STT confidence scoring, and automatic NPU→CPU fallback. Built on top of v1.2.0's Hydra configuration, hybrid RAG, and Sherpa-ONNX audio stack.

## 🚀 Key Features

- **Hydra Configuration**: Declarative YAML configs for models, tools, paths, audio, and hardware. Switch anything with a single CLI flag.
- **Dependency-Injected Agent**: Clean `Agent(llm, tools, audio)` architecture — no global state, fully modular.
- **Hybrid RAG**: Combines local FAISS (PDF) retrieval with PubMed biomedical search. Capability-routed: biomedical queries automatically engage PubMed.
- **Comparative Synthesis**: Multi-source evidence is analyzed for consensus, conflicts, and research gaps — not just retrieved.
- **Unified Evidence Layer**: All retrieval sources emit standardized `Evidence` dataclass objects for deduplication and ranking.
- **Sherpa-ONNX Audio Stack**: NPU-accelerated voice I/O with automatic fallbacks:

  | Profile | STT | TTS | Provider |
  |---------|-----|-----|----------|
  | `npu` | Zipformer Streaming | Kokoro (82M) | OpenVINO → Intel NPU |
  | `cuda` | Whisper float16 | Kokoro (82M) | CUDA → GPU |
  | `cpu` | Whisper INT8 | Piper VITS | CPU fallback |
  | `none` | Disabled | Disabled | Text-only |

- **Local Brain**: Powered by LM Studio or Ollama (OpenAI-compatible).
- **Intelligent Routing**: Graph-based router dispatches to hardware stats, research, Logseq, or general chat.
- **Hardware Monitoring**: Real-time CPU and battery stats (optimized for ROG Strix G16).
- **Persistent Memory**: Conversation history across restarts via SQLite.
- **Logseq Sync**: Auto-logs research insights and retrieves handwritten notes.
- **Offline First**: Aggressive model caching for fully offline usage.
- **Cross-Platform**: Path configs for Linux (ROG Strix), macOS, and Windows.

## 🏗️ Architecture

```
Hydra (configs/)
 ├── model   (lm_studio / ollama)
 ├── tools   (faiss / pubmed / logseq / research)
 ├── audio   (npu / cuda / cpu / none)
 └── paths   (strix / linux / mac / windows)
         │
         ▼
   Agent(llm, tools, audio)
         │
         ▼
   StateGraph (LangGraph)
         │
   START → summarize → router
                         ├── hardware_node → END
                         ├── llm_node → END
                         ├── logseq_node → END
                         ├── research_node → synthesis_node → END
                         └── agent_planning_node
                               └── retrieval_node (FAISS + PubMed hybrid)
                                     └── reasoning_node
                                           ├── retrieval_node (loop)
                                           └── generate_answer_node → END
         │
   Mic (STT) ←→ UI ←→ Speaker (TTS)
```

## 🛠 Project Structure

```text
ikaris_assistant/
├── configs/                    # Hydra configuration hierarchy
│   ├── main.yaml               # Master config switch
│   ├── model/
│   │   ├── lm_studio.yaml      # LM Studio backend
│   │   └── ollama.yaml         # Ollama backend
│   ├── audio/
│   │   ├── npu.yaml            # Intel NPU (Zipformer + Kokoro)
│   │   ├── cuda.yaml           # NVIDIA GPU (Whisper + Kokoro)
│   │   ├── cpu.yaml            # CPU fallback (Whisper INT8 + Piper)
│   │   └── none.yaml           # Text-only (disabled)
│   ├── paths/
│   │   ├── strix.yaml          # ROG Strix G16 (Linux)
│   │   ├── linux.yaml          # Generic Linux
│   │   ├── mac.yaml            # macOS
│   │   └── windows.yaml        # Windows
│   ├── tools/
│   │   ├── paper.yaml          # FAISS PDF tool
│   │   ├── research.yaml       # ArXiv downloader tool
│   │   ├── logseq.yaml         # Logseq notes tool
│   │   └── pubmed.yaml         # PubMed biomedical tool
│   └── ui/
│       └── default.yaml        # UI theme config
│
├── models/                     # ONNX model weights (gitignored)
│   ├── stt/                    # Speech-to-Text models
│   │   └── README.md           # Download instructions
│   ├── tts/                    # Text-to-Speech models
│   │   └── README.md           # Download instructions
│   └── vad/                    # Voice Activity Detection
│       └── README.md           # Download instructions
│
├── run.py                      # Hydra-powered entry point (GUI + CLI)
├── run_cli.py                  # DEPRECATED — use `python run.py mode=cli`
├── papers/                     # Drop your research PDFs here
│
└── src/
    ├── agent.py                # Agent class (DI-based graph builder)
    ├── evidence.py             # Unified Evidence dataclass
    ├── main.py                 # Node definitions, routing, GUI bootstrap
    ├── state.py                # LangGraph State definition
    ├── nodes/
    │   ├── llm_node.py             # General chat node
    │   ├── reasoning_node.py       # Agentic evidence evaluator
    │   ├── research_node.py        # ArXiv batch download node
    │   ├── retrieval_node.py       # Hybrid retrieval (FAISS + PubMed)
    │   └── synthesis_node.py       # Comparative synthesis engine
    ├── tools/
    │   ├── hardware.py             # System stats
    │   ├── paper_tool.py           # FAISS vector search (PaperTool)
    │   ├── research_tool.py        # ArXiv downloader (ResearchTool)
    │   ├── logseq_tool.py          # Logseq journal (LogseqTool)
    │   └── pubmed_tool.py          # PubMed biomedical (PubMedTool)
    ├── ui/
    │   ├── main_window.py          # PyQt5 researcher interface
    │   ├── chat_widget.py          # Chat panel
    │   ├── sidebar_widget.py       # Paper sidebar
    │   ├── status_bar.py           # System status bar
    │   ├── styles.py               # Dark theme
    │   └── workers.py              # Background QThread workers
    └── utils/
        ├── audio.py                # SherpaAudioStack (STT + TTS engine)
        ├── voice.py                # Legacy faster-whisper (fallback)
        ├── helpers.py              # Hardware detection
        ├── instantiators.py        # Hydra instantiation glue
        ├── llm_client.py           # LLM client functions
        └── summarizer.py           # Conversation compressor
```

## ⚙️ Setup

### 1. Environment
```bash
conda env create -f environment.yml
conda activate ikaris_env
pip install -r requirements.txt
```

### 2. Local LLM Server
Start **LM Studio** or **Ollama** and ensure the local server is running:
- LM Studio: `http://localhost:1234/v1`
- Ollama: `http://localhost:11434/v1`

### 3. Research Papers
Place PDFs in `ikaris_assistant/papers/`. Auto-indexed on first query.

### 4. PubMed (Optional)
```bash
export NCBI_API_KEY=your_real_key_here
```

### 5. Audio Models (Sherpa-ONNX)

Install Sherpa-ONNX:
```bash
pip install sherpa-onnx
```

Download models into `models/stt/` and `models/tts/`:
- See `models/stt/README.md` and `models/tts/README.md` for links.

**Validate your setup before Python:**
```bash
# Test STT (replace with your model paths)
sherpa-onnx-offline-asr \
  --nn-model=./models/stt/whisper-base-int8.onnx \
  --tokens=./models/stt/tokens.txt \
  --provider=openvino \
  --device=NPU

# If this works → Python will work.
```

## 🎤 Usage

### Run the GUI (default)
```bash
python run.py
```

### Run in CLI mode (no GUI, terminal REPL)
```bash
python run.py mode=cli
```
Inside the CLI you can type messages directly, press `v` + Enter for voice input, or `exit` to quit.

### Combine CLI with other overrides
```bash
python run.py mode=cli audio=none          # text-only terminal
python run.py mode=cli model=ollama        # use Ollama backend
python run.py mode=cli audio=cuda paths=linux
```

### Override configs via CLI (Hydra)
```bash
# Switch LLM backend
python run.py model=ollama

# Switch audio profile
python run.py audio=npu          # Intel NPU (Zipformer streaming + Kokoro)
python run.py audio=cuda         # NVIDIA GPU (Whisper float16 + Kokoro)
python run.py audio=cpu          # CPU fallback (Whisper INT8 + Piper)
python run.py audio=none         # Text-only, no mic

# Switch platform paths
python run.py paths=mac
python run.py paths=windows
python run.py paths=linux

# Use CPU instead of GPU for inference
python run.py device=cpu

# Combine overrides
python run.py model=ollama audio=cuda paths=linux

# Print resolved config (debugging)
python run.py --cfg job
```

### Chat Commands
- **General Chat**: Just type or speak.
- **Hardware**: Ask about "battery" or "cpu".
- **Research**: Ask questions about your papers (e.g., "What is Scaled Dot-Product Attention?").
- **Download Papers**: Paste ArXiv IDs (e.g., "download 1706.03762 2307.09288").
- **Logseq**: Every research insight is logged to your Logseq journal.
- **Biomedical**: Queries with medical/biological terms auto-trigger PubMed hybrid search.
- **Voice**: Click the mic button or use the voice shortcut (requires `audio=npu|cuda|cpu`).

## 🔊 Audio Profiles

| Profile | STT Engine | TTS Engine | Best For |
|---------|-----------|-----------|----------|
| **npu** | Zipformer (streaming, low-latency) | Kokoro 82M | Intel Core Ultra laptops with NPU |
| **cuda** | Whisper small (float16, high accuracy) | Kokoro 82M | NVIDIA GPU systems (RTX 5070 Ti) |
| **cpu** | Whisper base (INT8, lightweight) | Piper VITS | Any system, no GPU needed |
| **none** | Disabled | Disabled | Text-only, headless, CI/testing |

### Why these choices?
- **Zipformer** is designed for low-latency streaming — perfect for always-on mic on NPU.
- **Whisper INT8** via OpenVINO can hit the NPU/CPU efficiently for batch transcription.
- **Kokoro** (82M) is small enough for NPU yet sounds premium.
- **Piper** is rock-solid fallback when Kokoro glitches.
- **Sherpa-ONNX + OpenVINO** is currently the best local stack for Intel NPU + privacy + latency.

## 🧠 Audio v2 Features

### 1. Voice Activity Detection (Silero VAD)
Silero VAD gates the microphone so STT only processes actual speech. This saves power, improves accuracy, and makes the UX snappier. Download the model:
```bash
bash scripts/pull_models.sh vad
```

### 2. Partial Hypothesis Display
Zipformer streaming STT emits partial tokens as you speak. The UI shows live transcription (`🎤 ... hello how are`) that updates in real-time. Feels magical.

### 3. Confidence Scoring
Every transcription returns a confidence score (0.0–1.0) extracted from token probabilities:
- 🟢 ≥70% — high confidence
- 🟡 ≥40% — medium confidence
- 🔴 <40% — low confidence (consider asking user to repeat)

The confidence is stored in `IkarisState.stt_confidence` so downstream nodes can factor it in.

### 4. Auto-Switch STT
If the primary STT engine (NPU/CUDA) fails to load, the system automatically falls back to CPU Whisper INT8. No config change needed — the CPU models are already downloaded. The UI shows an ⚡ indicator when auto-switch occurs.

### v1.2.1 (Audio v2)
- **Silero VAD**: Voice Activity Detection gates microphone — no wasted compute on silence.
- **Partial Hypothesis**: Zipformer streaming emits live tokens to UI for real-time transcription display.
- **Confidence Scoring**: STT confidence (0.0–1.0) exposed to `IkarisState.stt_confidence` with 🟢/🟡/🔴 badges.
- **Auto-Switch STT**: If primary provider (NPU/CUDA) fails, automatic fallback to CPU Whisper INT8.
- **VoiceWorker QThread**: Voice input runs in background thread — UI never freezes during recording.
- **STTResult dataclass**: Rich return type with text, confidence, duration, provider, and fallback status.

### v1.2.0
- **Hydra Integration**: Full declarative config system (`configs/` hierarchy).
- **Agent Class**: `Agent(llm, tools, audio)` — dependency injection replaces all global state.
- **Sherpa-ONNX Audio**: NPU/CUDA/CPU audio profiles with Zipformer streaming STT, Whisper offline STT, Kokoro TTS, and Piper TTS fallback.
- **Evidence Dataclass**: Unified retrieval layer across FAISS, PubMed, and Logseq.
- **Hybrid RAG**: FAISS + PubMed merge ranking with deduplication and capability routing.
- **PubMedTool**: Real biomedical literature search via metapub (ESearch + EFetch + FindIt).
- **Synthesis Node**: Comparative analysis across multi-source evidence.
- **Capability Routing**: Biomedical intent detection triggers PubMed automatically.
- **Cross-Platform Configs**: Path profiles for Linux, macOS, and Windows.
- **Ollama Backend**: Alternative LLM backend support.
- **Observability**: Tool call logging with latency tracking via Hydra outputs.

### v1.1.0
- Wayland stability fixes.
- Offline model caching.
- PyQt5 GUI with streaming tokens.
- Background PDF indexing.

---
*Built for the ROG Strix G16. Runs anywhere.*
