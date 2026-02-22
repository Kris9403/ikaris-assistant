import hydra
from omegaconf import DictConfig
from src.utils.instantiators import instantiate_model, instantiate_tools, instantiate_audio
from src.agent import Agent


def _run_cli(cfg: DictConfig, agent):
    """Terminal REPL — no GUI, no PyQt5 dependency."""
    from langchain_core.messages import HumanMessage

    print("\n🦾 Ikaris CLI  (type 'exit' to quit, 'v' for voice)")
    print("─" * 52)

    config = {"configurable": {"thread_id": "cli_session"}}
    stt_confidence = 0.0

    while True:
        try:
            raw = input("\nYou ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down Ikaris…")
            break

        if raw.lower() in ("", "exit", "quit"):
            if raw.lower() in ("exit", "quit"):
                print("Shutting down Ikaris…")
                break
            continue

        # Voice shortcut
        if raw.lower() == "v":
            if agent.audio and agent.audio.has_stt:
                print("🎤  Listening…")
                result = agent.audio.listen()

                # Handle STTResult dataclass
                if hasattr(result, 'text'):
                    raw = result.text
                    stt_confidence = result.confidence

                    # Confidence badge
                    if result.confidence >= 0.7:
                        badge = f'🟢 {result.confidence:.0%}'
                    elif result.confidence >= 0.4:
                        badge = f'🟡 {result.confidence:.0%}'
                    else:
                        badge = f'🔴 {result.confidence:.0%}'

                    fallback_tag = ""
                    if result.is_fallback:
                        fallback_tag = f" ⚡ [auto-switched → {result.provider}]"

                    print(f"   You said ({badge}{fallback_tag}): {raw}")

                    if raw.startswith("Error"):
                        print(f"   ⚠  {raw}")
                        continue
                else:
                    # Legacy string return
                    raw = str(result)
                    stt_confidence = 0.5
                    print(f"   You said: {raw}")
            else:
                print("⚠  Audio not available. Type your message instead.")
                continue

        inputs = {
            "messages": [HumanMessage(content=raw)],
            "hardware_info": "",
            "summary": "",
            "loop_count": 0,
            "stt_confidence": stt_confidence,
        }

        # Reset confidence for next iteration (only meaningful for voice input)
        stt_confidence = 0.0

        for event in agent.app.stream(inputs, config=config):
            for value in event.values():
                msgs = value.get("messages", [])
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                elif isinstance(msgs, list):
                    continue
                else:
                    last = msgs

                content = last.content if hasattr(last, "content") else str(last)
                if content.strip():
                    print(f"\nIkaris ❯ {content}")

                # Optional TTS
                if agent.audio and agent.audio.has_tts and content.strip():
                    agent.audio.speak(content)


@hydra.main(version_base="1.3", config_path="configs", config_name="main")
def main(cfg: DictConfig):
    mode = cfg.get("mode", "gui")
    print(f"--- Ikaris OS Initialized on {cfg.device} (mode={mode}) ---")
    print(f"Logseq Path: {cfg.paths.logseq_path}")

    # Instantiate all components from Hydra config
    llm_client = instantiate_model(cfg)
    tools = instantiate_tools(cfg)
    audio = instantiate_audio(cfg)

    print(f"Audio Stack: {type(audio).__name__} (capabilities: {getattr(audio, 'capabilities', [])})")

    # Initialize Agent with full dependency injection
    agent = Agent(
        llm=llm_client,
        tools=tools,
        audio=audio,
    )

    if mode == "cli":
        _run_cli(cfg, agent)
    else:
        from src.main import start_agent_loop
        start_agent_loop(cfg, agent)


if __name__ == "__main__":
    main()
