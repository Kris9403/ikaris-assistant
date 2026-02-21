import hydra
from omegaconf import DictConfig
from src.utils.instantiators import instantiate_model, instantiate_tools, instantiate_audio
from src.agent import Agent

@hydra.main(version_base="1.3", config_path="configs", config_name="main")
def main(cfg: DictConfig):
    print(f"--- Ikaris OS Initialized on {cfg.device} ---")
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
        audio=audio
    )
    
    # Start the GUI loop
    from src.main import start_agent_loop
    start_agent_loop(cfg, agent)

if __name__ == "__main__":
    main()
