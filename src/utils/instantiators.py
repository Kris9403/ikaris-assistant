# src/utils/instantiators.py
import hydra
from omegaconf import DictConfig

def instantiate_model(cfg: DictConfig):
    return hydra.utils.instantiate(cfg.model)

def instantiate_tools(cfg: DictConfig):
    tools_list = []
    for tool_cfg in cfg.tools.values():
        if tool_cfg.get("enabled", True):
            tools_list.append(hydra.utils.instantiate(tool_cfg))
    return tools_list

def instantiate_audio(cfg: DictConfig):
    """
    Instantiate the audio stack from Hydra config.
    Returns NullAudioStack if audio._target_ is null or missing.
    """
    audio_cfg = cfg.get("audio", None)
    if audio_cfg is None or audio_cfg.get("_target_") is None:
        from src.utils.audio import NullAudioStack
        return NullAudioStack()
    return hydra.utils.instantiate(audio_cfg)
