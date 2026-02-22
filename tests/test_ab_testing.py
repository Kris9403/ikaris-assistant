import pytest
from src.ab_testing import ABTestManager, Experiment

def test_experiment_creation():
    exp = Experiment(name="test_exp", variants=["A", "B"])
    assert exp.name == "test_exp"
    assert exp.variants == ["A", "B"]
    assert exp.active_variant == "A" # Default to first

def test_manager_registration():
    manager = ABTestManager()
    manager.register_experiment("test_exp", ["A", "B"])
    assert "test_exp" in manager.experiments
    assert manager.get_variant("test_exp") == "A"

def test_manager_tracking():
    manager = ABTestManager()
    manager.register_experiment("test_exp", ["A", "B"])
    manager.track_metric("test_exp", "clicks", 1)
    assert manager.experiments["test_exp"].metrics["clicks"] == 1
