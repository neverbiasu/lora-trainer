"""ConfigManager tests"""
import pytest
from lora_trainer.config_manager import ConfigManager


def test_config_manager_init():
    """Test ConfigManager initialization"""
    manager = ConfigManager()
    assert manager.config_version == "0.1.0"


# TODO: add more tests
