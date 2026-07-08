import importlib, os
from src import config as config_mod


def test_local_dev_profile_merges():
    os.environ["CONFIG_PROFILE"] = "local-dev"
    config_mod.cfg._loaded = False
    config_mod.cfg._data = None
    config_mod.cfg.load()
    assert config_mod.cfg.get("ingestion.parser") == "simple"
    assert config_mod.cfg.get("retrieval.use_visual") is False
    del os.environ["CONFIG_PROFILE"]
    config_mod.cfg._loaded = False
    config_mod.cfg._data = None
