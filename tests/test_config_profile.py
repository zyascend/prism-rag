import os
from src import config as config_mod


def test_local_dev_profile_merges():
    os.environ["CONFIG_PROFILE"] = "local-dev"
    config_mod.cfg._loaded = False
    config_mod.cfg._data = None
    config_mod.cfg.load()
    assert config_mod.cfg.get("ingestion.parser") == "mineru"
    assert config_mod.cfg.get("retrieval.use_visual") is True
    assert config_mod.cfg.get("embedding.visual_backend") == "colqwen2"
    del os.environ["CONFIG_PROFILE"]
    config_mod.cfg._loaded = False
    config_mod.cfg._data = None
