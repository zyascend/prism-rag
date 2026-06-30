"""配置加载器 — 加载 models.yaml 并提供类型安全访问"""

import os
from pathlib import Path
from typing import Any
import yaml


class Config:
    """全局配置，单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self, path: str | None = None) -> "Config":
        if self._loaded:
            return self
        config_path = path or (Path(__file__).parent.parent / "config" / "models.yaml")
        try:
            with open(config_path) as f:
                self._data = yaml.safe_load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"Config file not found at {config_path}. "
                "Ensure config/models.yaml exists."
            ) from None
        except yaml.YAMLError as e:
            raise RuntimeError(f"Malformed YAML in {config_path}: {e}") from None
        # Auto-detect device if set to "auto"
        import torch
        for device_key in ["bge_device", "colpali_device"]:
            val = self._data["embedding"].get(device_key, "auto")
            if val == "auto":
                if torch.cuda.is_available():
                    self._data["embedding"][device_key] = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._data["embedding"][device_key] = "mps"
                else:
                    self._data["embedding"][device_key] = "cpu"
        self._loaded = True
        return self

    def __getattr__(self, key: str) -> Any:
        if not self._loaded:
            self.load()
        raise AttributeError(
            f"Config key '{key}' not found. "
            "Use typed properties like cfg.colpali_model_id or "
            "cfg.get('section.key') for dynamic access."
        )

    def get(self, key_path: str, default: Any = None) -> Any:
        """Access nested config with dotted path, e.g. cfg.get('storage.pgvector.host')"""
        if not self._loaded:
            self.load()
        keys = key_path.split(".")
        data = self._data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return default
        return data

    @property
    def colpali_model_id(self) -> str:
        return self._data["models"]["colpali"]

    @property
    def bge_model_id(self) -> str:
        return self._data["models"]["bge_embedding"]

    @property
    def reranker_model_id(self) -> str:
        return self._data["models"]["bge_reranker"]

    @property
    def bge_dim(self) -> int:
        return self._data["embedding"]["bge_dim"]


cfg = Config()
