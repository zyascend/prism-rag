"""配置加载器 — 加载 models.yaml 并提供类型安全访问"""

import os
from pathlib import Path
from typing import Any
import yaml
from dataclasses import dataclass


def deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in place."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


@dataclass
class ObservabilityConfig:
    """Observability 配置，从 YAML observability 段加载，缺失时使用默认值"""
    log_level: str = "INFO"
    log_file: str = "logs/app.jsonl"
    trace_enabled: bool = True
    dashboard_enabled: bool = True
    # 单条 Trace 的磁盘持久化路径（相对项目根，可被 YAML 覆盖）。
    # 置空字符串 "" 可关闭持久化（仅内存，进程重启后无法反查）。
    trace_persist_path: str = "logs/api_traces.jsonl"
    latency_p95_threshold_ms: int = 5000
    recall_at_5_min: float = 0.5
    faithfulness_min: float = 0.6
    rerank_score_min: float = 0.0
    context_relevancy_min: float = 0.05


@dataclass
class CacheConfig:
    """检索缓存配置，从 YAML cache 段加载，缺失时使用默认值。

    enabled: 全局开关，控制所有缓存层（L1/L3/L4）是否生效。
    max_size: 内存 LRU 容量上限（条目数）。
    ttl_seconds: 0 = 仅依赖 index_version 盐失效（推荐，正确性由版本保证）；
                 >0 仅作跨进程异常兜底，不作为正确性依赖。
    """

    enabled: bool = True
    max_size: int = 2048
    ttl_seconds: int = 0


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
        # 合并 profile（如 local-dev）：CONFIG_PROFILE=local-dev -> config/models.local-dev.yaml
        profile = os.environ.get("CONFIG_PROFILE")
        if profile:
            profile_path = Path(__file__).parent.parent / "config" / f"models.{profile}.yaml"
            if profile_path.exists():
                with open(profile_path) as pf:
                    deep_merge(self._data, yaml.safe_load(pf) or {})
        self._loaded = True
        return self

    def __getattr__(self, key: str) -> Any:
        if not self._loaded:
            self.load()
            # After loading, properties (bge_model_id, colpali_model_id, etc.)
            # may now succeed because _data is available.
            try:
                return object.__getattribute__(self, key)
            except AttributeError:
                pass
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

    def _ensure_loaded(self) -> None:
        """Lazily (re)load config if it has not been loaded or is uninitialized."""
        if not self._loaded or self._data is None:
            self.load()

    @property
    def colpali_model_id(self) -> str:
        self._ensure_loaded()
        return self._data["models"]["colpali"]

    @property
    def colqwen2_model_id(self) -> str:
        self._ensure_loaded()
        return self._data["models"]["colqwen2"]

    @property
    def bge_model_id(self) -> str:
        self._ensure_loaded()
        return self._data["models"]["bge_embedding"]

    @property
    def reranker_model_id(self) -> str:
        self._ensure_loaded()
        return self._data["models"]["bge_reranker"]

    @property
    def zerank_reranker_model_id(self) -> str:
        self._ensure_loaded()
        return self._data["models"]["zerank_reranker"]

    @property
    def llm_model_id(self) -> str:
        self._ensure_loaded()
        return self._data["models"]["llm"]

    @property
    def observability(self) -> ObservabilityConfig:
        """返回 observability 配置，YAML 缺失时回退到默认值"""
        raw = self.get("observability", {})
        alerting_raw = raw.get("alerting", {})
        return ObservabilityConfig(
            log_level=raw.get("log_level", "INFO"),
            log_file=raw.get("log_file", "logs/app.jsonl"),
            trace_enabled=raw.get("trace_enabled", True),
            dashboard_enabled=raw.get("dashboard_enabled", True),
            trace_persist_path=raw.get("trace_persist_path", "logs/api_traces.jsonl"),
            latency_p95_threshold_ms=alerting_raw.get("latency_p95_threshold_ms", 5000),
            recall_at_5_min=alerting_raw.get("recall_at_5_min", 0.5),
            faithfulness_min=alerting_raw.get("faithfulness_min", 0.6),
            rerank_score_min=alerting_raw.get("rerank_score_min", 0.0),
            context_relevancy_min=alerting_raw.get("context_relevancy_min", 0.05),
        )

    @property
    def cache(self) -> "CacheConfig":
        """返回 cache 配置，YAML 缺失时回退到默认值。"""
        raw = self.get("cache", {})
        return CacheConfig(
            enabled=raw.get("enabled", True),
            max_size=raw.get("max_size", 2048),
            ttl_seconds=raw.get("ttl_seconds", 0),
        )

    @property
    def bge_dim(self) -> int:
        self._ensure_loaded()
        return self._data["embedding"]["bge_dim"]


cfg = Config()
