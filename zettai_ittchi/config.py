from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load .env as early as possible
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"


class OutputConfig(BaseModel):
    default_mode: Literal["ghost", "clean", "debug", "audit"] = "ghost"


class DebateConfig(BaseModel):
    enabled: bool = True
    max_rounds: int = 2
    strict_unanimity: bool = True
    stream_internal: bool = False
    max_total_seconds: int = 25
    fallback_on_deadlock: bool = True
    malformed_vote_is_block: bool = True


class FallbackConfig(BaseModel):
    allow_single_model_pass_through: bool = False
    return_objections_on_deadlock: bool = True


class AgentConfig(BaseModel):
    name: str
    model: str
    role: str


class VirtualModelsConfig(BaseModel):
    default: str = "zettai-ittchi"
    presets: dict[str, str] = Field(default_factory=dict)


class PresetOverride(BaseModel):
    """Sparse override — only the fields present are merged."""
    output: dict[str, Any] | None = None
    debate: dict[str, Any] | None = None


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    agents: list[AgentConfig] = Field(default_factory=list)
    virtual_models: VirtualModelsConfig = Field(default_factory=VirtualModelsConfig)
    presets: dict[str, dict[str, Any]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------

_config: AppConfig | None = None


def _load_yaml() -> dict[str, Any]:
    config_path = _PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config() -> AppConfig:
    global _config
    if _config is None:
        raw = _load_yaml()
        _config = AppConfig(**raw)
    return _config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_preset(model_name: str) -> AppConfig:
    """Return an AppConfig with preset overrides applied for *model_name*.

    If *model_name* matches a key in ``virtual_models.presets``, the
    corresponding preset dict is deep-merged onto the base config.
    Otherwise the base config is returned unchanged.
    """
    cfg = get_config()
    preset_map = cfg.virtual_models.presets
    preset_key = preset_map.get(model_name)
    if preset_key is None:
        return cfg

    preset_data = cfg.presets.get(preset_key)
    if preset_data is None:
        return cfg

    base_dict = cfg.model_dump()
    merged = _deep_merge(base_dict, preset_data)
    return AppConfig(**merged)


def detected_api_keys() -> list[str]:
    """Return names of API-key env vars that have a non-empty value."""
    candidates = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
    ]
    return [k for k in candidates if os.environ.get(k)]
