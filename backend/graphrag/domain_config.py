"""YAML-backed domain configuration for extraction and corpus knowledge."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "domain_knowledge.yaml"


@lru_cache(maxsize=1)
def get_domain_knowledge() -> dict[str, Any]:
    config_path = Path(os.getenv("DOMAIN_KNOWLEDGE_PATH", _DEFAULT_CONFIG_PATH))
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Domain knowledge config must be a mapping: {config_path}")
    return data


def preload_domain_knowledge() -> None:
    get_domain_knowledge()
