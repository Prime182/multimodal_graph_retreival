"""Layer 2 entity records and extraction bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Layer2EntityRecord:
    entity_id: str
    entity_type: str
    label: str
    source_chunk_id: str
    mention_chunk_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.0
    extractor_model: str = "heuristic-v1"
    embedding: list[float] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Layer2DocumentRecord:
    paper_id: str
    extractor_model: str
    entities: list[Layer2EntityRecord] = field(default_factory=list)
    chunk_entity_ids: dict[str, list[str]] = field(default_factory=dict)
    chunk_salience_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
