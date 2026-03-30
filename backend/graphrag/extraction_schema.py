"""Extraction schema loading and inheritance for Phase 3."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml


_SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"


def _normalize_domain(domain: str) -> str:
    value = domain.strip().lower()
    if value.endswith(".yaml"):
        value = value[:-5]
    return value


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _merge_hint_values(base: Any, override: Any) -> Any:
    if base is None:
        return deepcopy(override)
    if override is None:
        return deepcopy(base)

    if isinstance(base, dict) and isinstance(override, dict):
        merged: dict[str, Any] = {}
        for key in base.keys() | override.keys():
            merged[key] = _merge_hint_values(base.get(key), override.get(key))
        return merged

    if isinstance(base, (list, tuple)) and isinstance(override, (list, tuple)):
        return _unique_strings([*map(str, base), *map(str, override)])

    return deepcopy(override)


class ValidationRule(BaseModel):
    """Declarative validation rule for a schema element."""

    model_config = ConfigDict(extra="ignore")

    rule_type: str
    field: str | None = None
    value: Any | None = None
    message: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rule_type", "field", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip().lower()


class EntitySchema(BaseModel):
    """Extraction expectations for an entity type."""

    model_config = ConfigDict(extra="ignore")

    type_name: str
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)
    validation_rules: list[ValidationRule] = Field(default_factory=list)
    extraction_hints: list[str] = Field(default_factory=list)

    @field_validator("type_name", mode="before")
    @classmethod
    def _normalize_type_name(cls, value: Any) -> str:
        return str(value).strip().lower()


class RelationSchema(BaseModel):
    """Extraction expectations for a relation type."""

    model_config = ConfigDict(extra="ignore")

    relation_name: str
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)
    validation_rules: list[ValidationRule] = Field(default_factory=list)
    extraction_hints: list[str] = Field(default_factory=list)

    @field_validator("relation_name", mode="before")
    @classmethod
    def _normalize_relation_name(cls, value: Any) -> str:
        return str(value).strip().lower()


class ExtractionSchema(BaseModel):
    """Top-level extraction schema with optional inheritance."""

    model_config = ConfigDict(extra="ignore")

    domain: str
    version: str
    extends: str | None = None
    entity_schemas: list[EntitySchema] = Field(default_factory=list)
    relation_schemas: list[RelationSchema] = Field(default_factory=list)
    hints: dict[str, Any] = Field(default_factory=dict)
    validation_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("domain", "extends", mode="before")
    @classmethod
    def _normalize_optional_domain(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return _normalize_domain(str(value))

    @field_validator("domain", mode="before")
    @classmethod
    def _normalize_domain_name(cls, value: Any) -> str:
        return _normalize_domain(str(value))

    @classmethod
    def load(cls, domain: str) -> "ExtractionSchema":
        """Load a schema from ``schemas/{domain}.yaml`` with inheritance."""
        normalized = _normalize_domain(domain)
        return cls._load_recursive(normalized, seen=frozenset())

    @classmethod
    def _load_recursive(cls, domain: str, seen: frozenset[str]) -> "ExtractionSchema":
        if domain in seen:
            cycle = " -> ".join((*seen, domain))
            raise ValueError(f"Cyclic schema inheritance detected: {cycle}")

        path = _SCHEMA_DIR / f"{domain}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found for domain '{domain}': {path}")

        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Schema file must contain a mapping: {path}")

        base_domain = raw.get("extends")
        current = cls.model_validate(raw)

        if not base_domain:
            return current

        base_schema = cls._load_recursive(_normalize_domain(str(base_domain)), seen | {domain})
        return cls._merge(base_schema, current)

    @classmethod
    def _merge(cls, base: "ExtractionSchema", override: "ExtractionSchema") -> "ExtractionSchema":
        entity_map = {schema.type_name: deepcopy(schema) for schema in base.entity_schemas}
        for schema in override.entity_schemas:
            existing = entity_map.get(schema.type_name)
            if existing is None:
                entity_map[schema.type_name] = deepcopy(schema)
                continue
            entity_map[schema.type_name] = EntitySchema(
                type_name=schema.type_name,
                required_fields=_unique_strings([*existing.required_fields, *schema.required_fields]),
                optional_fields=_unique_strings([*existing.optional_fields, *schema.optional_fields]),
                validation_rules=_merge_rules(existing.validation_rules, schema.validation_rules),
                extraction_hints=_unique_strings([*existing.extraction_hints, *schema.extraction_hints]),
            )

        relation_map = {schema.relation_name: deepcopy(schema) for schema in base.relation_schemas}
        for schema in override.relation_schemas:
            existing = relation_map.get(schema.relation_name)
            if existing is None:
                relation_map[schema.relation_name] = deepcopy(schema)
                continue
            relation_map[schema.relation_name] = RelationSchema(
                relation_name=schema.relation_name,
                source_types=_unique_strings([*existing.source_types, *schema.source_types]),
                target_types=_unique_strings([*existing.target_types, *schema.target_types]),
                required_fields=_unique_strings([*existing.required_fields, *schema.required_fields]),
                optional_fields=_unique_strings([*existing.optional_fields, *schema.optional_fields]),
                validation_rules=_merge_rules(existing.validation_rules, schema.validation_rules),
                extraction_hints=_unique_strings([*existing.extraction_hints, *schema.extraction_hints]),
            )

        merged_hints = _merge_hint_values(base.hints, override.hints)
        merged_validation_metadata = _merge_hint_values(base.validation_metadata, override.validation_metadata)

        return cls(
            domain=override.domain or base.domain,
            version=override.version or base.version,
            extends=override.extends or base.extends,
            entity_schemas=list(entity_map.values()),
            relation_schemas=list(relation_map.values()),
            hints=merged_hints if isinstance(merged_hints, dict) else {},
            validation_metadata=merged_validation_metadata if isinstance(merged_validation_metadata, dict) else {},
        )

    def entity_schema(self, type_name: str) -> EntitySchema | None:
        normalized = _normalize_domain(type_name)
        for schema in self.entity_schemas:
            if schema.type_name == normalized:
                return schema
        return None

    def relation_schema(self, relation_name: str) -> RelationSchema | None:
        normalized = _normalize_domain(relation_name)
        for schema in self.relation_schemas:
            if schema.relation_name == normalized:
                return schema
        return None

    def get_hints(self, key: str, default: Any | None = None) -> Any:
        """Return the configured hint value for ``key``."""
        if key not in self.hints:
            return deepcopy(default)
        return deepcopy(self.hints[key])

    def get_hint_list(self, key: str) -> tuple[str, ...]:
        """Return a hint value as a normalized tuple of strings."""
        value = self.get_hints(key, default=[])
        if isinstance(value, list):
            return tuple(_unique_strings([str(item) for item in value]))
        if isinstance(value, tuple):
            return tuple(_unique_strings([str(item) for item in value]))
        if isinstance(value, dict):
            flattened: list[str] = []
            for item in value.values():
                if isinstance(item, (list, tuple)):
                    flattened.extend(str(entry) for entry in item)
                elif item not in (None, ""):
                    flattened.append(str(item))
            return tuple(_unique_strings(flattened))
        if value in (None, ""):
            return ()
        return (str(value),)

    def get_validation_metadata(self, key: str, default: Any | None = None) -> dict[str, Any]:
        """Return validation metadata for an entity or relation type."""
        value = self.validation_metadata.get(key, default if default is not None else {})
        if isinstance(value, dict):
            return deepcopy(value)
        return {}


def _merge_rules(base: list[ValidationRule], override: list[ValidationRule]) -> list[ValidationRule]:
    seen: set[tuple[Any, ...]] = set()
    merged: list[ValidationRule] = []
    for rule in [*base, *override]:
        payload = rule.model_dump(mode="python")
        marker = (
            payload.get("rule_type"),
            payload.get("field"),
            payload.get("value"),
            payload.get("message"),
            tuple(sorted((payload.get("parameters") or {}).items())),
        )
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(rule)
    return merged


@lru_cache(maxsize=1)
def load_schema(domain: str) -> ExtractionSchema:
    """Cached convenience wrapper for schema loading."""
    return ExtractionSchema.load(domain)
