"""Layer 2 entity extraction for the GraphRAG pipeline."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha1
import json
import os
import re
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from .config import Phase1Settings
from .corpus import enrich_entity
from .domain_config import get_domain_knowledge
from .embeddings import TextEmbedder, build_entity_embedder
from .entities import Layer2DocumentRecord, Layer2EntityRecord
from .extraction_cache import get_extraction_cache
from .extraction_schema import ExtractionSchema, load_schema
from .gemini import GeminiError, generate_json, gemini_available
from .models import ChunkRecord, PaperRecord, SectionRecord
from .tracing import get_tracing_manager

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    END = START = StateGraph = None

# ---------------------------------------------------------------------------
# Lazy config accessors
# All module-level constants that used to be hardcoded are now loaded once
# from domain_knowledge.yaml on first use via these helpers.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    """Return the extraction section of domain_knowledge.yaml."""
    return get_domain_knowledge().get("extraction", {})


def _schema_from_context(context: dict[str, Any] | None) -> ExtractionSchema:
    schema = (context or {}).get("schema")
    if isinstance(schema, ExtractionSchema):
        return schema
    return load_schema("general")


def _payload_value(payload: dict[str, Any] | BaseModel, field_name: str) -> Any:
    if isinstance(payload, BaseModel):
        getter = getattr
    else:
        getter = lambda obj, key, default=None: obj.get(key, default)  # noqa: E731

    aliases = [field_name]
    if field_name == "label":
        aliases.append("name")
    if field_name == "name":
        aliases.append("label")
    for alias in aliases:
        value = getter(payload, alias, None)
        if value not in (None, "", [], {}):
            return value
    return None


def detect_domain(paper: PaperRecord) -> str:
    """Choose an extraction schema domain from journal, keywords, and abstract."""
    candidates = ["biomedical", "physics"]
    journal_text = " ".join(
        part for part in [
            paper.journal.name if paper.journal else "",
            paper.journal.code if paper.journal else "",
        ] if part
    )
    combined = " ".join(
        part for part in [
            paper.title,
            journal_text,
            " ".join(paper.keywords),
            paper.abstract,
        ] if part
    ).lower()
    if not combined:
        return "general"

    best_domain = "general"
    best_score = 0
    for domain in candidates:
        schema = load_schema(domain)
        score = 0

        for hint_key in ("biomedical_entity_markers", "concept_patterns", "method_suffixes", "quantity_patterns"):
            hint_value = schema.get_hints(hint_key)
            if isinstance(hint_value, dict):
                values = []
                for item in hint_value.values():
                    if isinstance(item, list):
                        values.extend(str(entry).lower() for entry in item)
                    elif item not in (None, ""):
                        values.append(str(item).lower())
            else:
                values = [str(item).lower() for item in schema.get_hint_list(hint_key)]
            for value in values:
                if value and value in combined:
                    score += 2

        for entity_schema in schema.entity_schemas:
            for hint in entity_schema.extraction_hints:
                if hint.lower() in combined:
                    score += 1

        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


def _extraction_model_version(schema: ExtractionSchema, use_gemini: bool) -> str:
    model_name = os.getenv("EXTRACT_MODEL", "gemini-2.5-flash") if use_gemini else _LOCAL_MODEL
    return f"{model_name}::{schema.domain}::{schema.version}"


@lru_cache(maxsize=1)
def _known_method_terms() -> tuple[str, ...]:
    terms = {
        str(term).strip()
        for term in _cfg().get("biomedical_methods", [])
        if str(term).strip()
    }
    terms.update(
        str(term).strip()
        for term in _method_canonical_map().keys()
        if str(term).strip()
    )
    terms.update(
        str(term).strip()
        for term in _method_aliases().keys()
        if str(term).strip()
    )
    return tuple(sorted(terms, key=len, reverse=True))


@lru_cache(maxsize=1)
def _known_method_pattern() -> re.Pattern[str]:
    """Build a YAML-backed method term matcher."""
    alternation = "|".join(
        re.escape(term).replace(r"\ ", r"\s+")
        for term in _known_method_terms()
    )
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _method_suffixes() -> tuple[str, ...]:
    suffixes = {
        str(suffix).strip().lower()
        for suffix in _cfg().get("generic_method_suffixes", [])
        if str(suffix).strip()
    }
    return tuple(sorted(suffixes, key=len, reverse=True))


@lru_cache(maxsize=1)
def _method_suffix_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for suffix in _method_suffixes():
        patterns.append(
            (
                suffix,
                re.compile(
                    rf"\b(?:[A-Za-z][A-Za-z0-9-]*\s+){{0,4}}[A-Za-z][A-Za-z0-9-]*-?{re.escape(suffix)}\b",
                    re.IGNORECASE,
                ),
            )
        )
    return patterns


@lru_cache(maxsize=1)
def _biomedical_method_re() -> re.Pattern[str]:
    """Compatibility wrapper for callers that still import the old helper."""
    return _known_method_pattern()


@lru_cache(maxsize=1)
def _method_fragment_exclusions() -> frozenset[str]:
    return frozenset(_cfg().get("method_fragment_exclusions", []))


@lru_cache(maxsize=1)
def _generic_method_names() -> frozenset[str]:
    return frozenset(_cfg().get("generic_method_names", []))


@lru_cache(maxsize=1)
def _metric_regex_cache() -> list[tuple[str, re.Pattern[str]]]:
    """Compile metric label patterns once, sorted longest-first."""
    labels: list[str] = _cfg().get("metric_labels", [])
    sorted_labels = sorted(set(labels), key=len, reverse=True)
    return [
        (
            label,
            re.compile(
                r"(?<![A-Za-z0-9])"
                + re.escape(label).replace(r"\ ", r"\s+")
                + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            ),
        )
        for label in sorted_labels
    ]


@lru_cache(maxsize=1)
def _metric_expansions() -> dict[str, str]:
    return {str(k): str(v) for k, v in _cfg().get("metric_expansions", {}).items()}


@lru_cache(maxsize=1)
def _method_type_keywords() -> dict[str, list[str]]:
    return {
        str(k): [str(t) for t in v]
        for k, v in _cfg().get("method_type_keywords", {}).items()
    }


@lru_cache(maxsize=1)
def _method_type_overrides() -> dict[str, str]:
    return {str(k): str(v) for k, v in _cfg().get("method_type_overrides", {}).items()}


@lru_cache(maxsize=1)
def _method_canonical_map() -> dict[str, str]:
    return {str(k): str(v) for k, v in _cfg().get("method_canonical_map", {}).items()}


@lru_cache(maxsize=1)
def _concept_canonical_map() -> dict[str, str]:
    return {str(k): str(v) for k, v in _cfg().get("concept_canonical_map", {}).items()}


@lru_cache(maxsize=1)
def _concept_alias_map() -> dict[str, list[str]]:
    return {
        str(k): [str(a) for a in v]
        for k, v in _cfg().get("concept_alias_map", {}).items()
    }


@lru_cache(maxsize=1)
def _method_aliases() -> dict[str, list[str]]:
    return {
        str(k): [str(a) for a in v]
        for k, v in _cfg().get("method_aliases", {}).items()
    }


@lru_cache(maxsize=1)
def _concept_text_aliases() -> dict[str, list[str]]:
    return {
        str(k): [str(a) for a in v]
        for k, v in _cfg().get("concept_text_aliases", {}).items()
    }


@lru_cache(maxsize=1)
def _dataset_keywords() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(keyword).strip()
                for keyword in _cfg().get("dataset_keywords", [])
                if str(keyword).strip()
            },
            key=len,
            reverse=True,
        )
    )


@lru_cache(maxsize=1)
def _dataset_keyword_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for keyword in _dataset_keywords():
        pattern = re.escape(keyword).replace(r"\ ", r"\s+")
        patterns.append((keyword, re.compile(rf"\b{pattern}\b", re.IGNORECASE)))
    return patterns


@lru_cache(maxsize=1)
def _quantity_patterns() -> dict[str, tuple[str, ...]]:
    return {
        str(name): tuple(
            str(pattern).strip().lower()
            for pattern in patterns
            if str(pattern).strip()
        )
        for name, patterns in _cfg().get("quantity_patterns", {}).items()
    }


@lru_cache(maxsize=1)
def _concept_patterns() -> tuple[str, ...]:
    patterns = {
        str(pattern).strip().lower()
        for pattern in _cfg().get("concept_patterns", [])
        if str(pattern).strip()
    }
    return tuple(sorted(patterns, key=len, reverse=True))


@lru_cache(maxsize=1)
def _concept_pattern_re() -> re.Pattern[str] | None:
    endings = _concept_patterns()
    if not endings:
        return None
    alternation = "|".join(re.escape(ending).replace(r"\ ", r"\s+") for ending in endings)
    return re.compile(
        rf"\b(?:[A-Za-z][A-Za-z0-9-]*\s+){{0,4}}(?:{alternation})\b",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Static regexes that are structural (not data-driven) — kept as module-level
# constants because they don't come from config and are cheap to compile once.
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_CONCEPT_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){1,4}|[A-Z]{2,}(?:-[A-Z0-9]+)?)\b"
)
_METHOD_PHRASE_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9-]{1,}\s+){0,4}(?:model|method|approach|framework|algorithm|pipeline|architecture|simulation|protocol|process|system|assay|technique|procedure|analysis|sequencing|microscopy|chromatography|spectroscopy|imaging|screening|measurement|quantification)\b",
    re.IGNORECASE,
)
_CLAIM_VERBS = re.compile(
    r"\b(achieves?|demonstrates?|shows?|reveals?|suggests?|indicates?|improves?|reduces?|increases?|decreases?|supports?|confirms?|enables?|outperforms?|leads to|results in|causes?)\b",
    re.IGNORECASE,
)
_LIMITATION_CUES = re.compile(
    r"\b(limitations?|however|drawback|cannot|fails to|challenge|challenges|weakness)\b",
    re.IGNORECASE,
)
_HYPOTHESIS_CUES = re.compile(
    r"\b(hypothesize|hypothesis|propose|proposed|we expect|future work|future studies|will)\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")
_P_VALUE_RE = re.compile(r"\bp[-\s]?value(?:s)?\s*(?:of|:|=|<|>|≤|≥)?\s*(\d*\.\d+)\b", re.IGNORECASE)
_STAR_PVALUE_RE = re.compile(r"\*+\s*p\s*value", re.IGNORECASE)
_PERCENT_VALUE_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_COUNT_VALUE_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})*|\d+)\s+(genes?|cells?|peaks?)\b", re.IGNORECASE)
_FOLD_CHANGE_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(?:fold(?:-change)?|fold change)\b", re.IGNORECASE)
_GENERIC_SYMBOL_RE = re.compile(
    r"\b(?:[A-Z]{2,}\d*(?:-[A-Z0-9αβγδ]+)?|[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9αβγδ]+)?)\b"
)
_EQUATION_PATTERNS = [
    re.compile(
        r"(?P<expr>(?:[A-Za-z][A-Za-z0-9_]*(?:\([^)]+\))?|\([^)]+\))\s*(?:=|≈|<=|>=|<|>)\s*(?:[A-Za-z0-9_+\-*/().^ ]+))"
    ),
    re.compile(
        r"(?P<expr>(?:[A-Za-z][A-Za-z0-9_]*(?:\([^)]+\))?(?:\s*[+\-*/]\s*[A-Za-z0-9_()]+)+))"
    ),
]

_DATASET_DEFINITION_RE = re.compile(
    r"\b(?P<label>[A-Za-z][A-Za-z0-9'’./+\-]*(?:\s+[A-Za-z0-9'’./+\-]+){0,8}?)\s*\((?P<abbr>[A-Za-z][A-Za-z0-9'’./+\-]{0,15})\)",
    re.IGNORECASE,
)
_DATASET_GROUP_RE = re.compile(
    r"\b(?P<label>(?:[A-Z][A-Za-z0-9-]*|[a-z][A-Za-z0-9-]*)(?:\s+(?:[A-Z][A-Za-z0-9-]*|[a-z][A-Za-z0-9-]*)){0,5}\s+(?:group|groups|cohort|cohorts|sample|samples|condition|conditions|arm|arms|strain|strains|line|lines|cell line|cell lines|cell|cells|macrophage|macrophages|patient|patients|subject|subjects))\b",
    re.IGNORECASE,
)
_DATASET_SCENARIO_RE = re.compile(r"\b(?:scenario|group|arm|cohort)\s+\d+\b", re.IGNORECASE)
_DATASET_COMPARISON_RE = re.compile(
    r"\b(?P<left>[A-Za-z][A-Za-z0-9'’./+\-]{0,20})\s*(?:vs\.?|versus)\s*(?P<right>[A-Za-z][A-Za-z0-9'’./+\-]{0,20})\b",
    re.IGNORECASE,
)
_DATASET_PAIR_RE = re.compile(
    r"\b(?:between|both)\s+(?P<left>[A-Za-z][A-Za-z0-9'’./+\-]{0,20})\s+and\s+(?P<right>[A-Za-z][A-Za-z0-9'’./+\-]{0,20})\b",
    re.IGNORECASE,
)
_EXPERIMENTAL_GROUP_DEFINITION_RE = re.compile(
    r"\b(?P<label>(?:[A-Za-z][A-Za-z0-9'’./+\-]*)(?:\s+[A-Za-z0-9'’./+\-]+){0,6}?\s+(?:group|groups|control|treated|untreated|arm|arms|cohort|cohorts|sample|samples|condition|conditions|patient|patients|subject|subjects|cell\s+line|cell\s+lines|line|lines|strain|strains))\s*\((?P<abbr>[A-Za-z][A-Za-z0-9'’./+\-]{0,15})\)",
    re.IGNORECASE,
)
_EXPERIMENTAL_GROUP_SCENARIO_RE = re.compile(
    r"\b(?P<label>(?:group|scenario|arm|cohort)\s+\d+)\b",
    re.IGNORECASE,
)

_LOCAL_MODEL = "heuristic-v2"

# ---------------------------------------------------------------------------
# Gemini prompt (schema-driven)
# ---------------------------------------------------------------------------

def _schema_entity_fields(entity_type: str, schema: ExtractionSchema) -> list[str]:
    entity_schema = schema.entity_schema(entity_type)
    if entity_schema is None:
        return []
    fields = list(entity_schema.required_fields) + list(entity_schema.optional_fields)
    normalized: list[str] = []
    for field_name in fields:
        normalized.append("name" if field_name == "label" else field_name)
    return _unique(normalized)


def _schema_entity_definition_lines(schema: ExtractionSchema) -> list[str]:
    lines: list[str] = []
    for entity_schema in schema.entity_schemas:
        hints = ", ".join(entity_schema.extraction_hints[:5]) if entity_schema.extraction_hints else "scientific entity"
        fields = ", ".join(_schema_entity_fields(entity_schema.type_name, schema))
        lines.append(f"- {entity_schema.type_name}: {hints}. Fields: {fields}")
    return lines


def _schema_return_shape(schema: ExtractionSchema) -> str:
    type_names = "|".join(entity_schema.type_name for entity_schema in schema.entity_schemas)
    return (
        "{\n"
        '  "entities": [\n'
        "    {\n"
        f'      "type": "{type_names}",\n'
        '      "name": "...",\n'
        '      "text": "...",\n'
        '      "claim_type": "finding|hypothesis|limitation|future_work",\n'
        '      "value": 0.0,\n'
        '      "unit": "...",\n'
        '      "dataset": "...",\n'
        '      "metric": "...",\n'
        '      "condition": "...",\n'
        '      "latex": "...",\n'
        '      "plain_desc": "...",\n'
        '      "is_loss_fn": false,\n'
        '      "aliases": [],\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ],\n"
        '  "salience_score": 0.0\n'
        "}"
    )


def _build_gemini_prompt(
    paper: PaperRecord,
    section: SectionRecord,
    chunk: ChunkRecord,
    schema: ExtractionSchema,
    *,
    validation_errors: list[str] | None = None,
    previous_payloads: list[dict[str, Any]] | None = None,
    salience_score: float | None = None,
) -> str:
    prompt_lines = [
        "You are a scientific knowledge extractor for research papers.",
        f"Domain schema: {schema.domain} v{schema.version}",
        "",
        "Task:",
        "- Extract only entities explicitly supported by the text chunk.",
        "- Prefer domain-appropriate scientific labels over generic words.",
        "- Do not infer beyond the text.",
        "- Omit absent entities instead of inventing placeholders.",
        "",
        "Entity definitions:",
        *_schema_entity_definition_lines(schema),
        "",
        "Rules:",
        "- Keep aliases only when they are explicit in the chunk.",
        "- Use confidence in the range 0.0 to 1.0.",
        "- Return ONLY valid JSON.",
    ]
    if validation_errors:
        prompt_lines.extend(
            [
                "",
                "Previous extraction failed validation. Fix the issues below:",
                *[f"- {error}" for error in validation_errors],
            ]
        )
    if previous_payloads:
        prompt_lines.extend(
            [
                "",
                "Previous JSON:",
                json.dumps(
                    {"entities": previous_payloads, "salience_score": salience_score or 0.0},
                    indent=2,
                ),
            ]
        )
    prompt_lines.extend(
        [
            "",
            "Return ONLY valid JSON with this schema:",
            _schema_return_shape(schema),
            "",
            f"Paper title: {paper.title}",
            f"Section: {section.title}",
            f"Chunk ID: {chunk.chunk_id}",
            "",
            f"Text:\n{chunk.text}",
        ]
    )
    return "\n".join(prompt_lines)


# ---------------------------------------------------------------------------
# Pydantic models for Gemini output validation (unchanged)
# ---------------------------------------------------------------------------

class _GeminiEntityPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None
    label: str | None = None
    text: str | None = None
    claim_type: str | None = None
    value: float | None = None
    unit: str | None = None
    dataset: str | None = None
    metric: str | None = None
    condition: str | None = None
    latex: str | None = None
    plain_desc: str | None = None
    is_loss_fn: bool | None = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str, info: ValidationInfo) -> str:
        normalized = _normalize(value).lower()
        schema = _schema_from_context(info.context)
        allowed_types = {entity_schema.type_name for entity_schema in schema.entity_schemas}
        if normalized not in allowed_types:
            raise ValueError(f"Unsupported entity type: {value}")
        return normalized

    @field_validator("aliases", mode="before")
    @classmethod
    def _coerce_aliases(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        raise ValueError("aliases must be a list of strings")

    @model_validator(mode="after")
    def _validate_shape(self, info: ValidationInfo) -> "_GeminiEntityPayload":
        schema = _schema_from_context(info.context)
        entity_schema = schema.entity_schema(self.type)
        if entity_schema is None:
            raise ValueError(f"Unsupported entity type: {self.type}")
        for field_name in entity_schema.required_fields:
            value = _payload_value(self, field_name)
            if value in (None, "", [], {}):
                raise ValueError(f"{self.type} entities require {field_name}")
        return self


class _GeminiChunkExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entities: list[_GeminiEntityPayload] = Field(default_factory=list)
    salience_score: float = Field(default=0.0, ge=0.0, le=1.0)


class _ExtractionGraphState(TypedDict):
    paper: PaperRecord
    section: SectionRecord
    chunk: ChunkRecord
    schema: ExtractionSchema
    model_name: str
    model_version: str
    raw_payloads: list[dict[str, Any]]
    validated_payloads: list[dict[str, Any]]
    salience_score: float
    validation_errors: list[str]
    extraction_quality_score: float
    retry_count: int


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_key(text: str) -> str:
    """Lowercase + collapse whitespace — used for merge keys and map lookups."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _clean_method_candidate(candidate: str) -> str:
    cleaned = _normalize(candidate)
    while True:
        next_cleaned = re.sub(
            r"^(?:we|used|use|using|and|or|the|a|an|by|with|via)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" ,;:-")
        if next_cleaned == cleaned:
            return cleaned
        cleaned = next_cleaned


def _slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text or "unknown"


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def _canonicalize_method_name(candidate: str) -> str:
    return _method_canonical_map().get(_normalize_key(candidate), candidate.strip())


def _canonicalize_concept_name(candidate: str) -> str:
    lowered = _normalize_key(candidate).replace("−", "-")
    return _concept_canonical_map().get(lowered, candidate.strip())


# ---------------------------------------------------------------------------
# Method type classification — reads from YAML, no hardcoded token lists
# ---------------------------------------------------------------------------

def _method_type(label: str, text: str) -> str:
    canonical = _canonicalize_method_name(label)
    overrides = _method_type_overrides()
    if canonical in overrides:
        return overrides[canonical]

    combined = f"{canonical} {text}".lower()
    keywords = _method_type_keywords()

    for category in ("statistical", "computational", "experimental"):
        if any(token in combined for token in keywords.get(category, [])):
            return category

    # computational fallback
    if any(token in combined for token in keywords.get("computational_fallback", [])):
        return "computational"

    return "experimental"


# ---------------------------------------------------------------------------
# Method validation — reads exclusion sets from YAML
# ---------------------------------------------------------------------------

def _is_valid_method(candidate: str, context: str = "") -> bool:
    candidate = _normalize(candidate)
    lowered = candidate.lower()

    if lowered in _method_fragment_exclusions():
        return False

    # Temperature "4C" false positive
    if lowered == "4c" and ("°c" in context.lower() or "degrees" in context.lower()):
        return False

    single_word_exclusions = {
        "system", "process", "approach", "method", "analysis", "study", "work",
        "we", "it", "this", "our", "well", "in", "on", "by", "with", "to", "from",
        "model", "framework", "procedure", "technique",
    }
    if len(lowered.split()) == 1:
        if lowered in single_word_exclusions or lowered in set(_method_suffixes()):
            return False
        known_single = {"crispr", "elisa", "facs", "gwas", "hplc", "qpcr", "rna-seq", "dna-seq", "chip-seq"}
        return lowered in known_single or bool(
            re.search(r"seq|assay|technique|protocol", context, re.IGNORECASE)
        )

    fragment_signals = {
        "well with", "contradicted by", "regulates the", "we discovered",
        "in our analysis", "we found", "we used", "we performed", "we conducted",
        "shows that the", "demonstrates that", "indicates that", "reveals",
        "suggests", "well-known", "using ", "for this", "this assay", "this method",
        "our analysis",
    }
    for signal in fragment_signals:
        if candidate.startswith(signal):
            return False

    if lowered.endswith("ing") and not any(
        tech in lowered for tech in {"sequencing", "screening", "imaging", "processing", "mapping"}
    ):
        return False

    if len(candidate) < 5:
        return False

    if len(candidate.split()) >= 2:
        return True

    return True


# ---------------------------------------------------------------------------
# Alias finders — read entirely from YAML, no inline dicts
# ---------------------------------------------------------------------------

def _text_aliases(term: str, text_context: str) -> list[str]:
    """Extract abbreviation ↔ expansion pairs from the surrounding text."""
    aliases: set[str] = set()
    escaped_term = re.escape(term)
    if re.fullmatch(r"[A-Z0-9α-]+", term):
        full_name_pattern = re.compile(
            rf"([A-Z][A-Za-z0-9'/-]+(?:\s+[A-Z][A-Za-z0-9'/-]+){{1,8}})\s*\(\s*{escaped_term}\s*\)"
        )
        for match in full_name_pattern.finditer(text_context):
            aliases.add(_normalize(match.group(1)))
    else:
        abbrev_pattern = re.compile(rf"{escaped_term}\s*\(\s*([A-Z][A-Z0-9α-]{{1,12}})\s*\)")
        for match in abbrev_pattern.finditer(text_context):
            aliases.add(_normalize(match.group(1)))
    return list(aliases)


def _find_aliases_for_method(method_name: str, text_context: str) -> list[str]:
    """
    Alias discovery order:
      1. text-extracted abbreviation/expansion pairs
      2. corpus enrichment (MeSH / OLS)
      3. YAML method_aliases fallback
    """
    method_name = _canonicalize_method_name(method_name)
    aliases: set[str] = set(_text_aliases(method_name, text_context))

    try:
        corpus_match = enrich_entity(method_name, "method")
        if corpus_match and corpus_match.found:
            aliases.update(corpus_match.aliases)
    except Exception:
        pass

    # YAML fallback — scan configured variants against the text
    lowered_method = method_name.lower()
    lowered_context = text_context.lower()
    for variant in _method_aliases().get(lowered_method, []):
        if variant.lower() in lowered_context and variant.lower() != lowered_method:
            aliases.add(variant)

    # Acronym check
    words = method_name.split()
    if len(words) > 1:
        abbrev = "".join(w[0] for w in words).upper()
        if abbrev in text_context and abbrev != method_name:
            aliases.add(abbrev)

    return list(aliases)


def _find_aliases_for_concept(concept_name: str, text_context: str) -> list[str]:
    """
    Alias discovery order:
      1. text-extracted abbreviation/expansion pairs
      2. corpus enrichment
      3. YAML concept_alias_map (known authoritative names)
      4. YAML concept_text_aliases (variant spellings to scan)
      5. m6A-specific variant spellings via regex
    """
    concept_name = _canonicalize_concept_name(concept_name)
    aliases: set[str] = set(_text_aliases(concept_name, text_context))

    try:
        corpus_match = enrich_entity(concept_name, "concept")
        if corpus_match and corpus_match.found:
            aliases.update(corpus_match.aliases)
    except Exception:
        pass

    # YAML authoritative alias map
    for alias in _concept_alias_map().get(concept_name, []):
        if alias.lower() in text_context.lower() or concept_name in _concept_canonical_map().values():
            aliases.add(alias)

    # YAML text-pattern aliases (scan chunk text for variants)
    lowered_concept = concept_name.lower()
    lowered_context = text_context.lower()
    for variant in _concept_text_aliases().get(lowered_concept, []):
        if variant.lower() in lowered_context and variant.lower() != lowered_concept:
            aliases.add(variant)

    # m6A variant spellings — regex-driven, not hardcoded strings
    if "methyladenosine" in lowered_concept or "m6a" in lowered_concept:
        for pattern in [r"m6[aA]", r"m⁶[aA]", r"[nN]6-methyl", r"[nN]6 methyl"]:
            for match in re.finditer(pattern, text_context):
                variant = match.group(0)
                if variant.lower() != lowered_concept:
                    aliases.add(variant)

    return list(aliases)


# ---------------------------------------------------------------------------
# Metric helpers — read from YAML
# ---------------------------------------------------------------------------

def _expand_metric_name(metric: str) -> str:
    return _metric_expansions().get(metric.lower().strip(), metric)


def _find_metric_match(sentence: str) -> tuple[str, int] | None:
    for label, pattern in _metric_regex_cache():
        match = pattern.search(sentence)
        if match:
            return label, match.end()
    return None


def _infer_metric_from_quantity_patterns(sentence: str) -> str | None:
    lowered = sentence.lower()
    if "%" in sentence or any(token in lowered for token in _quantity_patterns().get("percentage", ())):
        return "Percentage"
    if any(token in lowered for token in _quantity_patterns().get("fold_change", ())):
        return "Fold change"
    if any(token in lowered for token in _quantity_patterns().get("p_value", ())):
        return "P-value"
    if any(token in lowered for token in _quantity_patterns().get("concentration", ())):
        return "Concentration"
    return None


# ---------------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------------

def _register_dataset(
    mentions: dict[str, dict[str, Any]],
    label: str,
    *,
    aliases: list[str] | None = None,
    dataset_type: str = "experimental_group",
) -> None:
    normalized_label = _normalize(label)
    if not normalized_label:
        return
    payload = {
        "label": normalized_label,
        "aliases": _unique([alias for alias in (aliases or []) if alias and alias != normalized_label]),
        "dataset_type": dataset_type,
    }
    mentions.setdefault(normalized_label, payload)


def _looks_like_dataset(candidate: str) -> bool:
    lowered = candidate.lower()
    if not candidate:
        return False
    if lowered.startswith("scenario"):
        return True
    if any(char.isdigit() for char in candidate):
        return True
    keyword_set = {keyword.lower() for keyword in _dataset_keywords()}
    if lowered in keyword_set:
        return True
    if " " in candidate:
        leading = lowered.split()[0]
        if leading in {
            "control",
            "treated",
            "untreated",
            "wild",
            "wildtype",
            "knockout",
            "knockdown",
            "vehicle",
            "placebo",
            "baseline",
            "experimental",
            "treatment",
            "case",
            "patient",
            "sample",
            "group",
            "cohort",
            "arm",
            "strain",
            "line",
        }:
            return True
    if " " in candidate and any(
        noun in lowered
        for noun in {
            "group",
            "cohort",
            "sample",
            "samples",
            "condition",
            "conditions",
            "arm",
            "arms",
            "cell",
            "cells",
            "cell line",
            "cell lines",
            "macrophage",
            "macrophages",
            "line",
            "lines",
            "patient",
            "patients",
            "subject",
            "subjects",
            "strain",
            "strains",
        }
    ):
        leading = lowered.split()[0]
        if candidate[0].isupper() or leading in {
            "control",
            "treated",
            "untreated",
            "wild",
            "wildtype",
            "knockout",
            "knockdown",
            "vehicle",
            "placebo",
            "baseline",
            "experimental",
            "treatment",
            "case",
            "patient",
            "sample",
            "group",
            "cohort",
            "arm",
            "strain",
            "line",
        }:
            return True
    known_terms = {"imagenet", "cifar", "mnist", "squad", "coco", "glue", "wikidata", "wikipedia", "pubmed"}
    return lowered in known_terms


def _dataset_alias_patterns(registry: dict[str, dict[str, Any]]) -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for label, info in registry.items():
        aliases = [label, *[str(alias) for alias in info.get("aliases", []) if str(alias).strip()]]
        if not aliases:
            continue
        escaped = "|".join(
            re.escape(alias).replace(r"\ ", r"\s+")
            for alias in sorted({alias.strip() for alias in aliases if alias.strip()}, key=len, reverse=True)
        )
        patterns.append((label, re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE)))
    return patterns


def _extract_dataset_mentions(
    text: str,
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    mentions: dict[str, dict[str, Any]] = {}
    lowered_text = text.lower()

    if registry:
        for label, pattern in _dataset_alias_patterns(registry):
            if pattern.search(text):
                info = registry.get(label, {})
                _register_dataset(
                    mentions,
                    str(info.get("label", label)),
                    aliases=[str(alias) for alias in info.get("aliases", []) if str(alias).strip()],
                    dataset_type=str(info.get("dataset_type", "experimental_group")),
                )

    for match in _DATASET_SCENARIO_RE.finditer(text):
        scenario = _normalize(match.group(0))
        if scenario:
            _register_dataset(mentions, scenario, aliases=[scenario], dataset_type="scenario")

    for match in _DATASET_DEFINITION_RE.finditer(text):
        label = _normalize(match.group("label"))
        label = re.sub(r"^(?:and|or|with|plus|the|a|an)\s+", "", label, flags=re.IGNORECASE)
        abbr = _normalize(match.group("abbr"))
        if not label or not abbr:
            continue
        if not _looks_like_dataset(label):
            continue
        if re.search(rf"\b{re.escape(abbr)}\b", label, re.IGNORECASE):
            canonical = label
        else:
            canonical = f"{label} ({abbr})"
        _register_dataset(
            mentions,
            canonical,
            aliases=[label, abbr],
            dataset_type="experimental_group",
        )

    for _, pattern in _dataset_keyword_patterns():
        for match in pattern.finditer(text):
            candidate = _normalize(match.group(0))
            if candidate and _looks_like_dataset(candidate):
                _register_dataset(mentions, candidate, aliases=[candidate], dataset_type="experimental_group")

    for match in _DATASET_GROUP_RE.finditer(text):
        candidate = _normalize(match.group("label"))
        if candidate and _looks_like_dataset(candidate):
            _register_dataset(mentions, candidate, aliases=[candidate], dataset_type="experimental_group")

    for match in _DATASET_COMPARISON_RE.finditer(text):
        left = _normalize(match.group("left"))
        right = _normalize(match.group("right"))
        if left and _looks_like_dataset(left):
            _register_dataset(mentions, left, aliases=[left], dataset_type="experimental_group")
        if right and _looks_like_dataset(right):
            _register_dataset(mentions, right, aliases=[right], dataset_type="experimental_group")

    for match in _DATASET_PAIR_RE.finditer(text):
        left = _normalize(match.group("left"))
        right = _normalize(match.group("right"))
        if left and _looks_like_dataset(left):
            _register_dataset(mentions, left, aliases=[left], dataset_type="experimental_group")
        if right and _looks_like_dataset(right):
            _register_dataset(mentions, right, aliases=[right], dataset_type="experimental_group")

    if registry:
        for label, info in registry.items():
            if label in mentions:
                continue
            aliases = [str(alias) for alias in info.get("aliases", []) if str(alias).strip()]
            candidate_patterns = [label, *aliases]
            if any(re.search(rf"\b{re.escape(candidate)}\b", lowered_text, re.IGNORECASE) for candidate in candidate_patterns):
                _register_dataset(
                    mentions,
                    str(info.get("label", label)),
                    aliases=aliases,
                    dataset_type=str(info.get("dataset_type", "experimental_group")),
                )

    return list(mentions.values())


def _detect_experimental_groups(paper: PaperRecord) -> dict[str, dict[str, Any]]:
    """Build a per-document dataset registry from the abstract/title context."""
    context = _normalize(paper.abstract or "")
    if not context:
        context = _normalize(paper.title or "")
    registry: dict[str, dict[str, Any]] = {}
    if not context:
        return registry

    for match in _EXPERIMENTAL_GROUP_DEFINITION_RE.finditer(context):
        label = _normalize(match.group("label"))
        label = re.sub(r"^(?:and|or|with|plus|the|a|an)\s+", "", label, flags=re.IGNORECASE)
        abbr = _normalize(match.group("abbr"))
        if not label or not abbr:
            continue
        _register_dataset(
            registry,
            f"{label} ({abbr})",
            aliases=[label, abbr],
            dataset_type="experimental_group",
        )

    for match in _EXPERIMENTAL_GROUP_SCENARIO_RE.finditer(context):
        label = _normalize(match.group("label"))
        if label:
            _register_dataset(
                registry,
                label,
                aliases=[label],
                dataset_type="scenario",
            )

    return registry


# ---------------------------------------------------------------------------
# Entity base helpers
# ---------------------------------------------------------------------------

def _entity_base(
    entity_type: str,
    label: str,
    source_chunk_id: str,
    confidence: float,
    extractor_model: str,
    embedding: list[float],
    properties: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
    entity_id: str | None = None,
) -> Layer2EntityRecord:
    return Layer2EntityRecord(
        entity_id=entity_id or _stable_id(entity_type, label, source_chunk_id),
        entity_type=entity_type,
        label=_normalize(label),
        source_chunk_id=source_chunk_id,
        mention_chunk_ids=[source_chunk_id],
        aliases=_unique([alias for alias in (aliases or []) if alias]),
        confidence=round(max(0.0, min(confidence, 1.0)), 3),
        extractor_model=extractor_model,
        embedding=embedding,
        properties=properties or {},
    )


def _merge_entities(target: Layer2EntityRecord, incoming: Layer2EntityRecord) -> None:
    target.mention_chunk_ids = _unique([*target.mention_chunk_ids, *incoming.mention_chunk_ids])
    target.aliases = _unique([*target.aliases, *incoming.aliases])
    target.confidence = round(max(target.confidence, incoming.confidence), 3)
    if len(incoming.embedding) > len(target.embedding):
        target.embedding = incoming.embedding
    for key, value in incoming.properties.items():
        if value not in (None, "", [], {}):
            target.properties[key] = value


def _merge_key(entity_type: str, label: str) -> str:
    """
    Normalized merge key: type + lowercased label.
    Prevents "MeRIP-seq" and "merip-seq" from creating duplicate nodes.
    """
    return f"{entity_type}::{_normalize_key(label)}"


# ---------------------------------------------------------------------------
# Local heuristic extractors
# ---------------------------------------------------------------------------

def _is_valid_concept_candidate(candidate: str) -> bool:
    lowered = candidate.lower()
    if len(candidate) < 3:
        return False
    if lowered in {"abstract", "highlights", "introduction", "results", "discussion", "methods"}:
        return False
    if candidate.split()[0].endswith("ing") and any(token.isupper() for token in candidate.split()[1:]):
        return False
    if len(candidate.split()) > 1 and any(
        token in lowered
        for token in {"analysis", "assay", "blot", "cytometry", "seq", "sequencing", "crispr", "matrigel"}
    ):
        return False
    return True


def _local_concepts(
    paper: PaperRecord,
    section: SectionRecord,
    chunk: ChunkRecord,
    embedder: TextEmbedder,
) -> list[Layer2EntityRecord]:
    text = chunk.text
    lowered = text.lower()
    entities: dict[str, Layer2EntityRecord] = {}

    for keyword in paper.keywords:
        normalized_keyword = _canonicalize_concept_name(_normalize(keyword))
        if normalized_keyword and normalized_keyword.lower() in lowered:
            aliases = _find_aliases_for_concept(normalized_keyword, text)
            entity = _entity_base(
                "concept", normalized_keyword, chunk.chunk_id,
                confidence=0.92, extractor_model=_LOCAL_MODEL,
                embedding=embedder.embed(normalized_keyword),
                properties={"ontology": ""},
                aliases=aliases,
                entity_id=_stable_id("concept", _normalize_key(normalized_keyword)),
            )
            entities[entity.entity_id] = entity

    for match in _GENERIC_SYMBOL_RE.finditer(text):
        candidate = _canonicalize_concept_name(match.group(0))
        if not _is_valid_concept_candidate(candidate):
            continue
        aliases = _find_aliases_for_concept(candidate, text)
        entity = _entity_base(
            "concept", candidate, chunk.chunk_id,
            confidence=0.86, extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(candidate),
            properties={"ontology": ""},
            aliases=aliases,
            entity_id=_stable_id("concept", _normalize_key(candidate)),
        )
        entities.setdefault(entity.entity_id, entity)

    for match in _CONCEPT_PHRASE_RE.finditer(text):
        candidate = _canonicalize_concept_name(_normalize(match.group(0)))
        if not _is_valid_concept_candidate(candidate):
            continue
        aliases = _find_aliases_for_concept(candidate, text)
        entity = _entity_base(
            "concept", candidate, chunk.chunk_id,
            confidence=0.72 if candidate.isupper() else 0.62,
            extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(candidate),
            properties={"ontology": ""},
            aliases=aliases,
            entity_id=_stable_id("concept", _normalize_key(candidate)),
        )
        entities.setdefault(entity.entity_id, entity)

    concept_pattern_re = _concept_pattern_re()
    if concept_pattern_re is not None:
        for match in concept_pattern_re.finditer(text):
            candidate = _canonicalize_concept_name(_normalize(match.group(0)))
            if not _is_valid_concept_candidate(candidate):
                continue
            entity_id = _stable_id("concept", _normalize_key(candidate))
            if entity_id in entities:
                continue
            aliases = _find_aliases_for_concept(candidate, text)
            entity = _entity_base(
                "concept", candidate, chunk.chunk_id,
                confidence=0.6, extractor_model=_LOCAL_MODEL,
                embedding=embedder.embed(candidate),
                properties={"ontology": ""},
                aliases=aliases,
                entity_id=entity_id,
            )
            entities.setdefault(entity.entity_id, entity)

    if not entities and section.section_type in {"abstract", "introduction", "results", "discussion"}:
        fallback = _normalize(section.title)
        if fallback and fallback.lower() not in {"abstract", "highlights"}:
            aliases = _find_aliases_for_concept(fallback, text)
            entity = _entity_base(
                "concept", fallback, chunk.chunk_id,
                confidence=0.48, extractor_model=_LOCAL_MODEL,
                embedding=embedder.embed(fallback),
                properties={"ontology": ""},
                aliases=aliases,
                entity_id=_stable_id("concept", _normalize_key(fallback)),
            )
            entities[entity.entity_id] = entity

    return list(entities.values())


def _local_methods(
    paper: PaperRecord,
    section: SectionRecord,
    chunk: ChunkRecord,
    embedder: TextEmbedder,
) -> list[Layer2EntityRecord]:
    text = chunk.text
    entities: dict[str, Layer2EntityRecord] = {}

    for match in _known_method_pattern().finditer(text):
        candidate = _canonicalize_method_name(_clean_method_candidate(match.group(0)))
        if not _is_valid_method(candidate, text):
            continue
        aliases = _find_aliases_for_method(candidate, text)
        entity = _entity_base(
            "method", candidate, chunk.chunk_id,
            confidence=0.88 if section.section_type == "methods" else 0.78,
            extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(candidate),
            properties={
                "method_type": _method_type(candidate, text),
                "first_paper_id": paper.paper_id,
            },
            aliases=aliases,
            entity_id=_stable_id("method", _normalize_key(candidate)),
        )
        entities.setdefault(entity.entity_id, entity)

    for _, suffix_pattern in _method_suffix_patterns():
        for match in suffix_pattern.finditer(text):
            candidate = _canonicalize_method_name(_clean_method_candidate(match.group(0)))
            if not _is_valid_method(candidate, text):
                continue
            candidate_id = _stable_id("method", _normalize_key(candidate))
            if candidate_id in entities:
                continue
            aliases = _find_aliases_for_method(candidate, text)
            entity = _entity_base(
                "method", candidate, chunk.chunk_id,
                confidence=0.68 if section.section_type == "methods" else 0.55,
                extractor_model=_LOCAL_MODEL,
                embedding=embedder.embed(candidate),
                properties={
                    "method_type": _method_type(candidate, text),
                    "first_paper_id": paper.paper_id,
                },
                aliases=aliases,
                entity_id=candidate_id,
            )
            entities.setdefault(entity.entity_id, entity)

    for match in _METHOD_PHRASE_RE.finditer(text):
        candidate = _canonicalize_method_name(_clean_method_candidate(match.group(0)))
        if not _is_valid_method(candidate, text):
            continue
        candidate_id = _stable_id("method", _normalize_key(candidate))
        if candidate_id in entities:
            continue
        aliases = _find_aliases_for_method(candidate, text)
        entity = _entity_base(
            "method", candidate, chunk.chunk_id,
            confidence=0.68 if section.section_type == "methods" else 0.55,
            extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(candidate),
            properties={
                "method_type": _method_type(candidate, text),
                "first_paper_id": paper.paper_id,
            },
            aliases=aliases,
            entity_id=candidate_id,
        )
        entities.setdefault(entity.entity_id, entity)

    if section.section_type == "methods" and not entities:
        fallback = _canonicalize_method_name(_clean_method_candidate(section.title))
        if fallback and _is_valid_method(fallback, text):
            aliases = _find_aliases_for_method(fallback, text)
            entity = _entity_base(
                "method", fallback, chunk.chunk_id,
                confidence=0.52, extractor_model=_LOCAL_MODEL,
                embedding=embedder.embed(fallback),
                properties={
                    "method_type": _method_type(fallback, text),
                    "first_paper_id": paper.paper_id,
                },
                aliases=aliases,
                entity_id=_stable_id("method", _normalize_key(fallback)),
            )
            entities[entity.entity_id] = entity

    return list(entities.values())


def _claim_type(sentence: str) -> str:
    if _LIMITATION_CUES.search(sentence):
        return "limitation"
    if _HYPOTHESIS_CUES.search(sentence):
        return "future_work" if "future work" in sentence.lower() or "will" in sentence.lower() else "hypothesis"
    return "finding"


def _local_claims(chunk: ChunkRecord, embedder: TextEmbedder) -> list[Layer2EntityRecord]:
    entities: dict[str, Layer2EntityRecord] = {}
    biomedical_keywords = {
        "knockdown", "knockout", "overexpression", "mutation", "phosphorylation",
        "methylation", "acetylation", "polyubiquitination", "sumoylation", "palmitoylation",
        "ubiquitination", "expression", "transcript", "protein", "antibody", "inhibitor",
        "agonist", "antagonist", "phospho", "acetyl", "methyl", "ubiquitin",
    }
    for sentence in _split_sentences(chunk.text):
        stripped = sentence.strip()
        if not stripped or not stripped[0].isupper():
            continue
        has_claim_structure = (
            _CLAIM_VERBS.search(sentence)
            or _LIMITATION_CUES.search(sentence)
            or _HYPOTHESIS_CUES.search(sentence)
        )
        has_biomedical_keyword = any(kw in sentence.lower() for kw in biomedical_keywords)
        if not has_claim_structure and not has_biomedical_keyword:
            continue
        claim_text = _normalize(sentence)
        if len(claim_text.split()) < 4:
            continue
        claim_type = _claim_type(claim_text)
        confidence = 0.82 if _CLAIM_VERBS.search(claim_text) else 0.68 if has_biomedical_keyword else 0.55
        entity = _entity_base(
            "claim", claim_text, chunk.chunk_id,
            confidence=confidence, extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(claim_text),
            properties={"claim_type": claim_type, "text": claim_text},
            aliases=[],
            entity_id=_stable_id("claim", claim_text),
        )
        entities.setdefault(entity.entity_id, entity)
    return list(entities.values())


def _local_datasets(
    chunk: ChunkRecord,
    embedder: TextEmbedder,
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[Layer2EntityRecord]:
    entities: dict[str, Layer2EntityRecord] = {}
    for mention in _extract_dataset_mentions(chunk.text, registry=registry):
        candidate = mention["label"]
        entity = _entity_base(
            "dataset", candidate, chunk.chunk_id,
            confidence=0.84, extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(candidate),
            properties={"dataset_type": mention["dataset_type"], "text": f"Dataset: {candidate}"},
            aliases=mention.get("aliases", []),
            entity_id=_stable_id("dataset", candidate),
        )
        entities.setdefault(entity.entity_id, entity)
    return list(entities.values())


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _format_result_dataset(datasets: list[str]) -> str:
    if not datasets:
        return ""
    if len(datasets) == 1:
        return datasets[0]
    return " vs. ".join(datasets[:2]) if len(datasets) == 2 else "; ".join(datasets)


def _result_datasets(
    sentence: str,
    chunk_text: str,
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    sentence_labels = [item["label"] for item in _extract_dataset_mentions(sentence, registry=registry)]
    if sentence_labels:
        return sentence_labels
    chunk_labels = [item["label"] for item in _extract_dataset_mentions(chunk_text, registry=registry)]
    return chunk_labels[:2]


def _result_signature(value: float, metric: str, dataset: str, condition: str, chunk_id: str) -> str:
    return _stable_id("result", f"{value:.6f}", metric.lower(), dataset.lower(), condition.lower(), chunk_id)


def _build_result_entity(
    chunk: ChunkRecord,
    embedder: TextEmbedder,
    sentence: str,
    value: float,
    metric: str,
    datasets: list[str],
    unit: str = "",
    confidence: float = 0.8,
    metric_abbreviation: str = "",
    result_role: str = "primary",
) -> Layer2EntityRecord:
    dataset_label = _format_result_dataset(datasets)
    result_text = _normalize(sentence)
    result_label = f"{value:g} {metric}".strip()
    return _entity_base(
        "result", result_label, chunk.chunk_id,
        confidence=confidence, extractor_model=_LOCAL_MODEL,
        embedding=embedder.embed(result_text),
        properties={
            "value": value,
            "unit": unit,
            "dataset": dataset_label,
            "datasets": datasets,
            "metric": metric,
            "metric_abbreviation": metric_abbreviation or metric,
            "condition": dataset_label,
            "text": result_text,
            "result_role": result_role,
        },
        aliases=[],
        entity_id=_result_signature(value, metric, dataset_label, dataset_label, chunk.chunk_id),
    )


def _metric_category(metric: str) -> tuple[str, bool]:
    lowered = metric.lower()
    if any(token in lowered for token in {"accuracy", "f1", "precision", "recall", "auc", "bleu", "r2"}):
        return "classification", True
    if any(token in lowered for token in {"loss", "error", "perplexity", "rmse", "mae", "mape"}):
        return "efficiency", False
    if any(token in lowered for token in {"consensus", "agreement", "approval rate", "yes-vote share"}):
        return "decision", True
    return "other", True


def _local_results(
    section: SectionRecord,
    chunk: ChunkRecord,
    embedder: TextEmbedder,
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[Layer2EntityRecord]:
    if section.section_type == "methods":
        return []

    entities: dict[str, Layer2EntityRecord] = {}
    chunk_datasets = _result_datasets("", chunk.text, registry=registry)

    for sentence in _split_sentences(chunk.text):
        cleaned_sentence = re.sub(r"\[[^\]]+\]", "", sentence)
        cleaned_sentence = re.sub(r"\(\s*(?:Figure|Fig)\.?[^)]*\)", "", cleaned_sentence, flags=re.IGNORECASE)
        cleaned_sentence = re.sub(r"(?:Figure|Fig)\.?\s*S?\d+[A-Z]?(?:,\s*[^.)]+)?", "", cleaned_sentence, flags=re.IGNORECASE)
        cleaned_sentence = _normalize(cleaned_sentence)
        lowered = cleaned_sentence.lower()

        if "=" in cleaned_sentence and cleaned_sentence.count("(") > 2:
            continue
        if any(cue in lowered for cue in {"bootstrap", "confidence intervals", "resamples", "cat. no.", "catalog", "wavelength"}):
            continue
        datasets = _result_datasets(sentence, chunk.text, registry=registry) or chunk_datasets

        primary_found = False

        for match in _FOLD_CHANGE_RE.finditer(cleaned_sentence):
            entity = _build_result_entity(
                chunk, embedder, cleaned_sentence,
                value=float(match.group(1)), metric="Fold change",
                datasets=datasets, confidence=0.82, metric_abbreviation="fold-change",
            )
            entities.setdefault(entity.entity_id, entity)
            primary_found = True

        for match in _PERCENT_VALUE_RE.finditer(cleaned_sentence):
            window = lowered[max(0, match.start() - 60): min(len(lowered), match.end() + 80)]
            value = float(match.group(1))
            if "p value" in window or "top 5%" in window or "5% highest" in window:
                continue
            if "m6a" in window and any(token in window for token in {"percent", "percentage", "levels", "methylation"}):
                metric = "m6A percentage"
            elif "g2/m" in window or ("phase" in window and "cells" in window):
                metric = "G2/M phase percentage"
            elif any(token in window for token in {"up-regulated", "upregulated", "down-regulated", "downregulated", "differential"}):
                metric = "Differentially expressed gene percentage"
            elif "shared" in window and "genes" in lowered:
                metric = "Shared gene percentage"
            else:
                metric = "Percentage"
            entity = _build_result_entity(
                chunk, embedder, cleaned_sentence,
                value=value, metric=metric, datasets=datasets,
                unit="%", confidence=0.84, metric_abbreviation=metric.lower(),
            )
            entities.setdefault(entity.entity_id, entity)
            primary_found = True

        for match in _COUNT_VALUE_RE.finditer(cleaned_sentence):
            raw_value = match.group(1).replace(",", "")
            noun = match.group(2).lower()
            value = float(raw_value)
            window = lowered[max(0, match.start() - 70): min(len(lowered), match.end() + 80)]
            if noun.startswith("gene") and "shared" in window:
                metric = "Shared genes"
            elif noun.startswith("gene") and any(token in window for token in {"up-regulated", "upregulated", "downregulated", "differential", "de genes"}):
                metric = "Differentially expressed genes"
            elif noun.startswith("gene") and "top" in window and "ratio" in window:
                metric = "Genes with highest m6A ratio"
            elif noun.startswith("peak"):
                metric = "m6A peaks"
            elif noun.startswith("cell"):
                metric = "Cell count"
            else:
                metric = noun.capitalize()
            entity = _build_result_entity(
                chunk, embedder, cleaned_sentence,
                value=value, metric=metric, datasets=datasets,
                unit=noun, confidence=0.8, metric_abbreviation=noun,
            )
            entities.setdefault(entity.entity_id, entity)
            primary_found = True

        if _STAR_PVALUE_RE.search(cleaned_sentence):
            continue

        for match in _P_VALUE_RE.finditer(cleaned_sentence):
            if primary_found and any(token in lowered for token in {"corresponds to", "genes", "percentage", "ratio"}):
                continue
            entity = _build_result_entity(
                chunk, embedder, cleaned_sentence,
                value=float(match.group(1)), metric="P-value",
                datasets=datasets, confidence=0.66,
                metric_abbreviation="p-value", result_role="statistical_test",
            )
            entities.setdefault(entity.entity_id, entity)

        if primary_found:
            continue

        metric_match = _find_metric_match(cleaned_sentence)
        metric: str | None = None
        metric_end = 0
        if metric_match is not None:
            metric, metric_end = metric_match
        else:
            metric = _infer_metric_from_quantity_patterns(cleaned_sentence)
        if metric is None:
            continue

        metric_expanded = _expand_metric_name(metric)
        number_matches = list(_NUMERIC_RE.finditer(cleaned_sentence))
        if not number_matches:
            continue

        after_metric = [m for m in number_matches if m.start() >= metric_end] if metric_end else number_matches
        chosen_match = after_metric[0] if after_metric else number_matches[0]
        value = float(chosen_match.group(0))
        if metric_expanded == "Optical Density" and value > 5:
            continue
        if metric_expanded == "Dissociation Constant (Ki)":
            continue
        if metric_expanded == "P-value" and _STAR_PVALUE_RE.search(cleaned_sentence):
            continue

        unit = "%" if "%" in sentence else ""
        entity = _build_result_entity(
            chunk, embedder, cleaned_sentence,
            value=value, metric=metric_expanded, datasets=datasets,
            unit=unit, confidence=0.72, metric_abbreviation=metric,
            result_role="statistical_test" if metric_expanded == "P-value" else "primary",
        )
        entities.setdefault(entity.entity_id, entity)

    return list(entities.values())


def _local_equations(chunk: ChunkRecord, embedder: TextEmbedder) -> list[Layer2EntityRecord]:
    entities: dict[str, Layer2EntityRecord] = {}
    for sentence in _split_sentences(chunk.text):
        if "=" not in sentence and "≈" not in sentence and "≤" not in sentence and "≥" not in sentence:
            continue
        if not re.search(r"\b[A-Za-z][A-Za-z0-9_]*(?:\([^)]+\))?\b", sentence):
            continue
        expr = _normalize(sentence)
        if len(expr) < 8:
            continue
        entity = _entity_base(
            "equation", expr, chunk.chunk_id,
            confidence=0.55, extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(expr),
            properties={
                "latex": expr,
                "plain_desc": expr,
                "is_loss_fn": bool(re.search(r"\bloss\b|cross-entropy|negative log likelihood", expr, re.IGNORECASE)),
                "domain": "mathematics",
            },
            aliases=[],
            entity_id=_stable_id("equation", expr),
        )
        entities.setdefault(entity.entity_id, entity)
    return list(entities.values())


# ---------------------------------------------------------------------------
# Salience scoring
# ---------------------------------------------------------------------------

def _heuristic_salience(chunk: ChunkRecord, entities: list[Layer2EntityRecord]) -> float:
    if not entities:
        return 0.15

    concept_count = sum(1 for e in entities if e.entity_type == "concept")
    method_count = sum(1 for e in entities if e.entity_type == "method")
    claim_count = sum(1 for e in entities if e.entity_type == "claim")
    result_count = sum(1 for e in entities if e.entity_type == "result")
    dataset_count = sum(1 for e in entities if e.entity_type == "dataset")
    equation_count = sum(1 for e in entities if e.entity_type == "equation")

    avg_confidence = sum(e.confidence for e in entities) / len(entities)

    section_bonus = 0.0
    chunk_type_lower = chunk.chunk_type.lower()
    if "results" in chunk_type_lower or "findings" in chunk_type_lower:
        section_bonus = 0.15
    elif "discussion" in chunk_type_lower:
        section_bonus = 0.10
    elif "methods" in chunk_type_lower or "materials" in chunk_type_lower:
        section_bonus = 0.05

    score = 0.10
    if result_count > 0 and dataset_count > 0:
        score += 0.35 * min(result_count / 3.0, 1.0)
    elif result_count > 0:
        score += 0.22 * min(result_count / 3.0, 1.0)
    if claim_count > 0:
        score += 0.25 * min(claim_count / 2.0, 1.0)
    if method_count > 0:
        score += 0.15 * min(method_count / 2.0, 1.0)
    if concept_count > 0:
        score += 0.08 * min(concept_count / 3.0, 1.0)
    if dataset_count > 0:
        score += 0.03 * min(dataset_count / 2.0, 1.0)
    if equation_count > 0:
        score += 0.06 * min(equation_count / 2.0, 1.0)

    score += 0.10 * avg_confidence
    score += min(chunk.word_count / 180.0, 0.12)
    score += section_bonus

    return round(max(min(score, 0.95), 0.10), 2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _apply_schema_validation_rules(
    payload: dict[str, Any],
    *,
    schema: ExtractionSchema,
    chunk_text: str,
) -> list[str]:
    entity_type = _normalize(str(payload.get("type", ""))).lower()
    entity_schema = schema.entity_schema(entity_type)
    if entity_schema is None:
        return [f"Unsupported entity type: {entity_type}"]

    errors: list[str] = []
    for rule in entity_schema.validation_rules:
        field_name = rule.field or ""
        value = payload.get(field_name)
        if field_name == "label":
            value = payload.get("name", value)

        if rule.rule_type == "min_length":
            target = str(value or "").strip()
            if len(target) < int(rule.value or 0):
                errors.append(rule.message or f"{entity_type} {field_name} is too short.")
        elif rule.rule_type == "min_words":
            target = str(value or "").strip()
            if len(target.split()) < int(rule.value or 0):
                errors.append(rule.message or f"{entity_type} {field_name} is too short.")
        elif rule.rule_type == "numeric_value":
            if not isinstance(value, (int, float)):
                errors.append(rule.message or f"{entity_type} {field_name} must be numeric.")
        elif rule.rule_type == "metric_present":
            if not str(value or "").strip():
                errors.append(rule.message or f"{entity_type} requires a metric.")
        elif rule.rule_type == "biomedical_method":
            method_name = _normalize(str(payload.get("name", "")))
            if method_name and not _is_valid_method(method_name, chunk_text):
                errors.append(rule.message or f"Method '{method_name}' is malformed.")
        elif rule.rule_type == "physics_unit":
            if str(payload.get("metric", "")).strip() and not str(payload.get("unit", "")).strip():
                errors.append(rule.message or "Physics results should include units when available.")

    return errors


def _validate_extracted_payloads(
    payloads: list[dict[str, Any]],
    *,
    chunk_text: str,
    schema: ExtractionSchema,
) -> tuple[list[dict[str, Any]], list[str], float]:
    if not payloads:
        return [], ["No entities were extracted."], 0.0

    valid_payloads: list[dict[str, Any]] = []
    errors: list[str] = []

    for payload in payloads:
        entity_type = _normalize(str(payload.get("type", ""))).lower()
        label = _normalize(str(payload.get("name") or payload.get("text") or payload.get("latex") or ""))
        entity_errors: list[str] = []

        entity_schema = schema.entity_schema(entity_type)
        if entity_schema is None:
            errors.append(f"Unsupported entity type: {entity_type}")
            continue

        for field_name in entity_schema.required_fields:
            value = _payload_value(payload, field_name)
            if value in (None, "", [], {}):
                entity_errors.append(f"{entity_type} entities require {field_name}.")

        if entity_type == "method":
            method_name = _normalize(str(payload.get("name", "")))
            if method_name.lower() in _generic_method_names() or not _is_valid_method(method_name, chunk_text):
                entity_errors.append(f"Method '{method_name or label}' is too generic or malformed.")
        elif entity_type == "claim":
            claim_text = _normalize(str(payload.get("text", "")))
            if len(claim_text.split()) < 5:
                entity_errors.append(f"Claim '{claim_text or label}' is too short.")
        elif entity_type == "concept":
            concept_name = _normalize(str(payload.get("name", "")))
            if len(concept_name.split()) > 6:
                entity_errors.append(f"Concept '{concept_name or label}' is suspiciously long.")
        elif entity_type == "result":
            value = payload.get("value")
            if not isinstance(value, (int, float)):
                entity_errors.append(f"Result '{label}' is missing a numeric value.")

        confidence = payload.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            entity_errors.append(f"Entity '{label}' has invalid confidence '{confidence}'.")

        entity_errors.extend(_apply_schema_validation_rules(payload, schema=schema, chunk_text=chunk_text))

        if entity_errors:
            errors.extend(entity_errors)
            continue
        valid_payloads.append(payload)

    quality = len(valid_payloads) / len(payloads) if payloads else 0.0
    return valid_payloads, errors, round(quality, 3)


# ---------------------------------------------------------------------------
# LangGraph extraction pipeline (unchanged logic, uses new helpers)
# ---------------------------------------------------------------------------

def _extract_node(state: _ExtractionGraphState) -> _ExtractionGraphState:
    gemini_output = _gemini_extract_chunk_entities(
        state["paper"],
        state["section"],
        state["chunk"],
        state["schema"],
    )
    if gemini_output is None:
        return {**state, "raw_payloads": [], "validated_payloads": [], "salience_score": 0.0}
    payloads, salience_score = gemini_output
    return {**state, "raw_payloads": payloads, "validated_payloads": [], "salience_score": salience_score}


def _validation_node(state: _ExtractionGraphState) -> _ExtractionGraphState:
    valid_payloads, errors, quality = _validate_extracted_payloads(
        state["raw_payloads"],
        chunk_text=state["chunk"].text,
        schema=state["schema"],
    )
    return {**state, "validated_payloads": valid_payloads, "validation_errors": errors, "extraction_quality_score": quality}


def _should_retry_extraction(state: _ExtractionGraphState) -> str:
    if state["retry_count"] >= 2:
        return "finalize"
    if not state["validated_payloads"]:
        return "self_correct"
    if state["extraction_quality_score"] < 0.6:
        return "self_correct"
    if len(state["validation_errors"]) > 1:
        return "self_correct"
    return "finalize"


def _self_correction_node(state: _ExtractionGraphState) -> _ExtractionGraphState:
    import time
    # Back off before retrying to avoid immediate rate-limit cascades.
    time.sleep(0.5 * (state["retry_count"] + 1))

    correction_prompt = _build_gemini_prompt(
        state["paper"],
        state["section"],
        state["chunk"],
        state["schema"],
        validation_errors=state["validation_errors"],
        previous_payloads=state["raw_payloads"],
        salience_score=state["salience_score"],
    )
    try:
        corrected = _GeminiChunkExtractionPayload.model_validate(
            generate_json(correction_prompt, model_name=state["model_name"], temperature=0.1),
            context={"schema": state["schema"]},
        )
        return {
            **state,
            "raw_payloads": [e.model_dump(exclude_none=True, mode="json") for e in corrected.entities],
            "salience_score": corrected.salience_score,
            "retry_count": state["retry_count"] + 1,
        }
    except (GeminiError, ValidationError, ValueError, TypeError, json.JSONDecodeError):
        return {**state, "retry_count": state["retry_count"] + 1}


def _finalize_extraction_node(state: _ExtractionGraphState) -> _ExtractionGraphState:
    final_payloads = state["validated_payloads"] or state["raw_payloads"]
    return {**state, "raw_payloads": final_payloads}


@lru_cache(maxsize=1)
def _build_chunk_extraction_graph() -> Any | None:
    if StateGraph is None or START is None or END is None:
        return None
    graph = StateGraph(_ExtractionGraphState)
    graph.add_node("extract", _extract_node)
    graph.add_node("validate", _validation_node)
    graph.add_node("self_correct", _self_correction_node)
    graph.add_node("finalize", _finalize_extraction_node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        _should_retry_extraction,
        {"self_correct": "self_correct", "finalize": "finalize"},
    )
    graph.add_edge("self_correct", "validate")
    graph.add_edge("finalize", END)
    return graph.compile(name="chunk_extraction_graph")


def _run_chunk_extraction_graph(
    paper: PaperRecord,
    section: SectionRecord,
    chunk: ChunkRecord,
    schema: ExtractionSchema,
) -> tuple[list[dict[str, Any]], float] | None:
    model_name = os.getenv("EXTRACT_MODEL", "gemini-2.5-flash")
    model_version = _extraction_model_version(schema, True)
    cache = get_extraction_cache()
    cached = cache.get(chunk.chunk_id, chunk.text, model_version)
    if isinstance(cached, dict):
        cached_payloads = cached.get("entities")
        if isinstance(cached_payloads, list):
            return cached_payloads, float(cached.get("salience_score", 0.0) or 0.0)

    graph = _build_chunk_extraction_graph()
    if graph is None:
        result = _gemini_extract_chunk_entities(paper, section, chunk, schema)
        if result is not None:
            cache.set(
                chunk.chunk_id,
                chunk.text,
                model_version,
                {"entities": result[0], "salience_score": result[1]},
            )
        return result
    final_state = graph.invoke({
        "paper": paper,
        "section": section,
        "chunk": chunk,
        "schema": schema,
        "model_name": model_name,
        "model_version": model_version,
        "raw_payloads": [],
        "validated_payloads": [],
        "salience_score": 0.0,
        "validation_errors": [],
        "extraction_quality_score": 0.0,
        "retry_count": 0,
    })
    payloads = final_state["validated_payloads"] or final_state["raw_payloads"]
    if not payloads:
        return None
    cache.set(
        chunk.chunk_id,
        chunk.text,
        model_version,
        {"entities": payloads, "salience_score": final_state["salience_score"]},
    )
    return payloads, final_state["salience_score"]


def _gemini_extract_chunk_entities(
    paper: PaperRecord,
    section: SectionRecord,
    chunk: ChunkRecord,
    schema: ExtractionSchema,
) -> tuple[list[dict[str, Any]], float] | None:
    if not gemini_available():
        return None
    model_name = os.getenv("EXTRACT_MODEL", "gemini-2.5-flash")
    prompt = _build_gemini_prompt(paper, section, chunk, schema)
    try:
        tracer = get_tracing_manager()
        payload = _GeminiChunkExtractionPayload.model_validate(
            generate_json(prompt, model_name=model_name, temperature=0.1),
            context={"schema": schema},
        )
        tracer.log_llm_call(
            name="entity_extraction",
            model=model_name,
            prompt=prompt[:500],
            response=json.dumps(payload.model_dump(mode="json"))[:500],
            metadata={"paper_id": paper.paper_id, "chunk_id": chunk.chunk_id, "section_title": section.title},
        )
        return [e.model_dump(exclude_none=True, mode="json") for e in payload.entities], payload.salience_score
    except (GeminiError, ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        tracer = get_tracing_manager()
        if tracer.enabled:
            tracer.langfuse.event(
                name="entity_extraction_error",
                output={"error": str(exc)},
                metadata={"paper_id": paper.paper_id, "chunk_id": chunk.chunk_id},
            )
        return None


def _gemini_to_entity(
    raw: dict[str, Any],
    chunk: ChunkRecord,
    embedder: TextEmbedder,
) -> Layer2EntityRecord | None:
    entity_type = _normalize(str(raw.get("type", ""))).lower()
    if entity_type not in {"concept", "method", "claim", "result", "equation", "dataset"}:
        return None

    if entity_type in {"concept", "method", "dataset"}:
        label = _normalize(str(raw.get("name") or raw.get("label") or ""))
    elif entity_type == "claim":
        label = _normalize(str(raw.get("text", "")))
    elif entity_type == "result":
        value = raw.get("value")
        metric = _normalize(str(raw.get("metric", "")))
        label = f"{value} {metric}".strip()
    else:
        label = _normalize(str(raw.get("latex", "")))

    if not label:
        return None

    properties = {k: v for k, v in raw.items() if k not in {"type", "name", "text", "aliases", "confidence"}}
    if entity_type == "claim":
        properties.setdefault("claim_type", raw.get("claim_type", "finding"))
        properties["text"] = label
    elif entity_type == "result":
        properties.setdefault("text", raw.get("text", label))
        properties.setdefault("dataset", raw.get("dataset", ""))
        properties.setdefault("metric", raw.get("metric", ""))
        properties.setdefault("condition", raw.get("condition", ""))
    elif entity_type == "equation":
        properties.setdefault("latex", raw.get("latex", label))
        properties.setdefault("plain_desc", raw.get("plain_desc", label))
        properties.setdefault("is_loss_fn", raw.get("is_loss_fn", False))
        properties.setdefault("domain", raw.get("domain", "mathematics"))
    elif entity_type == "dataset":
        properties.setdefault("dataset_type", raw.get("dataset_type", "experimental_group"))
        properties.setdefault("text", raw.get("text", label))
    else:
        properties.setdefault("ontology", raw.get("ontology", ""))

    aliases = [str(a) for a in raw.get("aliases", []) if str(a).strip()]
    confidence = float(raw.get("confidence", 0.5) or 0.5)

    if entity_type == "result":
        entity_id = _result_signature(
            float(raw.get("value", 0.0) or 0.0),
            str(raw.get("metric", "")),
            str(raw.get("dataset", "")),
            str(raw.get("condition", "")),
            chunk.chunk_id,
        )
    else:
        entity_id = _stable_id(entity_type, _normalize_key(label))

    return _entity_base(
        entity_type, label, chunk.chunk_id,
        confidence=confidence,
        extractor_model=os.getenv("EXTRACT_MODEL", "gemini-2.0-flash"),
        embedding=embedder.embed(label),
        properties=properties,
        aliases=aliases,
        entity_id=entity_id,
    )


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_layer2(
    paper: PaperRecord,
    settings: Phase1Settings | None = None,
    use_gemini: bool | None = None,
    update_chunks: bool = True,
) -> Layer2DocumentRecord:
    """Extract Layer 2 entities from a Layer 1 paper record."""

    tracer = get_tracing_manager()
    settings = settings or Phase1Settings.from_env()
    use_gemini = use_gemini if use_gemini is not None else os.getenv("USE_GEMINI_EXTRACTION", "0") == "1"
    schema = load_schema(detect_domain(paper))
    embedder = build_entity_embedder(dim=settings.embedding_dim, prefer_remote=use_gemini)
    section_lookup = {section.section_id: section for section in paper.sections}
    dataset_registry = _detect_experimental_groups(paper)

    # global_entities keyed by normalized merge key (not raw entity_id)
    # to catch label-case duplicates like "MeRIP-seq" vs "merip-seq".
    global_entities: dict[str, Layer2EntityRecord] = {}
    chunk_entity_ids: dict[str, list[str]] = {}
    chunk_salience_scores: dict[str, float] = {}
    extractor_model = _extraction_model_version(schema, use_gemini)

    with tracer.trace(
        name="extract_layer2",
        input_data={"paper_id": paper.paper_id, "chunk_count": len(paper.chunks), "schema_domain": schema.domain},
    ) as trace_ctx:
        for chunk in paper.chunks:
            section = section_lookup.get(chunk.section_id)
            if section is None:
                continue

            raw_entities: list[Layer2EntityRecord] = []
            gemini_salience: float | None = None

            if use_gemini:
                gemini_output = _run_chunk_extraction_graph(paper, section, chunk, schema)
                if gemini_output is not None:
                    raw_payloads, gemini_salience = gemini_output
                    for raw in raw_payloads:
                        entity = _gemini_to_entity(raw, chunk, embedder)
                        if entity is not None:
                            raw_entities.append(entity)

            if not raw_entities:
                raw_entities.extend(_local_concepts(paper, section, chunk, embedder))
                raw_entities.extend(_local_methods(paper, section, chunk, embedder))
                raw_entities.extend(_local_claims(chunk, embedder))
                raw_entities.extend(_local_datasets(chunk, embedder, registry=dataset_registry))
                raw_entities.extend(_local_results(section, chunk, embedder, registry=dataset_registry))
                raw_entities.extend(_local_equations(chunk, embedder))

            chunk_entity_ids[chunk.chunk_id] = [e.entity_id for e in raw_entities]

            for entity in raw_entities:
                key = _merge_key(entity.entity_type, entity.label)
                if key in global_entities:
                    _merge_entities(global_entities[key], entity)
                else:
                    global_entities[key] = entity

            salience = gemini_salience if gemini_salience is not None else _heuristic_salience(chunk, raw_entities)
            chunk_salience_scores[chunk.chunk_id] = salience
            if update_chunks:
                chunk.salience_score = max(chunk.salience_score, salience)

        entity_types: dict[str, int] = {}
        for entity in global_entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1

        tracer.log_entity_extraction(
            paper_id=paper.paper_id,
            entity_count=len(global_entities),
            entity_types=entity_types,
        )
        trace_ctx["output"] = {
            "entity_count": len(global_entities),
            "entity_types": entity_types,
            "schema_domain": schema.domain,
        }

    result = Layer2DocumentRecord(
        paper_id=paper.paper_id,
        extractor_model=extractor_model,
        entities=sorted(global_entities.values(), key=lambda e: (e.entity_type, e.label.lower())),
        chunk_entity_ids=chunk_entity_ids,
        chunk_salience_scores=chunk_salience_scores,
    )
    tracer.flush()
    return result
