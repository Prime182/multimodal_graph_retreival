"""Layer 2 entity extraction for the GraphRAG pipeline."""

from __future__ import annotations

from hashlib import sha1
import json
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .config import Phase1Settings
from .corpus import enrich_entity, get_hierarchy, CorpusMatch
from .embeddings import TextEmbedder, build_entity_embedder
from .entities import Layer2DocumentRecord, Layer2EntityRecord
from .gemini import generate_json, gemini_available
from .models import ChunkRecord, PaperRecord, SectionRecord
from .tracing import get_tracing_manager


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_CONCEPT_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){1,4}|[A-Z]{2,}(?:-[A-Z0-9]+)?)\b"
)
_METHOD_PHRASE_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9-]{1,}\s+){0,4}(?:model|method|approach|framework|algorithm|pipeline|architecture|simulation|protocol|process|system|assay|technique|procedure|analysis|sequencing|microscopy|chromatography|spectroscopy|imaging|screening|measurement|quantification)\b",
    re.IGNORECASE,
)
_BIOMEDICAL_METHOD_RE = re.compile(
    r"\b(?:chip-?seq|rna-?seq|dna-?seq|crispra?|crispr/cas9?|meriP|flow-?cytometry?|mass-?spec|lc-?ms|gc-?ms|hplc|immunoblot|western-?blot|immunofluorescence|facs|co-?ip|yeast-?two-?hybrid|elisa|qpcr|qrt-?pcr|rt-?pcr|rnai|sirna|morpholino|knockout|knockdown|conditional-?knockout|transgenic|reporter-?assay|luciferase-?assay|chromatin-?ip|cut-?and-?run|atac-?seq|dnase-?seq|capture-?c|hi-?c|3c|4c|5c|single-?cell-?seq|10x-?genomics?|single-?cell-?rna|bisulfite-?seq|methylation-?array|gwas|exome-?seq|whole-?genome|microarray|proteomics|metabolomics|lipidomics|glycomics|deseq2?|trimmomatic|bowtie2?|bwa|samtools|bedtools|picard|gatk|biopython|matrigel|cell-?invasion|trans-?well)\b",
    re.IGNORECASE,
)
_KNOWN_BIOMEDICAL_METHODS = {
    "DESeq2": ["deseq2", "deseq", "differential expression analysis"],
    "trimmomatic": ["trimmomatic", "read trimming"],
    "bowtie2": ["bowtie2", "bowtie", "sequence alignment"],
    "BWA": ["bwa", "burrows-wheeler aligner"],
    "SAMtools": ["samtools", "sam tools", "bam processing"],
    "BEDtools": ["bedtools", "bed tools", "genomic arithmetic"],
    "Matrigel invasion assay": ["matrigel invasion", "invasion assay", "cell invasion"],
}
_METHOD_CANONICAL_MAP = {
    "a-seq and the macs2 algorithm": "MACS2",
    "cell cycle analysis": "Cell cycle analysis",
    "chip-seq": "ChIP-Seq",
    "crispr": "CRISPR",
    "crispr-cas9": "CRISPR",
    "crispr/cas9": "CRISPR",
    "crispra": "CRISPRa",
    "deseq": "DESeq2",
    "deseq2": "DESeq2",
    "elisa": "ELISA",
    "facs": "FACS",
    "facs analysis": "FACS",
    "flow cytometry": "Flow cytometry",
    "immunoblot": "Western blot",
    "matrigel": "Matrigel chamber assay",
    "matrigel chamber assay": "Matrigel chamber assay",
    "me-rip-seq": "MeRIP-seq",
    "merip": "MeRIP-seq",
    "merip-seq": "MeRIP-seq",
    "meripseq and data analysis": "MeRIP-seq",
    "meripseq": "MeRIP-seq",
    "meRIP": "MeRIP-seq",
    "model-based analysis": "MACS2",
    "macs2": "MACS2",
    "m6a sequencing": "m6A sequencing",
    "qpcr": "qPCR",
    "qrt-pcr": "qRT-PCR",
    "rt-pcr": "RT-PCR",
    "rna-seq": "RNA-Seq",
    "rnaseq": "RNA-Seq",
    "sequencing and bioinformatics analysis": "Bioinformatics analysis",
    "western blot": "Western blot",
    "western blot analysis": "Western blot analysis",
}
_METHOD_FRAGMENT_EXCLUSIONS = {
    "4c",
    "analysis",
    "approach",
    "assay",
    "bulk measurement",
    "ensuring precise detection and quantification",
    "for this assay",
    "in our analysis",
    "in this procedure",
    "materials and methods",
    "mammalian species and some model",
    "method",
    "our analysis",
    "per the manufacturer and sequencing",
    "procedure",
    "system",
    "the analysis",
    "the assay",
    "technique",
    "this assay",
    "uniform method",
    "well-rounded approach",
}
_METHOD_TYPE_OVERRIDES = {
    "bowtie2": "computational",
    "Cell cycle analysis": "experimental",
    "ChIP-Seq": "experimental",
    "CRISPR": "experimental",
    "CRISPRa": "experimental",
    "DESeq2": "statistical",
    "ELISA": "experimental",
    "FACS": "experimental",
    "Flow cytometry": "experimental",
    "MACS2": "computational",
    "MeRIP-seq": "experimental",
    "Matrigel chamber assay": "experimental",
    "qPCR": "experimental",
    "qRT-PCR and western blot analysis": "experimental",
    "qRT-PCR": "experimental",
    "RNA-Seq": "experimental",
    "Western blot": "experimental",
    "Western blot analysis": "experimental",
}
_CLAIM_VERBS = re.compile(
    r"\b(achieves?|demonstrates?|shows?|reveals?|suggests?|indicates?|improves?|reduces?|increases?|decreases?|supports?|confirms?|enables?|outperforms?|leads to|results in|causes?)\b",
    re.IGNORECASE,
)
_METRIC_LABELS = [
    "yes-vote share",
    "yes-vote",
    "yes-rate",
    "adoption rate",
    "adoption",
    "approval rate",
    "global consensus",
    "consensus",
    "agreement",
    "support",
    "polarization",
    "attitude",
    "opinion",
    "performance",
    "accuracy",
    "precision",
    "recall",
    "f1 score",
    "f1",
    "bleu",
    "auc",
    "loss",
    "error",
    "perplexity",
    "rmse",
    "mae",
    "mape",
    "r2",
    # Biomedical metrics
    "p-value",
    "p value",
    "fold-change",
    "fold change",
    "log2-fold-change",
    "log2 fold change",
    "ec50",
    "ic50",
    "ki",
    "kd",
    "km",
    "vmax",
    "significance",
    "statistical significance",
    "expression level",
    "gene expression",
    "mrna level",
    "protein level",
    "phosphorylation",
    "methylation",
    "acetylation",
    "ubiquitination",
    "sumoylation",
    "palmitoylation",
    "proliferation rate",
    "proliferation",
    "cell viability",
    "cell survival",
    "apoptosis",
    "cell death",
    "differentiation",
    "migration",
    "invasion",
    "migration rate",
    "invasion rate",
    "wound closure",
    "binding affinity",
    "dissociation constant",
    "association rate",
    "fluorescence intensity",
    "intensity",
    "signal intensity",
    "rfu",
    "relative fluorescence",
    "luminescence",
    "absorbance",
    "optical density",
    "od",
    "percent phosphorylated",
    "percent methylated",
    "percent acetylated",
    "half-life",
    "half life",
    "t1/2",
    "area under curve",
    "concentration",
    "micromolar",
    "nanomolar",
    "picomolar",
    "copy number",
    "read count",
    "transcript",
    "exon",
    "splice junction",
    "g2/m phase",
    "g2 m phase",
    "cell cycle phase",
    "m6a percentage",
    "methylation percentage",
]

# Metric abbreviation expansions
_METRIC_EXPANSIONS = {
    "od": "Optical Density",
    "ki": "Dissociation Constant (Ki)",
    "kd": "Dissociation Constant (Kd)",
    "km": "Michaelis Constant (Km)",
    "vmax": "Maximum Velocity (Vmax)",
    "ec50": "Half-maximal Effective Concentration (EC50)",
    "ic50": "Half-maximal Inhibitory Concentration (IC50)",
    "rfu": "Relative Fluorescence Units",
    "auc": "Area Under Curve",
    "rmse": "Root Mean Square Error",
    "mae": "Mean Absolute Error",
    "mape": "Mean Absolute Percentage Error",
    "r2": "Coefficient of Determination (R²)",
    "p-value": "P-value",
    "f1": "F1 Score",
    "bleu": "BLEU Score",
}
_METRIC_REGEX_CACHE = [
    (
        label,
        re.compile(
            r"(?<![A-Za-z0-9])"
            + re.escape(label).replace(r"\ ", r"\s+")
            + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    )
    for label in sorted(_METRIC_LABELS, key=len, reverse=True)
]
_LIMITATION_CUES = re.compile(
    r"\b(limitations?|however|drawback|cannot|fails to|challenge|challenges|weakness)\b",
    re.IGNORECASE,
)
_HYPOTHESIS_CUES = re.compile(
    r"\b(hypothesize|hypothesis|propose|proposed|we expect|future work|future studies|will)\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")
_DATASET_PATTERNS = [
    re.compile(r"\b(?:in|on|for|under)\s+(Scenario\s+\d+)\b", re.IGNORECASE),
    re.compile(r"\b(?:in|on|for|under)\s+([A-Z][A-Za-z0-9-]*(?:\s+[A-Z0-9][A-Za-z0-9-]*){0,4})\b"),
    # Biomedical: cell lines (all caps or with numbers)
    re.compile(r"\b(HEK293|HEK-293|293T|CHO|COS|COS7|CV-1|NIH-3T3|NIH3T3|HeLa|HepG2|A549|MCF7|BT474|MDCK|VERO|HT1080|Jurkat|Molt4|K562|HL60|U937|THP1|THP-1|Ba/F3|Ba/f3|BaF3|IL3|IL-3|IL3-dependent|IL-3-dependent)\b", re.IGNORECASE),
    # Biomedical: macrophage/immune cell types
    re.compile(r"\b([A-Z]{1,3}\d+\s+macrophages|bone\s+marrow\s+(?:derived\s+)?macrophages|bmdms|peritoneal\s+macrophages|alveolar\s+macrophages|m1\s+macrophages?|m2\s+macrophages?|primary\s+(?:human\s+)?macrophages?|THP-1|U937|dendritic\s+cells?|lymphocytes?|CD4\+|CD8\+)\b", re.IGNORECASE),
    # Biomedical: sample identifiers
    re.compile(r"\b(patient\s+cohort|sample\s+cohort|control\s+group|treated\s+group|wildtype|knockout|knockdown|transgenic|mutant)\b", re.IGNORECASE),
]
_DATASET_REGISTRY = {
    "a": {
        "label": "attenuated macrophages (A)",
        "aliases": ["A", "attenuated macrophages", "attenuated infected macrophages"],
        "dataset_type": "macrophage_state",
    },
    "am": {
        "label": "merozoite-producing attenuated macrophages (Am)",
        "aliases": ["Am", "attenuated merozoite-producing macrophages"],
        "dataset_type": "macrophage_state",
    },
    "bl20": {
        "label": "uninfected B cells (BL20)",
        "aliases": ["BL20", "non-infected bovine B cells", "uninfected bovine B cells"],
        "dataset_type": "b_cell_line",
    },
    "bl3": {
        "label": "uninfected B cells (BL3)",
        "aliases": ["BL3", "uninfected bovine B cells"],
        "dataset_type": "b_cell_line",
    },
    "ode": {
        "label": "Ode macrophages",
        "aliases": ["Ode", "Ode cell line"],
        "dataset_type": "macrophage_cell_line",
    },
    "scenario": {
        "label": "",
        "aliases": [],
        "dataset_type": "scenario",
    },
    "tbl20": {
        "label": "infected B cells (TBL20)",
        "aliases": ["TBL20", "infected bovine B cells", "Theileria-infected B cells"],
        "dataset_type": "b_cell_line",
    },
    "tbl3": {
        "label": "infected B cells (TBL3)",
        "aliases": ["TBL3", "Theileria-infected B cells"],
        "dataset_type": "b_cell_line",
    },
    "v": {
        "label": "virulent macrophages (V)",
        "aliases": ["V", "virulent macrophages", "virulent infected macrophages", "virulent bovine macrophages"],
        "dataset_type": "macrophage_state",
    },
    "vm": {
        "label": "merozoite-producing macrophages (Vm)",
        "aliases": ["Vm", "merozoite-producing macrophages", "merozoite-producing bovine macrophages"],
        "dataset_type": "macrophage_state",
    },
}
_DATASET_DEFINITION_PATTERNS = [
    (re.compile(r"\bvirulent(?:\s+infected|\s+bovine)?\s+macrophages?\s*\((V)\)", re.IGNORECASE), "v"),
    (re.compile(r"\battenuated(?:\s+infected|\s+bovine)?\s+macrophages?\s*\((A)\)", re.IGNORECASE), "a"),
    (re.compile(r"\bmerozoite-producing(?:\s+bovine)?\s+macrophages?\s*\((Vm)\)", re.IGNORECASE), "vm"),
    (re.compile(r"\bmerozoite-producing(?:\s+bovine)?\s+macrophages?\s*\((Am)\)", re.IGNORECASE), "am"),
    (re.compile(r"\binfected(?:\s+bovine)?\s+B[- ]?cells?\s*\((TBL20)\)", re.IGNORECASE), "tbl20"),
    (re.compile(r"\bnon-?infected(?:\s+bovine)?\s+B[- ]?cells?\s*\((BL20)\)", re.IGNORECASE), "bl20"),
    (re.compile(r"\buninfected(?:\s+bovine)?\s+B[- ]?cells?\s*\((BL20)\)", re.IGNORECASE), "bl20"),
]
_DATASET_DIRECT_PATTERNS = [
    (re.compile(r"\bOde\s+(?:cell\s+line|macrophages?)\b", re.IGNORECASE), "ode"),
    (re.compile(r"\bTBL20\b", re.IGNORECASE), "tbl20"),
    (re.compile(r"\bBL20\b", re.IGNORECASE), "bl20"),
    (re.compile(r"\bTBL3\b", re.IGNORECASE), "tbl3"),
    (re.compile(r"\bBL3\b", re.IGNORECASE), "bl3"),
    (re.compile(r"\bvirulent\s+(?:infected\s+)?macrophages?\b", re.IGNORECASE), "v"),
    (re.compile(r"\battenuated\s+(?:infected\s+)?macrophages?\b", re.IGNORECASE), "a"),
    (re.compile(r"\bmerozoite-producing\s+macrophages?\b", re.IGNORECASE), "vm"),
]
_DATASET_COMPARISON_RE = re.compile(r"\b(Vm|Am|TBL20|BL20|TBL3|BL3|V|A)\s*(?:vs\.?|versus)\s*(Vm|Am|TBL20|BL20|TBL3|BL3|V|A)\b", re.IGNORECASE)
_DATASET_PAIR_RE = re.compile(r"\b(?:between|both)\s+(BL20|TBL20|BL3|TBL3|V|A|Vm|Am)\s+and\s+(BL20|TBL20|BL3|TBL3|V|A|Vm|Am)\b", re.IGNORECASE)
_P_VALUE_RE = re.compile(r"\bp[-\s]?value(?:s)?\s*(?:of|:|=|<|>|≤|≥)?\s*(\d*\.\d+)\b", re.IGNORECASE)
_STAR_PVALUE_RE = re.compile(r"\*+\s*p\s*value", re.IGNORECASE)
_PERCENT_VALUE_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_COUNT_VALUE_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})*|\d+)\s+(genes?|cells?|peaks?)\b", re.IGNORECASE)
_FOLD_CHANGE_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(?:fold(?:-change)?|fold change)\b", re.IGNORECASE)
_GENE_SYMBOL_RE = re.compile(
    r"\b(?:ADAM19|ALKBH5|CCND1|CDC2|CDC25B|CDK1|CCNB1|CCNB2|BUB1|BUB1B|E2F4|FTO|HIF1A|HIF-1α|HNRNPA2B1|MACS2|METTL3|METTL14|MTOR|RBM15|SIRT5|SREBF1|WTAP|YTHDC1|YTHDF1|YTHDF3)\b",
    re.IGNORECASE,
)
_CONCEPT_CANONICAL_MAP = {
    "adam19": "ADAM19",
    "alkbh5": "ALKBH5",
    "e2f4": "E2F4",
    "fto": "FTO",
    "hif1a": "HIF-1α",
    "hif1α": "HIF-1α",
    "hif-1α": "HIF-1α",
    "hnrnpa2b1": "HNRNPA2B1",
    "macs2": "MACS2",
    "mettl3": "METTL3",
    "mettl14": "METTL14",
    "rbm15": "RBM15",
    "wtap": "WTAP",
    "ythdc1": "YTHDC1",
    "ythdf1": "YTHDF1",
    "ythdf3": "YTHDF3",
}
_CONCEPT_ALIAS_MAP = {
    "ADAM19": ["A disintegrin and metalloproteinase 19"],
    "ALKBH5": ["alkB homolog 5"],
    "HIF-1α": ["HIF1A", "hypoxia-inducible factor 1-alpha"],
    "HNRNPA2B1": ["heterogeneous nuclear ribonucleoprotein A2/B1"],
    "METTL3": ["methyltransferase like 3", "N6-adenosine-methyltransferase catalytic subunit"],
    "METTL14": ["methyltransferase like 14"],
    "RBM15": ["RNA-binding motif protein 15"],
    "WTAP": ["Wilms' Tumor 1-Associating Protein", "Wilms tumor 1-associating protein"],
    "YTHDC1": ["YTH domain containing 1"],
    "YTHDF1": ["YTH N6-methyladenosine RNA binding protein 1"],
    "YTHDF3": ["YTH N6-methyladenosine RNA binding protein 3"],
}
_EQUATION_PATTERNS = [
    re.compile(
        r"(?P<expr>(?:[A-Za-z][A-Za-z0-9_]*(?:\([^)]+\))?|\([^)]+\))\s*(?:=|≈|<=|>=|<|>)\s*(?:[A-Za-z0-9_+\-*/().^ ]+))"
    ),
    re.compile(
        r"(?P<expr>(?:[A-Za-z][A-Za-z0-9_]*(?:\([^)]+\))?(?:\s*[+\-*/]\s*[A-Za-z0-9_()]+)+))"
    ),
]
_LOCAL_MODEL = "heuristic-v2"
_GEMINI_PROMPT = """You are a scientific knowledge extractor. Given a text chunk from a research paper, extract all scientific entities.
Return ONLY valid JSON with this schema:
{
  "entities": [
    {
      "type": "concept|method|claim|result|equation",
      "name": "...",
      "text": "...",
      "claim_type": "finding|hypothesis|limitation|future_work",
      "value": 0.0,
      "unit": "...",
      "dataset": "...",
      "metric": "...",
      "condition": "...",
      "latex": "...",
      "plain_desc": "...",
      "is_loss_fn": false,
      "aliases": [],
      "confidence": 0.0
    }
  ]
}"""


class _GeminiEntityPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None
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
    def _validate_type(cls, value: str) -> str:
        normalized = _normalize(value).lower()
        if normalized not in {"concept", "method", "claim", "result", "equation"}:
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
    def _validate_shape(self) -> "_GeminiEntityPayload":
        if self.type in {"concept", "method"} and not self.name:
            raise ValueError(f"{self.type} entities require a name")
        if self.type == "claim" and not self.text:
            raise ValueError("claim entities require text")
        if self.type == "result":
            if self.value is None:
                raise ValueError("result entities require a numeric value")
            if not self.metric:
                raise ValueError("result entities require a metric")
        if self.type == "equation" and not self.latex:
            raise ValueError("equation entities require latex")
        return self


class _GeminiChunkExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: list[_GeminiEntityPayload] = Field(default_factory=list)
    salience_score: float = Field(default=0.0, ge=0.0, le=1.0)


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


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


def _canonicalize_method_name(candidate: str) -> str:
    lowered = _normalize(candidate).lower()
    return _METHOD_CANONICAL_MAP.get(lowered, candidate.strip())


def _canonicalize_concept_name(candidate: str) -> str:
    lowered = _normalize(candidate).lower()
    lowered = lowered.replace("−", "-")
    return _CONCEPT_CANONICAL_MAP.get(lowered, candidate.strip())


def _dataset_info(key: str, scenario_label: str | None = None) -> dict[str, Any]:
    if key == "scenario" and scenario_label:
        return {
            "label": scenario_label,
            "aliases": [scenario_label],
            "dataset_type": _DATASET_REGISTRY["scenario"]["dataset_type"],
        }
    return _DATASET_REGISTRY[key]


def _append_dataset(mentions: dict[str, dict[str, Any]], key: str, scenario_label: str | None = None) -> None:
    info = _dataset_info(key, scenario_label=scenario_label)
    label = info["label"]
    if not label:
        return
    aliases = _unique([alias for alias in info.get("aliases", []) if alias and alias != label])
    mentions[label] = {
        "label": label,
        "aliases": aliases,
        "dataset_type": info["dataset_type"],
    }


def _extract_dataset_mentions(text: str) -> list[dict[str, Any]]:
    mentions: dict[str, dict[str, Any]] = {}

    for match in _DATASET_PATTERNS[0].finditer(text):
        scenario = _normalize(match.group(1))
        if scenario:
            _append_dataset(mentions, "scenario", scenario_label=scenario)

    for pattern, key in _DATASET_DEFINITION_PATTERNS:
        if pattern.search(text):
            _append_dataset(mentions, key)

    for pattern, key in _DATASET_DIRECT_PATTERNS:
        if pattern.search(text):
            _append_dataset(mentions, key)

    for match in _DATASET_COMPARISON_RE.finditer(text):
        _append_dataset(mentions, match.group(1).lower())
        _append_dataset(mentions, match.group(2).lower())

    for match in _DATASET_PAIR_RE.finditer(text):
        _append_dataset(mentions, match.group(1).lower())
        _append_dataset(mentions, match.group(2).lower())

    for pattern in _DATASET_PATTERNS[2:]:
        for match in pattern.finditer(text):
            candidate = _normalize(match.group(0))
            if candidate and len(candidate) > 2 and _looks_like_dataset(candidate):
                mentions.setdefault(
                    candidate,
                    {
                        "label": candidate,
                        "aliases": [],
                        "dataset_type": "dataset",
                    },
                )

    return list(mentions.values())


def _text_aliases(term: str, text_context: str) -> list[str]:
    aliases: set[str] = set()
    escaped_term = re.escape(term)

    if re.fullmatch(r"[A-Z0-9α-]+", term):
        full_name_pattern = re.compile(rf"([A-Z][A-Za-z0-9'/-]+(?:\s+[A-Z][A-Za-z0-9'/-]+){{1,8}})\s*\(\s*{escaped_term}\s*\)")
        for match in full_name_pattern.finditer(text_context):
            aliases.add(_normalize(match.group(1)))
    else:
        abbrev_pattern = re.compile(rf"{escaped_term}\s*\(\s*([A-Z][A-Z0-9α-]{{1,12}})\s*\)")
        for match in abbrev_pattern.finditer(text_context):
            aliases.add(_normalize(match.group(1)))

    return list(aliases)


def _format_result_dataset(datasets: list[str]) -> str:
    if not datasets:
        return ""
    if len(datasets) == 1:
        return datasets[0]
    return " vs. ".join(datasets[:2]) if len(datasets) == 2 else "; ".join(datasets)


def _result_datasets(sentence: str, chunk_text: str) -> list[str]:
    sentence_labels = [item["label"] for item in _extract_dataset_mentions(sentence)]
    if sentence_labels:
        return sentence_labels
    chunk_labels = [item["label"] for item in _extract_dataset_mentions(chunk_text)]
    return chunk_labels[:2]


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
        "result",
        result_label,
        chunk.chunk_id,
        confidence=confidence,
        extractor_model=_LOCAL_MODEL,
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


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _is_valid_method(candidate: str, context: str = "") -> bool:
    """Validate if extracted candidate is a genuine technique name, not a fragment or phrase."""
    candidate = _normalize(candidate)
    lowered = candidate.lower()

    if lowered in _METHOD_FRAGMENT_EXCLUSIONS:
        return False

    if candidate != _canonicalize_method_name(candidate) and _canonicalize_method_name(candidate) in {"Matrigel chamber assay", "MeRIP-seq"}:
        return True

    if lowered == "4c" and ("°c" in context.lower() or "degrees" in context.lower()):
        return False
    
    # Reject single words that are clearly not techniques
    single_word_exclusions = {
        "system", "process", "approach", "method", "analysis", "study", "work",
        "we", "it", "this", "our", "well", "in", "on", "by", "with", "to", "from",
        "model", "framework", "procedure", "technique", "analysis", "assay",
    }
    
    # For single-word candidates, be very restrictive
    if len(lowered.split()) == 1:
        if lowered in single_word_exclusions:
            return False
        # Allow single words only if they're in known methods or very confidence-building terms
        known_single = {"crispr", "elisa", "facs", "gwas", "hplc", "qpcr", "rna-seq", "dna-seq", "chip-seq"}
        return lowered in known_single or re.search(r"seq|assay|technique|protocol", context, re.IGNORECASE) is not None
    
    # Reject fragments that start with weak predicates ending the previous sentence
    fragment_signals = {
        "well with", "contradicted by", "regulates the", "we discovered", "in our analysis",
        "we found", "we used", "we performed", "we conducted", "shows that the",
        "demonstrates that", "indicates that", "reveals", "suggests", "well-known",
        "using ", "for this", "this assay", "this method", "our analysis",
    }
    for signal in fragment_signals:
        if candidate.startswith(signal):
            return False
    
    # Reject candidates that look like verbal phrases (end in -ing without being a technique)
    if lowered.endswith("ing"):
        if not any(tech in lowered for tech in {"sequencing", "screening", "imaging", "processing", "mapping"}):
            return False
    
    # Reject fragments that are obviously incomplete
    if len(candidate) < 5:
        return False
    
    # Require at least 2+ tokens for general methods
    if len(candidate.split()) >= 2:
        return True
    
    return True


def _find_aliases_for_method(method_name: str, text_context: str) -> list[str]:
    """Find alternative names/aliases for identified methods from corpus and text.
    
    Strategy:
    1. Query biomedical corpora (MeSH, OLS) for authoritative aliases
    2. Fall back to hardcoded biomedical method aliases
    """
    method_name = _canonicalize_method_name(method_name)
    aliases: set[str] = set(_text_aliases(method_name, text_context))
    
    # First, try corpus enrichment
    try:
        corpus_match = enrich_entity(method_name, "method")
        if corpus_match and corpus_match.found:
            # Use corpus aliases (they're authoritative and current)
            aliases.update(corpus_match.aliases)
    except Exception:
        pass  # Graceful fallback if corpus unavailable
    
    # Fallback: hardcoded known aliases for common methods
    lowered_method = method_name.lower()
    lowered_context = text_context.lower()
    
    hardcoded_aliases = {
        "crispr": ["crispr-cas9", "crispr/cas9", "clustered regularly interspaced short palindromic repeats"],
        "deseq2": ["deseq", "differential expression analysis", "deseq2"],
        "matrigel invasion assay": ["invasion assay", "trans-well invasion", "cell invasion"],
        "chip-seq": ["chromatin immunoprecipitation sequencing", "chip sequencing"],
        "rna-seq": ["rnaseq", "rna sequencing", "whole transcriptome sequencing"],
        "western blot": ["western blotting", "immunoblot"],
        "elisa": ["enzyme-linked immunosorbent assay"],
        "qpcr": ["quantitative pcr", "real-time pcr", "qRT-PCR", "RT-PCR"],
        "facs": ["fluorescence activated cell sorting", "flow cytometry"],
        "flow cytometry": ["facs", "fluorescence activated cell sorting"],
        "immunofluorescence": ["immunofluorescence microscopy", "if"],
        "northern blot": ["northern blotting"],
        "merip-seq": ["me-rip-seq", "m6A-seq", "m6A sequencing"],
        "deseq2": ["DEseq", "Deseq2"],
    }
    
    # Try to find exact aliases in text
    for variant in hardcoded_aliases.get(lowered_method, []):
        if variant in lowered_context and variant != lowered_method:
            aliases.add(variant)
    
    # Look for abbreviations/expansions
    words = method_name.split()
    if len(words) > 1:
        # Check for acronym in text
        abbrev = ''.join(w[0] for w in words).upper()
        if abbrev in text_context and abbrev != method_name:
            aliases.add(abbrev)
    
    return list(aliases)


def _find_dataset_context(text: str) -> str | None:
    mentions = _extract_dataset_mentions(text)
    if mentions:
        return mentions[0]["label"]
    scenario_match = re.search(r"\bScenario\s+\d+\b", text, re.IGNORECASE)
    if scenario_match:
        candidate = _normalize(scenario_match.group(0))
        if candidate and _looks_like_dataset(candidate):
            return candidate
    return None


def _expand_metric_name(metric: str) -> str:
    """Expand abbreviated metric names to full form."""
    lowered = metric.lower().strip()
    return _METRIC_EXPANSIONS.get(lowered, metric)


def _local_datasets(
    chunk: ChunkRecord,
    embedder: TextEmbedder,
) -> list[Layer2EntityRecord]:
    """Extract Dataset entities from chunk text."""
    entities: dict[str, Layer2EntityRecord] = {}

    for mention in _extract_dataset_mentions(chunk.text):
        candidate = mention["label"]
        entity = _entity_base(
            "dataset",
            candidate,
            chunk.chunk_id,
            confidence=0.84,
            extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(candidate),
            properties={
                "dataset_type": mention["dataset_type"],
                "text": f"Dataset: {candidate}",
            },
            aliases=mention.get("aliases", []),
            entity_id=_stable_id("dataset", candidate),
        )
        entities.setdefault(entity.entity_id, entity)

    return list(entities.values())


def _looks_like_dataset(candidate: str) -> bool:
    lowered = candidate.lower()
    if lowered.startswith("scenario"):
        return True
    if any(char.isdigit() for char in candidate):
        return True
    if "-" in candidate:
        return True
    if candidate.upper() == candidate and len(candidate) >= 3:
        return True
    known_terms = {
        "imagenet",
        "cifar",
        "mnist",
        "squad",
        "coco",
        "glue",
        "wikidata",
        "wikipedia",
        "pubmed",
    }
    return lowered in known_terms


def _metric_category(metric: str) -> tuple[str, bool]:
    lowered = metric.lower()
    if any(token in lowered for token in {"accuracy", "f1", "precision", "recall", "auc", "bleu", "r2"}):
        return "classification", True
    if any(token in lowered for token in {"loss", "error", "perplexity", "rmse", "mae", "mape"}):
        return "efficiency", False
    if any(token in lowered for token in {"consensus", "agreement", "approval rate", "yes-vote share"}):
        return "decision", True
    return "other", True


def _find_metric_match(sentence: str) -> tuple[str, int] | None:
    for label, pattern in _METRIC_REGEX_CACHE:
        match = pattern.search(sentence)
        if match:
            return label, match.end()
    return None


def _result_signature(value: float, metric: str, dataset: str, condition: str, chunk_id: str) -> str:
    return _stable_id("result", f"{value:.6f}", metric.lower(), dataset.lower(), condition.lower(), chunk_id)


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


def _find_aliases_for_concept(concept_name: str, text_context: str) -> list[str]:
    """Find alternative names/aliases for identified concepts from corpus and text.
    
    Strategy:
    1. Query biomedical corpora (MeSH, OLS) for authoritative aliases
    2. Fall back to text pattern matching
    """
    concept_name = _canonicalize_concept_name(concept_name)
    aliases: set[str] = set(_text_aliases(concept_name, text_context))
    
    # First, try corpus enrichment
    try:
        corpus_match = enrich_entity(concept_name, "concept")
        if corpus_match and corpus_match.found:
            # Use corpus aliases (they're authoritative)
            aliases.update(corpus_match.aliases)
            # Store external IDs in properties (captured later in entity creation)
    except Exception:
        pass  # Graceful fallback if corpus unavailable
    
    for alias in _CONCEPT_ALIAS_MAP.get(concept_name, []):
        if alias.lower() in text_context.lower() or concept_name in _CONCEPT_CANONICAL_MAP.values():
            aliases.add(alias)

    # Then add text-based pattern matches as fallback
    lowered_concept = concept_name.lower()
    lowered_context = text_context.lower()
    
    # Common biomedical concept aliases (for text patterns)
    text_alias_patterns = {
        "m6a": ["n6-methyladenosine", "n6 methyladenosine", "m⁶a", "n6-methyl-adenosine"],
        "methylation": ["m6a modification", "dna methylation", "histone methylation"],
        "phosphorylation": ["protein phosphorylation", "kinase activity"],
        "acetylation": ["histone acetylation", "lysine acetylation"],
        "ubiquitination": ["protein ubiquitination", "ubiquitin modification"],
        "crispr": ["crispr-cas9", "crispr/cas9", "crispr technology"],
        "gene expression": ["mrna levels", "transcript levels", "protein expression"],
        "cell cycle": ["cell division", "cell proliferation"],
        "apoptosis": ["programmed cell death", "cell death"],
    }
    
    # Find matching text aliases
    for variant in text_alias_patterns.get(lowered_concept, []):
        if variant in lowered_context and variant != lowered_concept:
            aliases.add(variant)
    
    # Variant spellings for m6A
    if "methyladenosine" in lowered_concept or "m6a" in lowered_concept:
        patterns = [r"m6[aA]", r"m⁶[aA]", r"[nN]6-methyl", r"[nN]6 methyl"]
        for pattern in patterns:
            for match in re.finditer(pattern, text_context):
                variant = match.group(0)
                if variant.lower() != lowered_concept:
                    aliases.add(variant)
    
    return list(aliases)


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
                "concept",
                normalized_keyword,
                chunk.chunk_id,
                confidence=0.92,
                extractor_model=_LOCAL_MODEL,
                embedding=embedder.embed(normalized_keyword),
                properties={"ontology": ""},
                aliases=aliases,
                entity_id=_stable_id("concept", normalized_keyword),
            )
            entities[entity.entity_id] = entity

    for match in _GENE_SYMBOL_RE.finditer(text):
        candidate = _canonicalize_concept_name(match.group(0))
        aliases = _find_aliases_for_concept(candidate, text)
        entity = _entity_base(
            "concept",
            candidate,
            chunk.chunk_id,
            confidence=0.86,
            extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(candidate),
            properties={"ontology": ""},
            aliases=aliases,
            entity_id=_stable_id("concept", candidate),
        )
        entities.setdefault(entity.entity_id, entity)

    for match in _CONCEPT_PHRASE_RE.finditer(text):
        candidate = _canonicalize_concept_name(_normalize(match.group(0)))
        if not _is_valid_concept_candidate(candidate):
            continue
        aliases = _find_aliases_for_concept(candidate, text)
        entity = _entity_base(
            "concept",
            candidate,
            chunk.chunk_id,
            confidence=0.72 if candidate.isupper() else 0.62,
            extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(candidate),
            properties={"ontology": ""},
            aliases=aliases,
            entity_id=_stable_id("concept", candidate),
        )
        entities.setdefault(entity.entity_id, entity)

    if not entities and section.section_type in {"abstract", "introduction", "results", "discussion"}:
        fallback = _normalize(section.title)
        if fallback and fallback.lower() not in {"abstract", "highlights"}:
            aliases = _find_aliases_for_concept(fallback, text)
            entity = _entity_base(
                "concept",
                fallback,
                chunk.chunk_id,
                confidence=0.48,
                extractor_model=_LOCAL_MODEL,
                embedding=embedder.embed(fallback),
                properties={"ontology": ""},
                aliases=aliases,
                entity_id=_stable_id("concept", fallback),
            )
            entities[entity.entity_id] = entity

    return list(entities.values())


def _method_type(label: str, text: str) -> str:
    canonical = _canonicalize_method_name(label)
    if canonical in _METHOD_TYPE_OVERRIDES:
        return _METHOD_TYPE_OVERRIDES[canonical]

    lowered = f"{canonical} {text}".lower()
    if any(token in lowered for token in {"deseq", "statistical", "significance", "regression", "test"}):
        return "statistical"
    if any(token in lowered for token in {"bowtie", "samtools", "bedtools", "software", "algorithm", "alignment", "peak calling"}):
        return "computational"
    if any(
        token in lowered
        for token in {
            "assay",
            "blot",
            "cytometry",
            "crispr",
            "culture",
            "electroporation",
            "elisa",
            "experiment",
            "flow",
            "microscopy",
            "pcr",
            "sequencing",
            "transfection",
        }
    ):
        return "experimental"
    if any(token in lowered for token in {"simulation", "model", "architecture", "pipeline", "framework", "system"}):
        return "computational"
    return "experimental"


def _local_methods(
    paper: PaperRecord,
    section: SectionRecord,
    chunk: ChunkRecord,
    embedder: TextEmbedder,
) -> list[Layer2EntityRecord]:
    text = chunk.text
    entities: dict[str, Layer2EntityRecord] = {}
    
    # High-confidence biomedical methods first (these must pass validation)
    for match in _BIOMEDICAL_METHOD_RE.finditer(text):
        candidate = _canonicalize_method_name(match.group(0))
        if not _is_valid_method(candidate, text):
            continue
        
        # Find aliases and normalize
        aliases = _find_aliases_for_method(candidate, text)
        entity = _entity_base(
            "method",
            candidate,
            chunk.chunk_id,
            confidence=0.88 if section.section_type == "methods" else 0.78,
            extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(candidate),
            properties={
                "method_type": _method_type(candidate, text),
                "domain": "biomedical",
                "first_paper_id": paper.paper_id,
            },
            aliases=aliases,
            entity_id=_stable_id("method", candidate),
        )
        entities.setdefault(entity.entity_id, entity)
    
    # Lower-confidence general method phrases (stricter validation)
    for match in _METHOD_PHRASE_RE.finditer(text):
        candidate = _canonicalize_method_name(_normalize(match.group(0)))
        if not _is_valid_method(candidate, text):
            continue
        
        # Skip if already found as biomedical method
        candidate_id = _stable_id("method", candidate)
        if candidate_id in entities:
            continue
            
        aliases = _find_aliases_for_method(candidate, text)
        entity = _entity_base(
            "method",
            candidate,
            chunk.chunk_id,
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
        fallback = _canonicalize_method_name(_normalize(section.title))
        if fallback and _is_valid_method(fallback, text):
            aliases = _find_aliases_for_method(fallback, text)
            entity = _entity_base(
                "method",
                fallback,
                chunk.chunk_id,
                confidence=0.52,
                extractor_model=_LOCAL_MODEL,
                embedding=embedder.embed(fallback),
                properties={
                    "method_type": _method_type(fallback, text),
                    "first_paper_id": paper.paper_id,
                },
                aliases=aliases,
                entity_id=_stable_id("method", fallback),
            )
            entities[entity.entity_id] = entity

    return list(entities.values())


def _claim_type(sentence: str) -> str:
    if _LIMITATION_CUES.search(sentence):
        return "limitation"
    if _HYPOTHESIS_CUES.search(sentence):
        return "future_work" if "future work" in sentence.lower() or "will" in sentence.lower() else "hypothesis"
    return "finding"


def _local_claims(
    chunk: ChunkRecord,
    embedder: TextEmbedder,
) -> list[Layer2EntityRecord]:
    entities: dict[str, Layer2EntityRecord] = {}
    biomedical_keywords = {"knockdown", "knockout", "overexpression", "mutation", "phosphorylation", "methylation", "acetylation", "polyubiquitination", "sumoylation", "palmitoylation", "ubiquitination", "expression", "transcript", "protein", "antibody", "inhibitor", "agonist", "antagonist", "phospho", "acetyl", "methyl", "ubiquitin", "knockdown"}
    for sentence in _split_sentences(chunk.text):
        stripped = sentence.strip()
        if not stripped or not stripped[0].isupper():
            continue
        # Extract if has claim verbs, limitations, hypotheses, OR biomedical keywords in results section
        has_claim_structure = _CLAIM_VERBS.search(sentence) or _LIMITATION_CUES.search(sentence) or _HYPOTHESIS_CUES.search(sentence)
        has_biomedical_keyword = any(kw in sentence.lower() for kw in biomedical_keywords)
        if not has_claim_structure and not has_biomedical_keyword:
            continue
        claim_text = _normalize(sentence)
        # Lowered threshold from 6 to 4 words for biomedical text
        if len(claim_text.split()) < 4:
            continue
        claim_type = _claim_type(claim_text)
        confidence = 0.82 if _CLAIM_VERBS.search(claim_text) else 0.68 if has_biomedical_keyword else 0.55
        entity = _entity_base(
            "claim",
            claim_text,
            chunk.chunk_id,
            confidence=confidence,
            extractor_model=_LOCAL_MODEL,
            embedding=embedder.embed(claim_text),
            properties={"claim_type": claim_type, "text": claim_text},
            aliases=[],
            entity_id=_stable_id("claim", claim_text),
        )
        entities.setdefault(entity.entity_id, entity)
    return list(entities.values())


def _local_results(
    section: SectionRecord,
    chunk: ChunkRecord,
    embedder: TextEmbedder,
) -> list[Layer2EntityRecord]:
    if section.section_type == "methods":
        return []

    entities: dict[str, Layer2EntityRecord] = {}
    chunk_datasets = _result_datasets("", chunk.text)

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
        datasets = _result_datasets(sentence, chunk.text) or chunk_datasets
        if not datasets:
            continue

        primary_found = False

        for match in _FOLD_CHANGE_RE.finditer(cleaned_sentence):
            metric = "Fold change"
            entity = _build_result_entity(
                chunk,
                embedder,
                cleaned_sentence,
                value=float(match.group(1)),
                metric=metric,
                datasets=datasets,
                confidence=0.82,
                metric_abbreviation="fold-change",
            )
            entities.setdefault(entity.entity_id, entity)
            primary_found = True

        for match in _PERCENT_VALUE_RE.finditer(cleaned_sentence):
            window = lowered[max(0, match.start() - 60) : min(len(lowered), match.end() + 80)]
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
                chunk,
                embedder,
                cleaned_sentence,
                value=value,
                metric=metric,
                datasets=datasets,
                unit="%",
                confidence=0.84,
                metric_abbreviation=metric.lower(),
            )
            entities.setdefault(entity.entity_id, entity)
            primary_found = True

        for match in _COUNT_VALUE_RE.finditer(cleaned_sentence):
            raw_value = match.group(1).replace(",", "")
            noun = match.group(2).lower()
            value = float(raw_value)
            window = lowered[max(0, match.start() - 70) : min(len(lowered), match.end() + 80)]
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
                chunk,
                embedder,
                cleaned_sentence,
                value=value,
                metric=metric,
                datasets=datasets,
                unit=noun,
                confidence=0.8,
                metric_abbreviation=noun,
            )
            entities.setdefault(entity.entity_id, entity)
            primary_found = True

        if _STAR_PVALUE_RE.search(cleaned_sentence):
            continue

        for match in _P_VALUE_RE.finditer(cleaned_sentence):
            if primary_found and any(token in lowered for token in {"corresponds to", "genes", "percentage", "ratio"}):
                continue
            value = float(match.group(1))
            entity = _build_result_entity(
                chunk,
                embedder,
                cleaned_sentence,
                value=value,
                metric="P-value",
                datasets=datasets,
                confidence=0.66,
                metric_abbreviation="p-value",
                result_role="statistical_test",
            )
            entities.setdefault(entity.entity_id, entity)

        if primary_found:
            continue

        metric_match = _find_metric_match(cleaned_sentence)
        if metric_match is None:
            continue

        metric, metric_end = metric_match
        metric_expanded = _expand_metric_name(metric)
        number_matches = list(_NUMERIC_RE.finditer(cleaned_sentence))
        if not number_matches:
            continue

        after_metric = [match for match in number_matches if match.start() >= metric_end]
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
            chunk,
            embedder,
            cleaned_sentence,
            value=value,
            metric=metric_expanded,
            datasets=datasets,
            unit=unit,
            confidence=0.72,
            metric_abbreviation=metric,
            result_role="statistical_test" if metric_expanded == "P-value" else "primary",
        )
        entities.setdefault(entity.entity_id, entity)
    
    return list(entities.values())


def _local_equations(
    chunk: ChunkRecord,
    embedder: TextEmbedder,
) -> list[Layer2EntityRecord]:
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
            "equation",
            expr,
            chunk.chunk_id,
            confidence=0.55,
            extractor_model=_LOCAL_MODEL,
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


def _heuristic_salience(chunk: ChunkRecord, entities: list[Layer2EntityRecord]) -> float:
    """Calculate chunk salience with proper differentiation.
    
    Returns a score from ~0.1 (boilerplate) to ~0.95 (novel key finding).
    """
    if not entities:
        return 0.15  # Boilerplate chunks with no entities are low salience
    
    # Count entity types
    concept_count = sum(1 for entity in entities if entity.entity_type == "concept")
    method_count = sum(1 for entity in entities if entity.entity_type == "method")
    claim_count = sum(1 for entity in entities if entity.entity_type == "claim")
    result_count = sum(1 for entity in entities if entity.entity_type == "result")
    dataset_count = sum(1 for entity in entities if entity.entity_type == "dataset")
    equation_count = sum(1 for entity in entities if entity.entity_type == "equation")
    
    # Calculate average confidence
    avg_confidence = sum(entity.confidence for entity in entities) / len(entities)
    
    # Chunk position signal (results section is more important than methods/intro)
    section_bonus = 0.0
    chunk_type_lower = chunk.chunk_type.lower()
    if "results" in chunk_type_lower or "findings" in chunk_type_lower:
        section_bonus = 0.15
    elif "discussion" in chunk_type_lower:
        section_bonus = 0.10
    elif "methods" in chunk_type_lower or "materials" in chunk_type_lower:
        section_bonus = 0.05
    
    # Start with base score
    score = 0.10
    
    # Add contribution from entities (with realistic upper bounds per type)
    # Results with datasets are most important
    if result_count > 0 and dataset_count > 0:
        score += 0.35 * min(result_count / 3.0, 1.0)  # Cap at 3 results
    elif result_count > 0:
        score += 0.22 * min(result_count / 3.0, 1.0)  # Unlinked results worth less
    
    # Claims are important but not as much as results
    if claim_count > 0:
        score += 0.25 * min(claim_count / 2.0, 1.0)  # Cap at 2 claims
    
    # Methods add significant value
    if method_count > 0:
        score += 0.15 * min(method_count / 2.0, 1.0)
    
    # Concepts add modest value
    if concept_count > 0:
        score += 0.08 * min(concept_count / 3.0, 1.0)
    
    # Datasets add a small bonus
    if dataset_count > 0:
        score += 0.03 * min(dataset_count / 2.0, 1.0)
    
    # Equations add modest value
    if equation_count > 0:
        score += 0.06 * min(equation_count / 2.0, 1.0)
    
    # Confidence bonus (high-confidence extractions boost salience)
    score += 0.10 * avg_confidence
    
    # Length bonus (but not excessive - cap shorter chunks)
    length_factor = min(chunk.word_count / 180.0, 0.12)
    score += length_factor
    
    # Add section bonus
    score += section_bonus
    
    # Normalize and discretize to create differentiation
    # Return values across the full 0.1-0.95 range
    final_score = min(score, 0.95)
    
    # Round to 2 decimal places for better differentiation (not 3)
    return round(max(final_score, 0.10), 2)


def _gemini_extract_chunk_entities(
    paper: PaperRecord,
    section: SectionRecord,
    chunk: ChunkRecord,
) -> tuple[list[dict[str, Any]], float] | None:
    if not gemini_available():
        return None

    model_name = os.getenv("EXTRACT_MODEL", "gemini-2.5-flash")
    prompt = (
        f"{_GEMINI_PROMPT}\n\n"
        f"Paper title: {paper.title}\n"
        f"Section: {section.title}\n"
        f"Chunk ID: {chunk.chunk_id}\n\n"
        f"Text:\n{chunk.text}"
    )
    try:
        # Trace LLM call with LangFuse
        tracer = get_tracing_manager()

        payload = _GeminiChunkExtractionPayload.model_validate(
            generate_json(
                prompt,
                model_name=model_name,
                temperature=0.1,
            )
        )

        # Log LLM call to LangFuse
        tracer.log_llm_call(
            name="entity_extraction",
            model=model_name,
            prompt=prompt[:500],
            response=json.dumps(payload.model_dump(mode="json"))[:500],
            metadata={
                "paper_id": paper.paper_id,
                "chunk_id": chunk.chunk_id,
                "section_title": section.title,
            },
        )

        return [
            entity.model_dump(exclude_none=True, mode="json") for entity in payload.entities
        ], payload.salience_score
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:  # pragma: no cover - optional network-backed path
        tracer = get_tracing_manager()
        if tracer.enabled:
            tracer.langfuse.event(
                name="entity_extraction_error",
                output={"error": str(exc)},
                metadata={
                    "paper_id": paper.paper_id,
                    "chunk_id": chunk.chunk_id,
                },
            )
        return None


def _gemini_to_entity(
    raw: dict[str, Any],
    chunk: ChunkRecord,
    embedder: TextEmbedder,
) -> Layer2EntityRecord | None:
    entity_type = _normalize(str(raw.get("type", ""))).lower()
    if entity_type not in {"concept", "method", "claim", "result", "equation"}:
        return None

    if entity_type in {"concept", "method"}:
        label = _normalize(str(raw.get("name", "")))
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

    properties = {key: value for key, value in raw.items() if key not in {"type", "name", "text", "aliases", "confidence"}}
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
    else:
        properties.setdefault("ontology", raw.get("ontology", ""))

    aliases = [str(alias) for alias in raw.get("aliases", []) if str(alias).strip()]
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
        entity_id = _stable_id(entity_type, label)

    return _entity_base(
        entity_type,
        label,
        chunk.chunk_id,
        confidence=confidence,
        extractor_model=os.getenv("EXTRACT_MODEL", "gemini-2.0-flash"),
        embedding=embedder.embed(label),
        properties=properties,
        aliases=aliases,
        entity_id=entity_id,
    )


def extract_layer2(
    paper: PaperRecord,
    settings: Phase1Settings | None = None,
    use_gemini: bool | None = None,
    update_chunks: bool = True,
) -> Layer2DocumentRecord:
    """Extract Layer 2 entities from a Layer 1 paper record."""

    tracer = get_tracing_manager()
    settings = settings or Phase1Settings.from_env()
    embedder = build_entity_embedder(dim=settings.embedding_dim)
    use_gemini = use_gemini if use_gemini is not None else os.getenv("USE_GEMINI_EXTRACTION", "0") == "1"
    section_lookup = {section.section_id: section for section in paper.sections}
    global_entities: dict[str, Layer2EntityRecord] = {}
    chunk_entity_ids: dict[str, list[str]] = {}
    chunk_salience_scores: dict[str, float] = {}
    extractor_model = os.getenv("EXTRACT_MODEL", "gemini-2.5-flash") if use_gemini else _LOCAL_MODEL

    with tracer.trace(
        name="extract_layer2",
        input_data={"paper_id": paper.paper_id, "chunk_count": len(paper.chunks)},
    ) as trace_ctx:
        for chunk in paper.chunks:
            section = section_lookup.get(chunk.section_id)
            if section is None:
                continue

            raw_entities: list[Layer2EntityRecord] = []
            gemini_salience: float | None = None
            if use_gemini:
                gemini_output = _gemini_extract_chunk_entities(paper, section, chunk)
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
                raw_entities.extend(_local_datasets(chunk, embedder))
                raw_entities.extend(_local_results(section, chunk, embedder))
                raw_entities.extend(_local_equations(chunk, embedder))

            chunk_entity_ids[chunk.chunk_id] = [entity.entity_id for entity in raw_entities]
            for entity in raw_entities:
                if entity.entity_id in global_entities:
                    _merge_entities(global_entities[entity.entity_id], entity)
                else:
                    global_entities[entity.entity_id] = entity

            salience = gemini_salience if gemini_salience is not None else _heuristic_salience(chunk, raw_entities)
            chunk_salience_scores[chunk.chunk_id] = salience
            if update_chunks:
                chunk.salience_score = max(chunk.salience_score, salience)

        # Log entity extraction metrics
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
        }

    result = Layer2DocumentRecord(
        paper_id=paper.paper_id,
        extractor_model=extractor_model,
        entities=sorted(global_entities.values(), key=lambda entity: (entity.entity_type, entity.label.lower())),
        chunk_entity_ids=chunk_entity_ids,
        chunk_salience_scores=chunk_salience_scores,
    )
    
    tracer.flush()
    return result
