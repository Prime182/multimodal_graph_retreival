"""Layer 3 semantic edges and citation resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha1
import itertools
import re
from typing import Any, Iterable

from .corpus import get_hierarchy
from .embeddings import cosine_similarity
from .entities import Layer2DocumentRecord, Layer2EntityRecord
from .models import PaperRecord, ReferenceRecord


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_SUPPORT_CUES = {
    "support",
    "supports",
    "supported",
    "consistent with",
    "in line with",
    "confirms",
    "confirm",
    "demonstrates",
    "demonstrate",
    "indicates",
    "indicate",
    "improves",
    "improve",
    "increases",
    "increase",
    "higher",
    "better",
}
_CONTRADICT_CUES = {
    "contradict",
    "contradicts",
    "contradiction",
    "in contrast",
    "however",
    "yet",
    "but",
    "fails",
    "fail",
    "decreases",
    "decrease",
    "reduces",
    "reduce",
    "negative",
    "collapse",
    "limitation",
    "limited",
    "not",
}


@dataclass(slots=True)
class Layer3EdgeRecord:
    edge_id: str
    relation_type: str
    source_node_id: str
    source_node_type: str
    source_label: str
    target_node_id: str
    target_node_type: str
    target_label: str
    confidence: float
    source_chunk_id: str
    extractor_model: str
    evidence: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Layer3CorpusRecord:
    citation_edges: list[Layer3EdgeRecord] = field(default_factory=list)
    semantic_edges: list[Layer3EdgeRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _clip(value: float) -> float:
    return round(max(0.0, min(value, 1.0)), 3)


def _reference_title(reference: ReferenceRecord) -> str | None:
    if reference.title:
        return reference.title
    if reference.source_text:
        prefix = reference.source_text.split(".")[0]
        return prefix.strip() or None
    return None


def _reference_doi(reference: ReferenceRecord) -> str | None:
    if reference.doi:
        return reference.doi.lower().strip()
    if reference.source_text:
        match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", reference.source_text, re.IGNORECASE)
        if match:
            return match.group(0).lower().rstrip(".")
    return None


def _reference_stub_id(reference: ReferenceRecord) -> str:
    basis = _reference_doi(reference) or _normalize(_reference_title(reference) or reference.reference_id)
    return _stable_id("refpaper", basis)


def _match_reference_to_paper(reference: ReferenceRecord, papers_by_doi: dict[str, PaperRecord], papers_by_title: dict[str, PaperRecord]) -> PaperRecord | None:
    doi = _reference_doi(reference)
    if doi and doi in papers_by_doi:
        return papers_by_doi[doi]
    title = _reference_title(reference)
    if title:
        normalized = _normalize(title)
        if normalized in papers_by_title:
            return papers_by_title[normalized]
    return None


def build_citation_edges(papers: Iterable[PaperRecord]) -> list[Layer3EdgeRecord]:
    paper_list = list(papers)
    papers_by_doi = {paper.doi.lower(): paper for paper in paper_list if paper.doi}
    papers_by_title = {_normalize(paper.title): paper for paper in paper_list}
    edges: list[Layer3EdgeRecord] = []

    for paper in paper_list:
        for reference in paper.references:
            cited = _match_reference_to_paper(reference, papers_by_doi, papers_by_title)
            if cited is not None and cited.paper_id == paper.paper_id:
                continue
            target_id = cited.paper_id if cited is not None else _reference_stub_id(reference)
            target_label = cited.title if cited is not None else (_reference_title(reference) or reference.label or reference.doi or target_id)
            target_doi = cited.doi if cited is not None else _reference_doi(reference)
            confidence = 0.99 if cited is not None and target_doi and cited.doi and target_doi == cited.doi.lower() else 0.72 if cited is None else 0.88
            evidence = reference.source_text or reference.title or reference.label or reference.doi or ""
            edges.append(
                Layer3EdgeRecord(
                    edge_id=_stable_id("cite", paper.paper_id, target_id, reference.reference_id),
                    relation_type="CITES",
                    source_node_id=paper.paper_id,
                    source_node_type="Paper",
                    source_label=paper.title,
                    target_node_id=target_id,
                    target_node_type="Paper",
                    target_label=target_label,
                    confidence=confidence,
                    source_chunk_id="",
                    extractor_model="bibliography-parser",
                    evidence=evidence,
                    metadata={
                        "reference_id": reference.reference_id,
                        "reference_doi": target_doi,
                        "reference_title": _reference_title(reference),
                        "is_stub_target": cited is None,
                    },
                )
            )

    return edges


def _claim_text(entity: Layer2EntityRecord) -> str:
    return entity.properties.get("text") or entity.label


def _relation_from_cues(left: str, right: str, similarity: float) -> tuple[str | None, float]:
    combined = f"{left} {right}"
    support_hits = sum(1 for cue in _SUPPORT_CUES if cue in combined)
    contradict_hits = sum(1 for cue in _CONTRADICT_CUES if cue in combined)
    if similarity < 0.74:
        return None, 0.0
    if contradict_hits > support_hits:
        return "CONTRADICTS", _clip(0.52 + similarity * 0.38 + 0.05 * contradict_hits)
    if support_hits > 0:
        return "SUPPORTS", _clip(0.5 + similarity * 0.4 + 0.04 * support_hits)
    if similarity >= 0.9:
        return "SUPPORTS", _clip(0.45 + similarity * 0.45)
    return None, 0.0


def infer_claim_edges(layer2_docs: Iterable[Layer2DocumentRecord], similarity_threshold: float = 0.86) -> list[Layer3EdgeRecord]:
    docs = list(layer2_docs)
    claim_entities: list[tuple[str, Layer2EntityRecord]] = []
    for doc in docs:
        for entity in doc.entities:
            if entity.entity_type == "claim":
                claim_entities.append((doc.paper_id, entity))

    edges: list[Layer3EdgeRecord] = []
    for (paper_a, claim_a), (paper_b, claim_b) in itertools.combinations(claim_entities, 2):
        if paper_a == paper_b:
            continue
        similarity = cosine_similarity(claim_a.embedding, claim_b.embedding)
        if similarity < similarity_threshold:
            continue
        relation_type, confidence = _relation_from_cues(_claim_text(claim_a).lower(), _claim_text(claim_b).lower(), similarity)
        if relation_type is None:
            continue
        edge = Layer3EdgeRecord(
            edge_id=_stable_id("claim-rel", relation_type, claim_a.entity_id, claim_b.entity_id),
            relation_type=relation_type,
            source_node_id=claim_a.entity_id,
            source_node_type="Claim",
            source_label=_claim_text(claim_a),
            target_node_id=claim_b.entity_id,
            target_node_type="Claim",
            target_label=_claim_text(claim_b),
            confidence=confidence,
            source_chunk_id=claim_a.source_chunk_id,
            extractor_model="claim-relation-heuristic",
            evidence=f"similarity={similarity:.3f}",
            metadata={
                "source_paper_id": paper_a,
                "target_paper_id": paper_b,
                "similarity": round(similarity, 4),
            },
        )
        edges.append(edge)

    return edges


def infer_is_a_edges(layer2_docs: Iterable[Layer2DocumentRecord]) -> list[Layer3EdgeRecord]:
    """Create IS_A edges representing true ontological hierarchy: Child IS_A Parent.
    
    Strategy:
    1. Query biomedical corpora (OLS) for real ontological parents
    2. Try to match parents against concepts/methods in the document
    3. Fall back to known hierarchies if corpus unavailable
    """
    edges: list[Layer3EdgeRecord] = []
    
    # Known hierarchies for biomedical domain (fallback if corpus unavailable)
    known_hierarchies = [
        ("CRISPR-CAS9", "CRISPR", "Concept", 0.96),
        ("CRISPRa", "CRISPR", "Method", 0.97),
        ("FACS", "Flow cytometry", "Method", 0.96),
        ("MeRIP-seq", "RNA-Seq", "Method", 0.94),
        ("qRT-PCR", "qPCR", "Method", 0.95),
        ("qPCR", "PCR", "Concept", 0.93),
        ("RT-PCR", "PCR", "Concept", 0.92),
        ("DESeq2", "differential expression analysis", "Method", 0.91),
        ("bowtie2", "read alignment method", "Method", 0.89),
        ("MACS2", "peak calling method", "Method", 0.89),
    ]
    
    for doc in layer2_docs:
        doc_concepts = {_normalize(e.label): e for e in doc.entities if e.entity_type == "concept"}
        doc_methods: dict[str, Layer2EntityRecord] = {}
        for entity in doc.entities:
            if entity.entity_type != "method":
                continue
            doc_methods[_normalize(entity.label)] = entity
            for alias in entity.aliases:
                doc_methods.setdefault(_normalize(alias), entity)
        
        # Strategy 1: Try corpus-derived hierarchies for each method/concept
        entities_to_check = list(doc_methods.items())
        
        for entity_norm_label, entity in entities_to_check:
            try:
                # Query OLS for parent terms (try GO and other biomedical ontologies)
                parents = []
                for ontology in ["go", "chebi", "doid"]:
                    parents.extend(get_hierarchy(entity.label, ontology))
                    if parents:  # Stop at first successful ontology
                        break
                
                # Try to match parents against document entities
                for parent_label in parents[:3]:  # Limit to top 3 parents
                    parent_norm = _normalize(parent_label)
                    
                    # Look for parent in document
                    parent_entity = doc_concepts.get(parent_norm) or doc_methods.get(parent_norm)
                    
                    if parent_entity and parent_entity.entity_id != entity.entity_id:
                        edges.append(
                            Layer3EdgeRecord(
                                edge_id=_stable_id("is-a-corpus", entity.entity_id, parent_entity.entity_id),
                                relation_type="IS_A",
                                source_node_id=entity.entity_id,
                                source_node_type="Method",
                                source_label=entity.label,
                                target_node_id=parent_entity.entity_id,
                                target_node_type=parent_entity.entity_type.capitalize(),
                                target_label=parent_entity.label,
                                confidence=_clip(0.85),  # Corpus-derived hierarchies are high confidence
                                source_chunk_id=entity.source_chunk_id,
                                extractor_model="ontology-corpus",
                                evidence=f"From {ontology} ontology via OLS",
                                metadata={
                                    "paper_id": doc.paper_id,
                                    "hierarchy_type": "corpus_ontology",
                                    "ontology": ontology,
                                },
                            )
                        )
            except Exception:
                # Silently skip corpus lookup if unavailable
                pass
        
        # Strategy 2: Create edges based on known hierarchies (fallback)
        for specific, general, target_type, conf in known_hierarchies:
            specific_entity = doc_methods.get(_normalize(specific))
            if specific_entity is None:
                continue

            if target_type == "Concept":
                target_entity = doc_concepts.get(_normalize(general))
                target_node_id = target_entity.entity_id if target_entity else _stable_id("concept", general)
                target_label = target_entity.label if target_entity else general
            else:
                target_entity = doc_methods.get(_normalize(general))
                target_node_id = target_entity.entity_id if target_entity else _stable_id("method", general)
                target_label = target_entity.label if target_entity else general

            if target_node_id == specific_entity.entity_id:
                continue

            edges.append(
                Layer3EdgeRecord(
                    edge_id=_stable_id("is-a-known", specific_entity.entity_id, target_node_id),
                    relation_type="IS_A",
                    source_node_id=specific_entity.entity_id,
                    source_node_type="Method",
                    source_label=specific_entity.label,
                    target_node_id=target_node_id,
                    target_node_type=target_type,
                    target_label=target_label,
                    confidence=_clip(conf),
                    source_chunk_id=specific_entity.source_chunk_id,
                    extractor_model="ontology-hierarchy",
                    evidence=f"Known biomedical ontology: {specific} is a {general}",
                    metadata={
                        "paper_id": doc.paper_id,
                        "hierarchy_type": "known_biomedical",
                    },
                )
            )
    
    return edges


def infer_grounded_in_edges(layer2_docs: Iterable[Layer2DocumentRecord], papers_by_id: dict[str, PaperRecord]) -> list[Layer3EdgeRecord]:
    """Create GROUNDED_IN edges: Claim -> Chunk (audit trail for evidence)."""
    edges: list[Layer3EdgeRecord] = []
    for doc in layer2_docs:
        paper = papers_by_id.get(doc.paper_id)
        if paper is None:
            continue
        claims = [e for e in doc.entities if e.entity_type == "claim"]
        for claim in claims:
            # Find the actual chunk this claim comes from
            for chunk in paper.chunks:
                if chunk.chunk_id == claim.source_chunk_id:
                    edges.append(
                        Layer3EdgeRecord(
                            edge_id=_stable_id("grounded", claim.entity_id, chunk.chunk_id),
                            relation_type="GROUNDED_IN",
                            source_node_id=claim.entity_id,
                            source_node_type="Claim",
                            source_label=claim.label,
                            target_node_id=chunk.chunk_id,
                            target_node_type="Chunk",
                            target_label=f"Chunk: {chunk.text[:50]}...",
                            confidence=0.95,
                            source_chunk_id=claim.source_chunk_id,
                            extractor_model="chunk-mapping",
                            evidence=f"Claim directly extracted from this chunk",
                            metadata={"paper_id": doc.paper_id, "section_type": chunk.chunk_type},
                        )
                    )
                    break
    return edges


def infer_measured_on_edges(layer2_docs: Iterable[Layer2DocumentRecord]) -> list[Layer3EdgeRecord]:
    """Create MEASURED_ON edges: Result -> Dataset."""
    edges: list[Layer3EdgeRecord] = []
    for doc in layer2_docs:
        results = [e for e in doc.entities if e.entity_type == "result"]
        for result in results:
            datasets = result.properties.get("datasets") or []
            if not datasets and result.properties.get("dataset"):
                datasets = [result.properties["dataset"]]
            for dataset in datasets:
                edges.append(
                    Layer3EdgeRecord(
                        edge_id=_stable_id("measured", result.entity_id, dataset),
                        relation_type="MEASURED_ON",
                        source_node_id=result.entity_id,
                        source_node_type="Result",
                        source_label=result.label,
                        target_node_id=_stable_id("dataset", dataset),
                        target_node_type="Dataset",
                        target_label=dataset,
                        confidence=0.92,
                        source_chunk_id=result.source_chunk_id,
                        extractor_model="result-dataset-linking",
                        evidence=f"Result {result.label} measured on dataset {dataset}",
                        metadata={"paper_id": doc.paper_id, "value": result.properties.get("value")},
                    )
                )
    return edges


def infer_using_metric_edges(layer2_docs: Iterable[Layer2DocumentRecord]) -> list[Layer3EdgeRecord]:
    """Create USING_METRIC edges: Result -> Metric."""
    edges: list[Layer3EdgeRecord] = []
    for doc in layer2_docs:
        results = [e for e in doc.entities if e.entity_type == "result"]
        for result in results:
            metric = result.properties.get("metric")
            if metric:
                edges.append(
                    Layer3EdgeRecord(
                        edge_id=_stable_id("uses-metric", result.entity_id, metric),
                        relation_type="USING_METRIC",
                        source_node_id=result.entity_id,
                        source_node_type="Result",
                        source_label=result.label,
                        target_node_id=f"metric-{_normalize(metric[:20])}",
                        target_node_type="Metric",
                        target_label=metric,
                        confidence=0.95,
                        source_chunk_id=result.source_chunk_id,
                        extractor_model="result-metric-linking",
                        evidence=f"Result uses metric: {metric}",
                        metadata={"paper_id": doc.paper_id, "value": result.properties.get("value"), "unit": result.properties.get("unit")},
                    )
                )
    return edges


def build_layer3(
    papers: Iterable[PaperRecord],
    layer2_docs: Iterable[Layer2DocumentRecord],
) -> Layer3CorpusRecord:
    paper_list = list(papers)
    layer2_list = list(layer2_docs)
    papers_by_id = {paper.paper_id: paper for paper in paper_list}
    
    citation_edges = build_citation_edges(paper_list)
    semantic_edges = [
        *infer_is_a_edges(layer2_list),
        *infer_claim_edges(layer2_list),
        *infer_grounded_in_edges(layer2_list, papers_by_id),
        *infer_measured_on_edges(layer2_list),
        *infer_using_metric_edges(layer2_list),
    ]
    return Layer3CorpusRecord(citation_edges=citation_edges, semantic_edges=semantic_edges)
