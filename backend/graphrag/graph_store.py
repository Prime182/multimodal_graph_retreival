"""Neo4j persistence for the Layer 1 and Layer 2 GraphRAG records."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from .config import Phase1Settings
from .edges import Layer3CorpusRecord
from .entities import Layer2DocumentRecord
from .models import PaperRecord

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None


LAYER1_SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT paper_doi IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
    "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
    "CREATE CONSTRAINT author_orcid IF NOT EXISTS FOR (a:Author) REQUIRE a.orcid IS UNIQUE",
    "CREATE CONSTRAINT journal_id IF NOT EXISTS FOR (j:Journal) REQUIRE j.journal_id IS UNIQUE",
    "CREATE CONSTRAINT journal_issn IF NOT EXISTS FOR (j:Journal) REQUIRE j.issn IS UNIQUE",
    "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
]


LAYER2_SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (m:Method) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT result_id IF NOT EXISTS FOR (r:Result) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT metric_name IF NOT EXISTS FOR (m:Metric) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT equation_id IF NOT EXISTS FOR (e:Equation) REQUIRE e.id IS UNIQUE",
]


def _vector_index_statements(dim: int) -> list[str]:
    return [
        (
            "CREATE VECTOR INDEX paper_abstract_embedding IF NOT EXISTS "
            "FOR (p:Paper) ON (p.abstract_embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: "
            f"{dim}, `vector.similarity_function`: 'cosine'}}}}"
        ),
        (
            "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: "
            f"{dim}, `vector.similarity_function`: 'cosine'}}}}"
        ),
        (
            "CREATE VECTOR INDEX concept_embedding IF NOT EXISTS "
            "FOR (c:Concept) ON (c.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: "
            f"{dim}, `vector.similarity_function`: 'cosine'}}}}"
        ),
        (
            "CREATE VECTOR INDEX method_embedding IF NOT EXISTS "
            "FOR (m:Method) ON (m.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: "
            f"{dim}, `vector.similarity_function`: 'cosine'}}}}"
        ),
        (
            "CREATE VECTOR INDEX claim_embedding IF NOT EXISTS "
            "FOR (c:Claim) ON (c.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: "
            f"{dim}, `vector.similarity_function`: 'cosine'}}}}"
        ),
        (
            "CREATE VECTOR INDEX equation_embedding IF NOT EXISTS "
            "FOR (e:Equation) ON (e.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: "
            f"{dim}, `vector.similarity_function`: 'cosine'}}}}"
        ),
    ]


def _prune_none(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if value is not None}


def _to_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


class Neo4jGraphStore:
    """Persist Layer 1 records into Neo4j when the driver is available."""

    def __init__(self, settings: Phase1Settings | None = None) -> None:
        if GraphDatabase is None:
            raise RuntimeError(
                "The neo4j package is not installed. Install the optional dependency to use Neo4jGraphStore."
            )
        settings = settings or Phase1Settings.from_env()
        self._database = settings.neo4j_database
        self._embedding_dim = settings.embedding_dim
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            for statement in [
                *LAYER1_SCHEMA_STATEMENTS,
                *LAYER2_SCHEMA_STATEMENTS,
                *_vector_index_statements(self._embedding_dim),
            ]:
                session.run(statement)

    def upsert_paper(self, paper: PaperRecord) -> None:
        journal_row = _prune_none(asdict(paper.journal)) if paper.journal else None
        author_rows = [_prune_none(asdict(author)) for author in paper.authors]
        section_rows = [
            {
                "id": section.section_id,
                "paper_id": section.paper_id,
                "title": section.title,
                "section_type": section.section_type,
                "level": section.level,
                "ordinal": section.ordinal,
                "text": section.text,
                "key_sentence": section.key_sentence,
                "parent_section_id": section.parent_section_id,
            }
            for section in paper.sections
        ]
        chunk_rows = [
            {
                "id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "section_id": chunk.section_id,
                "ordinal": chunk.ordinal,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type,
                "word_count": chunk.word_count,
                "token_count": chunk.token_count,
                "salience_score": chunk.salience_score,
                "embedding": chunk.embedding,
                "prev_chunk_id": chunk.prev_chunk_id,
                "next_chunk_id": chunk.next_chunk_id,
            }
            for chunk in paper.chunks
        ]

        with self._driver.session(database=self._database) as session:
            session.execute_write(self._upsert_paper_tx, paper, journal_row)
            if author_rows:
                session.execute_write(self._upsert_authors_tx, paper.paper_id, author_rows)
            if section_rows:
                session.execute_write(self._upsert_sections_tx, paper.paper_id, section_rows)
            if chunk_rows:
                session.execute_write(self._upsert_chunks_tx, chunk_rows)

    def upsert_layer2(self, paper: PaperRecord, extraction: Layer2DocumentRecord) -> None:
        entity_rows = [entity.to_dict() for entity in extraction.entities]
        if not entity_rows:
            return
        with self._driver.session(database=self._database) as session:
            session.execute_write(self._upsert_layer2_tx, paper.paper_id, entity_rows)

    def upsert_layer3(self, layer3: Layer3CorpusRecord) -> None:
        with self._driver.session(database=self._database) as session:
            if layer3.citation_edges:
                session.execute_write(self._upsert_citation_edges_tx, [edge.to_dict() for edge in layer3.citation_edges])
            if layer3.semantic_edges:
                session.execute_write(self._upsert_semantic_edges_tx, [edge.to_dict() for edge in layer3.semantic_edges])

    @staticmethod
    def _upsert_paper_tx(tx: Any, paper: PaperRecord, journal_row: dict[str, Any] | None) -> None:
        tx.run(
            """
            MERGE (p:Paper {paper_id: $paper_id})
            SET p.title = $title,
                p.doi = $doi,
                p.pii = $pii,
                p.article_number = $article_number,
                p.published_year = $published_year,
                p.abstract = $abstract,
                p.highlights = $highlights,
                p.keywords = $keywords,
                p.metadata_json = $metadata_json,
                p.source_path = $source_path,
                p.abstract_embedding = $abstract_embedding
            """,
            paper_id=paper.paper_id,
            title=paper.title,
            doi=paper.doi,
            pii=paper.pii,
            article_number=paper.article_number,
            published_year=paper.published_year,
            abstract=paper.abstract,
            highlights=paper.highlights,
            keywords=paper.keywords,
            metadata_json=_to_json(paper.metadata),
            source_path=paper.source_path,
            abstract_embedding=paper.abstract_embedding,
        )
        if journal_row:
            tx.run(
                """
                MERGE (j:Journal {journal_id: $journal_id})
                SET j += $journal_row
                WITH j
                MATCH (p:Paper {paper_id: $paper_id})
                MERGE (p)-[:PUBLISHED_IN]->(j)
                """,
                journal_id=journal_row["journal_id"],
                journal_row=journal_row,
                paper_id=paper.paper_id,
            )

    @staticmethod
    def _upsert_authors_tx(tx: Any, paper_id: str, author_rows: list[dict[str, Any]]) -> None:
        tx.run(
            """
            MATCH (p:Paper {paper_id: $paper_id})
            UNWIND $authors AS author
            MERGE (a:Author {author_id: author.author_id})
            SET a += author
            MERGE (p)-[:AUTHORED_BY]->(a)
            """,
            paper_id=paper_id,
            authors=author_rows,
        )

    @staticmethod
    def _upsert_sections_tx(tx: Any, paper_id: str, section_rows: list[dict[str, Any]]) -> None:
        tx.run(
            """
            MATCH (p:Paper {paper_id: $paper_id})
            UNWIND $sections AS section
            MERGE (s:Section {id: section.id})
            SET s += section
            MERGE (p)-[:HAS_SECTION]->(s)
            """,
            paper_id=paper_id,
            sections=section_rows,
        )
        tx.run(
            """
            UNWIND $sections AS section
            WITH section
            WHERE section.parent_section_id IS NOT NULL
            MATCH (parent:Section {id: section.parent_section_id})
            MATCH (child:Section {id: section.id})
            MERGE (parent)-[:HAS_SUBSECTION]->(child)
            """,
            sections=section_rows,
        )

    @staticmethod
    def _upsert_chunks_tx(tx: Any, chunk_rows: list[dict[str, Any]]) -> None:
        tx.run(
            """
            UNWIND $chunks AS chunk
            MATCH (s:Section {id: chunk.section_id})
            MERGE (c:Chunk {id: chunk.id})
            SET c += chunk
            MERGE (s)-[:HAS_CHUNK]->(c)
            """,
            chunks=chunk_rows,
        )
        tx.run(
            """
            UNWIND $chunks AS chunk
            WITH chunk
            WHERE chunk.next_chunk_id IS NOT NULL
            MATCH (left:Chunk {id: chunk.id})
            MATCH (right:Chunk {id: chunk.next_chunk_id})
            MERGE (left)-[:NEXT]->(right)
            """,
            chunks=chunk_rows,
        )

    @staticmethod
    def _upsert_layer2_tx(tx: Any, paper_id: str, entity_rows: list[dict[str, Any]]) -> None:
        for entity in entity_rows:
            entity_type = entity.get("entity_type")
            if entity_type == "concept":
                tx.run(
                    """
                    MERGE (c:Concept {id: $id})
                    SET c.name = $label,
                        c.aliases = $aliases,
                        c.ontology = $ontology,
                        c.confidence = $confidence,
                        c.extractor_model = $extractor_model,
                        c.embedding = $embedding,
                        c.paper_id = $paper_id
                    WITH c
                    UNWIND $mention_chunk_ids AS chunk_id
                    MATCH (chunk:Chunk {id: chunk_id})
                    MERGE (chunk)-[m:MENTIONS]->(c)
                    SET m.confidence = $confidence,
                        m.source_chunk_id = $source_chunk_id,
                        m.extractor_model = $extractor_model
                    """,
                    id=entity["entity_id"],
                    label=entity["label"],
                    aliases=entity.get("aliases", []),
                    ontology=entity.get("properties", {}).get("ontology", ""),
                    confidence=entity.get("confidence", 0.5),
                    extractor_model=entity.get("extractor_model", "heuristic-v2"),
                    embedding=entity.get("embedding", []),
                    paper_id=paper_id,
                    mention_chunk_ids=entity.get("mention_chunk_ids", []),
                    source_chunk_id=entity.get("source_chunk_id", ""),
                )
            elif entity_type == "method":
                properties = entity.get("properties", {})
                tx.run(
                    """
                    MERGE (m:Method {id: $id})
                    SET m.name = $label,
                        m.aliases = $aliases,
                        m.method_type = $method_type,
                        m.domain = $domain,
                        m.confidence = $confidence,
                        m.extractor_model = $extractor_model,
                        m.embedding = $embedding,
                        m.paper_id = $paper_id
                    WITH m
                    UNWIND $mention_chunk_ids AS chunk_id
                    MATCH (chunk:Chunk {id: chunk_id})
                    MERGE (chunk)-[mn:MENTIONS]->(m)
                    SET mn.confidence = $confidence,
                        mn.source_chunk_id = $source_chunk_id,
                        mn.extractor_model = $extractor_model
                    """,
                    id=entity["entity_id"],
                    label=entity["label"],
                    aliases=entity.get("aliases", []),
                    method_type=properties.get("method_type", "computational"),
                    domain=properties.get("domain", "general"),
                    confidence=entity.get("confidence", 0.5),
                    extractor_model=entity.get("extractor_model", "heuristic-v2"),
                    embedding=entity.get("embedding", []),
                    paper_id=paper_id,
                    mention_chunk_ids=entity.get("mention_chunk_ids", []),
                    source_chunk_id=entity.get("source_chunk_id", ""),
                )
            elif entity_type == "claim":
                properties = entity.get("properties", {})
                tx.run(
                    """
                    MERGE (cl:Claim {id: $id})
                    SET cl.text = $text,
                        cl.claim_type = $claim_type,
                        cl.confidence = $confidence,
                        cl.extractor_model = $extractor_model,
                        cl.embedding = $embedding,
                        cl.paper_id = $paper_id
                    WITH cl
                    UNWIND $mention_chunk_ids AS chunk_id
                    MATCH (chunk:Chunk {id: chunk_id})
                    MERGE (chunk)-[mn:MENTIONS]->(cl)
                    SET mn.confidence = $confidence,
                        mn.source_chunk_id = $source_chunk_id,
                        mn.extractor_model = $extractor_model
                    """,
                    id=entity["entity_id"],
                    text=entity["label"],
                    claim_type=properties.get("claim_type", "finding"),
                    confidence=entity.get("confidence", 0.5),
                    extractor_model=entity.get("extractor_model", "heuristic-v2"),
                    embedding=entity.get("embedding", []),
                    paper_id=paper_id,
                    mention_chunk_ids=entity.get("mention_chunk_ids", []),
                    source_chunk_id=entity.get("source_chunk_id", ""),
                )
            elif entity_type == "result":
                properties = entity.get("properties", {})
                tx.run(
                    """
                    MERGE (r:Result {id: $id})
                    SET r.text = $text,
                        r.value = $value,
                        r.unit = $unit,
                        r.metric = $metric,
                        r.dataset = $dataset,
                        r.confidence = $confidence,
                        r.extractor_model = $extractor_model,
                        r.embedding = $embedding,
                        r.paper_id = $paper_id
                    WITH r
                    UNWIND $mention_chunk_ids AS chunk_id
                    MATCH (chunk:Chunk {id: chunk_id})
                    MERGE (chunk)-[mn:MENTIONS]->(r)
                    SET mn.confidence = $confidence,
                        mn.source_chunk_id = $source_chunk_id,
                        mn.extractor_model = $extractor_model
                    """,
                    id=entity["entity_id"],
                    text=entity["label"],
                    value=properties.get("value", 0.0),
                    unit=properties.get("unit", ""),
                    metric=properties.get("metric", ""),
                    dataset=properties.get("dataset", ""),
                    confidence=entity.get("confidence", 0.5),
                    extractor_model=entity.get("extractor_model", "heuristic-v2"),
                    embedding=entity.get("embedding", []),
                    paper_id=paper_id,
                    mention_chunk_ids=entity.get("mention_chunk_ids", []),
                    source_chunk_id=entity.get("source_chunk_id", ""),
                )
            elif entity_type == "equation":
                properties = entity.get("properties", {})
                tx.run(
                    """
                    MERGE (eq:Equation {id: $id})
                    SET eq.latex = $latex,
                        eq.plain_desc = $plain_desc,
                        eq.domain = $domain,
                        eq.is_loss_fn = $is_loss_fn,
                        eq.confidence = $confidence,
                        eq.extractor_model = $extractor_model,
                        eq.embedding = $embedding,
                        eq.paper_id = $paper_id
                    WITH eq
                    UNWIND $mention_chunk_ids AS chunk_id
                    MATCH (chunk:Chunk {id: chunk_id})
                    MERGE (chunk)-[mn:MENTIONS]->(eq)
                    SET mn.confidence = $confidence,
                        mn.source_chunk_id = $source_chunk_id,
                        mn.extractor_model = $extractor_model
                    """,
                    id=entity["entity_id"],
                    latex=properties.get("latex", entity["label"]),
                    plain_desc=properties.get("plain_desc", entity["label"]),
                    domain=properties.get("domain", "mathematics"),
                    is_loss_fn=bool(properties.get("is_loss_fn", False)),
                    confidence=entity.get("confidence", 0.5),
                    extractor_model=entity.get("extractor_model", "heuristic-v2"),
                    embedding=entity.get("embedding", []),
                    paper_id=paper_id,
                    mention_chunk_ids=entity.get("mention_chunk_ids", []),
                    source_chunk_id=entity.get("source_chunk_id", ""),
                )

    @staticmethod
    def _upsert_citation_edges_tx(tx: Any, edge_rows: list[dict[str, Any]]) -> None:
        serialized_edges = [{**edge, "metadata_json": _to_json(edge.get("metadata", {}))} for edge in edge_rows]
        tx.run(
            """
            UNWIND $edges AS edge
            MATCH (source:Paper {paper_id: edge.source_node_id})
            MERGE (target:Paper {paper_id: edge.target_node_id})
            ON CREATE SET target.title = edge.target_label,
                          target.doi = edge.metadata.reference_doi
            SET target.title = coalesce(target.title, edge.target_label),
                target.doi = coalesce(target.doi, edge.metadata.reference_doi)
            MERGE (source)-[r:CITES]->(target)
            SET r.edge_id = edge.edge_id,
                r.confidence = edge.confidence,
                r.source_chunk_id = edge.source_chunk_id,
                r.extractor_model = edge.extractor_model,
                r.evidence = edge.evidence,
                r.metadata_json = edge.metadata_json
            """,
            edges=serialized_edges,
        )

    @staticmethod
    def _upsert_semantic_edges_tx(tx: Any, edge_rows: list[dict[str, Any]]) -> None:
        for edge in edge_rows:
            serialized_edge = {**edge, "metadata_json": _to_json(edge.get("metadata", {}))}
            relation_type = edge.get("relation_type")
            if relation_type == "IS_A":
                source_node_type = edge.get("source_node_type", "Method")
                target_node_type = edge.get("target_node_type", "Concept")
                if source_node_type not in {"Method", "Concept"} or target_node_type not in {"Method", "Concept"}:
                    continue
                tx.run(
                    f"""
                    MERGE (source:{source_node_type} {{id: $source_node_id}})
                    ON CREATE SET source.name = $source_label
                    SET source.name = coalesce(source.name, $source_label)
                    MERGE (target:{target_node_type} {{id: $target_node_id}})
                    ON CREATE SET target.name = $target_label
                    SET target.name = coalesce(target.name, $target_label)
                    MERGE (source)-[r:IS_A]->(target)
                    SET r.edge_id = $edge_id,
                        r.confidence = $confidence,
                        r.source_chunk_id = $source_chunk_id,
                        r.extractor_model = $extractor_model,
                        r.evidence = $evidence,
                        r.metadata_json = $metadata_json
                    """,
                    **serialized_edge,
                )
            elif relation_type in {"SUPPORTS", "CONTRADICTS"}:
                tx.run(
                    f"""
                    MATCH (source:Claim {{id: $source_node_id}})
                    MATCH (target:Claim {{id: $target_node_id}})
                    MERGE (source)-[r:{relation_type}]->(target)
                    SET r.edge_id = $edge_id,
                        r.confidence = $confidence,
                        r.source_chunk_id = $source_chunk_id,
                        r.extractor_model = $extractor_model,
                        r.evidence = $evidence,
                        r.metadata_json = $metadata_json
                    """,
                    **serialized_edge,
                )
            elif relation_type == "GROUNDED_IN":
                # Claim is grounded in a specific chunk
                tx.run(
                    """
                    MATCH (source:Claim {id: $source_node_id})
                    MATCH (target:Chunk {id: $target_node_id})
                    MERGE (source)-[r:GROUNDED_IN]->(target)
                    SET r.edge_id = $edge_id,
                        r.confidence = $confidence,
                        r.source_chunk_id = $source_chunk_id,
                        r.extractor_model = $extractor_model,
                        r.evidence = $evidence,
                        r.metadata_json = $metadata_json
                    """,
                    **serialized_edge,
                )
            elif relation_type == "MEASURED_ON":
                # Result is measured on a dataset
                # First ensure Dataset node exists
                dataset_label = edge.get("target_label", "Unknown")
                tx.run(
                    """
                    MERGE (d:Dataset {id: $target_node_id})
                    SET d.name = $dataset_label
                    """,
                    target_node_id=edge.get("target_node_id"),
                    dataset_label=dataset_label,
                )
                # Then create the edge
                tx.run(
                    """
                    MATCH (source:Result {id: $source_node_id})
                    MATCH (target:Dataset {id: $target_node_id})
                    MERGE (source)-[r:MEASURED_ON]->(target)
                    SET r.edge_id = $edge_id,
                        r.confidence = $confidence,
                        r.source_chunk_id = $source_chunk_id,
                        r.extractor_model = $extractor_model,
                        r.evidence = $evidence,
                        r.metadata_json = $metadata_json
                    """,
                    **serialized_edge,
                )
            elif relation_type == "USING_METRIC":
                # Result uses a specific metric
                # First ensure Metric node exists
                metric_label = edge.get("target_label", "Unknown")
                tx.run(
                    """
                    MERGE (m:Metric {id: $target_node_id})
                    SET m.name = $metric_label
                    """,
                    target_node_id=edge.get("target_node_id"),
                    metric_label=metric_label,
                )
                # Then create the edge
                tx.run(
                    """
                    MATCH (source:Result {id: $source_node_id})
                    MATCH (target:Metric {id: $target_node_id})
                    MERGE (source)-[r:USING_METRIC]->(target)
                    SET r.edge_id = $edge_id,
                        r.confidence = $confidence,
                        r.source_chunk_id = $source_chunk_id,
                        r.extractor_model = $extractor_model,
                        r.evidence = $evidence,
                        r.metadata_json = $metadata_json
                    """,
                    **serialized_edge,
                )


def _metric_metadata(metric: str) -> tuple[str, bool]:
    lowered = metric.lower()
    if any(token in lowered for token in {"loss", "error", "perplexity", "rmse", "mae", "mape"}):
        return "efficiency", False
    if any(token in lowered for token in {"consensus", "agreement", "approval rate", "yes-vote share"}):
        return "decision", True
    return "classification", True
