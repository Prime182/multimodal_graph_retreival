"""Neo4j-backed retrieval using vector indexes and hybrid search."""

from __future__ import annotations

from typing import Any

from .config import Phase1Settings
from .models import SearchHit

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None


class GraphRetrieval:
    """Hybrid graph-based search using Neo4j vector indexes and Cypher."""

    def __init__(self, settings: Phase1Settings | None = None) -> None:
        if GraphDatabase is None:
            raise RuntimeError("neo4j package not installed")
        settings = settings or Phase1Settings.from_env()
        self._database = settings.neo4j_database
        self._embedding_dim = settings.embedding_dim
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._driver.close()

    def search_chunks(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[SearchHit]:
        """
        Vector similarity search over chunks using Neo4j vector index.
        Returns ranked chunks with context.
        """
        query = """
        CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding) AS (chunk, score)
        WHERE score >= $min_similarity
        MATCH (paper:Paper)-[:HAS_SECTION]->(section:Section)-[:HAS_CHUNK]->(chunk)
        RETURN 
            chunk.id AS chunk_id,
            section.id AS section_id,
            section.title AS section_title,
            paper.paper_id AS paper_id,
            paper.title AS paper_title,
            paper.doi AS doi,
            chunk.raw_text AS text,
            score
        ORDER BY score DESC
        LIMIT $top_k
        """

        with self._driver.session(database=self._database) as session:
            results = session.run(
                query,
                embedding=embedding,
                top_k=top_k,
                min_similarity=min_similarity,
            )
            hits: list[SearchHit] = []
            for record in results:
                hits.append(
                    SearchHit(
                        score=float(record["score"]),
                        paper_id=record["paper_id"],
                        paper_title=record["paper_title"],
                        section_id=record["section_id"],
                        chunk_id=record["chunk_id"],
                        section_title=record["section_title"],
                        text=record["text"],
                        doi=record["doi"],
                    )
                )
            return hits

    def search_papers(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[float, dict[str, Any]]]:
        """
        Vector similarity search over paper abstracts.
        Returns papers ranked by abstract similarity.
        """
        query = """
        CALL db.index.vector.queryNodes('paper_abstract_embedding', $top_k, $embedding) AS (paper, score)
        WHERE score >= $min_similarity
        RETURN
            paper.paper_id AS paper_id,
            paper.title AS title,
            paper.doi AS doi,
            paper.published_year AS year,
            paper.abstract AS abstract,
            score
        ORDER BY score DESC
        LIMIT $top_k
        """

        with self._driver.session(database=self._database) as session:
            results = session.run(
                query,
                embedding=embedding,
                top_k=top_k,
                min_similarity=min_similarity,
            )
            papers: list[tuple[float, dict[str, Any]]] = []
            for record in results:
                papers.append(
                    (
                        float(record["score"]),
                        {
                            "paper_id": record["paper_id"],
                            "title": record["title"],
                            "doi": record["doi"],
                            "year": record["year"],
                            "abstract": record["abstract"],
                        },
                    )
                )
            return papers

    def search_entities(
        self,
        entity_type: str,
        embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[float, dict[str, Any]]]:
        """
        Search entities by type and embedding similarity.
        Can target Concept, Method, Claim, Equation nodes.
        """
        node_label = entity_type.capitalize()
        index_name = f"{entity_type}_embedding"

        query = f"""
        CALL db.index.vector.queryNodes('{index_name}', $top_k, $embedding) AS (entity, score)
        WHERE score >= $min_similarity
        AND entity:{node_label}
        RETURN
            entity.id AS entity_id,
            entity.label AS label,
            entity.confidence AS confidence,
            entity.properties AS properties,
            score
        ORDER BY score DESC
        LIMIT $top_k
        """

        with self._driver.session(database=self._database) as session:
            results = session.run(
                query,
                top_k=top_k,
                min_similarity=min_similarity,
                embedding=embedding,
            )
            entities: list[tuple[float, dict[str, Any]]] = []
            for record in results:
                entities.append(
                    (
                        float(record["score"]),
                        {
                            "entity_id": record["entity_id"],
                            "label": record["label"],
                            "type": entity_type,
                            "confidence": record["confidence"],
                            "properties": record["properties"] or {},
                        },
                    )
                )
            return entities

    def get_claim_sources(self, claim_id: str) -> list[dict[str, Any]]:
        """
        Get all source chunks and papers for a claim.
        Useful for explainability/citation.
        """
        query = """
        MATCH (claim:Claim {id: $claim_id})-[:GROUNDED_IN]->(chunk:Chunk)
        MATCH (paper:Paper)-[:HAS_SECTION]->(section:Section)-[:HAS_CHUNK]->(chunk)
        RETURN
            paper.paper_id AS paper_id,
            paper.title AS paper_title,
            paper.doi AS doi,
            section.id AS section_id,
            section.title AS section_title,
            chunk.id AS chunk_id,
            chunk.raw_text AS chunk_text,
            chunk.salience_score AS salience
        """

        with self._driver.session(database=self._database) as session:
            results = session.run(query, claim_id=claim_id)
            sources: list[dict[str, Any]] = []
            for record in results:
                sources.append(
                    {
                        "paper_id": record["paper_id"],
                        "paper_title": record["paper_title"],
                        "doi": record["doi"],
                        "section_id": record["section_id"],
                        "section_title": record["section_title"],
                        "chunk_id": record["chunk_id"],
                        "chunk_text": record["chunk_text"],
                        "salience": record["salience"],
                    }
                )
            return sources

    def get_related_claims(self, claim_id: str, relation_type: str = "SUPPORTS") -> list[dict[str, Any]]:
        """
        Find claims that support or contradict a given claim.
        relation_type: "SUPPORTS" or "CONTRADICTS"
        """
        query = f"""
        MATCH (claim:Claim {{id: $claim_id}})-[:{relation_type}]->(related:Claim)
        RETURN
            related.id AS claim_id,
            related.label AS label,
            related.confidence AS confidence,
            related.claim_type AS claim_type
        LIMIT 10
        """

        with self._driver.session(database=self._database) as session:
            results = session.run(query, claim_id=claim_id)
            related: list[dict[str, Any]] = []
            for record in results:
                related.append(
                    {
                        "claim_id": record["claim_id"],
                        "label": record["label"],
                        "confidence": record["confidence"],
                        "type": record["claim_type"],
                    }
                )
            return related
