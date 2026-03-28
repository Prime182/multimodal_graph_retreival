"""Index and graph-metrics utilities for Phase 4."""

from __future__ import annotations

import sys
from typing import Any

from .config import Phase1Settings

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None
    Neo4jError = Exception


PROPERTY_INDEX_STATEMENTS = [
    "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.published_year)",
    "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
    "CREATE INDEX paper_field IF NOT EXISTS FOR (p:Paper) ON (p.field_of_study)",
    "CREATE INDEX section_type IF NOT EXISTS FOR (s:Section) ON (s.section_type)",
    "CREATE INDEX chunk_type IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_type)",
    "CREATE INDEX chunk_salience IF NOT EXISTS FOR (c:Chunk) ON (c.salience_score)",
    "CREATE INDEX claim_type IF NOT EXISTS FOR (c:Claim) ON (c.claim_type)",
    "CREATE INDEX result_dataset IF NOT EXISTS FOR (r:Result) ON (r.dataset)",
    "CREATE INDEX method_type IF NOT EXISTS FOR (m:Method) ON (m.type)",
    "CREATE INDEX author_h_index IF NOT EXISTS FOR (a:Author) ON (a.h_index)",
]

PAGERANK_PROJECT_QUERY = """
CALL gds.graph.project(
  'citation_graph',
  'Paper',
  'CITES'
)
YIELD graphName
"""

PAGERANK_WRITE_QUERY = """
CALL gds.pageRank.write(
  'citation_graph',
  {
    maxIterations: $max_iterations,
    dampingFactor: $damping_factor,
    writeProperty: 'pagerank'
  }
)
YIELD nodePropertiesWritten
"""

DROP_GRAPH_QUERY = "CALL gds.graph.drop('citation_graph')"

PAPER_COUNT_QUERY = """
MATCH (p:Paper)
RETURN count(p) AS node_count
"""

PAGERANK_INIT_FALLBACK_QUERY = """
MATCH (p:Paper)
SET p.pagerank = 1.0 / toFloat($node_count),
    p.__pagerank_out_degree = COUNT { (p)-[:CITES]->() }
"""

PAGERANK_STEP_FALLBACK_QUERY = """
MATCH (sink:Paper)
WHERE coalesce(sink.__pagerank_out_degree, 0) = 0
WITH coalesce(sum(sink.pagerank), 0.0) AS sink_rank
MATCH (p:Paper)
OPTIONAL MATCH (src:Paper)-[:CITES]->(p)
WITH p, sink_rank,
     coalesce(
         sum(
             CASE
                 WHEN src IS NULL OR coalesce(src.__pagerank_out_degree, 0) = 0 THEN 0.0
                 ELSE src.pagerank / toFloat(src.__pagerank_out_degree)
             END
         ),
         0.0
     ) AS incoming_rank
SET p.__pagerank_next =
    ((1.0 - $damping_factor) / toFloat($node_count))
    + ($damping_factor * sink_rank / toFloat($node_count))
    + ($damping_factor * incoming_rank)
"""

PAGERANK_APPLY_FALLBACK_QUERY = """
MATCH (p:Paper)
SET p.pagerank = p.__pagerank_next
REMOVE p.__pagerank_next
"""

PAGERANK_CLEANUP_FALLBACK_QUERY = """
MATCH (p:Paper)
REMOVE p.__pagerank_out_degree, p.__pagerank_next
"""


class GraphIndexManager:
    def __init__(self, settings: Phase1Settings | None = None) -> None:
        if GraphDatabase is None:
            raise RuntimeError("The neo4j package is not installed. Install the optional dependency to manage indexes.")
        self.settings = settings or Phase1Settings.from_env()
        self._driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    def ensure_property_indexes(self) -> None:
        with self._driver.session(database=self.settings.neo4j_database) as session:
            for statement in PROPERTY_INDEX_STATEMENTS:
                session.run(statement)

    def compute_pagerank(self, max_iterations: int = 20, damping_factor: float = 0.85) -> None:
        with self._driver.session(database=self.settings.neo4j_database) as session:
            try:
                session.run(PAGERANK_PROJECT_QUERY)
                session.run(
                    PAGERANK_WRITE_QUERY,
                    max_iterations=max_iterations,
                    damping_factor=damping_factor,
                )
            except Neo4jError as exc:
                if getattr(exc, "code", "") != "Neo.ClientError.Procedure.ProcedureNotFound":
                    raise
                print(
                    "Neo4j GDS plugin not available; falling back to Cypher PageRank.",
                    file=sys.stderr,
                )
                self._compute_pagerank_fallback(
                    session,
                    max_iterations=max_iterations,
                    damping_factor=damping_factor,
                )
            finally:
                try:
                    session.run(DROP_GRAPH_QUERY)
                except Exception:
                    pass

    def _compute_pagerank_fallback(
        self,
        session: Any,
        *,
        max_iterations: int,
        damping_factor: float,
    ) -> None:
        node_count = session.run(PAPER_COUNT_QUERY).single()["node_count"]
        if node_count == 0:
            return

        session.run(PAGERANK_INIT_FALLBACK_QUERY, node_count=node_count)
        try:
            for _ in range(max_iterations):
                session.run(
                    PAGERANK_STEP_FALLBACK_QUERY,
                    node_count=node_count,
                    damping_factor=damping_factor,
                )
                session.run(PAGERANK_APPLY_FALLBACK_QUERY)
        finally:
            session.run(PAGERANK_CLEANUP_FALLBACK_QUERY)
