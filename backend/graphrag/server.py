"""FastAPI application for GraphRAG search service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .circuit_breaker import get_circuit_breaker
from .config import Phase1Settings
from .embeddings import probe_embedding_backends
from .graph_retrieval import GraphRetrieval
from .search_service import GraphRAGSearchService
from .rag import QuerySynthesizer
from .tracing import get_tracing_manager


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"


class SearchRequest(BaseModel):
    """Request model for search queries."""

    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    """Response model for search results."""

    query: str
    text_hits: list[dict[str, Any]]
    table_hits: list[dict[str, Any]]
    figure_hits: list[dict[str, Any]]
    papers: list[dict[str, Any]]
    entities: dict[str, list[dict[str, Any]]]
    citations: list[dict[str, Any]]
    stats: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: str


class TracingHealthResponse(BaseModel):
    """Tracing health response."""

    status: str
    message: str
    enabled: bool


class GraphNeighborhoodResponse(BaseModel):
    """Graph debug response."""

    entity_id: str
    status: str
    message: str
    neighborhood: dict[str, Any]
    mentioning_chunks: list[str]


def create_app(
    input_dir: str | Path = "articles",
    use_gemini: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="GraphRAG API",
        description="Hybrid retrieval engine for scientific knowledge graphs",
        version="0.1.0",
    )

    # Add CORS middleware to allow frontend requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize search service on startup
    settings = Phase1Settings.from_env()
    search_service = GraphRAGSearchService(
        input_dir=input_dir,
        settings=settings,
        use_gemini=use_gemini,
        load_on_init=False,
    )
    graph_backend: GraphRetrieval | None = None
    graph_backend_error: str | None = None
    try:
        graph_backend = GraphRetrieval(settings=settings)
    except Exception as exc:
        graph_backend_error = str(exc)
        print(
            "⚠ Neo4j graph backend unavailable; using local retrieval fallback "
            f"for search and disabling /api/graph endpoints: {exc}"
        )

    # Initialize synthesis service
    synthesizer = QuerySynthesizer()

    embedding_health = probe_embedding_backends(
        dim=settings.embedding_dim,
        prefer_remote=use_gemini,
    )
    app.state.embedding_health = embedding_health

    import logging
    _log = logging.getLogger(__name__)

    if not embedding_health.get("remote_available") and not embedding_health.get("local_available"):
        raise RuntimeError(
            "FATAL: No embedding backend available at startup. "
            f"Remote error: {embedding_health.get('remote_error')}. "
            f"Local error: {embedding_health.get('local_error')}. "
            "Run with ALLOW_SBERT_MODEL_DOWNLOAD=1 once to cache the model."
        )

    if not embedding_health.get("local_available"):
        _log.warning(
            "⚠ SBERT local fallback is NOT cached (error: %s). "
            "If Gemini becomes unavailable, ALL embedding calls will fail. "
            "Set ALLOW_SBERT_MODEL_DOWNLOAD=1 and restart to pre-cache.",
            embedding_health.get("local_error"),
        )

    if hasattr(search_service, "start_background_load"):
        search_service.start_background_load()
    app.state.graph_backend_available = graph_backend is not None
    app.state.graph_backend_error = graph_backend_error

    async def close_graph_backend() -> None:
        """Close any optional graph backend connection cleanly."""

        if hasattr(search_service, "close"):
            search_service.close()
        if graph_backend is not None:
            graph_backend.close()

    app.router.add_event_handler("shutdown", close_graph_backend)

    # ─── Frontend Routes ───────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def serve_index() -> str:
        """Serve the main frontend page."""
        index_path = _FRONTEND_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend not found")
        return index_path.read_text()

    @app.get("/app.js", response_class=FileResponse)
    async def serve_app_js() -> FileResponse:
        """Serve the app JS."""
        app_path = _FRONTEND_DIR / "app.js"
        if not app_path.exists():
            raise HTTPException(status_code=404, detail="App JS not found")
        return FileResponse(app_path, media_type="application/javascript")

    @app.get("/styles.css", response_class=FileResponse)
    async def serve_styles_css() -> FileResponse:
        """Serve the CSS stylesheet."""
        styles_path = _FRONTEND_DIR / "styles.css"
        if not styles_path.exists():
            raise HTTPException(status_code=404, detail="Styles not found")
        return FileResponse(styles_path, media_type="text/css")

    # ─── API Routes ───────────────────────────────────────────────────────────
    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Check service health."""
        embedding_health = getattr(app.state, "embedding_health", {})
        backend_name = embedding_health.get("active_backend", "unknown")
        graph_status = "available" if graph_backend is not None else "unavailable"
        loading = bool(getattr(search_service, "loading", False))
        load_error = getattr(search_service, "load_error", None)
        paper_count = len(getattr(search_service, "papers", []))
        if load_error:
            return HealthResponse(
                status="error",
                message=(
                    f"GraphRAG service failed to load corpus "
                    f"(embeddings: {backend_name}, graph: {graph_status}): {load_error}"
                ),
            )
        if loading:
            return HealthResponse(
                status="loading",
                message=(
                    f"GraphRAG service is loading corpus in the background "
                    f"(papers loaded: {paper_count}, embeddings: {backend_name}, graph: {graph_status})"
                ),
            )
        return HealthResponse(
            status="ok",
            message=(
                f"GraphRAG service ready with {paper_count} papers "
                f"(embeddings: {backend_name}, graph: {graph_status})"
            ),
        )

    @app.get("/api/tracing/health", response_model=TracingHealthResponse)
    async def tracing_health() -> TracingHealthResponse:
        """Report LangFuse tracing availability."""
        tracer = get_tracing_manager()
        if tracer.enabled and tracer.langfuse is not None:
            return TracingHealthResponse(
                status="ok",
                message="LangFuse tracing is enabled and configured.",
                enabled=True,
            )
        return TracingHealthResponse(
            status="disabled",
            message="LangFuse tracing is disabled or unavailable.",
            enabled=False,
        )

    @app.get("/api/search", response_model=SearchResponse)
    async def search_get(
        q: str = Query(..., min_length=1, description="Search query"),
        top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
    ) -> SearchResponse:
        """Search the corpus (GET endpoint for compatibility)."""
        tracer = get_tracing_manager()
        
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        try:
            result = search_service.search(query=q, top_k=top_k)
            tracer.langfuse.event(
                name="api_search_get",
                input={"query": q, "top_k": top_k},
                output={
                    "text_hits": len(result.get("text_hits", [])),
                    "table_hits": len(result.get("table_hits", [])),
                    "figure_hits": len(result.get("figure_hits", [])),
                },
            ) if tracer.enabled else None
            return SearchResponse(**result)
        except Exception as e:
            tracer.langfuse.event(
                name="api_search_error",
                input={"query": q, "top_k": top_k},
                output={"error": str(e)},
            ) if tracer.enabled else None
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    @app.post("/api/search", response_model=SearchResponse)
    async def search_post(request: SearchRequest) -> SearchResponse:
        """Search the corpus (POST endpoint)."""
        tracer = get_tracing_manager()
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        try:
            result = search_service.search(query=request.query, top_k=request.top_k)
            tracer.langfuse.event(
                name="api_search_post",
                input={"query": request.query, "top_k": request.top_k},
                output={
                    "text_hits": len(result.get("text_hits", [])),
                    "table_hits": len(result.get("table_hits", [])),
                    "figure_hits": len(result.get("figure_hits", [])),
                },
            ) if tracer.enabled else None
            return SearchResponse(**result)
        except Exception as e:
            tracer.langfuse.event(
                name="api_search_error",
                input={"query": request.query, "top_k": request.top_k},
                output={"error": str(e)},
            ) if tracer.enabled else None
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    @app.get("/api/stats")
    async def get_stats() -> dict[str, Any]:
        """Get corpus statistics."""
        return {
            "paper_count": len(search_service.papers),
            "chunk_count": sum(len(p.chunks) for p in search_service.papers),
            "table_count": sum(len(p.tables) for p in search_service.papers),
            "figure_count": sum(len(p.figures) for p in search_service.papers),
            "section_count": sum(len(p.sections) for p in search_service.papers),
            "entity_count": sum(len(doc.entities) for doc in search_service.layer2_docs),
            "citation_count": len(search_service.layer3.citation_edges),
            "semantic_edge_count": len(search_service.layer3.semantic_edges),
        }

    @app.get("/api/corpus-misses")
    async def corpus_misses(
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        misses = search_service.corpus_misses(limit=limit)
        return {
            "misses": misses,
            "total_unique_misses": len(misses),
        }

    @app.get("/api/extraction-quality")
    async def extraction_quality(
        limit: int = Query(100, ge=1, le=500),
    ) -> dict[str, Any]:
        return search_service.extraction_quality_report(limit=limit)

    @app.get("/api/ingestion-status")
    async def ingestion_status(
        limit: int = Query(100, ge=1, le=500),
    ) -> dict[str, Any]:
        statuses = search_service.ingestion_status_report(limit=limit)
        return {
            "papers": statuses,
            "total": len(statuses),
        }

    @app.get("/api/circuit-breakers")
    async def circuit_breakers() -> dict[str, Any]:
        return {
            "services": get_circuit_breaker().snapshot(),
        }

    @app.get("/api/graph/entity/{entity_id}", response_model=GraphNeighborhoodResponse)
    async def graph_entity(
        entity_id: str,
        hops: int = Query(2, ge=0, le=5),
    ) -> GraphNeighborhoodResponse:
        """Expose a graph neighborhood for debugging and manual inspection."""
        if graph_backend is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Neo4j graph backend is unavailable. "
                    f"{graph_backend_error or 'No connection could be established.'}"
                ),
            )

        neighborhood = graph_backend.get_entity_neighborhood([entity_id], hops=hops)
        mentioning_chunks = graph_backend.get_chunks_mentioning_entities([entity_id])
        return GraphNeighborhoodResponse(
            entity_id=entity_id,
            status="ok",
            message=f"Neighborhood retrieved with {len(neighborhood.get('nodes', []))} nodes.",
            neighborhood=neighborhood,
            mentioning_chunks=mentioning_chunks,
        )

    @app.get("/api/papers")
    async def list_papers(
        skip: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=100),
    ) -> dict[str, Any]:
        """List papers in the corpus."""
        papers = search_service.papers[skip : skip + limit]
        return {
            "papers": [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "doi": p.doi,
                    "published_year": p.published_year,
                    "authors": [a.full_name for a in p.authors],
                    "abstract": p.abstract[:200] if p.abstract else None,
                }
                for p in papers
            ],
            "total": len(search_service.papers),
            "skip": skip,
            "limit": limit,
        }

    @app.post("/api/synthesize")
    async def synthesize(request: SearchRequest) -> dict[str, Any]:
        """
        Search the corpus and synthesize an answer using LLM.
        This is the RAG endpoint that combines retrieval + generative synthesis.
        """
        tracer = get_tracing_manager()
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        try:
            with tracer.trace(
                name="api_synthesize",
                input_data={"query": request.query, "top_k": request.top_k},
            ) as trace_ctx:
                synthesis_result = synthesizer.answer(
                    question=request.query,
                    search_fn=search_service.search,
                    top_k=request.top_k,
                    max_passages=5,
                )

                result = {
                    "query": request.query,
                    "answer": synthesis_result.get("answer"),
                    "sources": synthesis_result.get("sources", []),
                    "confidence": synthesis_result.get("confidence", 0.0),
                    "search_hits": synthesis_result.get("search_results", {}).get("text_hits", [])[:3],
                    "verification_result": synthesis_result.get("verification_result", {}),
                    "refined_query": synthesis_result.get("refined_query"),
                    "error": synthesis_result.get("error"),
                }
                
                trace_ctx["output"] = {
                    "answer_length": len(result["answer"]) if result["answer"] else 0,
                    "confidence": result["confidence"],
                }
                
                return result
        except Exception as e:
            tracer.langfuse.event(
                name="api_synthesize_error",
                output={"error": str(e)},
            ) if tracer.enabled else None
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

    return app


def serve_app(
    input_dir: str | Path = "articles",
    host: str = "127.0.0.1",
    port: int = 8000,
    use_gemini: bool = False,
    reload: bool = False,
) -> None:
    """Launch the FastAPI server."""
    import uvicorn

    app = create_app(input_dir=input_dir, use_gemini=use_gemini)
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
