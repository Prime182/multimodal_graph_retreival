"""FastAPI application for GraphRAG search service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .circuit_breaker import get_circuit_breaker
from .config import Phase1Settings
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
    )

    # Initialize synthesis service
    synthesizer = QuerySynthesizer()

    # ─── Frontend Routes ───────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def serve_index() -> str:
        """Serve the main frontend page."""
        index_path = _FRONTEND_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend not found")
        return index_path.read_text()

    @app.get("/app.js", response_class=str)
    async def serve_app_js() -> FileResponse:
        """Serve the app JS."""
        app_path = _FRONTEND_DIR / "app.js"
        if not app_path.exists():
            raise HTTPException(status_code=404, detail="App JS not found")
        return FileResponse(app_path, media_type="application/javascript")

    @app.get("/styles.css", response_class=str)
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
        return HealthResponse(
            status="ok",
            message=f"GraphRAG service ready with {len(search_service.papers)} papers",
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
