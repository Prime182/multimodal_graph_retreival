"""CLI entrypoints for the Phase 1 document spine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import chunk_article
from .config import Phase1Settings
from .edges import build_layer3
from .extraction import extract_layer2
from .graph_store import Neo4jGraphStore
from .indexing import GraphIndexManager
from .parser import default_registry
from .retrieval import LocalVectorIndex
from .server import serve_app


def _xml_paths(input_dir: str | Path) -> list[Path]:
    return sorted(Path(input_dir).glob("*.xml"))


def _load_corpus(input_dir: str | Path, settings: Phase1Settings | None = None, use_gemini: bool = False):
    settings = settings or Phase1Settings.from_env()
    papers = []
    layer2_docs = []
    for path in _xml_paths(input_dir):
        paper = chunk_article(default_registry.parse(path), settings=settings)
        papers.append(paper)
        layer2_docs.append(extract_layer2(paper, settings=settings, use_gemini=use_gemini))
    return papers, layer2_docs


def build_spine(args: argparse.Namespace) -> int:
    settings = Phase1Settings.from_env()
    papers, _ = _load_corpus(args.input_dir, settings=settings, use_gemini=False)
    payload = [paper.to_dict() for paper in papers]
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def search(args: argparse.Namespace) -> int:
    index = LocalVectorIndex.from_xml_paths([str(path) for path in _xml_paths(args.input_dir)])
    for hit in index.search(query=args.query, top_k=args.top_k):
        print(f"[{hit.score:.4f}] {hit.paper_title} :: {hit.section_title}")
        print(hit.text)
        print()
    return 0


def build_layer2(args: argparse.Namespace) -> int:
    settings = Phase1Settings.from_env()
    papers, layer2_docs = _load_corpus(args.input_dir, settings=settings, use_gemini=args.use_gemini)
    payload = [doc.to_dict() for doc in layer2_docs]
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def build_layer3_command(args: argparse.Namespace) -> int:
    settings = Phase1Settings.from_env()
    papers, layer2_docs = _load_corpus(args.input_dir, settings=settings, use_gemini=args.use_gemini)
    layer3 = build_layer3(papers, layer2_docs)
    payload = layer3.to_dict()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def load_neo4j(args: argparse.Namespace) -> int:
    settings = Phase1Settings.from_env()
    store = Neo4jGraphStore(settings=settings)
    try:
        store.ensure_schema()
        papers, layer2_docs = _load_corpus(args.input_dir, settings=settings, use_gemini=args.use_gemini)
        for paper, extraction in zip(papers, layer2_docs):
            store.upsert_paper(paper)
            store.upsert_layer2(paper, extraction)
        store.upsert_layer3(build_layer3(papers, layer2_docs))
    finally:
        store.close()
    return 0


def build_indexes(args: argparse.Namespace) -> int:
    manager = GraphIndexManager()
    try:
        manager.ensure_property_indexes()
    finally:
        manager.close()
    return 0


def compute_pagerank(args: argparse.Namespace) -> int:
    manager = GraphIndexManager()
    try:
        manager.ensure_property_indexes()
        manager.compute_pagerank(max_iterations=args.max_iterations, damping_factor=args.damping_factor)
    finally:
        manager.close()
    return 0


def serve(args: argparse.Namespace) -> int:
    serve_app(
        input_dir=args.input_dir,
        host=args.host,
        port=args.port,
        use_gemini=args.use_gemini,
    )
    return 0


def reset_circuit_breaker_cmd(args: argparse.Namespace) -> int:
    from .circuit_breaker import get_circuit_breaker
    cb = get_circuit_breaker()
    cb.reset(args.service)
    service_label = args.service or "all services"
    print(f"✓ Circuit breaker reset for {service_label}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer 1 and Layer 2 GraphRAG tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    spine_parser = subparsers.add_parser("build-spine", help="Parse XML articles into a JSON spine artifact")
    spine_parser.add_argument("--input-dir", default="articles")
    spine_parser.add_argument("--output")
    spine_parser.set_defaults(func=build_spine)

    search_parser = subparsers.add_parser("search", help="Run local chunk retrieval over the XML corpus")
    search_parser.add_argument("query")
    search_parser.add_argument("--input-dir", default="articles")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.set_defaults(func=search)

    layer2_parser = subparsers.add_parser("build-layer2", help="Extract Layer 2 entities into a JSON artifact")
    layer2_parser.add_argument("--input-dir", default="articles")
    layer2_parser.add_argument("--output")
    layer2_parser.add_argument("--use-gemini", action="store_true")
    layer2_parser.set_defaults(func=build_layer2)

    layer3_parser = subparsers.add_parser("build-layer3", help="Build citation and semantic edge artifacts")
    layer3_parser.add_argument("--input-dir", default="articles")
    layer3_parser.add_argument("--output")
    layer3_parser.add_argument("--use-gemini", action="store_true")
    layer3_parser.set_defaults(func=build_layer3_command)

    load_parser = subparsers.add_parser("load-neo4j", help="Load the parsed spine into Neo4j")
    load_parser.add_argument("--input-dir", default="articles")
    load_parser.add_argument("--use-gemini", action="store_true")
    load_parser.set_defaults(func=load_neo4j)

    indexes_parser = subparsers.add_parser("build-indexes", help="Create Neo4j property indexes for the graph")
    indexes_parser.set_defaults(func=build_indexes)

    pagerank_parser = subparsers.add_parser("compute-pagerank", help="Compute PageRank over the citation graph")
    pagerank_parser.add_argument("--max-iterations", type=int, default=20)
    pagerank_parser.add_argument("--damping-factor", type=float, default=0.85)
    pagerank_parser.set_defaults(func=compute_pagerank)

    serve_parser = subparsers.add_parser("serve", help="Serve the browser search frontend")
    serve_parser.add_argument("--input-dir", default="articles")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--use-gemini", action="store_true")
    serve_parser.set_defaults(func=serve)

    reset_cb_parser = subparsers.add_parser(
        "reset-circuit-breaker",
        help="Reset all circuit breaker failure counters (use after transient Gemini outages).",
    )
    reset_cb_parser.add_argument(
        "--service",
        default=None,
        help="Service name to reset (omit to reset all services).",
    )
    reset_cb_parser.set_defaults(func=reset_circuit_breaker_cmd)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
