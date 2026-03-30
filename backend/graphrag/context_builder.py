"""Context formatting for retrieved passages and graph summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

_WORD_LIMIT = 4000


class RankedPassageLike(Protocol):
    rank: int
    score: float
    paper_id: str
    paper_title: str
    section_id: str
    section_title: str
    text: str
    doi: str | None
    retrieval_method: str
    graph_context: Any | None
    previous_text: str | None
    next_text: str | None


@dataclass(slots=True)
class GenerationContext:
    query: str
    passages: list[RankedPassageLike]
    formatted_context: str
    passage_count: int
    total_tokens_estimate: int
    source_map: dict[int, str] = field(default_factory=dict)


def _token_count(text: str) -> int:
    return len(text.split())


def _truncate_words(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def _stringify_graph_context(graph_context: Any | None) -> str:
    if graph_context is None:
        return ""
    if isinstance(graph_context, str):
        return graph_context.strip()
    if isinstance(graph_context, dict):
        nodes = graph_context.get("nodes", []) or []
        edges = graph_context.get("edges", []) or []
        seeds = graph_context.get("seed_node_ids", []) or []
        lines: list[str] = []
        lines.append(f"Graph context: {len(nodes)} nodes, {len(edges)} edges")
        if seeds:
            lines.append(f"Seeds: {', '.join(str(seed) for seed in seeds[:5])}")
        if nodes:
            node_bits: list[str] = []
            for node in nodes[:4]:
                if isinstance(node, dict):
                    label = node.get("label") or node.get("name") or node.get("node_id")
                    node_type = node.get("node_type") or node.get("type")
                    if label and node_type:
                        node_bits.append(f"{label} ({node_type})")
                    elif label:
                        node_bits.append(str(label))
                else:
                    label = getattr(node, "label", None) or getattr(node, "name", None) or getattr(node, "node_id", None)
                    node_type = getattr(node, "node_type", None) or getattr(node, "type", None)
                    if label and node_type:
                        node_bits.append(f"{label} ({node_type})")
                    elif label:
                        node_bits.append(str(label))
            if node_bits:
                lines.append("Nodes: " + "; ".join(node_bits))
        if edges:
            edge_bits: list[str] = []
            for edge in edges[:4]:
                if isinstance(edge, dict):
                    relation = edge.get("relation_type") or edge.get("type")
                    source = edge.get("source_label") or edge.get("source_id")
                    target = edge.get("target_label") or edge.get("target_id")
                    if relation and source and target:
                        edge_bits.append(f"{source} -[{relation}]-> {target}")
                else:
                    relation = getattr(edge, "relation_type", None) or getattr(edge, "type", None)
                    source = getattr(edge, "source_label", None) or getattr(edge, "source_id", None)
                    target = getattr(edge, "target_label", None) or getattr(edge, "target_id", None)
                    if relation and source and target:
                        edge_bits.append(f"{source} -[{relation}]-> {target}")
            if edge_bits:
                lines.append("Edges: " + "; ".join(edge_bits))
        return "\n".join(lines)
    return str(graph_context).strip()


class ContextBuilder:
    """Formats ranked passages into an LLM-ready context window."""

    def build(
        self,
        query: str,
        passages: Sequence[RankedPassageLike],
        max_tokens: int = _WORD_LIMIT,
    ) -> GenerationContext:
        formatted_parts: list[str] = []
        source_map: dict[int, str] = {}
        remaining_tokens = max_tokens

        for index, passage in enumerate(passages, start=1):
            citation = f"[{index}] {passage.paper_title} — {passage.section_title}"
            block_lines = [citation, f"Retrieval: {passage.retrieval_method} (score={passage.score:.4f})"]

            graph_summary = _stringify_graph_context(passage.graph_context)
            if graph_summary:
                block_lines.append(graph_summary)

            if passage.previous_text:
                block_lines.append(f"Previous: {passage.previous_text.strip()}")
            block_lines.append(f"Current: {passage.text.strip()}")
            if passage.next_text:
                block_lines.append(f"Next: {passage.next_text.strip()}")

            block = "\n".join(line for line in block_lines if line)
            block_tokens = _token_count(block)
            if block_tokens > remaining_tokens:
                if remaining_tokens <= 0:
                    break
                block = _truncate_words(block, remaining_tokens)
                block_tokens = _token_count(block)

            formatted_parts.append(block)
            source_map[index] = citation
            remaining_tokens -= block_tokens
            if remaining_tokens <= 0:
                break

        return GenerationContext(
            query=query,
            passages=list(passages[: len(formatted_parts)]),
            formatted_context="\n\n---\n\n".join(formatted_parts),
            passage_count=len(formatted_parts),
            total_tokens_estimate=max_tokens - remaining_tokens,
            source_map=source_map,
        )

