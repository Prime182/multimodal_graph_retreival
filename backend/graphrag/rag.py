"""LLM-based query synthesis for generating answers from search results."""

from __future__ import annotations

from functools import lru_cache
import os
import re
from typing import Any, Callable, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from .gemini import generate_json, generate_text, gemini_available
from .tracing import get_tracing_manager

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - dependency is installed for Week 3
    END = START = StateGraph = None


SYNTHESIS_PROMPT_TEMPLATE = """You are a scientific knowledge synthesizer. Given a user's question and retrieved passages from research papers, synthesize a concise, factual answer.

IMPORTANT RULES:
1. Only use information from the provided passages.
2. Cite sources with [Source: paper title, year].
3. If sources contradict, acknowledge both perspectives.
4. If you're uncertain, say so.
5. Keep answers to 2-3 sentences maximum.

QUESTION: {question}

RETRIEVED PASSAGES:
{passages}

ANSWER:"""

VERIFICATION_PROMPT_TEMPLATE = """Given these source passages and an answer, determine if the answer is fully grounded.

SOURCES:
{passages}

ANSWER:
{answer}

Return ONLY valid JSON:
{{
  "grounded": true,
  "unsupported_claims": [],
  "confidence": 0.0,
  "reason": "brief justification"
}}"""

REFINEMENT_PROMPT_TEMPLATE = """You are refining a retrieval query for a scientific RAG system.

QUESTION:
{question}

PREVIOUS QUERY:
{current_query}

TOP RETRIEVED TITLES:
{titles}

VERIFICATION ISSUES:
{issues}

Return one short improved search query only. Do not explain it."""

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+-]*")


class _VerificationPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class _QueryGraphState(TypedDict):
    question: str
    current_query: str
    top_k: int
    max_passages: int
    search_fn: Callable[..., dict[str, Any]]
    search_results: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    answer: str | None
    sources: list[dict[str, Any]]
    synthesis_confidence: float
    verification_result: dict[str, Any]
    retry_count: int
    refined_query: str | None


def _format_passages(text_hits: list[dict[str, Any]], max_passages: int) -> tuple[list[dict[str, Any]], str]:
    selected_hits = text_hits[:max_passages]
    passages_text = "\n\n".join(
        [
            f"[{i+1}] {hit['paper_title']} ({hit.get('doi', 'N/A')})\n"
            f"Section: {hit['section_title']}\n"
            f"Text: {hit['text'][:300]}..."
            for i, hit in enumerate(selected_hits)
        ]
    )
    return selected_hits, passages_text


def _base_sources(text_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"paper": hit["paper_title"], "doi": hit.get("doi"), "section": hit["section_title"]}
        for hit in text_hits
    ]


def _base_confidence(text_hits: list[dict[str, Any]], max_passages: int) -> float:
    if not text_hits:
        return 0.0
    return round(min(len(text_hits) / max_passages, 1.0) * 0.8 + 0.2, 3)


def _heuristic_refine_query(question: str, search_results: dict[str, Any]) -> str:
    entity_terms: list[str] = []
    entities = search_results.get("entities", {})
    for entity_list in entities.values():
        for entity in entity_list[:2]:
            label = str(entity.get("label", "")).strip()
            if label:
                entity_terms.append(label)
        if len(entity_terms) >= 3:
            break

    if not entity_terms:
        terms = [token.lower() for token in _TOKEN_RE.findall(question) if len(token) > 3]
        return " ".join(terms[:8]) or question
    return f"{question} {' '.join(entity_terms[:3])}".strip()


class QuerySynthesizer:
    """Synthesize answers using a query-quality LangGraph loop."""

    def __init__(self, model: str = "models/gemini-2.0-flash") -> None:
        self.model = model
        self.enabled = gemini_available()

    def _synthesize_from_results(
        self,
        question: str,
        search_results: dict[str, Any],
        max_passages: int = 5,
    ) -> dict[str, Any]:
        tracer = get_tracing_manager()

        if not self.enabled:
            return {
                "answer": None,
                "sources": [],
                "confidence": 0.0,
                "warning": "Synthesis not available (Gemini API not configured)",
            }

        text_hits, passages_text = _format_passages(search_results.get("text_hits", []), max_passages)
        if not text_hits:
            return {
                "answer": "No results found to synthesize.",
                "sources": [],
                "confidence": 0.0,
            }

        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            question=question,
            passages=passages_text,
        )

        with tracer.trace(
            name="query_synthesis",
            input_data={"question": question, "passage_count": len(text_hits)},
        ) as trace_ctx:
            answer_text = generate_text(
                prompt,
                self.model,
                temperature=0.1,
            )
            tracer.log_llm_call(
                name="synthesis",
                model=self.model,
                prompt=prompt[:500],
                response=answer_text[:500],
                metadata={
                    "question": question[:100],
                    "passage_count": len(text_hits),
                },
            )

            result = {
                "answer": answer_text,
                "sources": _base_sources(text_hits),
                "confidence": _base_confidence(text_hits, max_passages),
            }
            trace_ctx["output"] = result
            return result

    def _verify_answer(
        self,
        answer: str | None,
        text_hits: list[dict[str, Any]],
        max_passages: int,
    ) -> dict[str, Any]:
        if not answer or not text_hits:
            return {
                "grounded": False,
                "unsupported_claims": [],
                "confidence": 0.0,
                "reason": "no answer or retrieved passages",
            }

        selected_hits, passages_text = _format_passages(text_hits, max_passages)
        prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            passages=passages_text,
            answer=answer,
        )

        try:
            payload = _VerificationPayload.model_validate(
                generate_json(
                    prompt,
                    self.model,
                    temperature=0.1,
                )
            )
            return payload.model_dump(mode="json")
        except Exception:
            return {
                "grounded": False,
                "unsupported_claims": [],
                "confidence": 0.0,
                "reason": "verification failed",
            }

    def _refine_query(
        self,
        question: str,
        current_query: str,
        search_results: dict[str, Any],
        verification_result: dict[str, Any],
    ) -> str:
        titles = "\n".join(
            f"- {hit.get('paper_title', '')}"
            for hit in search_results.get("text_hits", [])[:5]
            if hit.get("paper_title")
        ) or "- no retrieved titles"
        issues = "\n".join(
            f"- {item}"
            for item in verification_result.get("unsupported_claims", [])
        ) or f"- {verification_result.get('reason', 'low grounding confidence')}"
        prompt = REFINEMENT_PROMPT_TEMPLATE.format(
            question=question,
            current_query=current_query,
            titles=titles,
            issues=issues,
        )

        try:
            refined = generate_text(prompt, self.model, temperature=0.1).strip()
            refined = refined.splitlines()[0].strip()
            return refined or _heuristic_refine_query(question, search_results)
        except Exception:
            return _heuristic_refine_query(question, search_results)

    def _retrieve_node(self, state: _QueryGraphState) -> _QueryGraphState:
        query = state["refined_query"] or state["current_query"]
        search_results = state["search_fn"](query=query, top_k=state["top_k"])
        return {
            **state,
            "current_query": query,
            "search_results": search_results,
            "retrieved_chunks": search_results.get("text_hits", [])[: state["max_passages"]],
        }

    def _synthesize_node(self, state: _QueryGraphState) -> _QueryGraphState:
        synthesis = self._synthesize_from_results(
            question=state["question"],
            search_results=state["search_results"],
            max_passages=state["max_passages"],
        )
        return {
            **state,
            "answer": synthesis.get("answer"),
            "sources": synthesis.get("sources", []),
            "synthesis_confidence": float(synthesis.get("confidence", 0.0) or 0.0),
        }

    def _verify_node(self, state: _QueryGraphState) -> _QueryGraphState:
        verification_result = self._verify_answer(
            answer=state["answer"],
            text_hits=state["retrieved_chunks"],
            max_passages=state["max_passages"],
        )
        return {
            **state,
            "verification_result": verification_result,
        }

    def _should_refine(self, state: _QueryGraphState) -> str:
        if state["retry_count"] >= 2:
            return "respond"

        verification = state["verification_result"]
        if not state["retrieved_chunks"]:
            return "refine"
        if not verification.get("grounded", False):
            return "refine"
        if float(verification.get("confidence", 1.0) or 0.0) < 0.6:
            return "refine"
        return "respond"

    def _refine_query_node(self, state: _QueryGraphState) -> _QueryGraphState:
        refined_query = self._refine_query(
            question=state["question"],
            current_query=state["current_query"],
            search_results=state["search_results"],
            verification_result=state["verification_result"],
        )
        return {
            **state,
            "refined_query": refined_query,
            "retry_count": state["retry_count"] + 1,
        }

    def _respond_node(self, state: _QueryGraphState) -> _QueryGraphState:
        return state

    @lru_cache(maxsize=1)
    def _query_graph(self) -> Any | None:
        if StateGraph is None or START is None or END is None:
            return None

        graph = StateGraph(_QueryGraphState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("verify", self._verify_node)
        graph.add_node("refine_query", self._refine_query_node)
        graph.add_node("respond", self._respond_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "synthesize")
        graph.add_edge("synthesize", "verify")
        graph.add_conditional_edges(
            "verify",
            self._should_refine,
            {
                "refine": "refine_query",
                "respond": "respond",
            },
        )
        graph.add_edge("refine_query", "retrieve")
        graph.add_edge("respond", END)
        return graph.compile(name="query_quality_graph")

    def answer(
        self,
        question: str,
        search_fn: Callable[..., dict[str, Any]],
        top_k: int = 5,
        max_passages: int = 5,
    ) -> dict[str, Any]:
        tracer = get_tracing_manager()

        if not self.enabled:
            return {
                "query": question,
                "answer": None,
                "sources": [],
                "confidence": 0.0,
                "search_results": search_fn(query=question, top_k=top_k),
                "verification_result": {
                    "grounded": False,
                    "unsupported_claims": [],
                    "confidence": 0.0,
                    "reason": "synthesis not available",
                },
                "refined_query": None,
                "warning": "Synthesis not available (Gemini API not configured)",
            }

        graph = self._query_graph()
        if graph is None:
            search_results = search_fn(query=question, top_k=top_k)
            synthesis = self._synthesize_from_results(question, search_results, max_passages=max_passages)
            return {
                "query": question,
                "answer": synthesis.get("answer"),
                "sources": synthesis.get("sources", []),
                "confidence": synthesis.get("confidence", 0.0),
                "search_results": search_results,
                "verification_result": {},
                "refined_query": None,
            }

        with tracer.trace(
            name="query_answer_loop",
            input_data={"question": question, "top_k": top_k},
        ) as trace_ctx:
            final_state = graph.invoke(
                {
                    "question": question,
                    "current_query": question,
                    "top_k": top_k,
                    "max_passages": max_passages,
                    "search_fn": search_fn,
                    "search_results": {},
                    "retrieved_chunks": [],
                    "answer": None,
                    "sources": [],
                    "synthesis_confidence": 0.0,
                    "verification_result": {},
                    "retry_count": 0,
                    "refined_query": None,
                }
            )

            verification_confidence = float(
                final_state["verification_result"].get("confidence", 0.0) or 0.0
            )
            confidence = round(min(final_state["synthesis_confidence"], verification_confidence), 3)
            result = {
                "query": question,
                "answer": final_state["answer"],
                "sources": final_state["sources"],
                "confidence": confidence,
                "search_results": final_state["search_results"],
                "verification_result": final_state["verification_result"],
                "refined_query": final_state["refined_query"],
            }
            trace_ctx["output"] = {
                "confidence": confidence,
                "retry_count": final_state["retry_count"],
                "grounded": final_state["verification_result"].get("grounded", False),
            }
            return result

    def synthesize(
        self,
        question: str,
        search_results: dict[str, Any],
        max_passages: int = 5,
    ) -> dict[str, Any]:
        """Single-pass synthesis retained for direct callers."""
        tracer = get_tracing_manager()

        try:
            return self._synthesize_from_results(
                question=question,
                search_results=search_results,
                max_passages=max_passages,
            )
        except Exception as e:
            tracer.langfuse.event(
                name="synthesis_error",
                output={"error": str(e)},
            ) if tracer.enabled else None

            return {
                "answer": None,
                "sources": [],
                "confidence": 0.0,
                "error": f"Synthesis failed: {str(e)}",
            }

    def extract_claims(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Extract structured claims from arbitrary text using LLM.
        Returns list of {claim, confidence, evidence}.
        """
        tracer = get_tracing_manager()

        if not self.enabled:
            return []

        prompt = f"""Extract all factual claims from this text. For each claim, provide:
1. The claim itself (one sentence)
2. Confidence (0.0-1.0)
3. Evidence (supporting quote)

Format as JSON array: [
  {{"claim": "...", "confidence": 0.9, "evidence": "..."}}
]

TEXT: {text}

JSON:"""

        try:
            with tracer.trace(
                name="claim_extraction",
                input_data={"text_length": len(text)},
            ) as trace_ctx:
                claims_payload = generate_json(
                    prompt,
                    self.model,
                    temperature=0.1,
                )
                tracer.log_llm_call(
                    name="claim_extraction",
                    model=self.model,
                    prompt=prompt[:500],
                    response=str(claims_payload)[:500],
                    metadata={"text_length": len(text)},
                )

                if isinstance(claims_payload, list):
                    claims = claims_payload
                    trace_ctx["output"] = {"claim_count": len(claims)}
                    return claims
                return []
        except Exception as e:
            tracer.langfuse.event(
                name="claim_extraction_error",
                output={"error": str(e)},
            ) if tracer.enabled else None
            return []
