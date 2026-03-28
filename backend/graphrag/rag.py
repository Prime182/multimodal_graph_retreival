"""LLM-based query synthesis for generating answers from search results."""

from __future__ import annotations

import os
import re
from typing import Any

from .tracing import get_tracing_manager

try:
    try:
        import google.genai as genai  # type: ignore
    except ImportError:
        import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover
    genai = None


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


class QuerySynthesizer:
    """Synthesize answers using LLM from retrieved passages."""

    def __init__(self, model: str = "models/gemini-2.0-flash") -> None:
        if genai is None:
            self.enabled = False
            return

        self.enabled = True
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = model
            except Exception:
                self.enabled = False
        else:
            self.enabled = False

    def synthesize(
        self,
        question: str,
        search_results: dict[str, Any],
        max_passages: int = 5,
    ) -> dict[str, Any]:
        """
        Generate an answer by synthesizing top search results.

        Args:
            question: The user's query
            search_results: Output from search_service.search()
            max_passages: Max number of result passages to include

        Returns:
            Dict with 'answer', 'sources', 'confidence'
        """
        tracer = get_tracing_manager()
        
        if not self.enabled:
            return {
                "answer": None,
                "sources": [],
                "confidence": 0.0,
                "warning": "Synthesis not available (Gemini API not configured)",
            }

        # Extract top passages
        text_hits = search_results.get("text_hits", [])[:max_passages]
        if not text_hits:
            return {
                "answer": "No results found to synthesize.",
                "sources": [],
                "confidence": 0.0,
            }

        # Format passages
        passages_text = "\n\n".join(
            [
                f"[{i+1}] {hit['paper_title']} ({hit.get('doi', 'N/A')})\n"
                f"Section: {hit['section_title']}\n"
                f"Text: {hit['text'][:300]}..."
                for i, hit in enumerate(text_hits)
            ]
        )

        # Build prompt
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            question=question,
            passages=passages_text,
        )

        try:
            with tracer.trace(
                name="query_synthesis",
                input_data={"question": question, "passage_count": len(text_hits)},
            ) as trace_ctx:
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                answer_text = response.text.strip()

                # Log LLM call
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

                # Extract sources from answer
                sources = [
                    {"paper": hit["paper_title"], "doi": hit.get("doi"), "section": hit["section_title"]}
                    for hit in text_hits
                ]

                # Simple confidence based on result count and relevance
                confidence = min(len(text_hits) / max_passages, 1.0) * 0.8 + 0.2

                result = {
                    "answer": answer_text,
                    "sources": sources,
                    "confidence": round(confidence, 3),
                }
                
                trace_ctx["output"] = result
                return result
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
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)

                # Log LLM call
                tracer.log_llm_call(
                    name="claim_extraction",
                    model=self.model,
                    prompt=prompt[:500],
                    response=response.text[:500],
                    metadata={"text_length": len(text)},
                )

                # Extract JSON from response
                import json

                json_match = re.search(r"\[.*\]", response.text, re.DOTALL)
                if json_match:
                    claims = json.loads(json_match.group(0))
                    trace_ctx["output"] = {"claim_count": len(claims)}
                    return claims
                return []
        except Exception as e:
            tracer.langfuse.event(
                name="claim_extraction_error",
                output={"error": str(e)},
            ) if tracer.enabled else None
            return []
