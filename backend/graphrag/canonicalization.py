"""Entity canonicalization and deduplication for the GraphRAG corpus."""

from __future__ import annotations

import os
from typing import Any

from .entities import Layer2DocumentRecord, Layer2EntityRecord
from .embeddings import cosine_similarity
from .tracing import get_tracing_manager

try:
    try:
        import google.genai as genai  # type: ignore
    except ImportError:
        import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover
    genai = None


CANONICALIZE_PROMPT = """Given a list of entity aliases for '{entity_type}' entities, determine which ones refer to the same real-world concept.

Return a JSON object mapping canonical names to their aliases:
{{
  "canonical_name": ["alias1", "alias2", ...]
}}

ALIASES: {aliases}

JSON:"""


class EntityCanonicalizer:
    """Merge duplicate entities using similarity metrics and LLM."""

    def __init__(self, use_gemini: bool = False) -> None:
        self.use_gemini = use_gemini
        self.gemini_enabled = False

        if use_gemini and genai is not None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_enabled = True
                except Exception:
                    pass

    def canonicalize_corpus(
        self,
        layer2_docs: list[Layer2DocumentRecord],
        similarity_threshold: float = 0.85,
    ) -> dict[str, str]:
        """
        Build a canonicalization map: {entity_id -> canonical_entity_id}.
        Merges entities by type and similarity.

        Args:
            layer2_docs: All extracted entities from corpus
            similarity_threshold: Merge entities with similarity >= this

        Returns:
            Mapping of entity_id -> canonical_entity_id
        """
        tracer = get_tracing_manager()
        
        canonical_map: dict[str, str] = {}
        entity_groups: dict[str, list[tuple[str, Layer2EntityRecord]]] = {}

        with tracer.trace(
            name="entity_canonicalization",
            input_data={"doc_count": len(layer2_docs)},
        ) as trace_ctx:
            # Group entities by type
            for doc in layer2_docs:
                for entity in doc.entities:
                    key = entity.entity_type
                    if key not in entity_groups:
                        entity_groups[key] = []
                    entity_groups[key].append((doc.paper_id, entity))

            # Merge within each entity type
            merge_stats: dict[str, int] = {}
            for entity_type, entities in entity_groups.items():
                merges = self._find_merges(entity_type, entities, similarity_threshold)
                canonical_map.update(merges)
                merge_stats[entity_type] = len(merges)

            # Log canonicalization metrics
            tracer.langfuse.event(
                name="canonicalization_complete",
                output={
                    "mapping_size": len(canonical_map),
                    "entity_types": len(entity_groups),
                },
                metadata={
                    "mapping_size": len(canonical_map),
                    **{f"merges_{k}": v for k, v in merge_stats.items()},
                },
            ) if tracer.enabled else None
            
            trace_ctx["output"] = {
                "mapping_size": len(canonical_map),
                "entity_types": len(entity_groups),
            }

        tracer.flush()
        return canonical_map

    def _find_merges(
        self,
        entity_type: str,
        entities: list[tuple[str, Layer2EntityRecord]],
        similarity_threshold: float,
    ) -> dict[str, str]:
        """Find entity merges within a single entity type."""
        canonical_map: dict[str, str] = {}
        seen_entities: dict[str, str] = {}  # {entity_id -> canonical_id}

        if self.gemini_enabled and len(entities) > 10:
            # Use LLM for large groups
            groups = self._llm_group_entities(entity_type, entities)
        else:
            # Use embedding similarity for small groups
            groups = self._similarity_group_entities(entities, similarity_threshold)

        for group in groups:
            if not group:
                continue

            # First entity in group is canonical
            canonical_id = group[0][1].entity_id
            for paper_id, entity in group:
                canonical_map[entity.entity_id] = canonical_id
                seen_entities[entity.entity_id] = canonical_id

        return canonical_map

    def _similarity_group_entities(
        self,
        entities: list[tuple[str, Layer2EntityRecord]],
        threshold: float,
    ) -> list[list[tuple[str, Layer2EntityRecord]]]:
        """Group entities by embedding similarity."""
        groups: list[list[tuple[str, Layer2EntityRecord]]] = []
        assigned: set[str] = set()

        for i, (paper_a, entity_a) in enumerate(entities):
            if entity_a.entity_id in assigned:
                continue

            current_group = [(paper_a, entity_a)]
            assigned.add(entity_a.entity_id)

            for j, (paper_b, entity_b) in enumerate(entities[i + 1 :], start=i + 1):
                if entity_b.entity_id in assigned:
                    continue

                # Check embedding similarity
                if entity_a.embedding and entity_b.embedding:
                    sim = cosine_similarity(entity_a.embedding, entity_b.embedding)
                    if sim >= threshold:
                        current_group.append((paper_b, entity_b))
                        assigned.add(entity_b.entity_id)
                # Check label similarity
                elif self._label_similar(entity_a.label, entity_b.label):
                    current_group.append((paper_b, entity_b))
                    assigned.add(entity_b.entity_id)

            groups.append(current_group)

        return groups

    def _llm_group_entities(
        self,
        entity_type: str,
        entities: list[tuple[str, Layer2EntityRecord]],
    ) -> list[list[tuple[str, Layer2EntityRecord]]]:
        """Use LLM to group similar entities."""
        if not self.gemini_enabled:
            return [[(paper, entity)] for paper, entity in entities]

        # Extract unique labels
        unique_labels = list({entity.label for _, entity in entities})[:50]  # Limit to 50

        prompt = CANONICALIZE_PROMPT.format(
            entity_type=entity_type,
            aliases=", ".join(unique_labels),
        )

        try:
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            response = model.generate_content(prompt)

            import json
            import re

            json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
            if json_match:
                groups_dict = json.loads(json_match.group(0))

                # Map labels to entities
                label_to_entity = {entity.label: (paper, entity) for paper, entity in entities}

                # Build groups from LLM output
                groups: list[list[tuple[str, Layer2EntityRecord]]] = []
                for canonical, aliases in groups_dict.items():
                    group = []
                    if canonical in label_to_entity:
                        group.append(label_to_entity[canonical])

                    for alias in aliases:
                        if alias in label_to_entity and label_to_entity[alias] not in group:
                            group.append(label_to_entity[alias])

                    if group:
                        groups.append(group)

                return groups
        except Exception:
            pass

        # Fallback to single-entity groups
        return [[(paper, entity)] for paper, entity in entities]

    def _label_similar(self, label_a: str, label_b: str, threshold: float = 0.8) -> bool:
        """Simple string similarity using character overlap."""
        if label_a.lower() == label_b.lower():
            return True

        # Check if one is a substring/expansion of the other
        a_lower = label_a.lower()
        b_lower = label_b.lower()

        if a_lower in b_lower or b_lower in a_lower:
            return True

        # Check acronym matching (E.g., "BERT" vs "Bidirectional Encoder...")
        a_words = label_a.split()
        b_words = label_b.split()

        a_acronym = "".join(w[0].upper() for w in a_words if w)
        b_acronym = "".join(w[0].upper() for w in b_words if w)

        if a_acronym and b_acronym and (a_acronym == b_acronym or a_acronym in b_acronym):
            return True

        return False

    def apply_canonicalization(
        self,
        layer2_docs: list[Layer2DocumentRecord],
        canonical_map: dict[str, str],
    ) -> list[Layer2DocumentRecord]:
        """
        Apply canonicalization map to update entity references.
        Returns new Layer2DocumentRecord objects with canonical IDs.
        """
        from copy import deepcopy

        canonicalized = deepcopy(layer2_docs)

        for doc in canonicalized:
            for entity in doc.entities:
                if entity.entity_id in canonical_map:
                    canonical_id = canonical_map[entity.entity_id]
                    # Set canonical ID but keep original as alias
                    if entity.label not in entity.aliases:
                        entity.aliases.append(entity.label)
                    entity.entity_id = canonical_id

        return canonicalized
