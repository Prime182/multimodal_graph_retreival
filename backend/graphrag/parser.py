"""Elsevier XML parser for the Phase 1 document spine."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from lxml import etree

from .models import (
    AuthorRecord,
    FigureRecord,
    JournalRecord,
    PaperRecord,
    ReferenceRecord,
    SectionRecord,
    TableRecord,
)


_PARSER = etree.XMLParser(recover=True, huge_tree=True)
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_FORMULA_TAGS = {"ce:formula", "ce:inline-equation", "ce:equation", "ce:math", "mml:math"}


def _norm(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def _node_text(node: etree._Element | None) -> str:
    if node is None:
        return ""
    return _norm(" ".join(part for part in node.itertext()))


def _first_child(node: etree._Element | None, tag: str) -> etree._Element | None:
    if node is None:
        return None
    for child in node:
        if child.tag == tag:
            return child
    return None


def _children(node: etree._Element | None, tag: str) -> list[etree._Element]:
    if node is None:
        return []
    return [child for child in node if child.tag == tag]


def _iter_descendants(node: etree._Element | None, tag: str) -> Iterable[etree._Element]:
    if node is None:
        return ()
    return (element for element in node.iter() if element.tag == tag)


def _slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text or "unknown"


def _scoped_id(paper_id: str, raw_id: str | None, prefix: str) -> str:
    value = raw_id or prefix
    return _slug(f"{paper_id}-{value}")


def _first_sentence(text: str) -> str | None:
    text = _norm(text)
    if not text:
        return None
    match = re.search(r"(.+?[.!?])(?:\s|$)", text)
    if match:
        return match.group(1).strip()
    words = text.split()
    return " ".join(words[: min(24, len(words))])


def _section_type(title: str) -> str:
    normalized = _slug(title)
    if not normalized:
        return "body"
    patterns = [
        ("abstract", "abstract"),
        ("highlight", "highlights"),
        ("introduction", "introduction"),
        ("background", "background"),
        ("related-work", "related_work"),
        ("literature-review", "related_work"),
        ("method", "methods"),
        ("materials-and-methods", "methods"),
        ("result", "results"),
        ("discussion", "discussion"),
        ("conclusion", "conclusion"),
        ("appendix", "appendix"),
        ("supplement", "supplement"),
        ("acknowledg", "acknowledgements"),
    ]
    for needle, kind in patterns:
        if needle in normalized:
            return kind
    return "body"


def _build_paper_id(doi: str | None, journal_code: str | None, article_id: str | None) -> str:
    if doi:
        return _slug(doi)
    return _slug("-".join(part for part in [journal_code, article_id] if part))


def _extract_year(root: etree._Element) -> int | None:
    for element in root.iter():
        if element.tag == "ce:copyright":
            year = element.attrib.get("year")
            if year and year.isdigit():
                return int(year)
        if element.tag == "ce:date-accepted":
            text = _node_text(element)
            match = re.search(r"(19|20)\d{2}", text)
            if match:
                return int(match.group(0))
    return None


def _parse_journal(item_info: etree._Element | None) -> JournalRecord | None:
    if item_info is None:
        return None
    journal_code = _node_text(_first_child(item_info, "jid")) or None
    if journal_code is None:
        return None
    return JournalRecord(
        journal_id=_slug(journal_code),
        code=journal_code,
        name=journal_code,
    )


def _parse_affiliations(head: etree._Element | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if head is None:
        return result
    for node in _iter_descendants(head, "ce:textfn"):
        text = _node_text(node)
        if text:
            node_id = node.attrib.get("id")
            if node_id:
                result[node_id] = text
    return result


def _parse_authors(head: etree._Element | None, paper_id: str) -> list[AuthorRecord]:
    author_group = _first_child(head, "ce:author-group")
    if author_group is None:
        return []
    affiliation_lookup = _parse_affiliations(head)
    authors: list[AuthorRecord] = []
    for ordinal, node in enumerate(_children(author_group, "ce:author"), start=1):
        given_name = _node_text(_first_child(node, "ce:given-name")) or None
        surname = _node_text(_first_child(node, "ce:surname")) or None
        full_name = _norm(" ".join(part for part in [given_name or "", surname or ""] if part))
        if not full_name:
            full_name = _node_text(node) or f"Author {ordinal}"
        email = None
        affiliations: list[str] = []
        for child in node:
            if child.tag == "ce:e-address" and child.attrib.get("type") == "email":
                email = _node_text(child) or email
            if child.tag == "ce:cross-ref":
                ref_id = child.attrib.get("refid")
                if ref_id and ref_id in affiliation_lookup:
                    affiliations.append(affiliation_lookup[ref_id])
        author_id = node.attrib.get("author-id") or node.attrib.get("id")
        if not author_id:
            author_id = f"{paper_id}-author-{ordinal}"
        authors.append(
            AuthorRecord(
                author_id=_slug(author_id),
                full_name=full_name,
                given_name=given_name,
                surname=surname,
                email=email,
                affiliations=affiliations,
            )
        )
    return authors


def _parse_abstracts(head: etree._Element | None) -> tuple[str, list[str]]:
    abstract_text = ""
    highlights: list[str] = []
    for node in _children(head, "ce:abstract"):
        sections = _children(node, "ce:abstract-sec")
        text = _norm(" ".join(_node_text(section) for section in sections)) if sections else _node_text(node)
        if not text:
            continue
        if node.attrib.get("class") == "author-highlights":
            parts = [part.strip("• ").strip() for part in text.split("•") if part.strip()]
            highlights.extend(parts)
        elif not abstract_text:
            abstract_text = text.replace("Abstract:", "", 1).replace("Abstract", "", 1).strip()
    return abstract_text, highlights


def _parse_keywords(head: etree._Element | None) -> list[str]:
    keywords_node = _first_child(head, "ce:keywords")
    if keywords_node is None:
        return []
    keywords: list[str] = []
    for node in _children(keywords_node, "ce:keyword"):
        keyword = _node_text(node)
        if not keyword:
            continue
        parts = keyword.split()
        if len(parts) <= 3 and any(len(part) == 1 for part in parts):
            keyword = "".join(parts)
        keywords.append(keyword)
    return keywords


def _first_descendant_text(node: etree._Element | None, tags: set[str]) -> str | None:
    if node is None:
        return None
    for child in node.iter():
        if child.tag in tags:
            text = _node_text(child)
            if text:
                return text
    return None


def _parse_references(tail: etree._Element | None) -> list[ReferenceRecord]:
    bibliography = _first_child(tail, "ce:bibliography")
    if bibliography is None:
        return []
    references: list[ReferenceRecord] = []
    for node in _children(_first_child(bibliography, "ce:bibliography-sec"), "ce:bib-reference"):
        reference_id = node.attrib.get("id") or f"ref-{len(references) + 1}"
        label = _node_text(_first_child(node, "ce:label")) or None
        source_text = _node_text(_first_child(node, "ce:source-text")) or None
        reference_container = _first_child(node, "sb:reference")
        title = None
        url = None
        doi = None
        if reference_container is not None:
            doi = _first_descendant_text(reference_container, {"ce:doi"}) or None
            url = _first_descendant_text(reference_container, {"sb:url", "ce:url"}) or None
            title = _first_descendant_text(
                reference_container,
                {"sb:maintitle", "sb:title", "ce:title"},
            ) or None
        if title is None:
            title = _first_descendant_text(node, {"sb:maintitle", "sb:title", "ce:title"}) or None
        references.append(
            ReferenceRecord(
                reference_id=_slug(reference_id),
                label=label,
                title=title,
                doi=doi,
                url=url,
                source_text=source_text,
            )
        )
    return references


def _math_text(node: etree._Element | None) -> str:
    if node is None:
        return ""
    text = _node_text(node)
    if text:
        return text
    alt = node.attrib.get("alttext") or node.attrib.get("altimg") or ""
    return _norm(alt.replace(".svg", "").replace("_", " ").replace("-", " "))


def _equation_placeholder(node: etree._Element | None) -> str:
    if node is None:
        return ""
    label = _node_text(_first_child(node, "ce:label"))
    math_text = _math_text(node)
    detail = " ".join(part for part in [label, math_text] if part).strip()
    return f"[Equation {detail}]".strip() if detail else "[Equation]"


def _figure_placeholder(node: etree._Element | None) -> str:
    if node is None:
        return ""
    label = _node_text(_first_child(node, "ce:label")) or "Figure"
    caption = _first_sentence(_node_text(_first_child(node, "ce:caption")) or "")
    return f"[{label}: {caption}]" if caption else f"[{label}]"


def _table_placeholder(node: etree._Element | None) -> str:
    if node is None:
        return ""
    label = _node_text(_first_child(node, "ce:label")) or "Table"
    caption = _first_sentence(_node_text(_first_child(node, "ce:caption")) or "")
    return f"[{label}: {caption}]" if caption else f"[{label}]"


def _render_node_text(node: etree._Element | None) -> str:
    if node is None:
        return ""

    parts: list[str] = [node.text or ""]
    for child in node:
        replacement = ""
        if child.tag == "ce:display":
            display_parts = []
            for grandchild in child:
                if grandchild.tag == "ce:figure":
                    display_parts.append(_figure_placeholder(grandchild))
                elif grandchild.tag == "ce:table":
                    display_parts.append(_table_placeholder(grandchild))
                elif grandchild.tag in _FORMULA_TAGS:
                    display_parts.append(_equation_placeholder(grandchild))
                else:
                    text = _render_node_text(grandchild)
                    if text:
                        display_parts.append(text)
            replacement = " ".join(part for part in display_parts if part)
        elif child.tag == "ce:figure":
            replacement = _figure_placeholder(child)
        elif child.tag == "ce:table":
            replacement = _table_placeholder(child)
        elif child.tag in _FORMULA_TAGS:
            replacement = _equation_placeholder(child)
        else:
            replacement = _render_node_text(child)
        if replacement:
            parts.append(f" {replacement} ")
        parts.append(child.tail or "")
    return _norm("".join(parts))


def _extract_table_rows(node: etree._Element | None) -> list[list[str]]:
    rows: list[list[str]] = []
    if node is None:
        return rows
    for row in node.iter("row"):
        values = [_render_node_text(entry) for entry in _children(row, "entry")]
        cleaned = [value for value in values if value]
        if cleaned:
            rows.append(cleaned)
    return rows


def _table_text(node: etree._Element, label: str | None, caption: str | None, alt_text: str | None) -> str:
    lines: list[str] = []
    if label:
        lines.append(label)
    if caption:
        lines.append(caption)
    if alt_text and alt_text != caption:
        lines.append(alt_text)

    tgroup = _first_child(node, "tgroup")
    for index, row in enumerate(_extract_table_rows(_first_child(tgroup, "thead")), start=1):
        lines.append(f"Header {index}: {' | '.join(row)}")
    for index, row in enumerate(_extract_table_rows(_first_child(tgroup, "tbody")), start=1):
        lines.append(f"Row {index}: {' | '.join(row)}")

    if not lines:
        fallback = _render_node_text(node)
        if fallback:
            lines.append(fallback)
    return _norm(" ".join(lines))


def _parse_tables(root: etree._Element, paper_id: str) -> list[TableRecord]:
    tables: list[TableRecord] = []
    for ordinal, node in enumerate(_iter_descendants(root, "ce:table"), start=1):
        table_id = _scoped_id(paper_id, node.attrib.get("id"), f"table-{ordinal}")
        label = _node_text(_first_child(node, "ce:label")) or None
        caption_node = _first_child(node, "ce:caption")
        caption = _node_text(caption_node) or None
        alt_text = _node_text(_first_child(node, "ce:alt-text")) or None
        text = _table_text(node, label=label, caption=caption, alt_text=alt_text)
        tgroup = _first_child(node, "tgroup")
        columns = None
        if tgroup is not None:
            cols = tgroup.attrib.get("cols")
            if cols and cols.isdigit():
                columns = int(cols)
        rows = sum(1 for child in node.iter() if child.tag == "row")
        tables.append(
            TableRecord(
                table_id=table_id,
                paper_id=paper_id,
                ordinal=ordinal,
                label=label,
                caption=caption,
                text=text,
                rows=rows or None,
                columns=columns,
            )
        )
    return tables


def _parse_figures(root: etree._Element, paper_id: str) -> list[FigureRecord]:
    figures: list[FigureRecord] = []
    for ordinal, node in enumerate(_iter_descendants(root, "ce:figure"), start=1):
        figure_id = _scoped_id(paper_id, node.attrib.get("id"), f"figure-{ordinal}")
        label = _node_text(_first_child(node, "ce:label")) or None
        caption = _node_text(_first_child(node, "ce:caption")) or None
        alt_text = _node_text(_first_child(node, "ce:alt-text")) or None
        link = _first_child(node, "ce:link")
        source_ref = None
        if link is not None:
            source_ref = link.attrib.get("{http://www.w3.org/1999/xlink}href") or link.attrib.get("locator")
        text = _norm(" ".join(part for part in [label or "", caption or "", alt_text or ""] if part))
        figures.append(
            FigureRecord(
                figure_id=figure_id,
                paper_id=paper_id,
                ordinal=ordinal,
                label=label,
                caption=caption,
                alt_text=alt_text,
                text=text,
                placeholder_uri=f"placeholder://figure/{paper_id}/{figure_id}",
                source_ref=source_ref,
            )
        )
    return figures


def _direct_paragraphs(section: etree._Element) -> list[str]:
    paragraphs: list[str] = []
    for child in section:
        if child.tag in {"ce:para", "ce:simple-para", "ce:list-item"}:
            text = _render_node_text(child)
            if text:
                paragraphs.append(text)
        elif child.tag == "ce:list":
            for list_item in _children(child, "ce:list-item"):
                text = _render_node_text(list_item)
                if text:
                    paragraphs.append(text)
    return paragraphs


def _build_section_record(
    node: etree._Element,
    paper_id: str,
    ordinal: int,
    level: int,
    parent_section_id: str | None,
) -> SectionRecord:
    title = _node_text(_first_child(node, "ce:section-title")) or f"Section {ordinal}"
    label = _node_text(_first_child(node, "ce:label"))
    if label and not title.startswith(label):
        title = f"{label} {title}".strip()
    paragraphs = _direct_paragraphs(node)
    text = _norm(" ".join(paragraphs))
    raw_section_id = node.attrib.get("id") or f"section-{ordinal}"
    section_id = _scoped_id(paper_id, raw_section_id, f"section-{ordinal}")
    return SectionRecord(
        section_id=section_id,
        paper_id=paper_id,
        title=title,
        section_type=_section_type(title),
        level=level,
        ordinal=ordinal,
        text=text,
        paragraphs=paragraphs,
        key_sentence=_first_sentence(text),
        parent_section_id=parent_section_id,
    )


def _collect_body_sections(
    node: etree._Element,
    paper_id: str,
    level: int,
    parent_section_id: str | None,
    start_ordinal: int,
) -> tuple[list[SectionRecord], int]:
    records: list[SectionRecord] = []
    ordinal = start_ordinal
    for child in _children(node, "ce:section"):
        record = _build_section_record(
            node=child,
            paper_id=paper_id,
            ordinal=ordinal,
            level=level,
            parent_section_id=parent_section_id,
        )
        records.append(record)
        ordinal += 1
        nested, ordinal = _collect_body_sections(
            node=child,
            paper_id=paper_id,
            level=level + 1,
            parent_section_id=record.section_id,
            start_ordinal=ordinal,
        )
        records.extend(nested)
    return records, ordinal


def parse_article(path: str | Path) -> PaperRecord:
    """Parse a local Elsevier XML article into a Layer 1 paper record."""

    source_path = Path(path)
    root = etree.parse(str(source_path), _PARSER).getroot()
    item_info = _first_child(root, "item-info")
    head = _first_child(root, "head")
    if head is None:
        head = _first_child(root, "simple-head")
    body = _first_child(root, "body")
    title = _node_text(_first_child(head, "ce:title")) or source_path.stem
    doi = _node_text(_first_child(item_info, "ce:doi")) or None
    journal_code = _node_text(_first_child(item_info, "jid")) or None
    article_number = _node_text(_first_child(item_info, "ce:article-number")) or None
    pii = _node_text(_first_child(item_info, "ce:pii")) or None
    article_id = _node_text(_first_child(item_info, "aid")) or source_path.stem
    paper_id = _build_paper_id(doi=doi, journal_code=journal_code, article_id=article_id)
    authors = _parse_authors(head, paper_id=paper_id)
    journal = _parse_journal(item_info)
    abstract, highlights = _parse_abstracts(head)
    keywords = _parse_keywords(head)
    references = _parse_references(_first_child(root, "tail"))
    tables = _parse_tables(root, paper_id=paper_id)
    figures = _parse_figures(root, paper_id=paper_id)

    sections: list[SectionRecord] = []
    if abstract:
        sections.append(
            SectionRecord(
                section_id=_scoped_id(paper_id, "abstract", "abstract"),
                paper_id=paper_id,
                title="Abstract",
                section_type="abstract",
                level=1,
                ordinal=0,
                text=abstract,
                paragraphs=[abstract],
                key_sentence=_first_sentence(abstract),
            )
        )
    if highlights:
        joined = " ".join(highlights)
        sections.append(
            SectionRecord(
                section_id=_scoped_id(paper_id, "highlights", "highlights"),
                paper_id=paper_id,
                title="Highlights",
                section_type="highlights",
                level=1,
                ordinal=1 if not sections else len(sections),
                text=joined,
                paragraphs=highlights,
                key_sentence=_first_sentence(joined),
            )
        )

    body_sections_node = _first_child(body, "ce:sections")
    start_ordinal = len(sections)
    body_sections, _ = _collect_body_sections(
        node=body_sections_node if body_sections_node is not None else body,
        paper_id=paper_id,
        level=1,
        parent_section_id=None,
        start_ordinal=start_ordinal,
    )
    if not body_sections and body is not None:
        body_paragraphs = _direct_paragraphs(body_sections_node if body_sections_node is not None else body)
        body_text = _norm(" ".join(body_paragraphs))
        if body_text:
            body_sections.append(
                SectionRecord(
                    section_id=_scoped_id(paper_id, "body", "body"),
                    paper_id=paper_id,
                    title="Body",
                    section_type="body",
                    level=1,
                    ordinal=start_ordinal,
                    text=body_text,
                    paragraphs=body_paragraphs,
                    key_sentence=_first_sentence(body_text),
                )
            )
    sections.extend(body_sections)

    metadata = {
        "journal_code": journal_code,
        "article_id": article_id,
        "figure_count": len(figures),
        "table_count": len(tables),
        "equation_count": sum(1 for element in root.iter() if element.tag in _FORMULA_TAGS or element.tag == "ce:display"),
        "reference_count": sum(1 for element in root.iter() if element.tag == "ce:bib-reference"),
        "source_format": "elsevier_xml",
    }

    return PaperRecord(
        paper_id=paper_id,
        source_path=str(source_path),
        title=title,
        doi=doi,
        pii=pii,
        article_number=article_number,
        published_year=_extract_year(root),
        abstract=abstract,
        highlights=highlights,
        keywords=keywords,
        journal=journal,
        authors=authors,
        references=references,
        tables=tables,
        figures=figures,
        sections=sections,
        metadata=metadata,
    )

# ---------------------------------------------------------------------------
# Parser abstraction and registry (Protocol-based extensibility)
# ---------------------------------------------------------------------------

from typing import Protocol, runtime_checkable


@runtime_checkable
class ArticleParser(Protocol):
    """Protocol for pluggable article parsers."""
    def can_parse(self, path: Path) -> bool: ...
    def parse(self, path: Path) -> PaperRecord: ...


class ElsevierParser:
    """Wraps the existing Elsevier ce: namespace parse_article()."""
    def can_parse(self, path: Path) -> bool:
        try:
            root = etree.parse(str(path), _PARSER).getroot()
            return root.tag in {"full-text-retrieval-response", "article"} and any(
                child.tag == "item-info" for child in root
            )
        except Exception:
            return False

    def parse(self, path: Path) -> PaperRecord:
        return parse_article(path)


class JATSParser:
    """Handles PubMed / arXiv JATS XML (root tag: <article>)."""
    def can_parse(self, path: Path) -> bool:
        try:
            root = etree.parse(str(path), _PARSER).getroot()
            local = root.tag.split("}")[-1] if "}" in root.tag else root.tag
            return local == "article" and root.get("article-type") is not None
        except Exception:
            return False

    def parse(self, path: Path) -> PaperRecord:
        root = etree.parse(str(path), _PARSER).getroot()

        def _get(xpath: str) -> str:
            try:
                el = root.find(xpath)
                return _norm(" ".join(el.itertext())) if el is not None else ""
            except Exception:
                return ""

        title = _get(".//article-title") or path.stem
        abstract = _get(".//abstract")
        doi = _get(".//article-id[@pub-id-type='doi']") or None
        paper_id = _build_paper_id(doi=doi, journal_code=None, article_id=path.stem)

        # Build minimal PaperRecord — sections from body
        body = root.find(".//body")
        sections: list[SectionRecord] = []
        if abstract:
            sections.append(SectionRecord(
                section_id=_scoped_id(paper_id, "abstract", "abstract"),
                paper_id=paper_id, title="Abstract",
                section_type="abstract", level=1, ordinal=0,
                text=abstract, paragraphs=[abstract],
                key_sentence=_first_sentence(abstract),
            ))
        if body is not None:
            for i, sec in enumerate(body.findall(".//sec")):
                sec_title = _norm(" ".join(sec.find("title").itertext())) if sec.find("title") is not None else f"Section {i+1}"
                paras = [_norm(" ".join(p.itertext())) for p in sec.findall(".//p")]
                text = " ".join(paras)
                if text:
                    sections.append(SectionRecord(
                        section_id=_scoped_id(paper_id, f"sec-{i}", f"sec-{i}"),
                        paper_id=paper_id, title=sec_title,
                        section_type=_section_type(sec_title),
                        level=1, ordinal=i+1,
                        text=text, paragraphs=paras,
                        key_sentence=_first_sentence(text),
                    ))

        return PaperRecord(
            paper_id=paper_id, source_path=str(path),
            title=title, doi=doi,
            abstract=abstract, sections=sections,
            metadata={"source_format": "jats_xml"},
        )


class ParserRegistry:
    """Registry for pluggable article parsers."""
    def __init__(self) -> None:
        self._parsers: list[ArticleParser] = []

    def register(self, parser: ArticleParser) -> None:
        self._parsers.append(parser)

    def parse(self, path: Path) -> PaperRecord:
        for parser in self._parsers:
            if parser.can_parse(path):
                return parser.parse(path)
        raise ValueError(
            f"No registered parser can handle '{path}'. "
            "Supported formats: Elsevier XML, JATS/PubMed XML."
        )


# Default registry — used by cli.py and search_service.py
default_registry = ParserRegistry()
default_registry.register(ElsevierParser())
default_registry.register(JATSParser())