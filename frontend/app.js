const state = {
  lastQuery: "",
};

const els = {
  form: document.getElementById("search-form"),
  input: document.getElementById("query-input"),
  textHits: document.getElementById("text-hits"),
  tableHits: document.getElementById("table-hits"),
  figureHits: document.getElementById("figure-hits"),
  paperHits: document.getElementById("paper-hits"),
  entities: document.getElementById("entities"),
  citations: document.getElementById("citations"),
  summary: document.getElementById("result-summary"),
  statPapers: document.getElementById("stat-papers"),
  statChunks: document.getElementById("stat-chunks"),
  statTables: document.getElementById("stat-tables"),
  statCitations: document.getElementById("stat-citations"),
  suggested: document.getElementById("suggested-searches"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderEmpty(node, label) {
  node.innerHTML = `<div class="empty">${escapeHtml(label)}</div>`;
}

function renderLoading(node, label) {
  node.innerHTML = `<div class="loading">${escapeHtml(label)}</div>`;
}

function entityBadge(entityType) {
  return `<span class="tag">${escapeHtml(entityType)}</span>`;
}

function renderTextHits(hits) {
  if (!hits.length) {
    renderEmpty(els.textHits, "No passage matched this query.");
    return;
  }
  els.textHits.innerHTML = hits
    .map((hit) => {
      const entities = (hit.entities || [])
        .slice(0, 5)
        .map(
          (entity) => `
            <span class="tag">${escapeHtml(entity.entity_type)}: ${escapeHtml(entity.label)}</span>
          `
        )
        .join("");
      const context = hit.context || {};
      return `
        <article class="hit-card">
          <div class="hit-topline">
            <span class="score-pill">${(hit.score * 100).toFixed(1)}%</span>
            <span>${escapeHtml(hit.paper_title)}</span>
            <span>•</span>
            <span>${escapeHtml(hit.section_title || hit.section_type || "Section")}</span>
          </div>
          <h3>${escapeHtml(hit.paper_title)}</h3>
          <p class="excerpt">${escapeHtml(hit.text)}</p>
          ${entities ? `<div class="hit-topline">${entities}</div>` : ""}
          ${
            context.previous || context.next
              ? `<div class="context">
                  ${context.previous ? `<div><strong>Prev</strong> ${escapeHtml(context.previous)}</div>` : ""}
                  <div><strong>Hit</strong> ${escapeHtml(context.current || hit.text)}</div>
                  ${context.next ? `<div><strong>Next</strong> ${escapeHtml(context.next)}</div>` : ""}
                </div>`
              : ""
          }
        </article>
      `;
    })
    .join("");
}

function renderTables(tables) {
  if (!tables.length) {
    renderEmpty(els.tableHits, "No table matched this query.");
    return;
  }
  els.tableHits.innerHTML = tables
    .map(
      (table) => `
        <article class="table-card">
          <div class="mini-topline">
            <span class="score-pill">${(table.score * 100).toFixed(1)}%</span>
            <span>${escapeHtml(table.paper_title)}</span>
            ${table.label ? `<span class="tag">${escapeHtml(table.label)}</span>` : ""}
          </div>
          <h3>${escapeHtml(table.caption || table.label || "Table")}</h3>
          <p>${escapeHtml(table.text)}</p>
        </article>
      `
    )
    .join("");
}

function renderFigures(figures) {
  if (!figures.length) {
    renderEmpty(els.figureHits, "No figure matched this query.");
    return;
  }
  els.figureHits.innerHTML = figures
    .map(
      (figure) => `
        <article class="table-card">
          <div class="mini-topline">
            <span class="score-pill">${(figure.score * 100).toFixed(1)}%</span>
            <span>${escapeHtml(figure.paper_title)}</span>
            ${figure.label ? `<span class="tag">${escapeHtml(figure.label)}</span>` : ""}
          </div>
          <h3>${escapeHtml(figure.caption || figure.label || "Figure")}</h3>
          <p>${escapeHtml(figure.text || figure.alt_text || "Image placeholder only")}</p>
          <p class="entity-metadata">${escapeHtml(figure.placeholder_uri || "placeholder://figure")}</p>
        </article>
      `
    )
    .join("");
}

function renderPapers(papers) {
  if (!papers.length) {
    renderEmpty(els.paperHits, "No article-level match for this query.");
    return;
  }
  els.paperHits.innerHTML = papers
    .map(
      (paper) => `
        <article class="citation-card">
          <div class="mini-topline">
            <span class="score-pill">${(paper.score * 100).toFixed(1)}%</span>
            <span>${paper.published_year || "Year?"}</span>
          </div>
          <h3>${escapeHtml(paper.title)}</h3>
          <p>${escapeHtml((paper.abstract || "").slice(0, 260))}${paper.abstract && paper.abstract.length > 260 ? "..." : ""}</p>
          <div class="entity-metadata">
            <span>${escapeHtml(paper.journal?.name || "Unknown journal")}</span>
            <span>•</span>
            <span>${escapeHtml((paper.authors || []).slice(0, 3).join(", "))}</span>
          </div>
        </article>
      `
    )
    .join("");
}

function renderEntities(groups) {
  const entries = Object.entries(groups || {});
  if (!entries.length) {
    renderEmpty(els.entities, "No structured entities matched this query.");
    return;
  }
  els.entities.innerHTML = entries
    .map(([type, items]) => {
      const cards = (items || [])
        .slice(0, 5)
        .map((entity) => {
          const label = entity.entity_type === "equation" ? entity.properties?.latex || entity.label : entity.label;
          const extra =
            entity.entity_type === "result"
              ? `Value ${entity.properties?.value ?? "?"} ${entity.properties?.unit || ""} on ${entity.properties?.dataset || "dataset"}`
              : entity.entity_type === "claim"
                ? entity.properties?.claim_type || "claim"
                : entity.properties?.method_type || entity.properties?.ontology || "entity";
          return `
            <article class="entity-card">
              <div class="entity-title">
                <span class="entity-label">${escapeHtml(label)}</span>
                ${entityBadge(entity.entity_type)}
              </div>
              <div class="entity-metadata">
                <span>${escapeHtml(entity.paper_title)}</span>
                <span>•</span>
                <span>${escapeHtml(extra)}</span>
              </div>
              ${
                entity.entity_type === "equation"
                  ? `<p class="equation">${escapeHtml(entity.properties?.plain_desc || entity.properties?.latex || entity.label)}</p>`
                  : `<p>${escapeHtml(entity.chunk_text || "")}</p>`
              }
            </article>
          `;
        })
        .join("");
      return `
        <div class="entity-group">
          <h3>${escapeHtml(type)}</h3>
          ${cards}
        </div>
      `;
    })
    .join("");
}

function renderCitations(citations) {
  if (!citations.length) {
    renderEmpty(els.citations, "No local citation edge matched this query.");
    return;
  }
  els.citations.innerHTML = citations
    .map(
      (citation) => `
        <article class="citation-card">
          <div class="mini-topline">
            <span class="score-pill">${(citation.confidence * 100).toFixed(1)}%</span>
            <span>${escapeHtml(citation.relation_type)}</span>
          </div>
          <h3>${escapeHtml(citation.source_paper_title)}</h3>
          <p>cites <strong>${escapeHtml(citation.target_paper_title)}</strong></p>
          <p>${escapeHtml(citation.evidence || "")}</p>
        </article>
      `
    )
    .join("");
}

function updateStats(stats) {
  els.statPapers.textContent = stats.paper_count ?? "-";
  els.statChunks.textContent = stats.chunk_count ?? "-";
  els.statTables.textContent = stats.table_count ?? "-";
  els.statCitations.textContent = stats.citation_count ?? "-";
}

function updateSummary(payload) {
  const chunkCount = payload.text_hits?.length ?? 0;
  const tableCount = payload.table_hits?.length ?? 0;
  const figureCount = payload.figure_hits?.length ?? 0;
  const citationCount = payload.citations?.length ?? 0;
  els.summary.textContent = `Returned ${chunkCount} text hits, ${tableCount} tables, ${figureCount} figures, and ${citationCount} citations for "${payload.query}".`;
}

async function runSearch(query) {
  const trimmed = query.trim();
  if (!trimmed) {
    return;
  }
  state.lastQuery = trimmed;
  els.input.value = trimmed;
  renderLoading(els.textHits, "Searching passages...");
  renderLoading(els.tableHits, "Searching tables...");
  renderLoading(els.figureHits, "Searching figures...");
  renderLoading(els.entities, "Loading entities...");
  renderLoading(els.citations, "Loading citations...");

  const response = await fetch(`/api/search?q=${encodeURIComponent(trimmed)}&top_k=6`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Search failed");
  }
  updateStats(payload.stats || {});
  updateSummary(payload);
  renderPapers(payload.papers || []);
  renderTextHits(payload.text_hits || []);
  renderTables(payload.table_hits || []);
  renderFigures(payload.figure_hits || []);
  renderEntities(payload.entities || {});
  renderCitations(payload.citations || []);
  const url = new URL(window.location.href);
  url.searchParams.set("q", trimmed);
  window.history.replaceState({}, "", url);
}

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await runSearch(els.input.value);
  } catch (error) {
    renderEmpty(els.textHits, String(error.message || error));
    renderEmpty(els.tableHits, "Search unavailable.");
    renderEmpty(els.figureHits, "Search unavailable.");
    renderEmpty(els.paperHits, "Search unavailable.");
    renderEmpty(els.entities, "Search unavailable.");
    renderEmpty(els.citations, "Search unavailable.");
  }
});

els.suggested.addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) {
    return;
  }
  els.input.value = button.textContent || "";
  try {
    await runSearch(els.input.value);
  } catch (error) {
    renderEmpty(els.textHits, String(error.message || error));
  }
});

const initialQuery = new URL(window.location.href).searchParams.get("q");
if (initialQuery) {
  runSearch(initialQuery).catch((error) => {
    renderEmpty(els.textHits, String(error.message || error));
  });
} else {
  updateStats({ paper_count: "-", chunk_count: "-", table_count: "-", citation_count: "-" });
  renderEmpty(els.paperHits, "Articles will appear here.");
  renderEmpty(els.textHits, "Enter a query to begin.");
  renderEmpty(els.tableHits, "Tables will appear here.");
  renderEmpty(els.figureHits, "Figures will appear here.");
  renderEmpty(els.entities, "Claims, results, and equations will appear here.");
  renderEmpty(els.citations, "Resolved citations will appear here.");
}
