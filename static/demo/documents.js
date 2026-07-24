// static/demo/documents.js
(function () {
  "use strict";
  const { apiUrl, bootNav, escapeHtml, checkHealth } = window.PrismDemo;

  function showError(msg) {
    const el = document.getElementById("error");
    if (!msg) {
      el.classList.remove("show");
      el.textContent = "";
      return;
    }
    el.textContent = msg;
    el.classList.add("show");
  }

  function basename(path) {
    if (!path) return "—";
    const parts = String(path).split(/[/\\]/);
    return parts[parts.length - 1] || path;
  }

  async function loadDocs() {
    showError(null);
    const tbody = document.getElementById("doc-tbody");
    tbody.innerHTML = `<tr><td colspan="8" class="muted">加载中…</td></tr>`;
    try {
      const r = await fetch(apiUrl("/documents"));
      let data;
      try {
        data = await r.json();
      } catch (_) {
        throw new Error("非 JSON · HTTP " + r.status);
      }
      if (!r.ok) throw new Error(data.detail || "HTTP " + r.status);
      renderStats(data.stats || {});
      renderTable(data.documents || []);
    } catch (e) {
      showError("加载文档库失败: " + (e.message || e));
      tbody.innerHTML = `<tr><td colspan="8" class="muted">无法加载。请确认 API 已启动（CONFIG_PROFILE=local-dev）。</td></tr>`;
    }
  }

  function renderStats(st) {
    document.getElementById("st-docs").textContent = st.num_documents ?? "—";
    document.getElementById("st-pages").textContent = st.num_pages ?? "—";
    document.getElementById("st-chunks").textContent = st.num_chunks ?? "—";
    document.getElementById("st-tables").textContent = st.num_table_chunks ?? "—";
    document.getElementById("st-faiss").textContent =
      st.index_pages != null ? `${st.index_pages}` : "—";
    document.getElementById("st-visual").textContent = st.use_visual ? "ON" : "OFF";
    document.getElementById("corpus-extra").innerHTML =
      `<span><b>FAISS size</b> ${st.index_size_mb ?? 0} MB</span>` +
      `<span><b>BM25</b> ${st.bm25_ready ? "ready" : "not ready"}</span>` +
      `<span><b>documents 表行</b> ${st.num_document_rows ?? "—"}</span>`;
  }

  function renderTable(docs) {
    const tbody = document.getElementById("doc-tbody");
    if (!docs.length) {
      tbody.innerHTML =
        `<tr><td colspan="8" class="muted">库为空。去 <a href="embed.html">嵌入页</a> 上传 PDF。</td></tr>`;
      return;
    }
    tbody.innerHTML = docs
      .map((d) => {
        const src = basename(d.source_path);
        const range =
          d.page_from != null && d.page_to != null
            ? `${d.page_from}–${d.page_to}`
            : "—";
        const when = d.created_at
          ? new Date(d.created_at).toLocaleString()
          : "—";
        const askHref =
          `index.html?doc_id=${encodeURIComponent(d.doc_id)}`;
        return (
          `<tr>` +
          `<td><code>${escapeHtml(d.doc_id)}</code></td>` +
          `<td title="${escapeHtml(d.source_path || "")}">${escapeHtml(src)}</td>` +
          `<td>${d.num_pages}</td>` +
          `<td>${d.num_chunks}</td>` +
          `<td>${d.num_tables}</td>` +
          `<td>${range}</td>` +
          `<td class="muted">${escapeHtml(when)}</td>` +
          `<td><a href="${askHref}">问答</a></td>` +
          `</tr>`
        );
      })
      .join("");
  }

  document.getElementById("btn-refresh").addEventListener("click", () => {
    checkHealth();
    loadDocs();
  });

  bootNav("docs");
  loadDocs();
})();
