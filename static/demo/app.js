// static/demo/app.js
(function () {
  "use strict";

  const state = {
    mode: "demo", // "demo" | "live"
    apiBase: "",
    health: null,
    query: "",
    loading: false,
    error: null,
    response: null,
    traceId: null,
    lastDocId: null,
    filterByDoc: false,
    fixtures: null,
    metrics: null,
  };

  const $ = (id) => document.getElementById(id);

  function showError(msg) {
    state.error = msg;
    const el = $("error");
    if (!msg) {
      el.classList.remove("show");
      el.textContent = "";
      return;
    }
    el.textContent = msg;
    el.classList.add("show");
  }

  function setLoading(on) {
    state.loading = on;
    $("loading").hidden = !on;
    $("btn-ask").disabled = on;
  }

  function baseUrl() {
    const raw = ($("api-base").value || "").trim().replace(/\/$/, "");
    return raw;
  }

  function apiUrl(path) {
    const b = baseUrl();
    if (!b) return path.startsWith("/") ? path : `/${path}`;
    return `${b}${path.startsWith("/") ? path : `/${path}`}`;
  }

  function renderMetrics() {
    const root = $("metrics");
    root.innerHTML = "";
    if (!state.metrics || !state.metrics.chips) return;
    state.metrics.chips.forEach((c) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.textContent = `${c.label} ${c.value}`;
      btn.title = c.detail;
      btn.addEventListener("click", () => {
        const box = $("chip-detail");
        const open = box.classList.contains("open") && box.dataset.label === c.label;
        if (open) {
          box.classList.remove("open");
          return;
        }
        box.dataset.label = c.label;
        box.textContent = c.detail;
        box.classList.add("open");
      });
      root.appendChild(btn);
    });
    $("footnote").textContent = state.metrics.footnote || "";
  }

  function renderPresets() {
    const root = $("presets");
    root.innerHTML = "";
    if (!state.fixtures) return;
    state.fixtures.presets.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = p.label || p.id;
      if ((p.id || "").includes("reject") || (p.label || "").toLowerCase().includes("out")) {
        btn.classList.add("reject");
      }
      btn.addEventListener("click", () => {
        $("query").value = p.query;
        ask();
      });
      root.appendChild(btn);
    });
  }

  function renderRouteList(el, items) {
    el.innerHTML = "";
    if (!items || !items.length) {
      el.innerHTML = "<li class='muted'>(empty)</li>";
      return;
    }
    items.forEach((it) => {
      const li = document.createElement("li");
      const score = typeof it.score === "number" ? it.score.toFixed(3) : it.score;
      li.textContent = `${it.chunk_id || "?"} · p${it.page_id} · ${score}`;
      el.appendChild(li);
    });
  }

  function renderCitations(citations) {
    const root = $("citations");
    root.innerHTML = "";
    if (!citations || !citations.length) {
      root.innerHTML = "<p class='muted'>No citations</p>";
      return;
    }
    citations.forEach((c, i) => {
      const div = document.createElement("div");
      div.className = "citation";
      const page = c.page_number != null ? c.page_number : c.page_id;
      div.innerHTML =
        `<div class="meta"><b>C${i + 1}</b> · page ${page} · ` +
        `${c.doc_id || "—"} · <code>${c.chunk_id || ""}</code></div>` +
        `<div>${escapeHtml(c.snippet || "")}</div>`;
      root.appendChild(div);
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderResponse(resp, traceId) {
    state.response = resp;
    state.traceId = traceId || (resp && resp._demo_trace_id) || null;
    $("answer").textContent = (resp && resp.answer) || "(empty answer)";
    renderCitations(resp && resp.citations);
    const rt = (resp && resp.retrieval_trace) || {};
    renderRouteList($("trace-bm25"), rt.bm25_top5);
    renderRouteList($("trace-dense"), rt.dense_top5);
    renderRouteList($("trace-visual"), rt.visual_top5);

    const sr = (resp && resp.self_rag) || { enabled: false };
    const cr = (resp && resp.crag) || { enabled: false };
    $("eng-meta").innerHTML =
      `<div>Trace-Id: <code>${escapeHtml(state.traceId || "—")}</code>` +
      (state.traceId && state.mode === "live"
        ? ` · <a href="${apiUrl("/trace/" + state.traceId)}" target="_blank" rel="noopener">open /trace</a>`
        : "") +
      `</div>` +
      `<div>self_rag.enabled=${sr.enabled}` +
      (sr.passed != null ? ` passed=${sr.passed}` : "") +
      (sr.score != null ? ` score=${sr.score}` : "") +
      (sr.final_action ? ` action=${escapeHtml(String(sr.final_action))}` : "") +
      `</div>` +
      `<div>crag.enabled=${cr.enabled}` +
      (cr.applied != null ? ` applied=${cr.applied}` : "") +
      (cr.final_action ? ` action=${escapeHtml(String(cr.final_action))}` : "") +
      `</div>`;
    const dump = Object.assign({}, resp || {});
    delete dump._demo_trace_id;
    $("eng-json").textContent = JSON.stringify(dump, null, 2);
  }

  function setMode(mode) {
    state.mode = mode;
    $("mode-demo").classList.toggle("active", mode === "demo");
    $("mode-live").classList.toggle("active", mode === "live");
    const live = mode === "live";
    $("pdf-file").disabled = !live;
    $("btn-upload").disabled = !live;
    $("filter-doc").disabled = !live || !state.lastDocId;
    if (live) {
      checkHealth();
    } else {
      $("health").textContent = "health: (demo)";
      $("health").className = "health";
    }
  }

  async function loadData() {
    const [fx, mx] = await Promise.all([
      fetch("fixtures.json").then((r) => {
        if (!r.ok) throw new Error("fixtures.json " + r.status);
        return r.json();
      }),
      fetch("metrics.json").then((r) => {
        if (!r.ok) throw new Error("metrics.json " + r.status);
        return r.json();
      }),
    ]);
    state.fixtures = fx;
    state.metrics = mx;
    renderMetrics();
    renderPresets();
  }

  async function askDemo(query) {
    const resp = state.fixtures.responses[query];
    if (!resp) {
      throw new Error("Demo 模式仅支持预设问句（请点上方 chips，或切换 Live）");
    }
    // 深拷贝避免后续污染
    return JSON.parse(JSON.stringify(resp));
  }

  async function askLive(query) {
    const body = {
      query,
      k: 5,
      use_rerank: true,
    };
    if (state.filterByDoc && state.lastDocId) {
      body.doc_id = state.lastDocId;
    }
    const r = await fetch(apiUrl("/ask"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const traceId = r.headers.get("X-Trace-Id") || r.headers.get("x-trace-id");
    let data;
    try {
      data = await r.json();
    } catch (_) {
      throw new Error("Live /ask 返回非 JSON，HTTP " + r.status);
    }
    if (!r.ok) {
      const detail = data.detail || JSON.stringify(data);
      throw new Error("Live /ask " + r.status + ": " + detail);
    }
    return { data, traceId };
  }

  async function ask() {
    const query = ($("query").value || "").trim();
    if (!query) {
      showError("请输入问题");
      return;
    }
    showError(null);
    setLoading(true);
    try {
      if (state.mode === "demo") {
        const data = await askDemo(query);
        renderResponse(data, data._demo_trace_id || null);
      } else {
        const { data, traceId } = await askLive(query);
        renderResponse(data, traceId);
      }
    } catch (e) {
      showError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  async function checkHealth() {
    const el = $("health");
    el.textContent = "health: …";
    el.className = "health";
    try {
      const r = await fetch(apiUrl("/health"));
      if (!r.ok) throw new Error("HTTP " + r.status);
      const j = await r.json();
      state.health = j;
      el.textContent = `health: ok · pages=${j.index_pages ?? "—"}`;
      el.classList.add("ok");
    } catch (e) {
      state.health = null;
      el.textContent = "health: fail (" + (e.message || e) + ")";
      el.classList.add("bad");
    }
  }

  async function uploadPdf() {
    if (state.mode !== "live") return;
    const fileInput = $("pdf-file");
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      showError("请选择 PDF 文件");
      return;
    }
    if (!/\.pdf$/i.test(file.name)) {
      showError("仅支持 PDF");
      return;
    }
    showError(null);
    $("btn-upload").disabled = true;
    $("upload-status").textContent = "uploading…";
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      const r = await fetch(apiUrl("/ingest"), { method: "POST", body: fd });
      let data;
      try {
        data = await r.json();
      } catch (_) {
        throw new Error("ingest 非 JSON HTTP " + r.status);
      }
      if (!r.ok) {
        throw new Error("ingest " + r.status + ": " + (data.detail || JSON.stringify(data)));
      }
      state.lastDocId = data.doc_id;
      $("filter-doc").disabled = false;
      $("upload-status").textContent =
        `doc_id=${data.doc_id} · pages=${data.num_pages} · chunks=${data.num_chunks}`;
    } catch (e) {
      showError(e.message || String(e));
      $("upload-status").textContent = "upload failed";
    } finally {
      $("btn-upload").disabled = false;
    }
  }

  function wire() {
    $("mode-demo").addEventListener("click", () => setMode("demo"));
    $("mode-live").addEventListener("click", () => setMode("live"));
    $("btn-ask").addEventListener("click", ask);
    $("query").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") ask();
    });
    $("btn-upload").addEventListener("click", uploadPdf);
    $("filter-doc").addEventListener("change", (ev) => {
      state.filterByDoc = !!ev.target.checked;
    });
    $("api-base").addEventListener("change", () => {
      if (state.mode === "live") checkHealth();
    });
    document.addEventListener("click", (ev) => {
      if (!ev.target.classList.contains("chip")) {
        /* keep detail until chip click toggles */
      }
    });
  }

  async function main() {
    wire();
    setMode("demo");
    try {
      await loadData();
    } catch (e) {
      showError("加载 fixtures/metrics 失败: " + (e.message || e));
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
