// static/demo/app.js
// Hybrid: Demo fixtures OR Live API. Upload always available → forces Live + doc-scoped ask.
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
      btn.setAttribute("aria-expanded", "false");
      btn.addEventListener("click", () => {
        const box = $("chip-detail");
        const open = box.classList.contains("open") && box.dataset.label === c.label;
        if (open) {
          box.classList.remove("open");
          btn.setAttribute("aria-expanded", "false");
          return;
        }
        root.querySelectorAll(".chip").forEach((b) => b.setAttribute("aria-expanded", "false"));
        box.dataset.label = c.label;
        box.textContent = c.detail;
        box.classList.add("open");
        btn.setAttribute("aria-expanded", "true");
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
      btn.title = state.mode === "live"
        ? "Live 模式：会向 API 发真实问题（非 fixture）"
        : "Demo 模式：回放预录答案";
      btn.addEventListener("click", () => {
        $("query").value = p.query;
        ask();
      });
      root.appendChild(btn);
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function scoreNorm(items) {
    if (!items || !items.length) return [];
    const scores = items.map((it) => Number(it.score) || 0);
    const max = Math.max.apply(null, scores.concat([0.0001]));
    const min = Math.min.apply(null, scores);
    const span = max - min || 1;
    return items.map((it) => {
      const s = Number(it.score) || 0;
      // relative bar within this route (not cross-route comparable)
      const pct = Math.round(((s - min) / span) * 100);
      return { it, pct: Math.max(8, pct), score: s };
    });
  }

  function showHitDetail(routeName, it, rank) {
    const title = $("hit-detail-title");
    const meta = $("hit-detail-meta");
    const body = $("hit-detail-body");
    if (!title || !meta || !body) return;
    const page = it.page_id != null ? it.page_id : "?";
    const score =
      typeof it.score === "number"
        ? Number.isInteger(it.score)
          ? String(it.score)
          : it.score.toFixed(4)
        : it.score;
    title.textContent = `${routeName} · #${rank} · p.${page}`;
    meta.textContent =
      `${it.chunk_type || "chunk"} · score=${score}` +
      (it.doc_id ? ` · doc=${it.doc_id}` : "") +
      (it.chunk_id ? ` · ${it.chunk_id}` : "");
    const text = (it.text || "").trim();
    body.textContent = text
      ? text
      : "（无文本预览：可能是 visual 仅页 id、旧缓存、或 chunk 无 text 字段）";
  }

  function clearHitActive() {
    document.querySelectorAll(".route-col li.hit-item.active").forEach((el) => {
      el.classList.remove("active");
    });
  }

  function renderRouteList(el, items, emptyHint, routeName) {
    el.innerHTML = "";
    if (!items || !items.length) {
      el.innerHTML = `<li class="empty-route">${escapeHtml(emptyHint || "本路无命中")}</li>`;
      return;
    }
    scoreNorm(items).forEach((row, idx) => {
      const it = row.it;
      const li = document.createElement("li");
      li.className = "hit-item";
      const page = it.page_id != null ? it.page_id : "?";
      const scoreStr =
        typeof row.score === "number" && !Number.isInteger(row.score)
          ? row.score.toFixed(3)
          : String(row.score);
      li.innerHTML =
        `<span class="hit-rank">${idx + 1}</span>` +
        `<span class="hit-page">p.${escapeHtml(String(page))}</span>` +
        `<span class="score-num">${escapeHtml(scoreStr)}</span>` +
        `<span class="hit-id" title="${escapeHtml(it.chunk_id || "")}">${escapeHtml(it.chunk_id || "—")}</span>` +
        `<div class="score-bar"><i style="width:${row.pct}%"></i></div>`;
      const show = () => {
        clearHitActive();
        li.classList.add("active");
        showHitDetail(routeName, it, idx + 1);
      };
      li.addEventListener("mouseenter", show);
      li.addEventListener("focus", show);
      li.addEventListener("click", show);
      li.tabIndex = 0;
      el.appendChild(li);
    });
  }

  function pageSet(items) {
    const s = new Set();
    (items || []).forEach((it) => {
      if (it && it.page_id != null) s.add(String(it.page_id));
    });
    return s;
  }

  function renderFuse(rt, citations) {
    const box = $("fuse-box");
    const bm = pageSet(rt.bm25_top5);
    const de = pageSet(rt.dense_top5);
    const vi = pageSet(rt.visual_top5);
    const all = new Set([].concat([...bm], [...de], [...vi]));
    const multi = [];
    all.forEach((p) => {
      const routes = [];
      if (bm.has(p)) routes.push("BM25");
      if (de.has(p)) routes.push("Dense");
      if (vi.has(p)) routes.push("Visual");
      if (routes.length >= 2) multi.push({ p, routes });
    });
    multi.sort((a, b) => b.routes.length - a.routes.length || Number(a.p) - Number(b.p));

    const citePages = new Set(
      (citations || []).map((c) => String(c.page_number != null ? c.page_number : c.page_id))
    );

    let html = "";
    if (!all.size) {
      html = `<p class="muted">尚无召回页。Ask 后会对照三路 top 的 page 交叉。</p>`;
    } else if (!multi.length) {
      html =
        `<p>三路 top 的 <b>page 无重叠</b>——融合更依赖 RRF 位次，精排压力更大。` +
        ` 各路独有页：BM25 ${bm.size} · Dense ${de.size} · Visual ${vi.size}。</p>`;
    } else {
      html =
        `<p><b>${multi.length}</b> 个 page 被 ≥2 路同时召回（交叉越强，通常越稳）：</p>` +
        `<div class="fuse-tags">` +
        multi
          .map((m) => {
            const used = citePages.has(m.p) ? " · 入引用" : "";
            const cls = m.routes.length >= 3 ? "fuse-tag" : "fuse-tag weak";
            return `<span class="${cls}"><span class="dot"></span>p.${escapeHtml(m.p)} · ${m.routes.join("+")}${used}</span>`;
          })
          .join("") +
        `</div>`;
    }
    if (citePages.size) {
      html +=
        `<p style="margin:10px 0 0" class="muted">最终 citations 覆盖页：` +
        [...citePages].map((p) => `p.${escapeHtml(p)}`).join(", ") +
        `（精排/生成后落地，不必等于各路 top1）</p>`;
    }
    html +=
      `<p style="margin:8px 0 0" class="muted">说明：API 只回各路 top5 + 最终答案；RRF/Rerank 中间列表未单独暴露，用「交叉页 + 引用页」反推融合结果。</p>`;
    box.innerHTML = html;
  }

  function setPipeStep(name, cls, subText) {
    const li = document.querySelector(`.pipe-step[data-step="${name}"]`);
    if (!li) return;
    li.classList.remove("idle", "done", "active", "warn", "skip");
    li.classList.add(cls || "idle");
    const map = {
      query: "pipe-query-sub",
      retrieve: "pipe-retrieve-sub",
      fuse: "pipe-fuse-sub",
      gates: "pipe-gates-sub",
      answer: "pipe-answer-sub",
    };
    const el = document.getElementById(map[name]);
    if (el && subText != null) el.textContent = subText;
  }

  function renderPipeline(resp) {
    if (!resp) {
      ["query", "retrieve", "fuse", "gates", "answer"].forEach((n) => setPipeStep(n, "idle"));
      return;
    }
    const q = resp.query || "";
    setPipeStep("query", "done", q.length > 42 ? q.slice(0, 40) + "…" : q || "ok");

    const rt = resp.retrieval_trace || {};
    const nB = (rt.bm25_top5 || []).length;
    const nD = (rt.dense_top5 || []).length;
    const nV = (rt.visual_top5 || []).length;
    setPipeStep(
      "retrieve",
      nB + nD + nV > 0 ? "done" : "warn",
      `BM25 ${nB} · Dense ${nD} · Visual ${nV}`
    );

    const cites = resp.citations || [];
    setPipeStep(
      "fuse",
      "done",
      cites.length ? `→ ${cites.length} citations` : "融合完成（无引用）"
    );

    const cr = resp.crag || { enabled: false };
    const sr = resp.self_rag || { enabled: false };
    let gateCls = "skip";
    let gateSub = "均关闭（默认）";
    if (cr.enabled || sr.enabled) {
      gateCls = "done";
      const bits = [];
      if (cr.enabled) bits.push(cr.applied ? "CRAG applied" : "CRAG on");
      if (sr.enabled) bits.push(sr.passed === false ? "Gate2 fail" : "Gate2 on");
      gateSub = bits.join(" · ");
      if (sr.enabled && sr.passed === false) gateCls = "warn";
    }
    setPipeStep("gates", gateCls, gateSub);

    const ans = (resp.answer || "").trim();
    const rejected = /enough information|cannot answer|not enough/i.test(ans);
    const ctxLen = ((resp && resp.context) || "").length;
    setPipeStep(
      "answer",
      rejected ? "warn" : ans ? "done" : "warn",
      rejected
        ? "拒答/信息不足"
        : ans
          ? `答案 ${ans.length}c · ctx ${ctxLen}c · ${cites.length} cite`
          : "空答案"
    );
  }

  function renderGates(resp) {
    const cr = (resp && resp.crag) || { enabled: false };
    const sr = (resp && resp.self_rag) || { enabled: false };
    const crCard = $("gate-crag");
    const srCard = $("gate-selfrag");
    const crStatus = crCard.querySelector(".gate-status");
    const crDetail = crCard.querySelector(".gate-detail");
    const srStatus = srCard.querySelector(".gate-status");
    const srDetail = srCard.querySelector(".gate-detail");

    crCard.className = "gate-card " + (cr.enabled ? (cr.applied ? "on" : "warn") : "off");
    if (!cr.enabled) {
      crStatus.className = "gate-status off";
      crStatus.textContent = "OFF · 默认关闭";
      crDetail.textContent = "生成前证据纠错未启用（云上实验阴性后保持关）";
    } else {
      crStatus.className = "gate-status " + (cr.applied ? "ok" : "off");
      crStatus.textContent = cr.applied
        ? `ON · applied · ${cr.final_action || "—"}`
        : `ON · not applied · ${cr.skip_reason || cr.final_action || "—"}`;
      crDetail.textContent =
        `q: ${cr.query_original || "—"} → ${cr.query_used || "—"}` +
        (cr.num_relevant != null ? ` · relevant=${cr.num_relevant}` : "") +
        (cr.sufficient != null ? ` · sufficient=${cr.sufficient}` : "");
    }

    srCard.className = "gate-card " + (sr.enabled ? (sr.passed === false ? "warn" : "on") : "off");
    if (!sr.enabled) {
      srStatus.className = "gate-status off";
      srStatus.textContent = "OFF · 默认关闭";
      srDetail.textContent = "生成后忠实性门未启用；可配置 trigger=low_rerank";
    } else {
      const bad = sr.passed === false;
      srStatus.className = "gate-status " + (bad ? "bad" : "ok");
      srStatus.textContent =
        `ON · passed=${sr.passed}` +
        (sr.score != null ? ` · score=${sr.score}` : "") +
        (sr.final_action ? ` · ${sr.final_action}` : "") +
        (sr.attempts != null ? ` · attempts=${sr.attempts}` : "");
      const detail = (sr.attempts_detail && sr.attempts_detail[0]) || {};
      const uns = detail.unsupported || [];
      srDetail.textContent = uns.length
        ? `unsupported: ${uns.slice(0, 3).join("; ")}`
        : detail.action
          ? `last action=${detail.action}`
          : "无 unsupported 明细";
    }
  }

  function renderCitations(citations) {
    const root = $("citations");
    root.innerHTML = "";
    if (!citations || !citations.length) {
      root.innerHTML = "<p class='muted'>No citations — 拒答或证据未回链</p>";
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

  function renderContext(resp) {
    // 完整入模 context 仍放左侧折叠；页级详情主交互在右侧三路 hover
    const body = $("context-body");
    const summary = $("context-summary");
    if (!body || !summary) return;
    const ctx = (resp && resp.context) || "";
    if (!ctx) {
      body.textContent = "（空 context：拒答 / 无检索结果 / 或旧缓存未存 context）";
      summary.textContent = "完整入模 Context · 空 · 页级请 hover 右侧三路";
      return;
    }
    body.textContent = ctx;
    const nChars = ctx.length;
    const blocks = ctx.split(/\n\s*\n/).filter((s) => s.trim()).length;
    summary.textContent =
      `完整入模 Context · ${nChars} chars · ~${blocks} blocks（折叠；页级预览 hover 右侧）`;
  }

  function renderResponse(resp, traceId) {
    state.response = resp;
    state.traceId = traceId || (resp && resp._demo_trace_id) || null;
    $("answer").textContent = (resp && resp.answer) || "(empty answer)";
    renderContext(resp);
    renderCitations(resp && resp.citations);
    const rt = (resp && resp.retrieval_trace) || {};
    renderRouteList(
      $("trace-bm25"),
      rt.bm25_top5,
      "无 BM25 命中（库空或 query 无词面重叠）",
      "BM25"
    );
    renderRouteList(
      $("trace-dense"),
      rt.dense_top5,
      "无 Dense 命中（BGE/pgvector 未返回）",
      "Dense"
    );
    renderRouteList(
      $("trace-visual"),
      rt.visual_top5,
      "Visual 空：未开 visual / FAISS 无页 / 本 query 无视觉命中",
      "Visual"
    );
    // 默认展示第一路第一条，方便不用 hover 也能看到文本
    const first =
      (rt.bm25_top5 && rt.bm25_top5[0]) ||
      (rt.dense_top5 && rt.dense_top5[0]) ||
      (rt.visual_top5 && rt.visual_top5[0]);
    if (first) {
      const routeName = rt.bm25_top5 && rt.bm25_top5[0] ? "BM25" : rt.dense_top5 && rt.dense_top5[0] ? "Dense" : "Visual";
      showHitDetail(routeName, first, 1);
      const listId =
        routeName === "BM25" ? "trace-bm25" : routeName === "Dense" ? "trace-dense" : "trace-visual";
      const firstLi = $(listId) && $(listId).querySelector("li.hit-item");
      if (firstLi) firstLi.classList.add("active");
    } else if ($("hit-detail-body")) {
      $("hit-detail-title").textContent = "Page / chunk 详情";
      $("hit-detail-meta").textContent = "无召回命中";
      $("hit-detail-body").textContent = "本次三路均为空。";
    }
    renderFuse(rt, (resp && resp.citations) || []);
    renderPipeline(resp);
    renderGates(resp);

    const modeLabel = state.mode === "live" ? "Live API" : "Demo fixture";
    $("eng-meta").innerHTML =
      `<span><b>模式</b> ${modeLabel}</span>` +
      `<span><b>Trace</b> <code>${escapeHtml(state.traceId || "—")}</code>` +
      (state.traceId && state.mode === "live"
        ? ` <a href="${apiUrl("/trace/" + state.traceId)}" target="_blank" rel="noopener">打开</a>`
        : "") +
      `</span>` +
      `<span><b>doc 过滤</b> ${
        state.filterByDoc && state.lastDocId
          ? `<code>${escapeHtml(state.lastDocId)}</code>`
          : "off"
      }</span>`;
    const dump = Object.assign({}, resp || {});
    delete dump._demo_trace_id;
    $("eng-json").textContent = JSON.stringify(dump, null, 2);
  }

  function setMode(mode, opts) {
    opts = opts || {};
    state.mode = mode;
    $("mode-demo").classList.toggle("active", mode === "demo");
    $("mode-live").classList.toggle("active", mode === "live");
    // 上传始终可点；仅 Demo 时提示会自动切 Live
    $("pdf-file").disabled = false;
    $("btn-upload").disabled = false;
    $("filter-doc").disabled = !state.lastDocId;
    if (mode === "live") {
      if (!opts.skipHealth) checkHealth();
    } else {
      $("health").textContent = "health: (demo fixtures)";
      $("health").className = "health";
    }
    renderPresets();
  }

  function bindDocFilter(docId, enable) {
    state.lastDocId = docId || null;
    state.filterByDoc = !!enable && !!docId;
    const cb = $("filter-doc");
    cb.disabled = !docId;
    cb.checked = state.filterByDoc;
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
      throw new Error("Demo 模式仅支持预设 chips。上传 PDF 或切换 Live 可真实检索。");
    }
    return JSON.parse(JSON.stringify(resp));
  }

  async function askLive(query) {
    const body = {
      query,
      k: 5,
      use_rerank: true,
    };
    // 有上传文档时默认带 doc_id（filter 勾选）—— 上传后立刻问本篇
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
      return true;
    } catch (e) {
      state.health = null;
      el.textContent = "health: fail (" + (e.message || e) + ")";
      el.classList.add("bad");
      return false;
    }
  }

  async function ensureLiveForUpload() {
    if (state.mode !== "live") {
      setMode("live", { skipHealth: true });
    }
    const ok = await checkHealth();
    if (!ok) {
      throw new Error(
        "API 不可达：请在仓库根启动 " +
          "`CONFIG_PROFILE=local-dev python scripts/run_api.py`，" +
          "并确保 `make db`（pgvector）已起。当前页面需同源 /demo 或填 API base。"
      );
    }
  }

  async function uploadPdf() {
    const fileInput = $("pdf-file");
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      showError("请先选择 PDF 文件");
      return;
    }
    if (!/\.pdf$/i.test(file.name)) {
      showError("仅支持 PDF");
      return;
    }
    showError(null);
    $("btn-upload").disabled = true;
    $("upload-status").textContent = "上传并编码中…（本地 BGE / 解析，请稍候）";
    try {
      await ensureLiveForUpload();
      const fd = new FormData();
      // 字段名必须是 file，与 FastAPI UploadFile 参数一致
      fd.append("file", file, file.name);
      const r = await fetch(apiUrl("/ingest"), { method: "POST", body: fd });
      let data;
      try {
        data = await r.json();
      } catch (_) {
        throw new Error("ingest 非 JSON HTTP " + r.status);
      }
      if (!r.ok) {
        const detail = data.detail || JSON.stringify(data);
        throw new Error("ingest " + r.status + ": " + detail);
      }
      // 上传成功 → 立刻绑定本篇，勾选过滤，可直接提问
      bindDocFilter(data.doc_id, true);
      const pages = data.num_pages;
      const chunks = data.num_chunks;
      const note =
        pages === 0 && chunks === 0
          ? "（内容哈希命中，已有索引，可直接问）"
          : "";
      $("upload-status").textContent =
        `✓ 已入库 doc_id=${data.doc_id} · pages=${pages} · chunks=${chunks} ${note} → 直接提问即可`;
      $("query").placeholder = "文档已就绪，输入问题后点 Ask（已限当前 doc）";
      $("query").focus();
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
    // 选文件后若仍在 Demo，轻提示
    $("pdf-file").addEventListener("change", () => {
      const f = $("pdf-file").files && $("pdf-file").files[0];
      if (f) {
        $("upload-status").textContent = `已选 ${f.name} — 点 Upload 入库后即可检索`;
      }
    });
    $("filter-doc").addEventListener("change", (ev) => {
      state.filterByDoc = !!ev.target.checked;
      if (state.filterByDoc && !state.lastDocId) {
        showError("尚无上传文档，无法按 doc 过滤");
        ev.target.checked = false;
        state.filterByDoc = false;
      }
    });
    $("api-base").addEventListener("change", () => {
      if (state.mode === "live") checkHealth();
    });
  }

  function applyQueryParams() {
    // 从文档库/嵌入页跳转：?doc_id=xxx 自动 Live + 过滤
    try {
      const params = new URLSearchParams(location.search || "");
      const docId = params.get("doc_id");
      if (docId) {
        bindDocFilter(docId, true);
        $("upload-status").textContent =
          `已锁定 doc_id=${docId}（来自链接）· 提问将只检索该文档`;
      }
    } catch (_) { /* ignore */ }
  }

  async function main() {
    wire();
    if (window.PrismDemo && window.PrismDemo.bootNav) {
      window.PrismDemo.bootNav("ask");
    }
    // 上传控件始终可用（不再 disabled）
    $("pdf-file").disabled = false;
    $("btn-upload").disabled = false;
    try {
      await loadData();
    } catch (e) {
      showError("加载 fixtures/metrics 失败: " + (e.message || e));
    }
    applyQueryParams();
    // 优先 Live：本机 API 起来就能上传；失败再回退 Demo fixtures
    setMode("live", { skipHealth: true });
    const ok = await checkHealth();
    if (!ok) {
      setMode("demo");
      if (!$("upload-status").textContent.includes("doc_id=")) {
        $("upload-status").textContent =
          "API 未就绪：chips 用 Demo；或到「嵌入」页等 API 启动后再入库";
      }
    } else if (!$("upload-status").textContent.includes("doc_id=")) {
      $("upload-status").textContent =
        "Live 就绪：也可在「嵌入」页看编码进度 · 或本页快速 Upload";
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
