// static/demo/embed.js — async ingest + poll
(function () {
  "use strict";
  const { apiUrl, bootNav, escapeHtml, checkHealth } = window.PrismDemo;

  let pollTimer = null;
  let currentJobId = null;

  const PHASE_ORDER = [
    "queued",
    "hash",
    "schema",
    "start",
    "parse",
    "chunk",
    "embed_text",
    "write_pg",
    "embed_visual",
    "bm25",
    "faiss",
    "index",
    "done",
    "error",
  ];

  // map internal phases → track chips
  const PHASE_CHIP = {
    queued: "queued",
    hash: "parse",
    schema: "parse",
    start: "queued",
    parse: "parse",
    chunk: "chunk",
    embed_text: "embed_text",
    write_pg: "write_pg",
    embed_visual: "embed_visual",
    bm25: "bm25",
    faiss: "bm25",
    index: "bm25",
    done: "done",
    error: "done",
  };

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

  function setProgress(pct, phase, message) {
    const p = Math.max(0, Math.min(100, Number(pct) || 0));
    document.getElementById("progress-fill").style.width = p + "%";
    document.getElementById("progress-pct").textContent = Math.round(p) + "%";
    document.getElementById("progress-phase").textContent = phase || "—";
    document.getElementById("progress-msg").textContent = message || "";
    highlightPhases(phase, statusFromPhase(phase, p));
  }

  function statusFromPhase(phase, pct) {
    if (phase === "error") return "error";
    if (phase === "done" || pct >= 100) return "done";
    return "running";
  }

  function highlightPhases(phase, status) {
    const chipKey = PHASE_CHIP[phase] || phase;
    const order = ["queued", "parse", "chunk", "embed_text", "write_pg", "embed_visual", "bm25", "done"];
    const idx = order.indexOf(chipKey);
    document.querySelectorAll(".phase-track .phase").forEach((el) => {
      el.classList.remove("active", "done", "error");
      const key = el.getAttribute("data-phase");
      const i = order.indexOf(key);
      if (status === "error" && key === "done") {
        el.classList.add("error");
        return;
      }
      if (idx < 0) return;
      if (i < idx) el.classList.add("done");
      else if (i === idx) el.classList.add(status === "error" ? "error" : "active");
      if (status === "done" && key === "done") {
        el.classList.remove("active");
        el.classList.add("done");
      }
    });
  }

  function renderEvents(events) {
    const ol = document.getElementById("event-log");
    if (!events || !events.length) {
      ol.innerHTML = `<li class="muted">尚无事件</li>`;
      return;
    }
    const slice = events.slice(-40);
    ol.innerHTML = slice
      .map((ev) => {
        const t = ev.t ? new Date(ev.t * 1000).toLocaleTimeString() : "";
        return (
          `<li>` +
          `<span class="ev-time">${escapeHtml(t)}</span> ` +
          `<code>${escapeHtml(ev.phase || "")}</code> ` +
          `<span class="ev-pct">${Math.round(ev.pct || 0)}%</span> ` +
          `<span>${escapeHtml(ev.message || "")}</span>` +
          `</li>`
        );
      })
      .join("");
    ol.scrollTop = ol.scrollHeight;
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function pollOnce(jobId) {
    const r = await fetch(apiUrl("/ingest/jobs/" + encodeURIComponent(jobId)));
    let data;
    try {
      data = await r.json();
    } catch (_) {
      throw new Error("job 状态非 JSON · " + r.status);
    }
    if (!r.ok) throw new Error(data.detail || "HTTP " + r.status);

    setProgress(data.pct, data.phase, data.message);
    renderEvents(data.events);
    document.getElementById("job-meta").innerHTML =
      `<span><b>job</b> <code>${escapeHtml(data.job_id)}</code></span>` +
      `<span><b>doc_id</b> <code>${escapeHtml(data.doc_id)}</code></span>` +
      `<span><b>file</b> ${escapeHtml(data.filename || "")}</span>` +
      `<span><b>status</b> ${escapeHtml(data.status)}</span>`;

    if (data.status === "done") {
      stopPoll();
      document.getElementById("btn-start").disabled = false;
      const res = data.result || {};
      document.getElementById("done-actions").hidden = false;
      document.getElementById("done-summary").textContent =
        `✓ 完成 · pages=${res.num_pages ?? "—"} · chunks=${res.num_chunks ?? "—"}`;
      document.getElementById("link-ask").href =
        `index.html?doc_id=${encodeURIComponent(data.doc_id)}`;
      setProgress(100, "done", data.message || "完成");
    } else if (data.status === "error") {
      stopPoll();
      document.getElementById("btn-start").disabled = false;
      showError("嵌入失败: " + (data.error || data.message || "unknown"));
      setProgress(100, "error", data.error || data.message || "error");
    }
    return data;
  }

  function startPoll(jobId) {
    stopPoll();
    currentJobId = jobId;
    pollTimer = setInterval(() => {
      pollOnce(jobId).catch((e) => {
        showError(e.message || String(e));
        stopPoll();
        document.getElementById("btn-start").disabled = false;
      });
    }, 500);
    pollOnce(jobId).catch((e) => showError(e.message || String(e)));
  }

  async function startEmbed() {
    showError(null);
    document.getElementById("done-actions").hidden = true;
    const fileInput = document.getElementById("pdf-file");
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      showError("请先选择 PDF");
      return;
    }
    if (!/\.pdf$/i.test(file.name)) {
      showError("仅支持 PDF");
      return;
    }
    const ok = await checkHealth();
    if (!ok) {
      showError("API 不可达，无法嵌入。请启动 local-dev API。");
      return;
    }
    document.getElementById("btn-start").disabled = true;
    setProgress(0, "queued", "上传文件中…");
    renderEvents([]);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      const r = await fetch(apiUrl("/ingest/jobs"), { method: "POST", body: fd });
      let data;
      try {
        data = await r.json();
      } catch (_) {
        throw new Error("创建 job 失败 · HTTP " + r.status);
      }
      if (!r.ok) throw new Error(data.detail || "HTTP " + r.status);
      setProgress(1, "queued", "已排队 job=" + data.job_id);
      startPoll(data.job_id);
    } catch (e) {
      showError(e.message || String(e));
      document.getElementById("btn-start").disabled = false;
    }
  }

  document.getElementById("btn-start").addEventListener("click", startEmbed);
  document.getElementById("pdf-file").addEventListener("change", () => {
    const f = document.getElementById("pdf-file").files[0];
    document.getElementById("file-hint").textContent = f
      ? `已选 ${f.name} (${(f.size / 1024).toFixed(1)} KB)`
      : "选择 PDF 后点击开始。";
  });

  bootNav("embed");
})();
