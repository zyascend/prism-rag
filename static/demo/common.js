// static/demo/common.js — shared nav, API helpers
(function (global) {
  "use strict";

  function apiBase() {
    const el = document.getElementById("api-base");
    const raw = el ? (el.value || "").trim().replace(/\/$/, "") : "";
    return raw;
  }

  function apiUrl(path) {
    const b = apiBase();
    if (!b) return path.startsWith("/") ? path : `/${path}`;
    return `${b}${path.startsWith("/") ? path : `/${path}`}`;
  }

  function pageName() {
    const p = (location.pathname || "").split("/").pop() || "index.html";
    if (!p || p === "demo" || p === "") return "index.html";
    return p;
  }

  function renderNav(active) {
    const nav = document.getElementById("main-nav");
    if (!nav) return;
    const items = [
      { href: "index.html", id: "ask", label: "问答" },
      { href: "documents.html", id: "docs", label: "文档库" },
      { href: "embed.html", id: "embed", label: "嵌入" },
    ];
    nav.innerHTML = items
      .map((it) => {
        const cls = it.id === active ? "nav-link active" : "nav-link";
        return `<a class="${cls}" href="${it.href}">${it.label}</a>`;
      })
      .join("");
  }

  async function checkHealth(el) {
    if (!el) el = document.getElementById("health");
    if (!el) return false;
    el.textContent = "health: …";
    el.className = "health";
    try {
      const r = await fetch(apiUrl("/health"));
      if (!r.ok) throw new Error("HTTP " + r.status);
      const j = await r.json();
      el.textContent = `health: ok · pages=${j.index_pages ?? "—"}`;
      el.classList.add("ok");
      return true;
    } catch (e) {
      el.textContent = "health: fail (" + (e.message || e) + ")";
      el.classList.add("bad");
      return false;
    }
  }

  function wireApiBase() {
    const input = document.getElementById("api-base");
    if (!input) return;
    input.addEventListener("change", () => checkHealth());
  }

  function bootNav(active) {
    renderNav(active);
    wireApiBase();
    checkHealth();
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  global.PrismDemo = {
    apiUrl,
    apiBase,
    checkHealth,
    bootNav,
    escapeHtml,
    pageName,
  };
})(window);
