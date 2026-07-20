/* Shared-state client for the OCT–T1 inline figures.
   Each figure calls OpticNerve.mount({figId, divId}); all figures + the
   iframed dashboard stay in sync via the URL + the "opticnerve:state" bus. */
(function () {
  "use strict";
  const ORDER = ["exclude", "stat", "band", "mac", "disc", "mode"];
  const DEF = {exclude: "", stat: "R2m", band: "T1_mean_015",
               mac: "All_1_3_gcc", disc: "All_um_", mode: "avg"};

  function apiBase() {
    const m = document.querySelector('meta[name="opticnerve-api"]');
    return (window.OPTICNERVE_API_BASE || (m && m.content) || "http://localhost:3000")
      .replace(/\/$/, "");
  }
  function readParams() {
    const q = new URLSearchParams(location.search), p = {};
    ORDER.forEach(k => { p[k] = q.has(k) ? q.get(k) : DEF[k]; });
    return p;
  }
  function serialize(p) { return ORDER.map(k => `${k}=${encodeURIComponent(p[k])}`).join("&"); }
  function equal(a, b) { return serialize(a) === serialize(b); }
  function writeURL(p) {
    history.replaceState(null, "", location.pathname + "?" + serialize(p) + location.hash);
  }

  const registry = [];          // {figId, divId, apply, last}
  let applying = false;         // loop guard

  function fetchAndRender(entry, p) {
    return fetch(`${apiBase()}/opticnerve/${entry.figId}?${serialize(p)}`)
      .then(r => r.json())
      .then(j => {
        Plotly.react(entry.divId, j.figure.data, j.figure.layout);
        entry.last = p;
        wireClicks(entry);
      });
  }

  // Fig 3 wedge clicks set mac/disc (toggle back to default on re-click).
  function wireClicks(entry) {
    if (entry.figId !== "fig3") return;
    const gd = document.getElementById(entry.divId);
    if (gd._onWired) return;
    gd._onWired = true;
    gd.on("plotly_click", ev => {
      if (applying) return;            // ignore clicks while a broadcast is in flight
      const pt = ev.points && ev.points[0];
      const m = pt && (pt.data.meta || (pt.customdata && pt.customdata[0]));
      if (!m) return;
      const p = readParams();
      if (m.endsWith("_gcc")) p.mac = (p.mac === m) ? DEF.mac : m;
      else if (m.endsWith("_um_")) p.disc = (p.disc === m) ? DEF.disc : m;
      broadcast(p);
    });
  }

  // Originate a state change: write URL, re-render everyone, dispatch once.
  function broadcast(p) {
    writeURL(p);
    applying = true;
    Promise.all(registry.map(e => fetchAndRender(e, p))).finally(() => {
      applying = false;
      window.dispatchEvent(new CustomEvent("opticnerve:state", {detail: p}));
    });
  }

  // Receive external state (another figure, the dashboard bridge, or popstate).
  function receive(p) {
    if (applying) return;
    registry.forEach(e => { if (!e.last || !equal(e.last, p)) fetchAndRender(e, p); });
  }
  window.addEventListener("opticnerve:state", e => receive(e.detail));
  window.addEventListener("popstate", () => receive(readParams()));

  window.OpticNerve = {
    mount: function (opts) {
      const entry = {figId: opts.figId, divId: opts.divId, last: null};
      registry.push(entry);
      fetchAndRender(entry, readParams());
    },
    _readParams: readParams, _broadcast: broadcast   // exposed for the bridge/tests
  };

  // --- iframe bridge: relay between the same-page bus and the Dash iframe ---
  const ALLOW = (window.OPTICNERVE_DASH_ORIGIN || "").replace(/\/$/, "");
  function dashFrame() { return document.querySelector('iframe[data-opticnerve-dash]'); }

  // bus change -> tell the iframe
  window.addEventListener("opticnerve:state", e => {
    const f = dashFrame();
    if (f && f.contentWindow) {
      f.contentWindow.postMessage(
        {type: "opticnerve:state", search: "?" + serialize(e.detail)}, ALLOW || "*");
    }
  });
  // iframe change -> drive the bus
  window.addEventListener("message", e => {
    if (ALLOW && e.origin !== ALLOW) return;         // origin allow-check
    const d = e.data || {};
    if (d.type !== "opticnerve:state" || typeof d.search !== "string") return;
    const q = new URLSearchParams(d.search.replace(/^\?/, "")), p = {};
    ORDER.forEach(k => { p[k] = q.has(k) ? q.get(k) : DEF[k]; });
    const cur = readParams();
    if (!equal(cur, p)) broadcast(p);
  });
})();
