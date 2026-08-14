"""
OCT – T1 correlation dashboard  (Plotly Dash app)

Recreates the three figures of the static HTML dashboard, plus two regression
modes selectable in the sidebar:

  * "Average + OLS"  -> average both eyes per subject, ordinary least squares
  * "LME"            -> per-eye data, linear mixed-effects model with a random
                        intercept per subject; Figure 2 plots the marginal T1
                        (observed T1 minus the subject random intercept).

Also serves, for embedding in paper.md:
  /opticnerve/<figid>   the figure as a Plotly JSON spec  (figid: fig1|fig2|fig3|all)
  /figure/<figid>       a standalone page rendering that figure

Run locally:   python app.py                (http://127.0.0.1:8050)
Deploy:        gunicorn app:server           (see render.yaml)
"""

import json
import logging
from urllib.parse import parse_qsl

import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL, no_update
from flask import request, abort, jsonify
from flask_cors import CORS
from plotly.utils import PlotlyJSONEncoder

from opticnerve_core import (
    T1_BANDS, DEF_MAC, DEF_DISC, MAC_AVG, DISC_AVG, SUBJECTS, stat_val,
    stat_lbl, resolve_view, DEFAULTS, build_fig1, build_fig2, build_fig3,
    parse_params, ALL_EYES, EYES_OF, EYE_SEP, FIG_SIZE, PARAM_ORDER,
)

# The browser needs the same state model as the Python. Injected into every
# script below rather than hand-copied, so the two can never drift.
JS_ORDER = json.dumps(list(PARAM_ORDER))
JS_DEFAULTS = json.dumps({k: (",".join(v) if isinstance(v, tuple) else v)
                          for k, v in DEFAULTS.items()})

# Browser twin of _toggle_point() below: the dashboard toggles a clicked Fig 2
# point through a Dash callback, but the standalone /figure/fig2 page has no
# Python behind it and must do the same edit to the `exclude` string itself.
# Self-contained on purpose (no browser APIs) so the tests can run it under node.
# Unlike the Python it needs no `mode`: a Fig 2 point's own token is already the
# subject in average mode and the eye in LME mode.
_TOGGLE_JS = ("""
function toggleExclude(exclude, subj, tok, ghost) {
  var SEP = __SEP__;
  var out = (exclude || "").split(",").filter(Boolean);
  if (ghost) {
    // Put it back by clearing every token that could still be keeping it out:
    // its bare subject token, plus the eye token(s) the click covers.
    var pre = subj + SEP;
    out = out.filter(function (t) {
      if (t === subj) return false;
      if (t.indexOf(pre) === 0 && (tok === subj || tok === t)) return false;
      return true;
    });
  } else if (out.indexOf(tok) === -1) {
    out.push(tok);
  }
  return out.sort().join(",");
}
""".replace("__SEP__", json.dumps(EYE_SEP)))

# Browser twin of toggle_band() below: check or uncheck one T1
# band. Order is selection order — a newly checked band goes last, so checking
# never steals the primary — and the set is never emptied. Self-contained (no
# browser APIs) so the tests can run it under node.
_BAND_JS = ("""
function toggleBand(band, name) {
  var ALL = __BANDS__;
  var DEFAULT = __DEFAULT__;
  var out = (band || "").split(",").filter(function (b) { return ALL.indexOf(b) !== -1; });
  // Junk name: keep whatever survived filtering, but never hand back nothing.
  if (ALL.indexOf(name) === -1) return (out.length ? out : [DEFAULT]).join(",");
  var i = out.indexOf(name);
  if (i === -1) out.push(name);
  else if (out.length > 1) out.splice(i, 1);
  return out.join(",");
}
""".replace("__BANDS__", json.dumps([k for k, _, _ in T1_BANDS]))
   .replace("__DEFAULT__", json.dumps(DEFAULTS["band"][0])))

T1_KEYS = [k for k, _, _ in T1_BANDS]


def toggle_band(bands, name):
    """Check or uncheck one T1 band. Twin of toggleBand() in _BAND_JS.

    Order is selection order and bands[0] is the primary, so a newly checked
    band goes LAST — checking never steals the primary. The set is never
    emptied: Figure 2 must always have something to draw."""
    out = [b for b in (bands or []) if b in T1_KEYS]
    if name not in T1_KEYS:
        return out or list(DEFAULTS["band"])
    if name not in out:
        return out + [name]
    return out if len(out) == 1 else [b for b in out if b != name]


# ============================================================================
# AVERAGES TABLE — clickable rows drive Figure 2
# ============================================================================
def build_avg_table(view):
    excluded, mode, stat = view["excluded"], view["mode"], view["stat"]

    def cell(metric, region, band):
        r = view["avg_fits"][band][region][metric]
        val = stat_val(r, stat)
        pf = view["avg_pf"][band][region][metric]
        sig = bool(r and not np.isnan(pf) and pf < 0.05)
        txt = ("—" if (val is None or np.isnan(val)) else f"{val:.2f}") + ("*" if sig else "")
        return html.Td(txt, className="aval sig" if sig else "aval")

    def rows(group_label, items, region, selected):
        out = []
        for i, (metric, name) in enumerate(items):
            tds = [cell(metric, region, band) for band, _, _ in T1_BANDS]
            region_td = ([html.Td(group_label, className="region", rowSpan=len(items))]
                         if i == 0 else [])
            out.append(html.Tr(
                region_td + [html.Td(name, className="anm")] + tds,
                id={"type": "avgrow", "metric": metric, "region": region},
                n_clicks=0,
                className="avgrow sel" if metric == selected else "avgrow"))
        return out

    bands = view["bands"]
    header = html.Thead(html.Tr(
        [html.Th("T₁ sector", className="corner", colSpan=2)]
        + [html.Th(lbl, className=("prim" if k == view["band"] else
                                   "on" if k in bands else ""))
           for k, lbl, _ in T1_BANDS]))
    body = html.Tbody(
        rows("Macula (GCC)", MAC_AVG, "mac", view["mac"])
        + rows("Optic Disc (RNFL)", DISC_AVG, "disc", view["disc"]))
    return [
        html.Div([f"OCT × T₁ band correlation · {stat_lbl(stat)} ",
                  html.Span("— click a row to use that sector in Figure 2", className="avghint")],
                 className="avgtitle"),
        html.Table([header, body], className="avgtbl"),
    ]


# ============================================================================
# DASH APP  (layout + callbacks)
# ============================================================================
app = Dash(__name__, title="OCT – T₁ FigServe",eager_loading=True)
server = app.server                      # gunicorn entry point (Render)
CORS(server, resources={r"/opticnerve/*": {"origins": "*"}})

_BUILDERS = {"fig1": build_fig1, "fig2": build_fig2, "fig3": build_fig3}

# Extra DPI multiplier for the modebar's "download as png" (600 dpi / 96 css dpi).
PNG_SCALE = 600 / 96


def _coerce_list_params(body):
    """JSON clients naturally send list-valued params (band: ["a", "b"]),
    but parse_params() expects the comma-string a query string always gives.
    Join lists back into that shape so both inputs behave the same."""
    return {k: (",".join(v) if isinstance(v, list) else v) for k, v in body.items()}


def _build_payload(figid, p, view):
    """Shared by /opticnerve and /api/generate_plots: the params+figure(s)
    JSON body, identical either way."""
    payload = {"figid": figid,
               "params": {**p, "exclude": list(p["exclude"]), "band": list(p["band"])},
               "n": view["n"]}
    if figid == "all":
        payload.update({k: _BUILDERS[k](view).to_dict() for k in _BUILDERS})
    else:
        payload["figure"] = _BUILDERS[figid](view).to_dict()
    return payload


@server.route("/opticnerve/<figid>")
def opticnerve(figid):
    if figid not in _BUILDERS and figid != "all":
        abort(404)
    p = parse_params(request.args)
    view = resolve_view(**p)
    payload = _build_payload(figid, p, view)
    return server.response_class(
        json.dumps(payload, cls=PlotlyJSONEncoder), mimetype="application/json")


@server.route("/api/generate_plots", methods=["POST"])
def api_generate_plots():
    """Stateless figure generation: full params in, figure JSON out."""
    body = request.get_json(silent=True) or {}
    figid = body.get("figid", "fig1")
    if figid not in _BUILDERS and figid != "all":
        abort(404)
    try:
        p = parse_params(_coerce_list_params(body))
        view = resolve_view(**p)
        payload = _build_payload(figid, p, view)
    except Exception:
        logging.exception("error in api_generate_plots")
        return jsonify({"error": "internal error generating plots"}), 500
    return server.response_class(
        json.dumps(payload, cls=PlotlyJSONEncoder), mimetype="application/json")


# ---------------------------------------------------------------------------
# Standalone live figure pages, for embedding in the article as <iframe>s.
# Each page is driven by its own URL params plus postMessage from whoever embeds
# it, and is otherwise independent: the article's three figures and the dashboard
# iframe are same-origin, so a shared BroadcastChannel would gang them together
# (one click in Figure 2 re-rendering everything). The channel is therefore
# opt-in via ?sync=1 — the article does not pass it, so each figure's option
# panel stands alone; open /figure/fig2?sync=1 by hand to gang them deliberately.
# ---------------------------------------------------------------------------
_FIG_CLIENT = r"""
(function () {
  "use strict";
__TOGGLE__
__BANDTOGGLE__
  var FIGID = "__FIGID__";
  var NATIVE_W = __FIGW__, NATIVE_H = __FIGH__, PNG_SCALE = __PNGSCALE__;
  var ORDER = __ORDER__, DEF = __DEFAULTS__;
  var API = window.location.origin;
  // Read once at load: broadcast() rewrites the query from ORDER alone, so the
  // param does not survive the first click in the address bar (it does not need
  // to -- bc is already made or not by then).
  var SYNC = new URLSearchParams(window.location.search).has("sync");
  var bc = (SYNC && "BroadcastChannel" in window) ? new BroadcastChannel("opticnerve") : null;
  var applying = false, last = null;
  // MyST's theme forces every embedded <iframe> into its own responsive box and
  // discards the width/height we author, so the size can't be set from paper.md.
  // The figure instead renders on a fixed native canvas that this page scales to
  // fit whatever box the theme gave it, the way object-fit:contain would.
  function fitStage() {
    var scale = Math.min(window.innerWidth / NATIVE_W, window.innerHeight / NATIVE_H);
    var stage = document.getElementById("stage");
    stage.style.transform = "translate(-50%, -50%) scale(" + scale + ")";
  }
  window.addEventListener("resize", fitStage);
  fitStage();
  // CSS alone rounds the annotation chips on screen but toImage export
  // serializes the SVG without the external stylesheet, so it comes out
  // square. Setting rx/ry as real attributes makes it survive export too.
  new MutationObserver(function () {
    document.querySelectorAll("g.annotation rect").forEach(function (r) {
      r.setAttribute("rx", 6); r.setAttribute("ry", 6);
    });
  }).observe(document.getElementById("fig"), {childList: true, subtree: true});
  function readParams() {
    var q = new URLSearchParams(window.location.search), p = {};
    ORDER.forEach(function (k) { p[k] = q.has(k) ? q.get(k) : DEF[k]; });
    return p;
  }
  function serialize(p) {
    return ORDER.map(function (k) { return k + "=" + encodeURIComponent(p[k]); }).join("&");
  }
  function equal(a, b) { return serialize(a) === serialize(b); }
  function render(p) {
    return fetch(API + "/opticnerve/" + FIGID + "?" + serialize(p))
      .then(function (r) { return r.json(); })
      .then(function (j) {
        return Plotly.react("fig", j.figure.data, j.figure.layout,
                            {responsive:false, displaylogo:false,
                             toImageButtonOptions:{format:"png", filename:FIGID,
                                                   width:NATIVE_W, height:NATIVE_H,
                                                   scale:PNG_SCALE}});
      })
      .then(function () {
        last = p; wireClicks();
      })
      .catch(function (e) {
        document.getElementById("fig").textContent = "Could not load figure: " + e;
      });
  }
  function wireClicks() {
    if (FIGID !== "fig2" && FIGID !== "fig3") return;   // fig1 has nothing to click
    var gd = document.getElementById("fig");
    if (gd._wired) return; gd._wired = true;
    gd.on("plotly_click", function (ev) {
      if (applying) return;
      var pt = ev.points && ev.points[0];
      if (!pt) return;
      var p = readParams();
      if (FIGID === "fig2") {
        // [subj, tok, ghost] per point; the regression line carries none.
        var cd = pt.customdata;
        if (!cd || cd.length < 3) return;
        p.exclude = toggleExclude(p.exclude, cd[0], cd[1], cd[2]);
        broadcast(p);
        return;
      }
      var m = pt.data.meta || (pt.customdata && pt.customdata[0]);
      if (!m) return;
      if (m.slice(-4) === "_gcc") p.mac = (p.mac === m) ? DEF.mac : m;
      else if (m.slice(-4) === "_um_") p.disc = (p.disc === m) ? DEF.disc : m;
      else return;
      broadcast(p);
    });
    if (FIGID === "fig2") {
      gd.on("plotly_clickannotation", function (ev) {
        if (applying) return;
        var b = ev.annotation && ev.annotation.name;
        if (!b) return;
        var p = readParams();
        p.band = toggleBand(p.band, b);
        broadcast(p);
      });
    }
  }
  function broadcast(p) {
    try { window.history.replaceState(null, "", window.location.pathname + "?" + serialize(p)); } catch (e) {}
    applying = true;
    render(p).finally(function () {
      applying = false;
      if (bc) bc.postMessage({params: p});
      // Also tell the embedding page (cross-origin, so BroadcastChannel can't
      // reach it): the article's option panels mirror in-figure clicks from this.
      if (window.parent !== window) {
        window.parent.postMessage({opticnerve: true, figid: FIGID, params: p}, "*");
      }
    });
  }
  function receive(p) {
    if (applying) return;
    if (!last || !equal(last, p)) {
      try { window.history.replaceState(null, "", window.location.pathname + "?" + serialize(p)); } catch (e) {}
      render(p);
    }
  }
  if (bc) bc.onmessage = function (e) { var d = e.data || {}; if (d.params) receive(d.params); };
  // Params pushed by the embedding page (the article's option panels) — the
  // cross-origin twin of the BroadcastChannel path above. Values are validated
  // server-side by parse_params, so any origin is safe to accept. Missing keys
  // fall back to DEF so a partial push still serializes cleanly.
  window.addEventListener("message", function (e) {
    var d = e.data || {};
    if (!d.opticnerve || !d.params) return;
    var p = {};
    ORDER.forEach(function (k) { p[k] = d.params[k] != null ? d.params[k] : DEF[k]; });
    receive(p);
  });
  window.addEventListener("popstate", function () { receive(readParams()); });
  render(readParams());
})();
"""

_FIG_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCT-T1 __FIGID__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#fff}
#stage{position:absolute;top:50%;left:50%;width:__FIGW__px;height:__FIGH__px;
  transform-origin:center;}
#fig{width:100%;height:100%}
/* Plotly has no layout option for rounded annotation backgrounds, so the Fig 3
   stat chips get theirs from CSS. Keep in sync with app.index_string below —
   this page does not inherit the dashboard's stylesheet. */
g.annotation rect{rx:6px;ry:6px}</style></head>
<body><div id="stage"><div id="fig"></div></div>
<script>__CLIENT__</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Caption reset chip. MyST strips author <script> from the article page, so a
# button in a paper.md caption cannot reach the figures directly — but an
# <iframe> survives the sanitiser (that is how the figures get there). This
# route serves a caption-sized button that is a pure SENDER on the shared
# BroadcastChannel: the dashboard listens and resets itself, as do figure pages
# opened with ?sync=1. It starts hidden and appears only once it hears a
# non-default state, which is correct because nothing broadcasts on load.
# Unused by the article since each option panel grew its own Reset button.
# ---------------------------------------------------------------------------
_CHIP_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>reset</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:transparent;
  font:bold 11px Arial,Helvetica,sans-serif}
/* The iframe is exactly the button's size and clips overflow, so the attention
   grab is a colour flash, not a glow — a box-shadow would be cut off at the
   edge. Four alternating steps end back on the base red. */
#rst{display:none;width:100%;height:100%;padding:0;border:1px solid #a3161b;
  border-radius:4px;background:#d02a2f;color:#fff;font:inherit;letter-spacing:.3px;
  cursor:pointer;animation:flash 1s ease-in-out 4 alternate}
#rst:hover{background:#a3161b}
@keyframes flash{from{background:#d02a2f}to{background:#ff5a5f}}
@media (prefers-reduced-motion:reduce){#rst{animation:none}}</style></head>
<body><button id="rst" title="Reset every figure to its default view">&#8634; Reset</button>
<script>
(function () {
  "use strict";
  if (!("BroadcastChannel" in window)) return;
  var ORDER = __ORDER__, DEF = __DEFAULTS__;
  var btn = document.getElementById("rst"), bc = new BroadcastChannel("opticnerve");
  function ser(p) {
    return ORDER.map(function (k) {
      return k + "=" + encodeURIComponent(p[k] == null ? DEF[k] : p[k]);
    }).join("&");
  }
  var DEFSER = ser(DEF);
  bc.onmessage = function (e) {
    var p = (e.data || {}).params;
    if (p) btn.style.display = (ser(p) === DEFSER) ? "none" : "block";
  };
  // A channel never delivers a sender its own post, so hide on click ourselves.
  btn.onclick = function () { bc.postMessage({params: DEF}); btn.style.display = "none"; };
})();
</script></body></html>"""


@server.route("/resetchip")
def reset_chip():
    return server.response_class(
        _CHIP_PAGE.replace("__ORDER__", JS_ORDER).replace("__DEFAULTS__", JS_DEFAULTS),
        mimetype="text/html")


@server.route("/figure/<figid>")
def figure_page(figid):
    if figid not in _BUILDERS:
        abort(404)
    w, h = FIG_SIZE[figid]
    subs = {"__FIGID__": figid, "__FIGW__": str(w), "__FIGH__": str(h),
            "__PNGSCALE__": str(PNG_SCALE), "__ORDER__": JS_ORDER,
            "__DEFAULTS__": JS_DEFAULTS, "__TOGGLE__": _TOGGLE_JS,
            "__BANDTOGGLE__": _BAND_JS}
    client = _FIG_CLIENT
    page = _FIG_PAGE
    for k, v in subs.items():
        client, page = client.replace(k, v), page.replace(k, v)
    return server.response_class(page.replace("__CLIENT__", client), mimetype="text/html")


GRAPH_CFG = lambda name, w, h: dict(
    scrollZoom=False, displaylogo=False, displayModeBar="hover", responsive=False,
    modeBarButtonsToRemove=["select2d", "lasso2d", "zoom2d", "pan2d", "zoomIn2d",
                            "zoomOut2d", "autoScale2d", "resetScale2d"],
    toImageButtonOptions=dict(format="png", filename=name, width=w, height=h, scale=PNG_SCALE))

# One option per checkbox, interleaved subject / OD / OS so a 3-column CSS grid
# lays them out as one row per subject. Values are exactly the exclusion tokens
# parse_params understands: bare = whole subject, dotted = one eye.
SUBJ_OPTIONS = []
for _s in SUBJECTS:
    SUBJ_OPTIONS.append({"label": _s, "value": _s})
    SUBJ_OPTIONS += [{"label": _e, "value": f"{_s}{EYE_SEP}{_e}"} for _e in EYES_OF[_s]]
ALL_TOKENS = tuple(o["value"] for o in SUBJ_OPTIONS)

app.layout = html.Div(id="root", children=[
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="postbridge-dummy"),
    dcc.Store(id="all-tokens", data=list(ALL_TOKENS)),   # for the clientside URL writer
    html.H1(["Figure Server for OCT – T", html.Sub("1"), " Correlation"]),
    dcc.Store(id="sel", data={"mac": DEF_MAC, "disc": DEF_DISC}),
    dcc.Store(id="bands", data=list(DEFAULTS["band"])),
    html.Div(id="app", children=[
      # ---- one fixed-size stage holding the figures AND the sidebar, scaled
      # to fit by fitStage() below ----
      html.Div(id="stagewrap", children=[
       html.Div(id="figures", children=[
        html.Div(className="leftcol", children=[
            html.Section(className="figpanel", children=[
                dcc.Graph(id="fig1", config=GRAPH_CFG("fig01_T1_profile", *FIG_SIZE["fig1"]))]),
            html.Section(className="figpanel statpanel", children=[html.Div(id="avgbox")]),
        ]),
        html.Div(className="rightcol", children=[
            html.Section(className="figpanel", children=[
                dcc.Graph(id="fig2", config=GRAPH_CFG("fig02_regression", *FIG_SIZE["fig2"]))]),
            html.Section(className="figpanel", children=[
                dcc.Graph(id="fig3", config=GRAPH_CFG("fig03_OCT_maps", *FIG_SIZE["fig3"]))]),
        ]),
        # sidebar last: scales with the figures, and stretches to exactly
        # fig2 + gap + fig3 tall
        html.Aside(id="sidebar", children=[
            html.H3("Subjects"),
            html.Div(id="subjhint", className="hint"),
            html.Div([html.Button("All", id="sel-all"), html.Button("None", id="sel-none")],
                     className="btnrow"),
            html.Div([html.Span(""), html.Span("OD"), html.Span("OS")],
                     id="subjhdr", className=f"subjhdr {DEFAULTS['mode']}"),
            dcc.Checklist(id="subjects", options=SUBJ_OPTIONS, value=list(ALL_TOKENS),
                          className=f"subjlist {DEFAULTS['mode']}"),
            html.Div(id="ncount", className="ncount"),
            html.Hr(),
            html.Div("Statistic", className="lbl"),
            dcc.RadioItems(id="stat", value=DEFAULTS["stat"],
                           options=[{"label": " R²", "value": "R2m"}, {"label": " R", "value": "Rm"}]),
            html.Hr(),
            html.Div("T₁ sector", className="lbl"),
            html.Div(id="bandgrid", children=[
                html.Button(lbl, id={"type": "bandbox", "band": k}, n_clicks=0,
                            className="bandbox")
                for k, lbl, _ in T1_BANDS]),
            html.Hr(),
            html.Div("Regression model", className="lbl"),
            dcc.RadioItems(id="mode", value=DEFAULTS["mode"],
                           options=[{"label": " Average + OLS", "value": "avg"},
                                    {"label": " LME (per-eye, marginal T₁)", "value": "lme"}]),
            html.Hr(),
            html.Div([html.Button("Reset all", id="reset")], className="btnrow"),
        ]),
       ]),
      ]),
    ]),
])


def _qs_to_dict(search):
    return dict(parse_qsl((search or "").lstrip("?")))


def _excluded(included):
    """Checklist value -> canonical `exclude` tuple (see parse_params). A point
    is out if its subject box OR its own eye box is unchecked."""
    included = set(included or [])
    return tuple(sorted(t for t in ALL_TOKENS if t not in included))


def _included(exclude):
    ex = set(exclude)
    return [t for t in ALL_TOKENS if t not in ex]


# ---- URL (on load), All/None and Reset set the controls ----
@app.callback(
    Output("subjects", "value"), Output("stat", "value"),
    Output("mode", "value"), Output("sel", "data"),
    Input("url", "search"), Input("sel-all", "n_clicks"), Input("sel-none", "n_clicks"),
    Input("reset", "n_clicks"),
    State("stat", "value"), State("mode", "value"),
    State("sel", "data"), prevent_initial_call=False)
def _from_url(search, _a, _n, _r, stat, mode, sel):
    trig = ctx.triggered_id
    if trig == "sel-all":
        return list(ALL_TOKENS), stat, mode, sel
    if trig == "sel-none":
        return [], stat, mode, sel
    if trig == "reset":
        return (list(ALL_TOKENS), DEFAULTS["stat"], DEFAULTS["mode"],
                {"mac": DEF_MAC, "disc": DEF_DISC})
    p = parse_params(_qs_to_dict(search))
    return (_included(p["exclude"]), p["stat"], p["mode"],
            {"mac": p["mac"], "disc": p["disc"]})


# ---- the T1 band set: URL on load, Reset, a sidebar button, or a click on a
# Fig 2 band label. One callback owns the store so these can never race. ----
@app.callback(Output("bands", "data"),
              Input("url", "search"), Input("reset", "n_clicks"),
              Input({"type": "bandbox", "band": ALL}, "n_clicks"),
              Input("fig2", "clickAnnotationData"),
              State("bands", "data"), prevent_initial_call=False)
def _bands(search, _r, _clicks, click, bands):
    trig = ctx.triggered_id
    if trig == "reset":
        return list(DEFAULTS["band"])
    if isinstance(trig, dict) and trig.get("type") == "bandbox":
        # only a real click counts; ignore the n_clicks=0 firing on mount
        if not (ctx.triggered and ctx.triggered[0]["value"]):
            return no_update
        return toggle_band(bands, trig["band"])
    if trig == "fig2":
        name = ((click or {}).get("annotation") or {}).get("name")
        return toggle_band(bands, name) if name else no_update
    return list(parse_params(_qs_to_dict(search))["band"])


# ---- paint the grid: checked, and the primary marked ----
@app.callback(Output({"type": "bandbox", "band": ALL}, "className"),
              Input("bands", "data"))
def _paint_bands(bands):
    bands = list(bands or DEFAULTS["band"])
    # Dash resolves a pattern-matching ALL output by SORTED id order, which is
    # NOT T1_KEYS (layout order) — building the list positionally off T1_KEYS
    # silently swaps two buttons' painted state. Read the actual per-output id
    # from ctx.outputs_list so the class always lands on the band it names.
    return ["bandbox on prim" if o["id"]["band"] == bands[0] else
            "bandbox on" if o["id"]["band"] in bands else "bandbox"
            for o in ctx.outputs_list]


# ---- the panel shows only the level the active model can express: one box per
# SUBJECT for OLS (a point is a subject), one per EYE for the LME (a point is an
# eye). On a mode switch the state is translated to the visible level so the
# hidden one can never silently affect a fit: subject-out becomes both-eyes-out
# going to LME, and any-eye-out becomes subject-out going to OLS.
@app.callback(Output("subjects", "value", allow_duplicate=True),
              Output("subjects", "className"), Output("subjhdr", "className"),
              Output("subjhint", "children"),
              Input("mode", "value"), State("subjects", "value"),
              prevent_initial_call="initial_duplicate")
def _mode_granularity(mode, included):
    inc, value = set(included or []), []
    for s in SUBJECTS:
        eyes = [f"{s}{EYE_SEP}{e}" for e in EYES_OF[s]]
        if mode == "avg":
            value += ([s] if s in inc and all(e in inc for e in eyes) else []) + eyes
        else:
            value += [s] + [e for e in eyes if e in inc and s in inc]
    hint = ("Uncheck a subject to drop it" if mode == "avg" else
            "Uncheck an eye to drop it") + " — or click its point in Figure 2."
    return sorted(value), f"subjlist {mode}", f"subjhdr {mode}", hint


# ---- clicking a Figure 2 point drops it from the analysis (click again to
# restore). Average mode toggles the whole subject (one point = one subject),
# LME mode toggles that eye. Duplicate output of _from_url.
@app.callback(Output("subjects", "value", allow_duplicate=True),
              Input("fig2", "clickData"), State("mode", "value"),
              State("subjects", "value"), prevent_initial_call=True)
def _toggle_point(click, mode, included):
    pts = (click or {}).get("points") or []
    cd = pts[0].get("customdata") if pts else None
    if not isinstance(cd, (list, tuple)) or len(cd) < 3:
        return no_update                     # regression line, or clickData reset
    subj, tok, ghost = cd[0], cd[1], cd[2]
    included = list(included or [])
    if ghost:                                # put it back: clear every box that
        back = {subj} | {f"{subj}{EYE_SEP}{e}" for e in EYES_OF[subj]   # could be
                         if tok in (subj, f"{subj}{EYE_SEP}{e}")}       # keeping it out
        included += [t for t in back if t not in included]
    else:
        drop = subj if mode == "avg" else tok
        included = [t for t in included if t != drop]
    return sorted(included)


# ---- clicking a Fig 3 wedge or an averages row selects the Fig 2 sector ----
# Duplicate output: _from_url is the canonical writer of sel.data on load.
@app.callback(Output("sel", "data", allow_duplicate=True),
              Input("fig3", "clickData"),
              Input({"type": "avgrow", "metric": ALL, "region": ALL}, "n_clicks"),
              State("sel", "data"), prevent_initial_call=True)
def _select_sector(click, _rows, sel):
    trig = ctx.triggered_id
    metric, region = None, None
    if trig == "fig3" and click and click.get("points"):
        pt = click["points"][0]
        metric = pt.get("customdata", pt.get("meta"))     # per-point id, meta as fallback
        if isinstance(metric, (list, tuple)):             # some Plotly builds wrap customdata
            metric = metric[0] if metric else None
        if isinstance(metric, str):
            region = "mac" if metric.endswith("_gcc") else "disc"
        else:
            metric = None
    elif isinstance(trig, dict) and trig.get("type") == "avgrow":
        # only a real click counts; ignore the n_clicks=0 firings from a table rebuild
        if ctx.triggered and ctx.triggered[0]["value"]:
            metric, region = trig["metric"], trig["region"]
    # nothing selected: leave sel alone, rewriting it would race _from_url
    if not metric:
        return no_update
    sel = dict(sel)
    default = DEF_MAC if region == "mac" else DEF_DISC
    sel[region] = default if sel[region] == metric else metric
    return sel


# ---- main render ----
@app.callback(
    Output("fig1", "figure"), Output("fig2", "figure"), Output("fig3", "figure"),
    Output("avgbox", "children"), Output("ncount", "children"),
    Input("subjects", "value"), Input("stat", "value"),
    Input("bands", "data"), Input("mode", "value"), Input("sel", "data"))
def _render(included, stat, bands, mode, sel):
    view = resolve_view(exclude=_excluded(included), stat=stat,
                        band=tuple(bands or DEFAULTS["band"]),
                        mac=sel["mac"], disc=sel["disc"], mode=mode)
    count = (f"{view['n']}/{len(SUBJECTS)} subjects · "
             f"{view['n_eyes']}/{len(ALL_EYES)} eyes")
    return (build_fig1(view), build_fig2(view), build_fig3(view),
            build_avg_table(view), count)


# ---- push state OUT (clientside): write the URL + tell the figure iframes ----
# Runs on any control/click change. Writes the query string with
# history.replaceState (NOT a Dash Output on url.search -> no dependency cycle
# with _from_url). Output is a throwaway store.
app.clientside_callback(
    """
    function(included, stat, bands, mode, sel, allTokens) {
        // must match _excluded() in the Python: every unchecked token, sorted.
        var inc = included || [];
        var excl = (allTokens || []).filter(function (t) { return inc.indexOf(t) === -1; }).sort();
        var params = {exclude: excl.join(","), stat: stat,
                      band: (bands || []).join(","),
                      mac: (sel || {}).mac, disc: (sel || {}).disc, mode: mode};
        var search = "?" + __ORDER__.map(function (k) {
            return k + "=" + encodeURIComponent(params[k] == null ? "" : params[k]);
        }).join("&");
        try {
            window.history.replaceState(
                null, "", window.location.pathname + search + window.location.hash);
        } catch (e) {}
        // Sync to the sibling figure iframes (same origin). A single shared
        // channel instance never receives its own posts, so this cannot loop back.
        if ("BroadcastChannel" in window) {
            var bc = window.__ocbc || (window.__ocbc = new BroadcastChannel("opticnerve"));
            bc.postMessage({params: params});
        }
        return window.dash_clientside.no_update;
    }
    """.replace("__ORDER__", JS_ORDER),
    Output("postbridge-dummy", "data"),
    Input("subjects", "value"), Input("stat", "value"),
    Input("bands", "data"), Input("mode", "value"), Input("sel", "data"),
    State("all-tokens", "data"),
    prevent_initial_call=True,
)


# ---- light theme (kept inline so the app stays a single file) ----
app.index_string = ("""<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  html, body { height:100%; margin:0; overflow:hidden; }
  body { background:#fff; color:#222; font-family:Arial,Helvetica,sans-serif; }
  /* position:fixed pins #root to the viewport, escaping the plain-block divs
     Dash wraps the layout in — a flex chain from <body> cannot reach through
     them, so #app would size to the tall sidebar instead of the window. */
  #root { position:fixed; inset:0; display:flex; flex-direction:column; }
  h1 { flex:0 0 auto; text-align:center; font-size:22px; margin:14px 0 8px; }
  #app { flex:1 1 auto; min-height:0; display:flex; padding:0 16px 16px; }
  /* no height of its own: as a flex child of the stage it stretches to the
     stage's 626px, i.e. exactly fig2 + gap + fig3, and scales with it. */
  #sidebar { flex:0 0 210px; background:#f4f4f6; border-radius:6px;
    padding:14px; overflow-y:auto; box-sizing:border-box; }
  #sidebar h3 { margin:0 0 8px; font-size:16px; }
  #sidebar .hint { font-size:11px; color:#666; margin-bottom:8px; line-height:1.35; }
  #sidebar .lbl { font-size:13px; color:#555; margin-bottom:4px; }
  #sidebar hr { border:none; border-top:1px solid #ddd; margin:6px 0 5px; }
  .btnrow { display:flex; gap:6px; margin-bottom:8px; }
  .btnrow button { flex:1; background:#fff; color:#333; border:1px solid #bbb;
    border-radius:4px; padding:4px; font-size:12px; cursor:pointer; }
  .btnrow button:hover { background:#e6e6ea; }
  /* all sidebar OPTIONS share one font size (incl. the dropdown value + menu) */
  .subjlist label,
  #stat label, #mode label { font-size:12px; }
  #bandgrid { display:grid; grid-template-columns:1fr 1fr; gap:4px; }
  #bandgrid .bandbox { font-size:12px; text-align:left; padding:3px 6px;
    border:1px solid #bbb; border-radius:4px; background:#fff; color:#333;
    cursor:pointer; }
  #bandgrid .bandbox::before { content:"\\2610\\00a0"; }
  #bandgrid .bandbox:hover { background:#e6e6ea; }
  #bandgrid .bandbox.on { background:#d8e6f5; border-color:#8fb4d9; color:#111; }
  #bandgrid .bandbox.on::before { content:"\\2611\\00a0"; }
  /* the primary is the one Figure 3 and the averages table follow */
  #bandgrid .bandbox.prim { font-weight:bold; border-color:#1a5fb4; }
  /* one row per subject: the checklist's options are interleaved
     subject / OD / OS, so a 3-column grid lines them up. */
  .subjlist { display:grid; grid-template-columns:1fr 38px 38px; grid-auto-rows:16px;
    align-items:center; }
  .subjlist label { display:flex; align-items:center; cursor:pointer; color:#1a5fb4; }
  .subjlist input { width:12px; height:12px; margin:0 5px 0 0; }
  .subjhdr { display:grid; grid-template-columns:1fr 38px 38px; font-size:10px;
    color:#666; text-transform:uppercase; letter-spacing:.5px; margin-bottom:2px; }
  .subjhdr span { padding-left:15px; }
  /* OLS: subject boxes only (a point is a subject). The 3n+2/3n children are
     the OD/OS boxes of each row. */
  .subjhdr.avg { display:none; }
  .subjlist.avg { grid-template-columns:1fr; }
  .subjlist.avg label:nth-child(3n+2), .subjlist.avg label:nth-child(3n) { display:none; }
  /* LME: eye boxes only; column 1 keeps the name, as inert text. */
  .subjlist.lme label:nth-child(3n+1) { pointer-events:none; color:#555; }
  .subjlist.lme label:nth-child(3n+1) input { display:none; }
  #stat label { display:inline-flex; align-items:center; color:#222; margin-right:18px; cursor:pointer; }
  #mode label { display:flex; align-items:center; color:#222; margin-bottom:5px; cursor:pointer; }
  #stat input, #mode input { margin-right:7px; flex-shrink:0; }
  .ncount { font-size:12px; color:#1a7f37; margin-top:4px; }
  #sidebar .dash-dropdown { color:#111; }
  /* The figure area is ONE fixed 1475x626 canvas that fitStage() scales to fit
     #stagewrap. Every length below is an authored pixel, never a fraction of
     the window, so Plotly measures once at startup and never redraws. */
  #stagewrap { flex:1 1 auto; min-width:0; min-height:0; position:relative; overflow:hidden; }
  /*  575 = fig1 555 + 2x10 panel padding     658 = fig2/3 638 + 2x10
      626 = (402+20) + 16 gap + 188 table  ==  (285+20) + 16 + (285+20)
     1475 = 575 + 16 + 658 + 16 + 210 sidebar  <- both columns agree exactly */
  #figures { position:absolute; top:50%; left:50%; transform-origin:center;
    width:1475px; height:626px; display:flex; gap:16px; }
  .leftcol, .rightcol { display:flex; flex-direction:column; gap:16px; }
  /* no border/shadow: the panels are white on a white page, so the figures
     read as part of the article rather than as cards. */
  .figpanel { background:#fff; border-radius:6px; padding:10px; }
  /* 188 is what the left column needs to match the right (see the table above) */
  .statpanel { padding:12px; height:188px;
    overflow:hidden; box-sizing:border-box; }
  .avgtitle { font-size:14px; color:#222; margin-bottom:10px; }
  .avghint { font-size:11px; color:#666; }
  .avgtbl { border-collapse:collapse; width:100%; table-layout:fixed; }
  .avgtbl th { font-size:12px; font-weight:normal; color:#666; text-align:center; padding:2px 8px;
    text-transform:uppercase; letter-spacing:.4px; }
  .avgtbl th.corner { text-align:left; }
  .avgtbl th.on { color:#333; background:#eef4fb; }
  .avgtbl th.prim { color:#111; background:#d8e6f5; font-weight:bold; }
  .avgtbl td { padding:3px 8px; font-size:12px; text-align:center; }
  .avgtbl td.anm { text-align:left; color:#333; white-space:nowrap; }
  .avgtbl td.region { text-align:left; color:#555; font-size:12px; font-weight:bold;
    text-transform:uppercase; letter-spacing:.4px; vertical-align:middle; }
  .avgtbl tr.avgrow { cursor:pointer; }
  .avgtbl tr.avgrow:hover td { background:#e6e6ea; }
  .avgtbl tr.avgrow.sel td { background:#d8e6f5; }
  .avgtbl tr.avgrow.sel td.region { background:transparent; }
  .avgtbl .aval { color:#111; font-weight:600; font-variant-numeric:tabular-nums; }
  .avgtbl .aval.sig { color:#d02a2f; }
  g.annotation rect { rx:6px; ry:6px; }
</style></head>
<body>{%app_entry%}<footer>
<script>
(function () {
  // Absorb state changes from the sibling figure iframes (same origin) via the
  // shared BroadcastChannel, and reflect them into this app's URL so the normal
  // _from_url callback runs. Compares the 6 params SEMANTICALLY so encoding
  // differences never cause a spurious extra round-trip.
  if (!("BroadcastChannel" in window)) return;
  var ORDER = __ORDER__, DEF = __DEFAULTS__;
  function canon(search) {
    var q = new URLSearchParams((search || "").replace(/^[?]/, ""));
    return ORDER.map(function (k) {
      return k + "=" + (q.has(k) ? q.get(k) : DEF[k]);
    }).join("&");
  }
  var bc = window.__ocbc || (window.__ocbc = new BroadcastChannel("opticnerve"));
  bc.onmessage = function (e) {
    var d = e.data || {};
    if (!d.params) return;
    var p = d.params;
    var search = "?" + ORDER.map(function (k) {
      return k + "=" + encodeURIComponent(p[k] == null ? "" : p[k]);
    }).join("&");
    if (canon(search) !== canon(window.location.search)) {
      window.history.replaceState(null, "", window.location.pathname + search);
      window.dispatchEvent(new PopStateEvent("popstate"));
    }
  };
})();
</script>
<script>
(function () {
  // Fit the fixed stage into whatever room is left, the way object-fit:contain
  // would for an <img>: one CSS transform, centred. Nothing here touches Plotly
  // — the figures are drawn once at their native pixel size (autosize=False) and
  // are SVG, so they stay crisp at any scale and never re-measure.
  var NATIVE_W = 1475, NATIVE_H = 626;
  function fitStage() {
    var wrap = document.getElementById('stagewrap');
    var stage = document.getElementById('figures');
    if (!wrap || !stage) return;             // not mounted yet
    var w = wrap.clientWidth, h = wrap.clientHeight;
    if (!w || !h) return;
    var s = Math.min(w / NATIVE_W, h / NATIVE_H);
    stage.style.transform = 'translate(-50%, -50%) scale(' + s + ')';
  }
  // Dash mounts the layout after this script runs, so watch for the stage
  // appearing as well as for later resizes.
  var ro = new ResizeObserver(fitStage);
  new MutationObserver(function () {
    var wrap = document.getElementById('stagewrap');
    if (wrap && !wrap._fitted) { wrap._fitted = true; ro.observe(wrap); fitStage(); }
  }).observe(document.body, {childList: true, subtree: true});
  window.addEventListener('resize', fitStage);
})();
</script>
<script>
(function () {
  // Same fix as the standalone figure pages (see _FIG_PAGE): rx/ry on the
  // annotation rects has to be a real SVG attribute, not just CSS, or it's
  // lost when Plotly's toImage export serializes the SVG standalone.
  new MutationObserver(function () {
    document.querySelectorAll('g.annotation rect').forEach(function (r) {
      r.setAttribute('rx', 6); r.setAttribute('ry', 6);
    });
  }).observe(document.body, {childList: true, subtree: true});
})();
</script>
{%config%}{%scripts%}{%renderer%}</footer></body></html>"""
    .replace("__ORDER__", JS_ORDER).replace("__DEFAULTS__", JS_DEFAULTS))


if __name__ == "__main__":
    # 8050, not 3000: `myst start` takes 3000, and every URL in paper.md points
    # here. Keep the two in sync.
    app.run(debug=True, port=8050)
