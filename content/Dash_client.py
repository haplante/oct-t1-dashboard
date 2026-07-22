"""
OCT – T1 correlation dashboard  (Plotly Dash app)

Recreates the three figures of the static HTML dashboard, plus two regression
modes selectable in the sidebar:

  * "Average + OLS"  -> average both eyes per subject, ordinary least squares
  * "LME"            -> per-eye data, linear mixed-effects model with a random
                        intercept per subject; Figure 2 plots the marginal T1
                        (observed T1 minus the subject random intercept).

The code is intentionally written in the same plain style as the figure
notebooks so it can be read top to bottom.

Run locally:   python Dash_client.py       (opens http://127.0.0.1:8050)
Deploy:        gunicorn Dash_client:server  (see render.yaml)
"""

import numpy as np

from dash import Dash, dcc, html, Input, Output, State, ctx, ALL, no_update

from opticnerve_core import (
    T1_BANDS, DEF_MAC, DEF_DISC, MAC_AVG, DISC_AVG, SUBJECTS, stat_val,
    stat_lbl, resolve_view, DEFAULTS, build_fig1, build_fig2, build_fig3,
    parse_params, serialize_params, ALL_EYES, EYES_OF, EYE_SEP, FIG_SIZE,
)


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

    header = html.Thead(html.Tr(
        [html.Th("T₁ sector", className="corner", colSpan=2)]
        + [html.Th(lbl) for _, lbl, _ in T1_BANDS]))
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
app = Dash(__name__, title="OCT – T₁ Correlation")
server = app.server                      # gunicorn entry point (Render)

import json as _json

from flask import request, abort
from flask_cors import CORS
from plotly.utils import PlotlyJSONEncoder

CORS(server, resources={r"/opticnerve/*": {"origins": "*"}})

_BUILDERS = {"fig1": build_fig1, "fig2": build_fig2, "fig3": build_fig3}

# Native pixel canvas each standalone /figure/<figid> page renders at (no
# autosize/responsive) — the same canvas the builders draw on, and the same one
# the dashboard's own stage is laid out from. paper.md scales this down visually
# with a CSS transform to fit the article column — see the per-figure scale
# factors there, which must be recomputed (720 / width) if these change.
FIG_NATIVE_SIZE = FIG_SIZE

# PNG size for each standalone figure's own "download as png" modebar button
# (independent of FIG_NATIVE_SIZE / the on-screen paper.md scale). Edit these
# to change what gets downloaded; scale is the extra DPI multiplier (600/96
# matches the dashboard's own GRAPH_CFG export resolution).
PNG_EXPORT_SIZE = dict(FIG_SIZE)
PNG_EXPORT_SCALE = 600 / 96


@server.route("/opticnerve/<figid>")
def opticnerve(figid):
    if figid not in _BUILDERS and figid != "all":
        abort(404)
    p = parse_params(request.args)
    view = resolve_view(**p)
    payload = {"figid": figid, "params": {**p, "exclude": list(p["exclude"])},
               "n": view["n"]}
    if figid == "all":
        payload.update({k: _BUILDERS[k](view).to_dict() for k in _BUILDERS})
    else:
        payload["figure"] = _BUILDERS[figid](view).to_dict()
    return server.response_class(
        _json.dumps(payload, cls=PlotlyJSONEncoder), mimetype="application/json")


# ---------------------------------------------------------------------------
# Standalone live figure pages, for embedding in the article as <iframe>s.
# MyST won't run author <script>, so the same-page CustomEvent bus can't work
# there. Instead each figure is its own page served here; because every figure
# iframe AND the dashboard iframe are same-origin (this app), they sync directly
# via a BroadcastChannel — no parent-page JS required.
# ---------------------------------------------------------------------------
_FIG_CLIENT = r"""
(function () {
  "use strict";
  var FIGID = "__FIGID__";
  var NATIVE_W = __FIGW__, NATIVE_H = __FIGH__;
  var PNG_W = __PNGW__, PNG_H = __PNGH__, PNG_SCALE = __PNGSCALE__;
  var ORDER = ["exclude","stat","band","mac","disc","mode"];
  var DEF = {exclude:"",stat:"R2m",band:"T1_mean_015",mac:"All_1_3_gcc",disc:"All_um_",mode:"avg"};
  var API = window.location.origin;
  var bc = ("BroadcastChannel" in window) ? new BroadcastChannel("opticnerve") : null;
  var applying = false, last = null;
  // MyST's theme forces every embedded <iframe> into its own responsive box
  // (fixed padding-bottom aspect ratio, ignoring any width/height we author),
  // so we can't control the iframe's size from paper.md. Instead the figure
  // renders at a fixed native pixel canvas (crisp, like a static image) and
  // this page scales that canvas to fit whatever box the theme actually gave
  // it, the same way object-fit:contain would for an <img>.
  function fitStage() {
    var scale = Math.min(window.innerWidth / NATIVE_W, window.innerHeight / NATIVE_H);
    var stage = document.getElementById("stage");
    stage.style.transform = "translate(-50%, -50%) scale(" + scale + ")";
  }
  window.addEventListener("resize", fitStage);
  fitStage();
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
        // Fixed native canvas, no responsive/autosize: paper.md scales the
        // whole page visually via CSS transform instead, like a PNG would be,
        // so Plotly never needs to re-measure or redraw on resize.
        return Plotly.react("fig", j.figure.data, j.figure.layout,
                            {responsive:false, displaylogo:false,
                             toImageButtonOptions:{format:"png", filename:FIGID,
                                                   width:PNG_W, height:PNG_H, scale:PNG_SCALE}});
      })
      .then(function () {
        last = p; wireClicks();
      })
      .catch(function (e) {
        document.getElementById("fig").textContent = "Could not load figure: " + e;
      });
  }
  function wireClicks() {
    if (FIGID !== "fig3") return;
    var gd = document.getElementById("fig");
    if (gd._wired) return; gd._wired = true;
    gd.on("plotly_click", function (ev) {
      if (applying) return;
      var pt = ev.points && ev.points[0];
      var m = pt && (pt.data.meta || (pt.customdata && pt.customdata[0]));
      if (!m) return;
      var p = readParams();
      if (m.slice(-4) === "_gcc") p.mac = (p.mac === m) ? DEF.mac : m;
      else if (m.slice(-4) === "_um_") p.disc = (p.disc === m) ? DEF.disc : m;
      else return;
      broadcast(p);
    });
  }
  function broadcast(p) {
    try { window.history.replaceState(null, "", window.location.pathname + "?" + serialize(p)); } catch (e) {}
    applying = true;
    render(p).finally(function () {
      applying = false;
      if (bc) bc.postMessage({params: p});
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
/* Keep in sync with the same rule in app.index_string below. Plotly has no
   layout option for rounded annotation backgrounds, so the Fig 3 stat chips get
   their rounded corners from CSS on the SVG <rect>. This page has its own
   <style> and does not inherit the dashboard's, so without this the chips
   render square in paper.md while the dashboard shows them rounded. */
g.annotation rect{rx:6px;ry:6px}</style></head>
<body><div id="stage"><div id="fig"></div></div>
<script>__CLIENT__</script>
</body></html>"""


@server.route("/figure/<figid>")
def figure_page(figid):
    if figid not in _BUILDERS:
        abort(404)
    w, h = FIG_NATIVE_SIZE[figid]
    pw, ph = PNG_EXPORT_SIZE[figid]
    client = (_FIG_CLIENT.replace("__FIGID__", figid)
              .replace("__FIGW__", str(w)).replace("__FIGH__", str(h))
              .replace("__PNGW__", str(pw)).replace("__PNGH__", str(ph))
              .replace("__PNGSCALE__", str(PNG_EXPORT_SCALE)))
    page = (_FIG_PAGE.replace("__CLIENT__", client)
            .replace("__FIGID__", figid)
            .replace("__FIGW__", str(w)).replace("__FIGH__", str(h)))
    return server.response_class(page, mimetype="text/html")

GRAPH_CFG = lambda name, w, h: dict(
    scrollZoom=False, displaylogo=False, displayModeBar="hover", responsive=False,
    modeBarButtonsToRemove=["select2d", "lasso2d", "zoom2d", "pan2d", "zoomIn2d",
                            "zoomOut2d", "autoScale2d", "resetScale2d"],
    toImageButtonOptions=dict(format="png", filename=name, width=w, height=h, scale=600 / 96))

app.layout = html.Div(id="root", children=[
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="postbridge-dummy"),
    dcc.Store(id="all-subjects", data=SUBJECTS),   # for the clientside URL writer
    dcc.Store(id="all-eyes", data=list(ALL_EYES)),
    html.H1(["OCT – T", html.Sub("1"), " Correlation Figures"]),
    dcc.Store(id="sel", data={"mac": DEF_MAC, "disc": DEF_DISC}),
    html.Div(id="app", children=[
        # ---- sidebar ----
        html.Aside(id="sidebar", children=[
            html.H3("Subjects"),
            html.Div("Uncheck a subject to remove both its eyes from every figure. "
                     "Or click a point in Figure 2.", className="hint"),
            html.Div([html.Button("All", id="sel-all"), html.Button("None", id="sel-none")],
                     className="btnrow"),
            dcc.Checklist(id="subjects", options=[{"label": s, "value": s} for s in SUBJECTS],
                          value=SUBJECTS, className="subjlist"),
            html.Hr(),
            html.H3("Single eyes"),
            html.Div("Drops one eye only — changes the LME directly, and the "
                     "average of that subject in OLS.", className="hint"),
            dcc.Checklist(id="eyes", value=list(ALL_EYES), className="subjlist eyelist",
                          options=[{"label": t.replace(EYE_SEP, " "), "value": t}
                                   for t in ALL_EYES]),
            html.Div(id="ncount", className="ncount"),
            html.Hr(),
            html.Div("Statistic", className="lbl"),
            dcc.RadioItems(id="stat", value=DEFAULTS["stat"],
                           options=[{"label": " R²", "value": "R2m"}, {"label": " R", "value": "Rm"}]),
            html.Hr(),
            html.Div("T₁ sector", className="lbl"),
            dcc.Dropdown(id="t1band", clearable=False, value=DEFAULTS["band"],
                         options=[{"label": lbl, "value": k} for k, lbl, _ in T1_BANDS]),
            html.Hr(),
            html.Div("Regression model", className="lbl"),
            dcc.RadioItems(id="mode", value=DEFAULTS["mode"],
                           options=[{"label": " Average + OLS", "value": "avg"},
                                    {"label": " LME (per-eye, marginal T₁)", "value": "lme"}]),
            html.Hr(),
            html.Div([html.Button("Reset all", id="reset")], className="btnrow"),
        ]),
        # ---- figures: a fixed-size stage, scaled to fit by fitStage() below ----
        html.Div(id="stagewrap", children=[
            html.Div(id="figures", children=[
                html.Div(className="leftcol", children=[
                    html.Section(className="figpanel", children=[
                        dcc.Graph(id="fig1",
                                  config=GRAPH_CFG("fig01_T1_profile", *FIG_SIZE["fig1"]))]),
                    html.Section(className="figpanel statpanel", children=[
                        html.Div(id="avgbox")]),
                ]),
                html.Div(className="rightcol", children=[
                    html.Section(className="figpanel", children=[
                        dcc.Graph(id="fig2",
                                  config=GRAPH_CFG("fig02_regression", *FIG_SIZE["fig2"]))]),
                    html.Section(className="figpanel", children=[
                        dcc.Graph(id="fig3",
                                  config=GRAPH_CFG("fig03_OCT_maps", *FIG_SIZE["fig3"]))]),
                ]),
            ]),
        ]),
    ]),
])


from urllib.parse import parse_qsl


def _qs_to_dict(search):
    return dict(parse_qsl((search or "").lstrip("?")))


def _excluded(inc_subj, inc_eyes):
    """The two checklists -> one canonical `exclude` tuple (see parse_params).
    They are independent: a point is out if its subject OR its eye is unchecked.
    Keeping both spellings apart is what makes the URL round-trip exactly."""
    inc_subj, inc_eyes = set(inc_subj or []), set(inc_eyes or [])
    return tuple(sorted([s for s in SUBJECTS if s not in inc_subj]
                        + [t for t in ALL_EYES if t not in inc_eyes]))


# ---- URL (on load), All/None and Reset set the controls ----
@app.callback(
    Output("subjects", "value"), Output("eyes", "value"), Output("stat", "value"),
    Output("t1band", "value"), Output("mode", "value"), Output("sel", "data"),
    Input("url", "search"), Input("sel-all", "n_clicks"), Input("sel-none", "n_clicks"),
    Input("reset", "n_clicks"),
    State("stat", "value"), State("t1band", "value"), State("mode", "value"),
    State("sel", "data"), prevent_initial_call=False)
def _from_url(search, _a, _n, _r, stat, band, mode, sel):
    trig = ctx.triggered_id
    if trig == "sel-all":
        return SUBJECTS, list(ALL_EYES), stat, band, mode, sel
    if trig == "sel-none":
        return [], [], stat, band, mode, sel
    if trig == "reset":
        return (SUBJECTS, list(ALL_EYES), DEFAULTS["stat"], DEFAULTS["band"],
                DEFAULTS["mode"], {"mac": DEF_MAC, "disc": DEF_DISC})
    p = parse_params(_qs_to_dict(search))
    ex = set(p["exclude"])
    return ([s for s in SUBJECTS if s not in ex], [t for t in ALL_EYES if t not in ex],
            p["stat"], p["band"], p["mode"], {"mac": p["mac"], "disc": p["disc"]})


# ---- clicking a Figure 2 point drops it from the analysis (click again to
# restore). Average mode toggles the whole subject (one point = one subject),
# LME mode toggles that eye. Both checklists are duplicate outputs of _from_url.
@app.callback(Output("subjects", "value", allow_duplicate=True),
              Output("eyes", "value", allow_duplicate=True),
              Input("fig2", "clickData"), State("mode", "value"),
              State("subjects", "value"), State("eyes", "value"),
              prevent_initial_call=True)
def _toggle_point(click, mode, inc_subj, inc_eyes):
    pts = (click or {}).get("points") or []
    cd = pts[0].get("customdata") if pts else None
    if not isinstance(cd, (list, tuple)) or len(cd) < 3:
        return no_update, no_update          # regression line, or clickData reset
    subj, tok, ghost = cd[0], cd[1], cd[2]
    inc_subj, inc_eyes = list(inc_subj or []), list(inc_eyes or [])
    if ghost:                                # put it back: clear BOTH levels that
        if subj not in inc_subj:             # could be keeping it out
            inc_subj.append(subj)
        for e in EYES_OF[subj]:
            t = f"{subj}{EYE_SEP}{e}"
            if tok in (subj, t) and t not in inc_eyes:
                inc_eyes.append(t)
    elif mode == "avg":
        inc_subj = [s for s in inc_subj if s != tok]
    else:
        inc_eyes = [t for t in inc_eyes if t != tok]
    return sorted(inc_subj), sorted(inc_eyes)


# NOTE: writing the URL is done CLIENTSIDE (see the push-out callback near the
# bottom), NOT via a Dash Output on url.search. A server callback writing
# url.search would form a dependency cycle with _from_url (which reads url.search
# to set the controls): subjects.value -> url.search -> subjects.value. Writing
# the URL with history.replaceState from a clientside callback (dummy output)
# keeps the graph acyclic while still updating the address bar + notifying the
# iframe parent.


# ---- clicking a Fig 3 wedge or an averages row selects the Fig 2 sector ----
# sel.data is also written on load by _from_url (the canonical writer), so this
# click-driven writer is a duplicate output -> allow_duplicate=True (requires
# prevent_initial_call=True, already set).
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
        # only a genuine row click counts (n_clicks>0); ignore the n_clicks=0
        # firings when _render rebuilds the averages table, which would otherwise
        # spuriously toggle the selection.
        trigval = ctx.triggered[0]["value"] if ctx.triggered else None
        if trigval:
            metric, region = trig["metric"], trig["region"]
    # No genuine selection (e.g. clickData reset to None on re-render, or a table
    # rebuild): do NOT rewrite sel — returning it would race _from_url and clobber
    # a sector absorbed from the BroadcastChannel. Leave sel untouched.
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
    Input("subjects", "value"), Input("eyes", "value"), Input("stat", "value"),
    Input("t1band", "value"), Input("mode", "value"), Input("sel", "data"))
def _render(inc_subj, inc_eyes, stat, sel_band, mode, sel):
    view = resolve_view(exclude=_excluded(inc_subj, inc_eyes), stat=stat, band=sel_band,
                        mac=sel["mac"], disc=sel["disc"], mode=mode)
    count = (f"{view['n']} / {len(SUBJECTS)} subjects · "
             f"{view['n_eyes']} / {len(ALL_EYES)} eyes included")
    return (build_fig1(view), build_fig2(view), build_fig3(view),
            build_avg_table(view), count)


# ---- push state OUT (clientside): write the URL + notify the iframe parent ----
# Runs on any control/click change. Writes the query string with
# history.replaceState (NOT a Dash Output on url.search -> no dependency cycle
# with _from_url) and postMessages the new search to the parent window so the
# inline article figures (state_sync.js bus) stay in sync. Output is a throwaway
# store. exclude is derived from the full subject list (all-subjects store).
app.clientside_callback(
    """
    function(incSubj, incEyes, stat, band, mode, sel, allSubjects, allEyes) {
        // must match _excluded() in the Python: union of the two checklists,
        // subject tokens and eye tokens kept distinct, sorted.
        var is = incSubj || [], ie = incEyes || [];
        var excl = (allSubjects || []).filter(function (s) { return is.indexOf(s) === -1; })
            .concat((allEyes || []).filter(function (t) { return ie.indexOf(t) === -1; }))
            .sort();
        var params = {exclude: excl.join(","), stat: stat, band: band,
                      mac: (sel || {}).mac, disc: (sel || {}).disc, mode: mode};
        var ORDER = ["exclude", "stat", "band", "mac", "disc", "mode"];
        var qs = ORDER.map(function (k) {
            return k + "=" + encodeURIComponent(params[k] == null ? "" : params[k]);
        }).join("&");
        var search = "?" + qs;
        try {
            window.history.replaceState(
                null, "", window.location.pathname + search + window.location.hash);
        } catch (e) {}
        // Sync to the sibling figure iframes (same origin) via BroadcastChannel.
        // A single shared channel instance never receives its own posts, so this
        // does not loop back into this dashboard.
        if ("BroadcastChannel" in window) {
            var bc = window.__ocbc || (window.__ocbc = new BroadcastChannel("opticnerve"));
            bc.postMessage({params: params});
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("postbridge-dummy", "data"),
    Input("subjects", "value"), Input("eyes", "value"), Input("stat", "value"),
    Input("t1band", "value"), Input("mode", "value"), Input("sel", "data"),
    State("all-subjects", "data"), State("all-eyes", "data"),
    prevent_initial_call=True,
)


# ---- dark theme (kept inline so the app stays a single file) ----
app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  html, body { height:100%; margin:0; overflow:hidden; }
  body { background:#111; color:#eee; font-family:Arial,Helvetica,sans-serif; }
  /* Dash buries our layout under several plain-block divs of its own
     (#react-entry-point > #_dash-global-error-container > div >
     #_dash-app-content > #root), so a flex chain from <body> down cannot reach
     #app: it would size to its CONTENT (the tall sidebar) instead of to the
     viewport, which is what pushed the figures off the bottom of the screen.
     position:fixed lifts #root out of that chain entirely and pins it to the
     viewport, without this file having to know Dash's internal div names. */
  #root { position:fixed; inset:0; display:flex; flex-direction:column; }
  h1 { flex:0 0 auto; text-align:center; font-size:22px; margin:14px 0 8px; }
  /* row-reverse puts the sidebar on the RIGHT while keeping the controls first
     in the DOM (so keyboard/screen-reader order stays controls -> figures). */
  #app { flex:1 1 auto; min-height:0; display:flex; flex-direction:row-reverse;
    align-items:stretch; gap:16px; padding:0 16px 16px; }
  #sidebar { width:210px; flex:0 0 210px; background:#1e1e1e; border-radius:6px;
    padding:14px; min-height:0; overflow-y:auto; }
  #sidebar h3 { margin:0 0 8px; font-size:16px; }
  #sidebar .hint { font-size:12px; color:#999; margin-bottom:10px; line-height:1.4; }
  #sidebar .lbl { font-size:14px; color:#aaa; margin-bottom:6px; }
  #sidebar hr { border:none; border-top:1px solid #333; margin:14px 0 12px; }
  .btnrow { display:flex; gap:6px; margin-bottom:8px; }
  .btnrow button { flex:1; background:#2a2a2a; color:#ddd; border:1px solid #444;
    border-radius:4px; padding:4px; font-size:12px; cursor:pointer; }
  .btnrow button:hover { background:#3a3a3a; }
  /* all sidebar OPTIONS share one font size (incl. the dropdown value + menu) */
  .subjlist label,
  #stat label, #mode label,
  #t1band, #t1band * { font-size:12px; }
  .subjlist label { display:flex; align-items:center; padding:1px 0; cursor:pointer; color:#9ecbff; }
  .subjlist input { margin-right:7px; }
  .eyelist { display:grid; grid-template-columns:1fr 1fr; column-gap:4px; }
  #stat label { display:inline-flex; align-items:center; color:#ddd; margin-right:18px; cursor:pointer; }
  #mode label { display:flex; align-items:center; color:#ddd; margin-bottom:5px; cursor:pointer; }
  #stat input, #mode input { margin-right:7px; flex-shrink:0; }
  .ncount { font-size:12px; color:#7fd17f; margin-top:8px; }
  #sidebar .dash-dropdown { color:#111; }
  /* The figure area is ONE fixed 1249x626 canvas (see the size table in
     fitStage below) that fitStage() scales to fit #stagewrap, like squeezing a
     PNG. Every length here is an authored pixel, never a fraction of the
     window, so Plotly measures once at startup and never redraws on resize. */
  #stagewrap { flex:1 1 auto; min-width:0; min-height:0; position:relative; overflow:hidden; }
  /*  575 = fig1 555 + 2x10 panel padding     658 = fig2/3 638 + 2x10
      626 = (402+20) + 16 gap + 188 table  ==  (285+20) + 16 + (285+20)
     1249 = 575 + 16 gap + 658                 <- the two columns agree exactly */
  #figures { position:absolute; top:50%; left:50%; transform-origin:center;
    width:1249px; height:626px; display:flex; gap:16px; }
  .leftcol, .rightcol { display:flex; flex-direction:column; gap:16px; }
  .figpanel { background:#fff; border-radius:6px; padding:10px;
    box-shadow:0 1px 4px rgba(0,0,0,.4); }
  /* height is authored, not measured: 188 is what the left column needs to
     match the right one (see the size table above). The contents fit, so
     overflow is hidden rather than auto — no scrollbar inside the stage. */
  .statpanel { background:#1e1e1e; box-shadow:none; padding:12px; height:188px;
    overflow:hidden; box-sizing:border-box; }
  .avgtitle { font-size:14px; color:#eee; margin-bottom:10px; }
  .avghint { font-size:11px; color:#777; }
  .avgtbl { border-collapse:collapse; width:100%; table-layout:fixed; }
  .avgtbl th { font-size:12px; font-weight:normal; color:#999; text-align:center; padding:2px 8px;
    text-transform:uppercase; letter-spacing:.4px; }
  .avgtbl th.corner { text-align:left; }
  .avgtbl td { padding:3px 8px; font-size:12px; text-align:center; }
  .avgtbl td.anm { text-align:left; color:#ddd; white-space:nowrap; }
  .avgtbl td.region { text-align:left; color:#bbb; font-size:12px; font-weight:bold;
    text-transform:uppercase; letter-spacing:.4px; vertical-align:middle; }
  .avgtbl tr.avgrow { cursor:pointer; }
  .avgtbl tr.avgrow:hover td { background:#2a2a2a; }
  .avgtbl tr.avgrow.sel td { background:#26323f; }
  .avgtbl tr.avgrow.sel td.region { background:transparent; }
  .avgtbl .aval { color:#fff; font-weight:600; font-variant-numeric:tabular-nums; }
  .avgtbl .aval.sig { color:#ff383c; }
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
  var ORDER = ["exclude","stat","band","mac","disc","mode"];
  var DEF = {exclude:"", stat:"R2m", band:"T1_mean_015",
             mac:"All_1_3_gcc", disc:"All_um_", mode:"avg"};
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
  // Fit the fixed 1249x626 stage into whatever room is left beside the sidebar,
  // the way object-fit:contain would for an <img>: one CSS transform, uncapped
  // and centred. Nothing here touches Plotly — the figures are drawn once at
  // their native pixel size (autosize=False) and are SVG, so they stay crisp at
  // any scale and never re-measure. This replaces the old per-figure resize
  // relayout, which was also the only thing Fig 3's equal-aspect subplots could
  // get stuck on.
  var NATIVE_W = 1249, NATIVE_H = 626;
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
{%config%}{%scripts%}{%renderer%}</footer></body></html>"""


if __name__ == "__main__":
    app.run(debug=True, port=3000)
