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

from dash import Dash, dcc, html, Input, Output, State, ctx, ALL

from opticnerve_core import (
    T1_BANDS, DEF_MAC, DEF_DISC, MAC_AVG, DISC_AVG, SUBJECTS, stat_val,
    stat_lbl, resolve_view, DEFAULTS, build_fig1, build_fig2, build_fig3,
    parse_params, serialize_params,
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

GRAPH_CFG = lambda name, w, h: dict(
    scrollZoom=False, displaylogo=False, displayModeBar="hover", responsive=True,
    modeBarButtonsToRemove=["select2d", "lasso2d", "zoom2d", "pan2d", "zoomIn2d",
                            "zoomOut2d", "autoScale2d", "resetScale2d"],
    toImageButtonOptions=dict(format="png", filename=name, width=w, height=h, scale=600 / 96))

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="postbridge-dummy"),
    html.H1(["OCT – T", html.Sub("1"), " Correlation Figures"]),
    dcc.Store(id="sel", data={"mac": DEF_MAC, "disc": DEF_DISC}),
    html.Div(id="app", children=[
        # ---- sidebar ----
        html.Aside(id="sidebar", children=[
            html.H3("Subjects"),
            html.Div("Uncheck a subject to remove it from every figure.", className="hint"),
            html.Div([html.Button("All", id="sel-all"), html.Button("None", id="sel-none")],
                     className="btnrow"),
            dcc.Checklist(id="subjects", options=[{"label": s, "value": s} for s in SUBJECTS],
                          value=SUBJECTS, className="subjlist"),
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
        ]),
        # ---- figures ----
        html.Div(id="figures", children=[
            html.Div(className="leftcol", children=[
                html.Section(className="figpanel", children=[
                    dcc.Graph(id="fig1", style={"height": "100%"},
                              config=GRAPH_CFG("fig01_T1_profile", 864, 360))]),
                html.Section(className="figpanel statpanel", children=[
                    html.Div(id="avgbox")]),
            ]),
            html.Section(className="figpanel fig2", children=[
                dcc.Graph(id="fig2", style={"height": "100%"},
                          config=GRAPH_CFG("fig02_regression", 936, 408))]),
            html.Section(className="figpanel fig3", children=[
                dcc.Graph(id="fig3", style={"height": "100%"},
                          config=GRAPH_CFG("fig03_OCT_maps", 816, 408))]),
        ]),
    ]),
])


from urllib.parse import parse_qsl


def _qs_to_dict(search):
    return dict(parse_qsl((search or "").lstrip("?")))


# ---- URL (on load) and All/None buttons set the controls ----
@app.callback(
    Output("subjects", "value"), Output("stat", "value"), Output("t1band", "value"),
    Output("mode", "value"), Output("sel", "data"),
    Input("url", "search"), Input("sel-all", "n_clicks"), Input("sel-none", "n_clicks"),
    State("stat", "value"), State("t1band", "value"), State("mode", "value"),
    State("sel", "data"), prevent_initial_call=False)
def _from_url(search, _a, _n, stat, band, mode, sel):
    trig = ctx.triggered_id
    if trig == "sel-all":
        return SUBJECTS, stat, band, mode, sel
    if trig == "sel-none":
        return [], stat, band, mode, sel
    p = parse_params(_qs_to_dict(search))
    included = [s for s in SUBJECTS if s not in p["exclude"]]
    return included, p["stat"], p["band"], p["mode"], {"mac": p["mac"], "disc": p["disc"]}


# ---- every control change rewrites the URL query string ----
@app.callback(Output("url", "search"),
              Input("subjects", "value"), Input("stat", "value"),
              Input("t1band", "value"), Input("mode", "value"), Input("sel", "data"))
def _to_url(included, stat, band, mode, sel):
    excluded = tuple(sorted(set(SUBJECTS) - set(included or [])))
    return "?" + serialize_params({"exclude": excluded, "stat": stat, "band": band,
                                   "mac": sel["mac"], "disc": sel["disc"], "mode": mode})


# ---- clicking a Fig 3 wedge or an averages row selects the Fig 2 sector ----
# sel.data is also written on load by _from_url (the canonical writer), so this
# click-driven writer is a duplicate output -> allow_duplicate=True (requires
# prevent_initial_call=True, already set).
@app.callback(Output("sel", "data", allow_duplicate=True),
              Input("fig3", "clickData"),
              Input({"type": "avgrow", "metric": ALL, "region": ALL}, "n_clicks"),
              State("sel", "data"), prevent_initial_call=True)
def _select_sector(click, _rows, sel):
    sel = dict(sel)
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
        metric, region = trig["metric"], trig["region"]
    if metric:                                   # toggle: click the selected one to reset
        default = DEF_MAC if region == "mac" else DEF_DISC
        sel[region] = default if sel[region] == metric else metric
    return sel


# ---- main render ----
@app.callback(
    Output("fig1", "figure"), Output("fig2", "figure"), Output("fig3", "figure"),
    Output("avgbox", "children"), Output("ncount", "children"),
    Input("subjects", "value"), Input("stat", "value"), Input("t1band", "value"),
    Input("mode", "value"), Input("sel", "data"))
def _render(included, stat, sel_band, mode, sel):
    included = included or []
    excluded = tuple(sorted(set(SUBJECTS) - set(included)))
    view = resolve_view(exclude=excluded, stat=stat, band=sel_band,
                        mac=sel["mac"], disc=sel["disc"], mode=mode)
    count = f"{view['n']} / {len(SUBJECTS)} subjects included"
    return (build_fig1(view), build_fig2(view), build_fig3(view),
            build_avg_table(view), count)


# ---- postMessage bridge: emit this app's URL to the parent window (iframe host) ----
# NOTE (deviation from task-7 brief): the brief's Step 1 snippet outputs to
# Output("url", "search"), the same prop that `_to_url` above already writes.
# Two callbacks writing the same Output raises DuplicateCallbackOutput, and
# having url.search as BOTH the Input and Output of one callback additionally
# risks a same-input-output error. This callback instead only *reads*
# url.search (Input) and writes to a dedicated dummy dcc.Store
# ("postbridge-dummy") that nothing else uses, returning no_update.
app.clientside_callback(
    """
    function(search) {
        if (window.parent && window.parent !== window) {
            window.parent.postMessage(
                {type: "opticnerve:state", search: search || ""}, "*");
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("postbridge-dummy", "data"),
    Input("url", "search"),
)


# ---- dark theme (kept inline so the app stays a single file) ----
app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body { margin:0; background:#111; color:#eee; font-family:Arial,Helvetica,sans-serif; }
  h1 { text-align:center; font-size:22px; margin:14px 0 8px; }
  #app { display:flex; align-items:stretch; gap:16px; padding:0 16px 16px; height:calc(100vh - 60px); }
  #sidebar { width:210px; flex:0 0 210px; background:#1e1e1e; border-radius:6px;
    padding:14px; overflow-y:auto; }
  #sidebar h3 { margin:0 0 8px; font-size:14px; }
  #sidebar .hint { font-size:11px; color:#999; margin-bottom:10px; line-height:1.4; }
  #sidebar .lbl { font-size:12px; color:#aaa; margin-bottom:6px; }
  #sidebar hr { border:none; border-top:1px solid #333; margin:14px 0 12px; }
  .btnrow { display:flex; gap:6px; margin-bottom:8px; }
  .btnrow button { flex:1; background:#2a2a2a; color:#ddd; border:1px solid #444;
    border-radius:4px; padding:4px; font-size:11px; cursor:pointer; }
  .btnrow button:hover { background:#3a3a3a; }
  /* all sidebar OPTIONS share one font size (incl. the dropdown value + menu) */
  .subjlist label,
  #stat label, #mode label,
  #t1band .Select-control, #t1band .Select-value-label,
  #t1band .Select-placeholder, #t1band .Select-menu-outer,
  #t1band .Select-option, #t1band .VirtualizedSelectOption { font-size:12px; }
  .subjlist label { display:flex; align-items:center; padding:1px 0; cursor:pointer; color:#9ecbff; }
  .subjlist input { margin-right:7px; }
  #stat label { display:inline-block; color:#ddd; margin-right:18px; cursor:pointer; }
  #mode label { display:block; color:#ddd; margin-bottom:5px; cursor:pointer; }
  #stat input, #mode input { margin-right:7px; }
  .ncount { font-size:11px; color:#7fd17f; margin-top:8px; }
  #sidebar .dash-dropdown { color:#111; }
  #figures { flex:1 1 auto; min-width:0; display:grid; gap:16px;
    grid-template-columns:7fr 8fr; grid-template-rows:1fr 1fr;
    grid-template-areas:"f1 f2" "f1 f3"; }
  .leftcol { grid-area:f1; display:flex; flex-direction:column; gap:16px; min-height:0; }
  .leftcol .figpanel:first-child { flex:1 1 auto; min-height:0; }
  .fig2 { grid-area:f2; } .fig3 { grid-area:f3; }
  .figpanel { background:#fff; border-radius:6px; padding:10px; min-width:0; min-height:0;
    box-shadow:0 1px 4px rgba(0,0,0,.4); }
  .statpanel { background:#1e1e1e; box-shadow:none; padding:12px; overflow:auto; flex:0 0 auto; }
  .avgtitle { font-size:14px; color:#eee; margin-bottom:10px; }
  .avghint { font-size:11px; color:#777; }
  .avgtbl { border-collapse:collapse; width:100%; table-layout:fixed; }
  .avgtbl th { font-size:12px; font-weight:normal; color:#999; text-align:center; padding:2px 8px;
    text-transform:uppercase; letter-spacing:.4px; }
  .avgtbl th.corner { text-align:left; }
  .avgtbl td { padding:3px 8px; font-size:12px; text-align:center; }
  .avgtbl td.anm { text-align:left; color:#ddd; }
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
  // Compare the 6 params SEMANTICALLY (not raw string) so an encoding
  // difference between JS serialize() (encodeURIComponent) and Python
  // serialize_params() (raw) doesn't cause a spurious extra round-trip.
  var ORDER = ["exclude","stat","band","mac","disc","mode"];
  var DEF = {exclude:"", stat:"R2m", band:"T1_mean_015",
             mac:"All_1_3_gcc", disc:"All_um_", mode:"avg"};
  function canon(search) {
    var q = new URLSearchParams((search || "").replace(/^[?]/, ""));
    return ORDER.map(function (k) {
      return k + "=" + (q.has(k) ? q.get(k) : DEF[k]);
    }).join("&");
  }
  window.addEventListener("message", function (e) {
    var d = e.data || {};
    if (d.type !== "opticnerve:state" || typeof d.search !== "string") return;
    // reflect into this app's URL so the normal _from_url callback runs
    if (canon(d.search) !== canon(window.location.search)) {
      window.history.replaceState(null, "", window.location.pathname + d.search);
      window.dispatchEvent(new PopStateEvent("popstate"));
    }
  });
})();
</script>
{%config%}{%scripts%}{%renderer%}</footer></body></html>"""


if __name__ == "__main__":
    app.run(debug=True, port=3000)
