"""Shared source of truth for the OCT–T1 figures: params, stats, geometry,
resolve_view, and the Plotly builders. Imported by the Dash app, the Flask
route, and the three notebooks."""

# ---- state model -----------------------------------------------------------
DEFAULTS = {"exclude": (), "stat": "R2m", "band": "T1_mean_015",
            "mac": "All_1_3_gcc", "disc": "All_um_", "mode": "avg"}

_STATS = ("R2m", "Rm")
_BANDS = ("T1_mean_015", "T1_mean_05", "T1_mean_510", "T1_mean_1015")
_MODES = ("avg", "lme")
_ORDER = ("exclude", "stat", "band", "mac", "disc", "mode")


def parse_params(mapping):
    """Normalize a dict-like of raw query values into the 6 canonical params."""
    g = mapping.get
    exclude = g("exclude", "") or ""
    exclude = tuple(s for s in (x.strip() for x in exclude.split(",")) if s)
    pick = lambda key, allowed: (g(key) if g(key) in allowed else DEFAULTS[key])
    return {
        "exclude": exclude,
        "stat": pick("stat", _STATS),
        "band": pick("band", _BANDS),
        "mac": g("mac") or DEFAULTS["mac"],
        "disc": g("disc") or DEFAULTS["disc"],
        "mode": pick("mode", _MODES),
    }


def serialize_params(params):
    """Canonical query string, fixed key order, exclude joined by commas."""
    parts = []
    for k in _ORDER:
        v = params[k]
        v = ",".join(v) if k == "exclude" else v
        parts.append(f"{k}={v}")
    return "&".join(parts)
