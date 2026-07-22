"""Self-check for point-level exclusion (subject tokens vs eye tokens).

Run:  python test_exclude.py
"""
import numpy as np

from opticnerve_core import (ALL_EYES, DEF_MAC, DEFAULTS, EYES_OF, EYE_SEP, SUBJECTS,
                             build_fig1, build_fig2, fit, parse_params, resolve_view,
                             serialize_params, split_excluded)

band = DEFAULTS["band"]
s0 = fit((), DEF_MAC, band, "avg")["subj"][0]   # a subject that is actually plotted
eye0 = f"{s0}{EYE_SEP}{EYES_OF[s0][0]}"

# ---- tokens -----------------------------------------------------------------
p = parse_params({"exclude": f"{s0},{eye0},bogus,also.XX"})
assert p["exclude"] == tuple(sorted({s0, eye0})), p["exclude"]     # junk dropped
gone, out = split_excluded((s0,))
assert gone == {s0} and out == {f"{s0}{EYE_SEP}{e}" for e in EYES_OF[s0]}
assert split_excluded((eye0,))[0] == set()                        # one eye != subject out
assert f"exclude={s0},{eye0}" in serialize_params(p)               # URL round-trips both

# ---- the fits actually lose the points --------------------------------------
base = fit((), DEF_MAC, band, "avg")
one = fit((s0,), DEF_MAC, band, "avg")
assert len(one["x"]) == len(base["x"]) - 1
assert one["gsubj"] == (s0,) and len(one["gx"]) == 1               # ghost is drawn
assert base["gx"] == ()

lme = fit((eye0,), DEF_MAC, band, "lme")
assert len(lme["x"]) == len(fit((), DEF_MAC, band, "lme")["x"]) - 1
assert lme["gsubj"] == (s0,) and lme["geye"] == (EYES_OF[s0][0],)
assert np.isfinite(lme["gy"]).all()                               # marginal y is usable

# ---- figures build, ghosts carry a click token ------------------------------
v = resolve_view(exclude=(eye0,), mode="lme")
assert v["n"] == len(SUBJECTS) and v["n_eyes"] == len(ALL_EYES) - 1
f2 = build_fig2(v)
toks = [tuple(r) for t in f2.data if t.customdata is not None for r in t.customdata]
assert (s0, eye0, 1) in toks, "ghost point must be clickable back in"
assert any(t[2] == 0 for t in toks)
build_fig1(v)                                                     # per-eye profile filter
assert build_fig1(resolve_view(exclude=(s0,)))                    # whole subject out

print("ok")
