from pathlib import Path
import pandas as pd
from scipy.stats import linregress
from opticnerve_core import DEFAULTS, resolve_view


def _independent_ols_r2(sector, band):
    # avg-mode OLS on subject means, computed straight from the CSV, independent
    # of opticnerve_core — a reproducible anchor that survives analysis revisions.
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "data" / "data_merged.csv")
    d = (df.groupby("MRI_ID", as_index=False)[[sector, band]].mean()
           .dropna(subset=[sector, band]))
    return linregress(d[sector], d[band]).rvalue ** 2


def test_default_view_counts_all_subjects():
    v = resolve_view(**DEFAULTS)
    assert v["n"] == len(v["subjects"]) > 0
    assert v["params"]["mode"] == "avg"


def test_default_macula_matches_independent_ols():
    v = resolve_view(**DEFAULTS)
    r = v["mac_fits"]["All_1_3_gcc"]
    assert abs(r["R2"] - _independent_ols_r2("All_1_3_gcc", "T1_mean_015")) < 1e-9


def test_default_disc_matches_independent_ols():
    v = resolve_view(**DEFAULTS)
    r = v["disc_fits"]["All_um_"]
    assert abs(r["R2"] - _independent_ols_r2("All_um_", "T1_mean_015")) < 1e-9


def test_default_fits_are_valid():
    # analysis-agnostic sanity: fits exist and R² is a proper fraction
    v = resolve_view(**DEFAULTS)
    for r in (v["mac_fits"]["All_1_3_gcc"], v["disc_fits"]["All_um_"]):
        assert r is not None and 0.0 <= r["R2"] <= 1.0


def test_excluding_a_subject_changes_n():
    v0 = resolve_view(**DEFAULTS)
    some = v0["subjects"][0]
    v1 = resolve_view(**{**DEFAULTS, "exclude": (some,)})
    assert v1["n"] == v0["n"] - 1


def test_lme_mode_produces_a_fit():
    v = resolve_view(**{**DEFAULTS, "mode": "lme"})
    assert v["mac_fits"]["All_1_3_gcc"] is not None


def test_avg_fits_populated_for_displayed_metric():
    v = resolve_view(**DEFAULTS)
    r = v["avg_fits"]["T1_mean_015"]["mac"]["All_1_3_gcc"]
    assert isinstance(r, dict) and "R2" in r
