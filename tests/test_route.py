import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "content"))

import Dash_client  # noqa: E402


client = Dash_client.server.test_client()


def test_fig2_route_returns_plotly_spec():
    r = client.get("/opticnerve/fig2?exclude=&stat=R2m&band=T1_mean_015"
                   "&mac=All_1_3_gcc&disc=All_um_&mode=avg")
    assert r.status_code == 200
    body = r.get_json()
    assert body["figid"] == "fig2"
    assert "data" in body["figure"] and "layout" in body["figure"]
    assert body["params"]["mode"] == "avg"


def test_all_route_bundles_three_figures():
    r = client.get("/opticnerve/all")
    body = r.get_json()
    assert set(("fig1", "fig2", "fig3")).issubset(body)


def test_bad_params_clamp_to_defaults_no_500():
    r = client.get("/opticnerve/fig1?stat=bogus&band=bogus&mode=bogus")
    assert r.status_code == 200
    assert r.get_json()["params"]["stat"] == "R2m"


def test_bad_mac_disc_clamp_to_defaults_no_500():
    # a bad sector metric must not reach fit()/groupby (KeyError -> 500); it clamps.
    for fid in ("fig2", "fig3"):
        r = client.get(f"/opticnerve/{fid}?mac=BOGUS&disc=NOPE")
        assert r.status_code == 200, f"{fid} 500'd on bad mac/disc"
        p = r.get_json()["params"]
        assert p["mac"] == "All_1_3_gcc" and p["disc"] == "All_um_"


def test_all_route_returns_200():
    assert client.get("/opticnerve/all").status_code == 200


def test_unknown_figid_is_404():
    assert client.get("/opticnerve/figZ").status_code == 404
