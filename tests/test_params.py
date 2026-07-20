from opticnerve_core import DEFAULTS, parse_params, serialize_params


def test_empty_mapping_returns_defaults():
    assert parse_params({}) == DEFAULTS


def test_blank_exclude_is_empty_tuple():
    assert parse_params({"exclude": ""})["exclude"] == ()


def test_exclude_parsed_to_tuple():
    assert parse_params({"exclude": "Sub-0610,Sub-0611"})["exclude"] == ("Sub-0610", "Sub-0611")


def test_unknown_stat_falls_back_to_default():
    assert parse_params({"stat": "bogus"})["stat"] == "R2m"


def test_unknown_band_and_mode_fall_back():
    p = parse_params({"band": "nope", "mode": "nope"})
    assert p["band"] == "T1_mean_015"
    assert p["mode"] == "avg"


def test_valid_values_pass_through():
    p = parse_params({"stat": "Rm", "band": "T1_mean_05", "mode": "lme",
                      "mac": "N_1_3_gcc", "disc": "TS_um_"})
    assert (p["stat"], p["band"], p["mode"], p["mac"], p["disc"]) == \
           ("Rm", "T1_mean_05", "lme", "N_1_3_gcc", "TS_um_")


def test_serialize_is_canonical_and_ordered():
    qs = serialize_params(DEFAULTS)
    assert qs == "exclude=&stat=R2m&band=T1_mean_015&mac=All_1_3_gcc&disc=All_um_&mode=avg"


def test_roundtrip():
    p = {"exclude": ("Sub-0610",), "stat": "Rm", "band": "T1_mean_510",
         "mac": "S_3_6_gcc", "disc": "IT_um_", "mode": "lme"}
    assert parse_params(dict(zip(
        ["exclude", "stat", "band", "mac", "disc", "mode"],
        ["Sub-0610", "Rm", "T1_mean_510", "S_3_6_gcc", "IT_um_", "lme"]))) == p
