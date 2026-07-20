from Dash_client import _qs_to_dict


def test_qs_to_dict_parses_query_string():
    assert _qs_to_dict("?stat=Rm&band=T1_mean_05") == {"stat": "Rm", "band": "T1_mean_05"}


def test_qs_to_dict_handles_missing_leading_question_mark():
    assert _qs_to_dict("stat=Rm") == {"stat": "Rm"}


def test_qs_to_dict_handles_empty_and_none():
    assert _qs_to_dict("") == {}
    assert _qs_to_dict(None) == {}


def test_qs_to_dict_drops_blank_values():
    # parse_qsl's default keep_blank_values=False drops empty params (e.g. exclude=);
    # parse_params() still treats a missing "exclude" key the same as an empty one.
    assert _qs_to_dict("?exclude=&stat=R2m") == {"stat": "R2m"}
