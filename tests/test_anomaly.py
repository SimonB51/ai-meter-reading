from meter_reader.anomaly import check


def hist(*vals):  # newest first
    return [{"value": v} for v in vals]


def test_first_reading_has_no_consumption():
    assert check("123", []) == (None, None)


def test_normal_consumption():
    delta, flag = check("130", hist("120", "110", "100", "090"))
    assert delta == 10 and flag is None


def test_spike_is_flagged():
    delta, flag = check("300", hist("120", "110", "100", "090"))
    assert flag == "unusually_high_consumption"


def test_rollover_modulo_1000():
    delta, flag = check("005", hist("995", "985", "975", "965"))
    assert delta == 10 and flag is None


def test_backwards_is_flagged():
    _, flag = check("100", hist("120"))
    assert flag == "meter_went_backwards_or_misread"


def test_human_correction_takes_priority():
    delta, _ = check("130", [{"value": "999", "corrected_value": "120"}])
    assert delta == 10
