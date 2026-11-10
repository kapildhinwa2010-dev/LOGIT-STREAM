import numpy as np
import pytest

from logit_stream.stream import birth_certificate, layer_contributions

V = 10


def make_prefix_logits():
    p0 = np.zeros(V)
    p0[2] = 10.0
    p1 = p0.copy()
    p1[3] += 1.0
    p2 = p1.copy()
    p2[7] += 12.0
    p3 = p2.copy()
    p3[7] += 2.0
    return [p0, p1, p2, p3]


def test_layer_contributions_telescope():
    prefixes = make_prefix_logits()
    r = layer_contributions(prefixes, target=7)
    assert r["num_layers"] == 4
    assert r["final_logit"] == pytest.approx(14.0)
    assert r["sum_check"] == pytest.approx(r["final_logit"])
    assert sum(r["layer_increments"]) == pytest.approx(14.0)
    assert r["layer_increments"] == [0.0, 0.0, 12.0, 2.0]


def test_birth_layer():
    prefixes = make_prefix_logits()
    cert = birth_certificate(prefixes)
    assert cert["answer"] == 7
    assert cert["birth_layer"] == 2
    assert cert["layer_increments"][2] > 5.0
    assert abs(sum(cert["positive_share"]) - 1.0) < 1e-9
    assert cert["positive_share"][2] == pytest.approx(12.0 / 14.0)


def test_early_leader_is_not_answer():
    prefixes = make_prefix_logits()
    cert = birth_certificate(prefixes, target=2)
    assert cert["answer"] == 7
    assert cert["birth_layer"] == 2
    # target 2 had a strong early lead but loses ground
    assert cert["layer_increments"][0] == pytest.approx(10.0)


def test_target_out_of_range():
    prefixes = make_prefix_logits()
    with pytest.raises(ValueError):
        layer_contributions(prefixes, target=99)
    with pytest.raises(ValueError):
        birth_certificate(prefixes, target=-1)


def test_bad_shapes():
    with pytest.raises(ValueError):
        layer_contributions([], target=0)
    with pytest.raises(ValueError):
        layer_contributions([np.zeros(V), np.zeros(5)], target=0)
    with pytest.raises(ValueError):
        layer_contributions([np.zeros((2, 2))], target=0)


def test_all_negative_increments_share():
    decreasing = [np.zeros(V), -np.ones(V), -2 * np.ones(V)]
    cert = birth_certificate(decreasing, target=1)
    assert cert["positive_share"] == [0.0, 0.0, 0.0]
    assert cert["birth_layer"] is not None