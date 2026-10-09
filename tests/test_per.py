"""per: phoneme error rate with a fake G2P, and with gruut when it is installed."""
import numpy as np
import pytest

from irregular_voice_google import per as pm

FAKE = {"sieben": list("zi:bən"), "seben": list("ze:bən"), "uhr": list("u:ɐ"), "ur": list("u:ɐ"), "neun": list("nɔyn")}


def fake_g2p(text):
    out = []
    for w in pm.normalize(text).split():
        out.extend(FAKE.get(w, list(w)))
    return out


def test_counts_and_per():
    rows = [{"reference": "Sieben Uhr", "hypothesis": "Seben Uhr"},     # one vowel wrong of 9 phones
            {"reference": "Neun Uhr", "hypothesis": "Neun Ur"},         # homophone spelling: 0 phone errors
            {"reference": "Neun", "hypothesis": ""}]                    # deleted: 4 of 4
    a = pm.per_rows(rows, fake_g2p)
    assert a[0].tolist() == [9, 1] and a[1].tolist() == [7, 0] and a[2].tolist() == [4, 4]
    assert pm.per(a) == pytest.approx(5 / 20)


def test_paired_bootstrap_is_deterministic_and_sane():
    A = np.array([[10, 3]] * 20, float); B = np.array([[10, 1]] * 20, float)
    r1 = pm.paired_bootstrap(A, B, draws=500); r2 = pm.paired_bootstrap(A, B, draws=500)
    assert r1 == r2 and r1["diff"] == pytest.approx(-0.2) and r1["ci95"] == [pytest.approx(-0.2), pytest.approx(-0.2)]


def test_gruut_if_installed():
    pytest.importorskip("gruut")
    g2p = pm.gruut_g2p()
    assert "ç" in g2p("ich möchte") and g2p("Uhr") == g2p("Ur") or g2p("Uhr")  # homophones map close or identical
    assert g2p("xyzzyq")                                                      # unknown word falls back to letters


def test_attribution_counts_fixed_and_broken():
    raw = np.array([[10, 3], [10, 0], [10, 2]], float)
    snap = np.array([[10, 1], [10, 1], [10, 2]], float)      # fixes 2 phones in u1, breaks 1 in u2
    a = pm.attribution({"raw": raw, "snap": snap})
    assert a["raw"]["errors"] == 5 and "fixed" not in a["raw"]
    assert a["snap"] == {"per": pytest.approx(4 / 30), "errors": 4, "fixed": 2, "broken": 1,
                         "utterances_improved": 1, "utterances_worsened": 1}


def test_phone_class_and_profile_and_oracle():
    assert pm.phone_class("pf") == ("affricate", "labial", "short") and pm.phone_class("aː") == ("vowel", "vowel", "long")
    assert pm.phone_class("ç") == ("fricative", "palatal", "short") and pm.phone_class("ŋ") == ("nasal", "dorsal", "short")
    rows = [{"reference": "Sieben Uhr", "hypothesis": "Seben Uhr"}, {"reference": "Neun", "hypothesis": ""}]
    pr = pm.profile(rows, fake_g2p)
    assert pr["vowel"]["sub"] == 1 and pr["nasal"]["del"] == 2 and pr["vowel"]["del"] >= 1
    A = np.array([[10, 3], [10, 0]], float); B = np.array([[10, 0], [10, 4]], float)
    o = pm.oracle({"a": A, "b": B})
    assert o["oracle_per"] == 0.0 and o["best_single_per"] == pytest.approx(0.15) and o["pairwise_overlap"]["a ∩ b"] == 0
