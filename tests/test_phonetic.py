from irregular_voice_google.phonetic import code


def test_reference_codes():
    assert code("Wikipedia") == "3412"
    assert code("Müller-Lüdenscheidt") == "657 52682"
    assert code("Xaver") == "4837"


def test_near_misses_share_a_code():
    assert code("Seben") == code("Sieben") == "816"
    assert code("Aushalten") == code("Ausschalten")
    assert code("Luftansa") == code("Lufthansa")
    assert code("Berlin") != code("Bremen")
