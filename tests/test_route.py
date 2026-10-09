from sonic.route import medoid, rover, switch


def letters(text):
    return list(text.replace(" ", ""))


def test_switch_keeps_pivot_without_agreement():
    assert switch(["a b c", "a b d", "a x c"]) == 0


def test_switch_follows_two_agreeing_others():
    assert switch(["a b c", "a b d", "a b d"]) == 1


def test_rover_majority_and_tie_to_pivot():
    assert rover(["haus baum katze", "haus maus katze", "haus maus katze"]) == "haus maus katze"
    assert rover(["haus baum katze", "haus maus katze"]) == "haus baum katze"


def test_rover_drops_majority_deletion():
    assert rover(["haus baum katze", "haus katze", "haus katze"]) == "haus katze"


def test_medoid_picks_consensus_and_ties_to_pivot():
    assert medoid(["zzz", "abc", "abd", "abc"], letters) == 1
    assert medoid(["abc", "abd"], letters) == 0
