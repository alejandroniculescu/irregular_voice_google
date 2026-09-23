from irregular_voice_google.profile import Profile
from irregular_voice_google.questions import grammar, load_questions, match, prompt


def test_questions_load_values_from_lexicon():
    qs = load_questions()
    assert {"destination", "origin", "date", "airline", "confirm"} <= set(qs)
    assert "München" in qs["destination"].values  # as written, not normalized
    assert "Lufthansa" in qs["airline"].values


def test_grammar_lists_values_carriers_and_capitalised_variants():
    q = load_questions()["destination"]
    g = grammar(q)
    assert g.startswith('root ::= " " (before " ")? word (" " after)? [.!?]?')
    assert '"Hamburg"' in g and '"nach" | "Nach"' in g
    assert "day ::=" not in g
    assert "day ::=" in grammar(load_questions()["date"])


def test_prompt_ends_with_the_question():
    q = load_questions()["airline"]
    p = prompt(q, Profile(speaker="x", prompt_phrases=["Eins zwei.", "Drei vier.", "Fünf.", "Sechs."]))
    assert p.startswith("Eins zwei. Drei vier. Fünf. Lufthansa")
    assert p.endswith("Mit welcher Fluggesellschaft?")


def test_match_prefers_the_longest_value():
    q = load_questions()["confirm"]
    assert match(q, "Ja, richtig.") == "ja, richtig"
    assert match(load_questions()["destination"], "Ich möchte nach Berlin fliegen.") == "Berlin"
    assert match(load_questions()["destination"], "keine Ahnung") is None
