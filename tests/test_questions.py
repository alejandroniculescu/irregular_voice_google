from irregular_voice_google.profile import Profile
from irregular_voice_google.questions import Question, grammar, load_questions, match, prompt, resolve


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


def test_resolve_accepts_confident_exact_values_only():
    q = load_questions()["destination"]
    assert resolve(q, "Nach Berlin bitte.", min_p=0.9).action == "accept"
    assert resolve(q, "Nach Berlin bitte.", min_p=0.2).action == "confirm"
    assert resolve(q, "Nach Berlin bitte.").value == "Berlin"
    assert resolve(q, "Nach Berlin bitte.", min_p=0.9, flagged=True).action == "repeat"


def test_resolve_confirms_a_sound_alike():
    q = Question("n", "Welche Zahl?", ["sieben", "acht", "neun"])
    d = resolve(q, "Seben.", min_p=0.9)
    assert (d.action, d.value) == ("confirm", "sieben")
    assert resolve(q, "keine Ahnung").action == "repeat"
    q = load_questions()["airline"]
    assert resolve(q, "Mit Luftansa.").value == "Lufthansa"


def test_resolve_repeats_when_sound_alikes_are_ambiguous():
    q = Question("s", "Was?", ["einschalten", "ausschalten", "aushalten"])
    d = resolve(q, "Ausalten")
    assert d.action == "repeat" and set(d.candidates) == {"ausschalten", "aushalten"}
