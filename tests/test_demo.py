from irregular_voice_google import questions
from irregular_voice_google.demo import MAX_TRIES, Dialog
from irregular_voice_google.questions import Decision


def dialog():
    return Dialog(questions.load_questions())


def test_accepts_answers_then_reads_back_and_books():
    d = dialog()
    step = d.start()
    assert step["listen"] == "destination" and step["asking"] == "destination"
    for value in ["Wien", "München", "Mai", "Lufthansa"]:
        step = d.answer(Decision("accept", value))
    assert step["listen"] == "choice" and step["phase"] == "final"
    assert step["say"] == "Ein Flug von München nach Wien, am Mai, mit Lufthansa. Ist das richtig?"
    step = d.answer(Decision("accept", "ja"))
    assert step["listen"] is None and step["confirmed"] is True
    assert step["say"] == "Vielen Dank, Ihr Flug ist gebucht."


def test_confirm_yes_accepts_and_no_asks_again():
    d = dialog()
    d.start()
    step = d.answer(Decision("confirm", "Wien"))
    assert step["say"] == "Meinten Sie Wien?" and step["listen"] == "choice" and step["asking"] == "destination"
    step = d.answer(Decision("accept", "nein"))
    assert step["say"] == "Entschuldigung, bitte noch einmal." and step["listen"] == "destination"
    d.answer(Decision("confirm", "Wien"))
    step = d.answer(Decision("confirm", "ja"))
    assert step["booking"]["destination"] == "Wien" and step["listen"] == "origin"


def test_gives_up_after_max_tries_and_final_no_is_not_booked():
    d = dialog()
    d.start()
    for _ in range(MAX_TRIES):
        step = d.answer(Decision("repeat"))
    assert step["booking"]["destination"] is None and step["listen"] == "origin"
    assert step["say"].startswith("Entschuldigung, das überspringen wir.")
    for value in ["München", "Mai", "Lufthansa"]:
        step = d.answer(Decision("accept", value))
    assert "nach ?" in step["say"]
    step = d.answer(Decision("repeat"))
    assert step["listen"] is None and step["confirmed"] is False


def test_offers_up_to_three_choices_picked_by_number_or_name():
    d = dialog()
    d.start()
    step = d.answer(Decision("choose", None, ["Berlin", "Bern", "Bremen"]))
    assert step["say"] == "Meinten Sie eins: Berlin, zwei: Bern, oder drei: Bremen?"
    assert step["choices"] == ["Berlin", "Bern", "Bremen"] and step["listen"] == "choice"
    q = d.question("choice")
    step = d.answer(questions.resolve(q, "Die Zwei.", min_p=0.9))
    assert step["booking"]["destination"] == "Bern" and step["choices"] == []
    d.answer(Decision("choose", None, ["Hamburg", "Homburg"]))
    step = d.answer(questions.resolve(d.question("choice"), "Hamburg bitte", min_p=0.9))
    assert step["booking"]["origin"] == "Hamburg"


def test_ja_only_picks_a_single_choice_and_nein_asks_again():
    d = dialog()
    d.start()
    d.answer(Decision("choose", None, ["Berlin", "Bern"]))
    step = d.answer(questions.resolve(d.question("choice"), "Ja", min_p=0.9))
    assert step["say"] == "Entschuldigung, bitte noch einmal." and step["booking"]["destination"] is None
    d.answer(Decision("confirm", "Berlin"))
    step = d.answer(questions.resolve(d.question("choice"), "Keins davon", min_p=0.9))
    assert step["listen"] == "destination"


def test_taps_pick_a_choice_none_of_these_and_book():
    d = dialog()
    d.start()
    d.answer(Decision("choose", None, ["Berlin", "Bern"]))
    step = d.choose(None)
    assert step["listen"] == "destination" and step["say"] == "Entschuldigung, bitte noch einmal."
    d.answer(Decision("choose", None, ["Berlin", "Bern"]))
    step = d.choose("Berlin")
    assert step["booking"]["destination"] == "Berlin"
    for value in ["München", "Mai", "Lufthansa"]:
        step = d.answer(Decision("accept", value))
    assert step["phase"] == "final"
    step = d.choose("ja")
    assert step["confirmed"] is True and step["listen"] is None
