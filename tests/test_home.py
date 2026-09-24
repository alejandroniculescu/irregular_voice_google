from irregular_voice_google.demo import MAX_TRIES
from irregular_voice_google.home import HomeDialog


def say(d, step, text, p=0.9):
    """What the ear does with one take: the dialog decides, then answers."""
    return d.answer(d.decide(d.question(step["listen"]), text, p, 0.5, False))


def test_parses_several_commands_with_synonyms_and_carried_parts():
    d = HomeDialog()
    assert d.parse("Licht im Flur an, Büro dunkel.") == [
        {"device": "Licht", "room": "Flur", "action": "an"}, {"device": "Licht", "room": "Büro", "action": "aus"}]
    assert d.parse("Licht im Flur, Küche an") == [
        {"device": "Licht", "room": "Flur", "action": "an"}, {"device": "Licht", "room": "Küche", "action": "an"}]
    assert d.parse("Stehlampe start") == [{"device": "Lampe", "room": None, "action": "an"}]
    assert d.parse("Buche einen Flug nach Berlin.") == []
    assert d.parse("Ah, warte, jetzt hat es zu.") == []  # an action alone is no command


def test_whole_sentence_runs_and_updates_the_house():
    d = HomeDialog()
    step = d.start()
    assert step["listen"] == "command" and step["house"] == {}
    step = say(d, step, "Licht im Flur an, Büro dunkel.")
    assert step["say"] == "Okay: Licht im Flur an. Licht im Büro aus. Was noch?"
    assert step["house"] == {"Flur": {"Licht": "an"}, "Büro": {"Licht": "aus"}}
    assert step["listen"] == "command"


def test_missing_action_is_offered_as_choices_for_the_device():
    d = HomeDialog()
    step = say(d, d.start(), "Alle Lichter.")
    assert step["choices"] == ["an", "aus"] and step["listen"] == "choice" and step["asking"] == "action"
    assert step["say"].startswith("Licht überall:")
    step = say(d, step, "zwei")
    assert step["house"] == {"überall": {"Licht": "aus"}}


def test_missing_room_is_asked_and_a_tap_works_too():
    d = HomeDialog()
    step = say(d, d.start(), "Rollo runter")
    assert step["say"] == "Rollo runter: In welchem Raum?" and step["listen"] == "room"
    step = say(d, step, "im Schlafzimmer")
    assert step["house"] == {"Schlafzimmer": {"Rollo": "runter"}}
    step = say(d, step, "Heizung im Bad")
    step = d.choose("wärmer")
    assert step["house"]["Bad"] == {"Heizung": "wärmer"}


def test_unsure_sentence_is_read_back_and_nein_drops_it():
    d = HomeDialog()
    step = say(d, d.start(), "Licht im Flur an", p=0.2)
    assert step["say"] == "Meinten Sie Licht im Flur an?" and step["listen"] == "choice"
    step = say(d, step, "nein")
    assert step["house"] == {} and step["listen"] == "command"
    step = say(d, step, "Licht im Flur an", p=0.2)
    step = say(d, step, "ja")
    assert step["house"] == {"Flur": {"Licht": "an"}}


def test_unrelated_speech_is_asked_again_then_dropped():
    d = HomeDialog()
    step = d.start()
    for _ in range(MAX_TRIES):
        step = say(d, step, "Buche einen Flug nach Berlin.")
    assert step["say"].startswith("Das lassen wir.") and step["house"] == {}
