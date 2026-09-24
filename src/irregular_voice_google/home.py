"""Home control by voice: "Licht im Flur an", "Rollo im Büro runter".

Simpler than booking a flight: one sentence does one thing, and the result
shows at once. A command has three parts (device, room, action), each a list
of values with synonyms in ``resources/home_de.json`` ("dunkel" means "aus").

- One sentence can hold several commands: "Licht im Flur an, Büro dunkel"
  switches the hall light on and the office light off (a part left out is
  taken from the command before it).
- Only words that are spelled out count when a whole sentence is read, so an
  unrelated sentence does nothing and is asked again. When a part is missing,
  only that part is asked for ("In welchem Raum?"), and there the sound-alikes
  and near misses from ``questions.resolve`` apply, as in the flight dialog.
- A missing action is offered as choices fitting the device ("eins: an, oder
  zwei: aus?"), so a single word, a number or a tap is enough.
- If Whisper was unsure of the sentence, the commands are read back first
  ("Meinten Sie Licht im Flur an?").
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from irregular_voice_google import questions
from irregular_voice_google.demo import MAX_TRIES, Dialog
from irregular_voice_google.questions import Decision, Question

PARTS = ["device", "room", "action"]
ACTIONS = {"Licht": ["an", "aus"], "Lampe": ["an", "aus"], "Heizung": ["wärmer", "kälter", "aus"],
           "Rollo": ["hoch", "runter"], "Musik": ["an", "aus", "leiser"], "Fernseher": ["an", "aus", "leiser"]}
IN = {"Küche": "in der Küche", "überall": "überall", "Garten": "im Garten"}
ASK = "Was soll ich tun?"


def load_home(path: str | Path = "resources/home_de.json") -> tuple[dict[str, Question], dict[str, str]]:
    """The three parts as questions (values plus synonyms) and the synonym -> value map."""
    qs, canon = {}, {}
    for name, spec in json.loads(Path(path).read_text(encoding="utf-8")).items():
        canon.update(spec.get("aliases", {}))
        qs[name] = Question(name, spec["ask"], [*spec["values"], *spec.get("aliases", {})],
                            spec.get("before", []), spec.get("after", []))
    return qs, canon


def phrase(cmd: dict) -> str:
    """ "Licht im Flur an" """
    room = cmd["room"] and IN.get(cmd["room"], f"im {cmd['room']}")
    return " ".join(p for p in (cmd["device"], room, cmd["action"]) if p)


class HomeDialog(Dialog):
    """Commands one after another; ``house`` holds each device's last state per room."""

    title = "Zuhause per Stimme"
    fields = [("device", "Gerät"), ("room", "Raum"), ("action", "Aktion")]

    def __init__(self, path: str | Path = "resources/home_de.json"):
        self.parts, self.canon = load_home(path)
        super().__init__(self.parts)
        self.house: dict[str, dict[str, str]] = {}
        self.todo: list[dict] = []  # commands heard but not done yet; the first may lack a part
        self.slot: str | None = None  # the part asked or offered, None for the whole sentence
        self.log: list[dict] = []

    # --- what the page and the ear see ---
    def step(self, say: str) -> dict:
        cmd = self.todo[0] if self.todo else dict.fromkeys(PARTS)
        listen = "choice" if self.phase == "confirm" else self.slot or "command"
        return {"say": say, "listen": listen, "asking": self.slot, "phase": self.phase, "booking": dict(cmd),
                "choices": list(self.choices), "confirmed": None, "title": self.title, "fields": self.fields,
                "house": {room: dict(devices) for room, devices in self.house.items()}}

    def question(self, name: str) -> Question:
        if name == "command":
            return Question("command", ASK, [v for q in self.parts.values() for v in q.values])
        return super().question(name)

    def decide(self, q: Question, text: str, min_p: float, threshold: float, flagged: bool) -> Decision:
        if q.name == "command":
            cmds = [] if flagged else self.parse(text)
            if not cmds:
                return Decision("repeat")
            action = "accept" if min_p is None or min_p >= threshold else "confirm"
            return Decision(action, ", ".join(phrase(c) for c in cmds), commands=cmds)
        d = questions.resolve(q, text, min_p, threshold, flagged)
        d.value = self.canon.get(d.value, d.value)
        d.candidates = list(dict.fromkeys(self.canon.get(c, c) for c in d.candidates))
        return d

    def parse(self, text: str) -> list[dict]:
        """Commands named in ``text`` ("Licht im Flur an, Büro dunkel" is two); parts only if spelled out."""
        cmds, last = [], {}
        for clause in re.split(r"[,;.!?]|\bund\b|\bdann\b", text):
            found = {p: questions.match(self.parts[p], clause) for p in PARTS}
            if not any(found.values()) or not (found["device"] or found["room"] or cmds):
                continue  # an action alone ("jetzt hat es zu") only counts after a command
            cmd = {p: self.canon.get(v, v) if v else last.get(p) for p, v in found.items()}
            cmds.append(cmd)
            last = cmd
        for a, b in zip(reversed(cmds[:-1]), reversed(cmds[1:])):  # "Licht im Flur, Küche an": both on
            if not a["action"]:
                a["action"] = b["action"]
        return cmds

    # --- the conversation ---
    def start(self) -> dict:
        return self.step(ASK)

    def answer(self, d: Decision) -> dict:
        if self.phase == "confirm" and self.slot is None:  # "Meinten Sie Licht im Flur an?"
            return self._run() if self.picked(d) else self._retry()
        if self.phase == "confirm":
            return self._fill(value) if (value := self.picked(d)) else self._retry()
        if self.slot is None:  # a whole sentence
            if d.action == "repeat":
                return self._retry()
            self.todo = d.commands
            if d.action == "confirm":
                return self.offer([d.value])
            return self._run()
        if d.action == "accept":
            return self._fill(d.value)
        if d.action in ("confirm", "choose"):
            return self.offer(d.candidates if d.action == "choose" else [d.value])
        return self._retry()

    def choose(self, value: str | None) -> dict:
        if self.phase != "confirm":
            raise ValueError(f"nothing to choose in phase {self.phase}")
        if value is None or value not in self.choices:
            return self._retry()
        return self._run() if self.slot is None else self._fill(value)

    def _fill(self, value: str) -> dict:
        self.todo[0][self.slot] = value
        return self._run()

    def _run(self, done: str = "") -> dict:
        """Carries out the commands that are complete; asks for the first missing part."""
        self.phase, self.choices, self.slot = "ask", [], None
        said = [done] if done else []
        while self.todo:
            cmd = self.todo[0]
            missing = [p for p in PARTS if not cmd[p]]
            if missing:
                self.tries, self.slot = 0, missing[0]
                prefix = " ".join(said + [phrase(cmd) + ":"] if phrase(cmd) else said)
                if self.slot == "action":
                    return self.offer(ACTIONS.get(cmd["device"], ["an", "aus"]), lead=prefix.strip() or "Was genau")
                return self.step(f"{prefix} {self.parts[self.slot].ask}".strip())
            self.house.setdefault(cmd["room"], {})[cmd["device"]] = cmd["action"]
            self.log.append(dict(cmd))
            said.append(f"{phrase(cmd)}.")
            self.todo.pop(0)
        self.tries = 0
        return self.step(" ".join(["Okay:", *said, "Was noch?"]) if said else ASK)

    def _retry(self) -> dict:
        self.phase, self.choices, self.tries = "ask", [], self.tries + 1
        if self.tries < MAX_TRIES:
            return self.step("Entschuldigung, bitte noch einmal." + ("" if self.slot else f" {ASK}"))
        self.todo, self.slot, self.tries = [], None, 0
        return self.step(f"Das lassen wir. {ASK}")
