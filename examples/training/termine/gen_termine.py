#!/usr/bin/env python3
# Synthetic task for the llama-finetune evaluation: extract an appointment from a German message as JSON with a fixed
# schema. Writes termine-train.jsonl (700 chats for llama-finetune) and termine-test.jsonl (100 held-out prompts with the
# expected answer). Deterministic (seed 42): md5 train f1fc5656d900001f1cd2c68e5c0b70ed, test 0f5ce9372fbb0f720262a6633ffcaef3.
import json, random

random.seed(42)

SYSTEM = "Extrahiere den Termin aus der Nachricht als JSON."
VORNAMEN = ["Anna", "Lukas", "Sophie", "Maximilian", "Lena", "Paul", "Marie", "Felix", "Laura", "Jonas", "Hannah",
            "Tobias", "Katharina", "Florian", "Julia", "Stefan", "Theresa", "Michael", "Sarah", "Andreas", "Eva",
            "Dominik", "Verena", "Georg", "Magdalena", "Christoph", "Barbara", "Matthias", "Elisabeth", "Bernhard"]
NACHNAMEN = ["Huber", "Gruber", "Bauer", "Wagner", "Müller", "Pichler", "Steiner", "Moser", "Mayer", "Hofer",
             "Leitner", "Berger", "Fuchs", "Eder", "Fischer", "Schmid", "Winkler", "Weber", "Schwarz", "Maier"]
ORTE = ["Wien", "Graz", "Linz", "Salzburg", "Innsbruck", "Klagenfurt", "Villach", "Wels", "St. Pölten", "Dornbirn",
        "Steyr", "Wiener Neustadt", "Feldkirch", "Bregenz", "Leoben", "Krems", "Baden", "Amstetten"]
RAEUME = ["im Besprechungsraum 2", "im Café Central", "in der Kanzlei", "im Rathaus", "am Hauptbahnhof",
          "im Büro", "in der Ordination", "im Gemeindeamt", "in der Bibliothek", "online per Videocall"]
THEMEN = ["Projektbesprechung", "Steuerberatung", "Zahnarzttermin", "Vorstellungsgespräch", "Wohnungsbesichtigung",
          "Elternabend", "Jahresgespräch", "Vertragsunterzeichnung", "Kundentermin", "Teammeeting",
          "Behördentermin", "Werkstatttermin", "Physiotherapie", "Budgetplanung", "Abschlusspräsentation"]
MONATE = ["Jänner", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
          "November", "Dezember"]
TAGE_IM_MONAT = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def zeit_text(h, m):
    styles = [f"{h}:{m:02d} Uhr", f"{h}.{m:02d} Uhr"]
    if m == 0:
        styles += [f"{h} Uhr"]
    return random.choice(styles)


def datum_text(y, mo, d):
    styles = [f"{d}. {MONATE[mo - 1]} {y}", f"{d}.{mo}.{y}", f"{d:02d}.{mo:02d}.{y}"]
    return random.choice(styles)


def beispiel():
    name = f"{random.choice(VORNAMEN)} {random.choice(NACHNAMEN)}"
    y = random.choice([2026, 2027])
    mo = random.randint(1, 12)
    d = random.randint(1, TAGE_IM_MONAT[mo - 1])
    h = random.randint(7, 19)
    m = random.choice([0, 0, 15, 30, 45, 10, 20])
    stadt = random.choice(ORTE)
    raum = random.choice(RAEUME)
    thema = random.choice(THEMEN)
    dt, zt = datum_text(y, mo, d), zeit_text(h, m)
    ort = stadt if raum.startswith("online") else f"{stadt}, {raum.split(' ', 1)[1]}"
    templates = [
        f"Hallo! Ich möchte die {thema} mit {name} am {dt} um {zt} {raum} in {stadt} fixieren.",
        f"Bitte trag ein: {thema}, {dt}, {zt}, mit {name}, Ort: {stadt} ({raum}).",
        f"Servus, {name} hat sich gemeldet – die {thema} findet am {dt} um {zt} in {stadt} {raum} statt.",
        f"Erinnerung: Am {dt} ist um {zt} die {thema} mit {name}. Treffpunkt {raum} in {stadt}.",
        f"Kannst du für mich die {thema} mit {name} notieren? {stadt}, {raum}, {dt} ab {zt}.",
        f"Termin bestätigt: {name}, {thema}, {zt} am {dt}, {raum} in {stadt}. Danke!",
    ]
    msg = random.choice(templates)
    answer = {"person": name, "datum": f"{y:04d}-{mo:02d}-{d:02d}", "uhrzeit": f"{h:02d}:{m:02d}",
              "ort": ort, "thema": thema}
    return msg, json.dumps(answer, ensure_ascii=False)


def main():
    seen = set()
    rows = []
    while len(rows) < 800:
        msg, ans = beispiel()
        if msg not in seen:
            seen.add(msg)
            rows.append((msg, ans))
    with open("termine-train.jsonl", "w") as f:
        for msg, ans in rows[:700]:
            f.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": msg},
                                             {"role": "assistant", "content": ans}]}, ensure_ascii=False) + "\n")
    with open("termine-test.jsonl", "w") as f:
        for msg, ans in rows[700:]:
            f.write(json.dumps({"system": SYSTEM, "user": msg, "expected": json.loads(ans)}, ensure_ascii=False) + "\n")


main()
