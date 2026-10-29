# Generate synthetic "planted" PII inside neutral carrier sentences.
# Frozen with the plan (seed=909) - see plan.md. Public/synthetic only:
# all values come from Faker or hand-written synthetic generators, nothing
# is read from any annotated PII dataset or real registry.
import io
import json
import os
import random
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from faker import Faker  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")
SEED = 909
N_PER_TYPE = 40

BG_CITIES = [
    "Пловдив", "Варна", "Бургас", "Русе", "Стара Загора", "Плевен", "Сливен",
    "Добрич", "Шумен", "Перник", "Хасково", "Ямбол", "Пазарджик",
    "Благоевград", "Велико Търново", "Враца", "Габрово", "Асеновград", "Видин",
]
def bg_street(rng: random.Random) -> str:
    kind = rng.choice(["ул.", "бул."])
    name = rng.choice(
        ["Иван Вазов", "Г. С. Раковски", "Витоша", "Цар Освободител", "Христо Ботев"]
    )
    num = rng.randint(1, 180)
    return f"{kind} {name} {num}"


def bg_egn(rng: random.Random) -> str:
    """Synthetic EGN with a VALID checksum, not tied to any real person."""
    weights = (2, 4, 8, 5, 10, 9, 7, 3, 6)
    year = rng.randint(1950, 2009)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    yy = year % 100
    mm = month if year < 2000 else month + 40
    seq = rng.randint(0, 999)
    digits = [int(c) for c in f"{yy:02d}{mm:02d}{day:02d}{seq:03d}"]
    checksum = sum(d * w for d, w in zip(digits, weights)) % 11 % 10
    digits.append(checksum)
    return "".join(str(d) for d in digits)


def build_examples(lang: str, rng: random.Random, fk: Faker) -> list[dict]:
    rows: list[dict] = []

    def add(text: str, spans: list[tuple[str, int, int]]) -> None:
        rows.append({"source_text": text, "privacy_mask": [
            {"label": lab, "start": s, "end": e} for lab, s, e in spans
        ]})

    def place(template: str, **fills: str) -> tuple[str, dict[str, tuple[int, int]]]:
        text = template
        offsets: dict[str, tuple[int, int]] = {}
        # replace left-to-right so earlier offsets stay valid
        cursor = 0
        out = []
        i = 0
        while i < len(template):
            matched = False
            for key, val in fills.items():
                tok = "{" + key + "}"
                if template.startswith(tok, i):
                    start = len("".join(out))
                    out.append(val)
                    offsets[key] = (start, start + len(val))
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                out.append(template[i])
                i += 1
        return "".join(out), offsets

    # ---- PERSON (GIVENNAME + SURNAME) ----
    person_templates_en = [
        "Please contact {given} {surname} about the renewal.",
        "Meeting notes were shared with {given} {surname} yesterday.",
        "{given} {surname} will follow up on the request next week.",
        "Forward the attached form to {given} {surname} before Friday.",
    ]
    person_templates_bg = [
        "Моля, свържете се с {given} {surname} относно поръчката.",
        "Протоколът от срещата беше изпратен на {given} {surname}.",
        "{given} {surname} ще се обади за потвърждение следващата седмица.",
        "Изпратете формуляра на {given} {surname} до петък.",
    ]
    templates = person_templates_en if lang == "en" else person_templates_bg
    for n in range(N_PER_TYPE):
        given, surname = fk.first_name(), fk.last_name()
        tmpl = templates[n % len(templates)]
        text, off = place(tmpl, given=given, surname=surname)
        gs, ge = off["given"]
        ss, se = off["surname"]
        add(text, [("GIVENNAME", gs, ge), ("SURNAME", ss, se)])

    # ---- EMAIL ----
    email_templates_en = [
        "You can reach the account owner at {email} for billing questions.",
        "Please cc {email} on the next update.",
        "Confirmation was sent to {email} this morning.",
    ]
    email_templates_bg = [
        "Можете да се свържете на {email} за въпроси по фактурата.",
        "Моля, копирайте {email} в следващия имейл.",
        "Потвърждението беше изпратено на {email} тази сутрин.",
    ]
    templates = email_templates_en if lang == "en" else email_templates_bg
    for n in range(N_PER_TYPE):
        email = fk.email()
        tmpl = templates[n % len(templates)]
        text, off = place(tmpl, email=email)
        es, ee = off["email"]
        add(text, [("EMAIL", es, ee)])

    # ---- PHONE ----
    phone_templates_en = [
        "Call {phone} to reschedule the delivery.",
        "The support line is {phone}, open until 6pm.",
        "We tried reaching {phone} twice with no answer.",
    ]
    phone_templates_bg = [
        "Обадете се на {phone}, за да пренасрочите доставката.",
        "Линията за поддръжка е {phone}, работи до 18 часа.",
        "Опитахме се да се свържем на {phone} два пъти без отговор.",
    ]
    templates = phone_templates_en if lang == "en" else phone_templates_bg
    for n in range(N_PER_TYPE):
        phone = fk.phone_number()
        tmpl = templates[n % len(templates)]
        text, off = place(tmpl, phone=phone)
        ps, pe = off["phone"]
        add(text, [("TELEPHONENUM", ps, pe)])

    # ---- CREDIT CARD ----
    cc_templates_en = [
        "The charge on card {card} was declined at checkout.",
        "Please verify card {card} before the next billing cycle.",
    ]
    cc_templates_bg = [
        "Плащането с карта {card} беше отказано при финализиране.",
        "Моля, потвърдете карта {card} преди следващия цикъл на таксуване.",
    ]
    templates = cc_templates_en if lang == "en" else cc_templates_bg
    for n in range(N_PER_TYPE):
        card = fk.credit_card_number()
        tmpl = templates[n % len(templates)]
        text, off = place(tmpl, card=card)
        cs, ce = off["card"]
        add(text, [("CREDITCARDNUMBER", cs, ce)])

    # ---- DATE ----
    date_templates_en = [
        "The appointment is confirmed for {date}.",
        "Documents must be submitted by {date}.",
    ]
    date_templates_bg = [
        "Срещата е потвърдена за {date}.",
        "Документите трябва да бъдат подадени до {date}.",
    ]
    templates = date_templates_en if lang == "en" else date_templates_bg
    for n in range(N_PER_TYPE):
        date = fk.date()
        tmpl = templates[n % len(templates)]
        text, off = place(tmpl, date=date)
        ds, de = off["date"]
        add(text, [("DATE", ds, de)])

    # ---- CITY + STREET ----
    city_templates_en = [
        "The package was delivered to {street}, {city}.",
        "Our new office opens next month in {city}, on {street}.",
    ]
    city_templates_bg = [
        "Пратката беше доставена на {street}, {city}.",
        "Новият ни офис отваря следващия месец в {city}, на {street}.",
    ]
    templates = city_templates_en if lang == "en" else city_templates_bg
    for n in range(N_PER_TYPE):
        if lang == "en":
            city, street = fk.city(), fk.street_name()
        else:
            city, street = rng.choice(BG_CITIES), bg_street(rng)
        tmpl = templates[n % len(templates)]
        text, off = place(tmpl, city=city, street=street)
        cs, ce = off["city"]
        ss, se = off["street"]
        add(text, [("CITY", cs, ce), ("STREET", ss, se)])

    # ---- BG_EGN (bg only - see plan.md) ----
    if lang == "bg":
        egn_templates = [
            "ЕГН {egn} беше въведено погрешно във формуляра.",
            "Моля, потвърдете ЕГН {egn} при следващото посещение.",
        ]
        for n in range(N_PER_TYPE):
            egn = bg_egn(rng)
            tmpl = egn_templates[n % len(egn_templates)]
            text, off = place(tmpl, egn=egn)
            es, ee = off["egn"]
            add(text, [("IDCARDNUM", es, ee)])

    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for lang, locale in (("en", "en_US"), ("bg", "bg_BG")):
        rng = random.Random(SEED if lang == "en" else SEED + 1)
        fk = Faker(locale)
        fk.seed_instance(SEED if lang == "en" else SEED + 1)
        rows = build_examples(lang, rng, fk)
        path = os.path.join(OUT_DIR, f"sample_{lang}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{lang}: wrote {len(rows)} examples -> {path}")


if __name__ == "__main__":
    main()
