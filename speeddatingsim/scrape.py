#!/usr/bin/env python3

"""
Split Arthur Edward Waite's The Pictorial Key to the Tarot
into one text file per tarot card.

Requires:
    pip install requests
"""

import re
from pathlib import Path

import requests

BOOK_URL = "https://www.gutenberg.org/cache/epub/43548/pg43548.txt"
OUTPUT_DIR = Path(__file__).parent / "waite_cards"

MAJORS = [
    "THE FOOL",
    "THE MAGICIAN",
    "THE HIGH PRIESTESS",
    "THE EMPRESS",
    "THE EMPEROR",
    "THE HIEROPHANT",
    "THE LOVERS",
    "THE CHARIOT",
    "STRENGTH, OR FORTITUDE",
    "THE HERMIT",
    "WHEEL OF FORTUNE",
    "JUSTICE",
    "THE HANGED MAN",
    "DEATH",
    "TEMPERANCE",
    "THE DEVIL",
    "THE TOWER",
    "THE STAR",
    "THE MOON",
    "THE SUN",
    "THE LAST JUDGMENT",
    "THE WORLD",
]

SUITS = ["WAND", "CUP", "SWORD", "PENTACLE"]
VALUES = [
    "ACE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "PAGE",
    "KNIGHT",
    "QUEEN",
    "KING",
]


CARDS = []
for i, major in enumerate(MAJORS):
    slug = major.split(",")[0].title().replace(" ", "_")
    if slug.startswith("The_"):
        slug = slug[4:]
    if slug.startswith("Last_"):
        slug = slug[5:]
    if slug == "Judgment":
        slug = "Judgement"
    CARDS.append((f"{i:02d}_{i}_{slug}", major))
for i, suit in enumerate(SUITS):
    for j, value in enumerate(VALUES):
        k = len(MAJORS) + i * len(VALUES) + j
        name = j + 1 if j < 10 else value.title()
        CARDS.append(
            (f"{k:2d}_{suit.title()}_{name}", f"{suit}S. {value}.")
        )

print("Downloading book...")

text = requests.get(BOOK_URL, timeout=30).text

OUTPUT_DIR.mkdir(exist_ok=True)

# Find every heading position
positions = []

for fname, card in CARDS:
    m = re.search(rf"^.*{re.escape(card)}\s*$", text, re.MULTILINE)
    if m:
        positions.append((fname, m.start()))
    else:
        print(f"Not found: {card}")

positions.sort(key=lambda x: x[1])

for i, (fname, start) in enumerate(positions):
    end = positions[i + 1][1] if i + 1 < len(positions) else len(text)

    section = "\n".join(text[start:end].strip().splitlines()[2:])

    if match := re.search(rf"\[Illustration.*\]", section):
        section = section[:match.start()]

    filename = OUTPUT_DIR / f"{fname}.txt"
    filename.write_text(section.strip(), encoding="utf-8")

print(f"Wrote {len(positions)} files to {OUTPUT_DIR.resolve()}")