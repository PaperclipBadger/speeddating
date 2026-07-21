import dataclasses
import functools
import itertools
import pathlib
import re
from collections.abc import Iterator
from typing import ClassVar, Literal


TAROT_CARDS = []

SPLIT_AT = re.compile(r"(_Divinatory\s+Meanings_|_Reversed_)", re.MULTILINE)
ITALICS = re.compile(r"_(.*?)_", re.MULTILINE | re.DOTALL)


@dataclasses.dataclass
class TarotCard:
    INDICES: ClassVar[Iterator[int]] = iter(itertools.count())

    suit: Literal["Major", "Wand", "Cup", "Sword", "Pentacle"]
    rank: int
    name: str
    index: int = dataclasses.field(default_factory=INDICES.__next__)

    def __post_init__(self):
        TAROT_CARDS.append(self)
    
    @property
    def noun(self) -> str:
        if self.name.startswith("The "):
            return self.name[4:]
        return self.name

    @property
    def full_name(self) -> str:
        if self.suit == "Major":
            return f"{self.rank}. {self.name}"
        return f"{self.name} of {self.suit}s"

    @property
    def svg_filename(self) -> str:
        if self.suit == "Major":
            name_slug = self.noun.replace(" ", "_")
            return f"{self.index:02d}_{self.index}_{name_slug}.svg"
        else:
            if self.rank < 10:
                name_slug = str(self.rank + 1)
            else:
                name_slug = self.name
            return f"{self.index:02d}_{self.suit}_{name_slug}.svg"
    
    @functools.cached_property
    def description(self) -> str:
        here = pathlib.Path(__file__).parent
        desc_file = (here / "waite_cards" / self.svg_filename).with_suffix(".txt")
        with desc_file.open() as f:
            desc = f.read()
        desc = desc.replace("--", "&mdash;")
        desc = SPLIT_AT.sub(r"\n\n\g<1>", desc)
        desc = ITALICS.sub(r"<i>\g<1></i>", desc)
        return desc

TarotCard("Major", 0, "Fool")
TarotCard("Major", 1, "The Magician")
TarotCard("Major", 2, "The High Priestess")
TarotCard("Major", 3, "The Empress"),
TarotCard("Major", 4, "The Emperor"),
TarotCard("Major", 5, "The Hierophant"),
TarotCard("Major", 6, "The Lovers"),
TarotCard("Major", 7, "The Chariot"),
TarotCard("Major", 8, "Strength"),
TarotCard("Major", 9, "The Hermit"),
TarotCard("Major", 10, "Wheel of Fortune"),
TarotCard("Major", 11, "Justice"),
TarotCard("Major", 12, "The Hanged Man"),
TarotCard("Major", 13, "Death"),
TarotCard("Major", 14, "Temperance"),
TarotCard("Major", 15, "The Devil"),
TarotCard("Major", 16, "The Tower"),
TarotCard("Major", 17, "The Star"),
TarotCard("Major", 18, "The Moon"),
TarotCard("Major", 19, "The Sun"),
TarotCard("Major", 20, "Judgement"),
TarotCard("Major", 21, "The World"),
for suit in ("Wand", "Cup", "Sword", "Pentacle"):
    TarotCard(suit, 0, "Ace"),
    TarotCard(suit, 1, "Two"),
    TarotCard(suit, 2, "Three"),
    TarotCard(suit, 3, "Four"),
    TarotCard(suit, 4, "Five"),
    TarotCard(suit, 5, "Six"),
    TarotCard(suit, 6, "Seven"),
    TarotCard(suit, 7, "Eight"),
    TarotCard(suit, 8, "Nine"),
    TarotCard(suit, 9, "Ten"),
    TarotCard(suit, 10, "Page"),
    TarotCard(suit, 11, "Knight"),
    TarotCard(suit, 12, "Queen"),
    TarotCard(suit, 13, "King"),


if __name__ == "__main__":
    for card in TAROT_CARDS:
        print(card)
        print(card.svg_filename)