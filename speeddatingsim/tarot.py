import dataclasses
import itertools
from collections.abc import Iterator
from typing import ClassVar, Literal


TAROT_CARDS = []


@dataclasses.dataclass(slots=True, frozen=True)
class TarotCard:
    INDICES: ClassVar[Iterator[int]] = iter(itertools.count())

    suit: Literal["Major", "Wand", "Cup", "Sword", "Pentacle"]
    rank: int
    name: str
    description: str = ""
    upright: list[str] = dataclasses.field(default_factory=list)
    reversed: list[str] = dataclasses.field(default_factory=list)

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
                name_slug = str(self.rank)
            else:
                name_slug = self.name
            return f"{self.index:02d}_{self.suit}_{name_slug}.svg"
    


TarotCard(
    suit="Major",
    rank=0,
    name="Fool",
    description="""\
With light step, as if earth and its trammels had little power to restrain him, a young man in gorgeous vestments pauses at the brink of a precipice among the great heights of the world; he surveys the blue distance before him-its expanse of sky rather than the prospect below.

His act of eager walking is still indicated, though he is stationary at the given moment; his dog is still bounding. The edge which opens on the depth has no terror; it is as if angels were waiting to uphold him, if it came about that he leaped from the height. His countenance is full of intelligence and expectant dream.

He has a rose in one hand and in the other a costly wand, from which depends over his right shoulder a wallet curiously embroidered. He is a prince of the other world on his travels through this one-all amidst the morning glory, in the keen air. The sun, which shines behind him, knows whence he came, whither he is going, and how he will return by another path after many days.

He is the spirit in search of experience. Many symbols of the Instituted Mysteries are summarized in this card, which reverses, under high warrants, all the confusions that have preceded it.

In his Manual of Cartomancy, Grand Orient has a curious suggestion of the office of Mystic Fool, as apart of his process in higher divination; but it might call for more than ordinary gifts to put it into operation.

We shall see how the card fares according to the common arts of fortune-telling, and it will be an example, to those who can discern, of the fact, otherwise so evident, that the Trumps Major had no place originally in the arts of psychic gambling, when cards are used as the counters and pretexts. Of the circumstances under which this art arose we know, however, very little.

The conventional explanations say that the Fool signifies the flesh, the sensitive life, and by a peculiar satire its subsidiary name was at one time the alchemist, as depicting folly at the most insensate stage.
""",
    upright=["folly", "mania", "extravagance", "intoxication", "delirium", "frenzy", "bewrayment"],
    reversed=["negligence", "absence", "distribution", "carelessness", "apathy", "nullity", "vanity"],
)
TarotCard("Major", 1, "The Magician"),
TarotCard("Major", 2, "The High Priestess"),
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
itertools.chain.from_iterable(
    [
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
    ]
    for suit in ("Wand", "Cup", "Sword", "Pentacle")
)


if __name__ == "__main__":
    for card in TAROT_CARDS:
        print(card)
        print(card.svg_filename)