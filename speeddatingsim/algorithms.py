import dataclasses
from collections.abc import Generator, Iterable
from typing import Self, Type


class SpeedDatingSession[Person](Generator[list[tuple[Person, Person]], list[], None]):
    @classmethod
    def create(cls: Type[Self], cohort: Iterable[Person]) -> Self:
        pass

    def matchmake() -> 


    