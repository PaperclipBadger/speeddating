from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import enum
import collections
import dataclasses
import itertools
import random
from typing import Any, Final, TypeAlias, TypeVar

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.manifold import TSNE

from .mwmatching import maxWeightMatching
from .wordlists import NAMES

# sexuality: people have features, and a preference function
Feature: TypeAlias = str
Features: TypeAlias = set[Feature]
Preference: TypeAlias = Callable[[Features], bool]

T = TypeVar("T")

def batched(iterable: Iterable[T], n: int, *, strict: bool = False) -> Iterable[tuple[T, ...]]:
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


np.random.seed(100)
random.seed(100)

N_TRIALS: Final = 100
BUDGET: Final = 20
GRAPHS: Final = False
COHORT_SIZE: Final = 60


class Decision(enum.IntEnum):
    INCOMPATIBLE = -1
    NO = 1
    YES = 2


@dataclasses.dataclass
class OnlyIf:
    must_haves: set[Feature]

    def __call__(self, features: Features) -> bool:
        return self.must_haves <= features

    def __str__(self) -> str:
        if len(self.must_haves) > 1:
            return "&".join(self.must_haves)
        elif len(self.must_haves) == 1:
            return next(iter(self.must_haves))
        else:
            return "()"


@dataclasses.dataclass
class Or:
    subpreferences: list[Preference]

    def __call__(self, features: Features) -> bool:
        return any(p(features) for p in self.subpreferences)

    def __str__(self) -> str:
        subpref_strs = [str(subpref) for subpref in self.subpreferences]
        return "|".join(f"({s})" if "&" in s else s for s in subpref_strs)


unique_ids = iter(itertools.count())


@dataclasses.dataclass(unsafe_hash=True)
class Person:
    features: Features = dataclasses.field(hash=False, compare=False)
    preference: Preference = dataclasses.field(hash=False, compare=False)
    id: int = dataclasses.field(default_factory=unique_ids.__next__)
    hidden: float = dataclasses.field(
        default_factory=lambda: random.uniform(-np.pi, np.pi),
        hash=False,
        compare=False,
    )
    seeking: float = dataclasses.field(
        default_factory=lambda: random.uniform(-np.pi, np.pi),
        hash=False,
        compare=False,
    )
    pickiness: float = 0.5

    def __str__(self) -> str:
        return NAMES[self.id % len(NAMES)]

    @property
    def hidden_vec(self) -> np.ndarray:
        return np.array([np.sin(self.hidden), np.cos(self.hidden)])

    @property
    def seeking_vec(self) -> np.ndarray:
        return np.array([np.sin(self.seeking), np.cos(self.seeking)])

    def decide(self, other: Person) -> Decision:
        if self.preference(other.features) and other.preference(self.features):
            threshold = np.cos(np.pi * (1 - self.pickiness))
            if np.sum(self.seeking_vec * other.hidden_vec) > threshold:
                return Decision.YES
            else:
                return Decision.NO
        else:
            return Decision.INCOMPATIBLE


def make_natstats_cohort(n: int, smoothe: float = 0) -> list[Person]:
    cohort = []

    # https://www.ons.gov.uk/peoplepopulationandcommunity/culturalidentity/sexuality/bulletins/sexualidentityuk/2024
    # UK demographics:
    # gender split: 50/50
    # in 2024, for people 25–34:
    # - ~4.4% of men identified as gay
    # - ~2% of men identified as bi
    # - ~2% of women identified as lesbian
    # - ~4.5% of women identified as bi

    # trans stats are harder to come by, national stats are 0.5%-3% depending on estimate
    # 2021 census indicates equal incidence rates of trans men and women

    # https://www.them.us/story/cis-trans-dating
    # https://static1.squarespace.com/static/527403c4e4b02d3f058d2f18/t/5c520b6d6d2a7366fd3dd7e8/1548880749971/Transgender+Exclusion+-+Blair+&+Hoskin+2018CV.pdf
    # this article claims that
    # - 3.3% of straight men are open to dating a trans.
    #   1.4% were open to dating a trans man,
    #   1.4% were open to dating a trans woman,
    #   0.5% were open to dating both.
    # - 11.5% of gay men are open to dating a trans.
    #   8.2% were open to dating a trans man,
    #   0% were open to dating a trans woman,
    #   3.3% were open to dating both.
    # - 1.8% of straight women are open to dating a trans.
    #   1.5% were open to dating a trans man,
    #   0% were open to dating a trans woman,
    #   0.3% were open to dating both.
    # - 28.8% of gay women are open to dating a trans.
    #   9.9% were open to dating a trans man,
    #   9% were open to dating a trans woman,
    #   9.9% were open to dating both.
    # - 51.7% of bi/queer/nb people are open to dating a trans.
    #   2.5% were open to dating a trans woman,
    #   14.7% were open to dating a trans man,
    #   34.5% were open to dating both.

    def smoothen(weights: list[float]):
        n = len(weights)
        return [(1 - smoothe) * w + smoothe * (1 / n) for w in weights]

    for _ in range(n):
        sex = random.choice(["male", "female"])
        (trans,) = random.choices(["cis", "trans"], smoothen([0.97, 0.03]))
        match sex, trans:
            case "male", "cis":
                gender = "man"
            case "male", "trans":
                gender = "woman"
            case "female", "cis":
                gender = "woman"
            case "female", "trans":
                gender = "man"

        features = {sex, gender, trans}

        if gender == "man":
            sexuality_w = [0.935, 0.045, 0.02]
        elif gender == "woman":
            sexuality_w = [0.935, 0.02, 0.045]

        (sexuality,) = random.choices(["straight", "gay", "bi"], smoothen(sexuality_w))

        tsexualities = ["cis only", "cis+tmasc", "cis+tfemme", "all"]

        match trans, gender, sexuality:
            # common sense: trans people are fine dating other trans people
            # and the intellectually honest ones have congruent preferences
            # the survey had almost no trans people in it, so does not give stats
            case "trans", _, "straight" | "gay":
                target_gender = "woman"
                tsexuality_weights = [0, 3 / 7, 3 / 7, 1 / 7]
            case "trans", (_, "bi"):
                target_gender = "man"
                tsexuality_weights = [0, 0, 0, 1]
            case "cis", "man", "straight":
                target_gender = "woman"
                tsexuality_weights = [96.7, 1.4, 1.4, 0.5]
            case "cis", "man", "gay":
                target_gender = "man"
                tsexuality_weights = [88.5, 8.2, 0, 3.3]
            case "cis", "woman", "straight":
                target_gender = "man"
                tsexuality_weights = [98.2, 1.5, 0, 0.3]
            case "cis", "woman", "gay":
                target_gender = "woman"
                tsexuality_weights = [71.2, 9.9, 9, 9.9]
            case _:
                # bi, nb etc
                target_gender = None
                tsexuality_weights = [48.3, 2.5, 14.7, 34.5]

        (tsexuality,) = random.choices(tsexualities, smoothen(tsexuality_weights))

        if target_gender is not None:
            target_sex = "male" if target_gender == "man" else "woman"

            match target_gender, tsexuality:
                case "man", "cis+tmasc":
                    tsexuality = "gender essentialist"
                case "man", "cis+tfemme":
                    tsexuality = "sex essentialist"
                case "woman", "cis+tmasc":
                    tsexuality = "sex essentialist"
                case "woman", "cis+tfemme":
                    tsexuality = "gender essentialist"

            match tsexuality:
                case "cis only":
                    preference = OnlyIf({target_gender, target_sex})
                case "gender essentialist":
                    preference = OnlyIf({target_gender})
                case "sex essentialist":
                    preference = OnlyIf({target_sex})
                case "all":
                    preference = Or([OnlyIf({target_gender}), OnlyIf({target_sex})])
        else:
            match tsexuality:
                case "cis only":
                    preference = OnlyIf({"cis"})
                case "cis+tmasc":
                    preference = Or([OnlyIf({"man"}), OnlyIf({"female"})])
                case "cis+tfemme":
                    preference = Or([OnlyIf({"woman"}), OnlyIf({"male"})])
                case "all":
                    preference = OnlyIf(set())

        cohort.append(Person(features, preference))

    return cohort


def make_tinder_cohort(n: int = 60) -> list[Person]:
    # https://www.swipestats.io/blog/dating-app-statistics-gender
    # apps are about 60-40 man-woman, with Tinder a notable outlier at 80-20
    # median man swipes left 10,051 and right 5,096 = ~33% approval rating
    # median woman swipes left 19,553 and right 989 = ~4.8% approval rating

    # https://www.pewresearch.org/short-reads/2023/06/26/about-half-of-lesbian-gay-and-bisexual-adults-have-used-online-dating/
    # half of queers have used a dating app compared to 30% of straights
    # lets's... double the incidence of queer people in the population
    # assume gay dating dynamics are less cursed and smoothe the approval rate to 20%

    cohort = []

    for _ in range(n):
        (gender,) = random.choices(["man", "woman"], [3, 2])
        if gender == "man":
            sexuality_w = [0.87, 0.09, 0.04]
        elif gender == "woman":
            sexuality_w = [0.87, 0.04, 0.09]
        (sexuality,) = random.choices(["straight", "gay", "bi"], sexuality_w)

        match gender, sexuality:
            case "man", "straight":
                target = {"woman"}
                pickiness = 0.67
            case "woman", "straight":
                target = {"man"}
                pickiness = 0.95
            case "man", "gay":
                target = {"man"}
                pickiness = 0.8
            case "woman", "gay":
                target = {"woman"}
                pickiness = 0.8
            case _, "bi":
                target = set()
                pickiness = 0.8

        cohort.append(Person({gender, "cis"}, OnlyIf(target), pickiness=pickiness))

    return cohort


def make_crumpet_cohort(n: int = 60) -> list[Person]:
    # hypothesis: everyone comes to crumpet with a partner who matches their preferences
    # so not a balanced gender ratio per se

    # most are bi, then straight, then gay
    # maybe 25% of afab are transmasc, 25% nb
    # maybe 10% of amab are transfemme, 25% nb
    # lots people getting their boobs removed, not so many getting them put on

    # 50% are gender essentialist (nb inclusive)
    # 50% are sex essentialist
    cohort = []

    def make_crumpet():
        sex = random.choice(["male", "female"])
        if sex == "male":
            (gender,) = random.choices(["man", "woman", "nb"], [0.6, 0.1, 0.25])
        elif sex == "female":
            (gender,) = random.choices(["man", "woman", "nb"], [0.25, 0.5, 0.25])
        (sexuality,) = random.choices(["straight", "gay", "bi"], [0.3, 0.2, 0.5])

        tpref = random.choice(["gender essentialist", "sex essentialist"])

        match (gender, sexuality), tpref:
            case ("man", "straight") | ("woman", "gay"), "gender essentialist":
                preference = Or([OnlyIf({"woman"}), OnlyIf({"nb"})])
            case ("woman", "straight") | ("man", "gay"), "gender essentialist":
                preference = Or([OnlyIf({"man"}), OnlyIf({"nb"})])
            case ("man", "straight") | ("woman", "gay"), "sex essentialist":
                preference = OnlyIf({"female"})
            case ("woman", "straight") | ("man", "gay"), "sex essentialist":
                preference = OnlyIf({"male"})
            case _:
                preference = OnlyIf(set())

        match sex, gender:
            case ("male", "man") | ("female", "woman"):
                tstatus = "cis"
            case _:
                tstatus = "trans"

        return Person({sex, gender, tstatus}, preference)

    for _ in range(n // 2):
        # generate the crump
        crump = make_crumpet()

        # generate the sad sack who came with
        partner = make_crumpet()
        while not (
            crump.decide(partner) == Decision.YES
            and partner.decide(crump) == Decision.YES
        ):
            partner = make_crumpet()

        cohort.append(crump)
        cohort.append(partner)

    return cohort


def make_gay_paradise_cohort(n: int = 60) -> list[Person]:
    cohort = []

    featuress = [
        ("?", ("man", "woman")),
        ("=", ("poly", "mono")),
    ]
    for _ in range(n):
        my_features = {random.choice(feature) for _, feature in featuress}
        my_preferences = set()
        for kind, features in featuress:
            if kind == "?":
                my_preferences |= random.choice(
                    [set(), *({feature} for feature in features)]
                )
            if kind == "=":
                my_preferences |= my_features & set(features)
        cohort.append(Person({"cis"} | my_features, OnlyIf(my_preferences)))

    return cohort


def make_pan_cohort(n: int = 60) -> list[Person]:
    return [Person(set(), OnlyIf(set())) for _ in range(n)]


PersonID: TypeAlias = int


@dataclasses.dataclass
class Date:
    left: Person
    right: Person


def round_robin(cohort: list[Person], budget: int) -> list[Date]:
    # split into two groups and match
    # rotate one group until you run out of budget
    group_size = len(cohort) // 2
    lefts = cohort[group_size:]
    rights = cohort[:group_size]

    dates = []

    for i in range(budget):
        for j, left in enumerate(lefts):
            right = rights[(j + i) % group_size]
            dates.append(Date(left, right))

    return dates


def edmonds_blossom(
    cohort: list[Person],
    calibration_rounds: int,
    budget: int,
) -> list[Date]:
    if budget < calibration_rounds:
        return round_robin(cohort, budget)

    dates = []

    # associate each person with their cohort index (give each a number)
    # track their preferences over other participants
    n = len(cohort)

    # [i, j] = 1 if i likes j, 0 if not met, -1 if i dislikes j
    decisions = np.zeros((n, n), dtype=np.int8)

    def do_date(i: int, j: int) -> bool:
        assert decisions[i, j] == 0
        assert decisions[j, i] == 0
        date = Date(left=cohort[i], right=cohort[j])
        dates.append(date)
        decisions[i, j] = 1 if cohort[i].decide(cohort[j]) == Decision.YES else -1
        decisions[j, i] = 1 if cohort[j].decide(cohort[i]) == Decision.YES else -1

    # first n rounds: random matching
    # practically this'll be done with round robin
    for round in range(2):
        for i in range(n // 2):
            j = n // 2 + (i + round) % (n // 2)
            do_date(i, j)

    # now: everyone has matched with 2 people
    # the idea: predict decision based on past matches
    # e.g. if a dated b and c and d dated b and e,
    # then if a and d agree on b they will probably agree on c and e
    for round in range(2, budget):
        norms = np.linalg.norm(decisions, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = decisions / norms
        # S[i, j] is how similar i's preferences are to j's
        S = normed @ normed.T  # similarity matrix

        # shift decisions such that the mean is 0, without touching the 0s
        # this means that picky people's likes matter more
        # and nonpicky people's dislikes matter more
        means = decisions.sum(axis=1, keepdims=True) / (decisions != 0).sum(
            axis=1, keepdims=True
        )
        shifted = np.where(decisions != 0, decisions - means, 0)
        # recommendations_ij = sum_k S_ik D_kj
        recommendations = S @ shifted
        # print(recommendations[:8, :8].round(2))

        met = (decisions != 0).astype(np.float32)
        mutual = met @ met.T  # mutual[i, j] = num people who have dated both i and j
        uncertainty = 1 / (1 + mutual)

        # generate an (almost-perfect) matching
        yesses = decisions == 1
        n_matches = np.sum(yesses & yesses.T, axis=-1)
        # how many people said yes to me?
        hotness = np.sum(yesses, axis=0)
        # how many people did I say yes to?
        horniness = np.sum(yesses, axis=1)

        graph = []
        for i in range(n):
            for j in range(i + 1, n):
                if decisions[i, j] == 0:
                    # basic: match people who are recommended to each other
                    weight = min(
                        recommendations[i, j],  # * horniness[j] / (hotness[i] + 1),
                        recommendations[j, i],  # * horniness[i] / (hotness[j] + 1),
                    )
                    # # people are picky and don't give out likes very often
                    # # that can lead to some people languishing while the algo has poor info on them
                    # # adjust weights so that people who have no likes are matched with horny people

                    # algo works better without floating point error
                    graph.append((i, j, int(weight * 10000)))
        random.shuffle(graph)

        mate = maxWeightMatching(graph, maxcardinality=True)

        # perform the dates
        for i, j in enumerate(mate):
            if j < i:
                continue
            do_date(i, j)

    # tsne_plot(cohort, decisions)

    return dates


def tsne_plot(cohort: list[Person], decisions: np.ndarray) -> None:
    n = decisions.shape[0]

    embeddings = TSNE(2).fit_transform(decisions)

    fig = plt.figure()
    for i in range(n):
        for j in range(n):
            d = embeddings[j] - embeddings[i]
            if decisions[i, j] < 0 and decisions[j, i] < 0:
                # ew
                d = embeddings[j] - embeddings[i]
                plt.arrow(
                    embeddings[i, 0],
                    embeddings[i, 1],
                    d[0],
                    d[1],
                    color="red",
                    length_includes_head="true",
                    overhang=0.1,
                )
            if decisions[i, j] > 0 and decisions[j, i] < 0:
                # it's one-sided
                plt.arrow(
                    embeddings[i, 0],
                    embeddings[i, 1],
                    d[0],
                    d[1],
                    color="blue",
                    length_includes_head="true",
                    overhang=0.1,
                )
            if decisions[i, j] > 0 and decisions[j, i] > 0:
                # it's a match!
                plt.arrow(
                    embeddings[i, 0],
                    embeddings[i, 1],
                    d[0],
                    d[1],
                    color="green",
                    length_includes_head="true",
                    overhang=0.1,
                )

    sc = plt.scatter(embeddings[:, 0], embeddings[:, 1], marker=".")

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(
            "\n".join(
                [
                    " ".join(
                        [
                            next(iter(cohort[n].features & {"cis", "trans"})),
                            next(iter(cohort[n].features & {"man", "woman"})),
                            *(
                                cohort[n].features
                                - {"cis", "trans", "man", "woman", "male", "female"}
                            ),
                        ]
                    )
                    + "\n->"
                    + str(cohort[n].preference)
                    for n in ind["ind"]
                ]
            )
        )
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    ax = fig.gca()
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


def just_ask(cohort: list[Person], budget: int) -> None:
    # ask everyone for gender and gender preference
    # only match compatible people
    n = len(cohort)
    seen = np.zeros((n, n), dtype=np.bool)

    def compatible(a: Person, b: Person) -> bool:
        genders = {"man", "woman"}
        match b.preference:
            case OnlyIf():
                a_matches_b = (b.preference.must_haves & genders) <= a.features
            case Or():
                for preference in b.preference.subpreferences:
                    if target := preference.must_haves & genders:
                        if target <= a.features:
                            a_matches_b = True
                            break
                else:
                    a_matches_b = False
        match a.preference:
            case OnlyIf():
                b_matches_a = (a.preference.must_haves & genders) <= b.features
            case Or():
                for preference in a.preference.subpreferences:
                    if target := preference.must_haves & genders:
                        if target <= b.features:
                            b_matches_a = True
                            break
                else:
                    b_matches_a = False
        return a_matches_b and b_matches_a

    compatibility = np.array([[compatible(a, b) for b in cohort] for a in cohort])

    dates = []

    for _ in range(budget):
        graph = []

        for i in range(n):
            for j in range(i + 1, n):
                if seen[i, j] or not compatibility[i, j]:
                    continue

                graph.append((i, j, 1))

        random.shuffle(graph)

        mate = maxWeightMatching(graph, maxcardinality=True)
        for i, j in enumerate(mate):
            if j < i:
                continue
            seen[i, j] = True
            dates.append(Date(cohort[i], cohort[j]))

    return dates


def straight_round_robin(cohort: list[Person], budget: int) -> None:
    men = [person for person in cohort if "man" in person.features]
    women = [person for person in cohort if "woman" in person.features]

    # two circles, one for men and one for women
    # bigger circle moves every round
    # poker table or smth for the people who are waiting for a match
    dates = []

    if len(men) >= len(women):
        long = men
        short = women
    else:
        long = women
        short = men

    for i in range(min(budget, len(long))):
        rotated = itertools.islice(itertools.cycle(long), i, None)
        for left, right in zip(rotated, short):
            dates.append(Date(left, right))

    return dates


def report(cohorts: list[list[Person]], datess: list[list[Date]]):
    n_dates = [len(dates) for dates in datess]
    print(f"dates: [{min(n_dates)}, {max(n_dates)}] mean {np.mean(n_dates)}")
    n_dates_pp = [
        len(dates) * 2 / len(cohort) for cohort, dates in zip(cohorts, datess)
    ]
    print(
        f"dates per participant: [{min(n_dates_pp):.1f}, {max(n_dates_pp):.1f}] mean {np.mean(n_dates_pp):.1f}"
    )

    date_was_matchs = [
        [
            date.left.decide(date.right) == Decision.YES
            and date.right.decide(date.left) == Decision.YES
            for date in dates
        ]
        for dates in datess
    ]
    n_matches = [sum(date_was_match) for date_was_match in date_was_matchs]
    print(f"matches: [{min(n_matches)}, {max(n_matches)}] mean {np.mean(n_matches)}")

    if GRAPHS:
        for cohort, n, date_was_match in zip(cohorts, n_matches, date_was_matchs):
            round_matches = [
                sum(batch)
                for batch in batched(date_was_match, len(cohort) // 2)
            ]
            print(" ", n, "\t", graph(round_matches, length=BUDGET))

    n_matchess = []
    n_doubless = []
    n_selfcests = []

    print(
        f"matches per participant: ",
        format(np.mean([2 * n / len(c) for n, c in zip(n_matches, cohorts)]), ".1f"),
    )

    for cohort, dates in zip(cohorts, datess):
        n_matches = []
        n_doubles = 0
        n_selfcest = 0

        for person in cohort:
            partners = collections.Counter(
                itertools.chain(
                    (date.right for date in dates if date.left is person),
                    (date.left for date in dates if date.right is person),
                )
            )

            n_doubles += any(v > 1 for v in partners.values())
            n_selfcest += partners[person] > 0
            n_matches.append(
                int(
                    sum(
                        person.decide(partner) == Decision.YES
                        and partner.decide(person) == Decision.YES
                        for partner in partners
                    )
                )
            )

        if GRAPHS:
            match_counts = dict(sorted(collections.Counter(n_matches).items()))
            print(
                f" [{min(n_matches)}, {max(n_matches)}] mean {np.mean(n_matches):.1f}",
                "\t",
                graph([match_counts.get(i, 0) for i in range(BUDGET)], length=BUDGET),
            )

        n_matchess.append(n_matches)
        n_doubless.append(n_doubles)
        n_selfcests.append(n_selfcest)

    print(f"doubles: [{min(n_doubless)}, {max(n_doubless)}]")
    print(f"selfcest: [{min(n_selfcests)}, {max(n_selfcests)}]")


bars = "▁▂▃▄▅▆▇█"


def graph(
    values: Sequence[float], max_value: float | None = None, length: int = 10
) -> str:
    m = max(values) if max_value is None else max_value
    heights = itertools.islice(itertools.chain(values, itertools.repeat(0)), length)
    return "".join(bars[max(int(v * 8 / (m + 1)), 0)] for v in heights)


if __name__ == "__main__":
    cohorts = []
    datess = {
        "Round Robin": [],
        "Straight Round Robin": [],
        "Just Ask": [],
        "Recommender": [],
    }

    for _ in tqdm.trange(0, N_TRIALS, desc="Running trials", unit="trial"):
        # cohort = make_crumpet_cohort(COHORT_SIZE)
        # cohort = make_tinder_cohort(COHORT_SIZE)
        # cohort = make_natstats_cohort(COHORT_SIZE, smoothe=0.2)
        # cohort = make_gay_paradise_cohort(COHORT_SIZE)
        cohort = [
            *make_crumpet_cohort(COHORT_SIZE // 2),
            *make_natstats_cohort(COHORT_SIZE // 2, smoothe=0.1),
        ]
        random.shuffle(cohort)
        cohorts.append(cohort)

        datess["Round Robin"].append(round_robin(cohort, BUDGET))
        datess["Straight Round Robin"].append(straight_round_robin(cohort, BUDGET))
        datess["Just Ask"].append(just_ask(cohort, BUDGET))
        datess["Recommender"].append(edmonds_blossom(cohort, 2, BUDGET))

    print("Cohort stats:")
    print("-------------")

    likess = []
    likeds = []
    compatibless = []
    matchess = []

    for cohort in cohorts:
        likes = {p: 0 for p in cohort}
        liked = {p: 0 for p in cohort}
        compatibles = {p: 0 for p in cohort}
        matches = {p: 0 for p in cohort}
        for a, b in itertools.combinations(cohort, 2):
            if a.decide(b) == Decision.YES and b.decide(a) == Decision.YES:
                matches[a] += 1
                matches[b] += 1

            if a.decide(b) != Decision.INCOMPATIBLE:
                compatibles[a] += 1
            if b.decide(a) != Decision.INCOMPATIBLE:
                compatibles[b] += 1

            if a.decide(b) == Decision.YES:
                likes[a] += 1
                liked[b] += 1

            if b.decide(a) == Decision.YES:
                likes[b] += 1
                liked[a] += 1

        likess.append(likes)
        likeds.append(liked)
        compatibless.append(compatibles)
        matchess.append(matches)

    sizes = [len(cohort) for cohort in cohorts]
    print(f"size: [{min(sizes)}, {max(sizes)}] mean {np.mean(sizes)}")

    length = 40

    n_matches = [sum(cs.values()) // 2 for cs in matchess]
    print(f"matches: [{min(n_matches)}, {max(n_matches)}] mean {np.mean(n_matches)}")
    if GRAPHS:
        for n, cs in zip(n_matches, matchess):
            counts = dict(sorted(collections.Counter(cs.values()).items()))
            print(
                " ",
                n,
                "\t",
                graph([counts.get(i, 0) for i in range(length)], length=length),
            )

    n_compatibles = [sum(cs.values()) // 2 for cs in compatibless]
    print(
        f"compatibles: [{min(n_compatibles)}, {max(n_compatibles)}] mean {np.mean(n_compatibles)}"
    )
    if GRAPHS:
        for n, cs in zip(n_compatibles, compatibless):
            counts = dict(sorted(collections.Counter(cs.values()).items()))
            print(
                " ",
                n,
                "\t",
                graph([counts.get(i, 0) for i in range(length)], length=length),
            )

    n_likes = [sum(cs.values()) for cs in likess]
    print(f"likes: [{min(n_likes)}, {max(n_likes)}] mean {np.mean(n_likes)}")
    if GRAPHS:
        for n, cs in zip(n_likes, likess):
            counts = dict(sorted(collections.Counter(cs.values()).items()))
            print(
                " ",
                n,
                "\t",
                graph([counts.get(i, 0) for i in range(length)], length=length),
            )

    n_liked = [sum(cs.values()) for cs in likeds]
    print(f"liked: [{min(n_liked)}, {max(n_liked)}] mean {np.mean(n_liked)}")
    if GRAPHS:
        for n, cs in zip(n_liked, likeds):
            counts = dict(sorted(collections.Counter(cs.values()).items()))
            print(
                " ",
                n,
                "\t",
                graph([counts.get(i, 0) for i in range(length)], length=length),
            )

    for key in datess:
        print()
        print(key)
        print("-" * len(key))
        report(cohorts, datess[key])
