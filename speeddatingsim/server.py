import asyncio
import enum
import re
import functools
import random
from collections.abc import Awaitable

import numpy as np
import qrcode
import qrcode.image.svg
import jinja2.filters
from quart import Quart, Response, abort, render_template, redirect, jsonify, url_for, make_response, request
from pony import orm

from speeddatingsim.feistel import permute, unpermute
from speeddatingsim.mwmatching import maxWeightMatching
from speeddatingsim.tarot import TAROT_CARDS
from speeddatingsim.wordlists import ADJECTIVES, NOUNS, VERBS


db = orm.Database()


class SessionStatus(enum.IntEnum):
    PENDING = enum.auto()
    ACTIVE = enum.auto()
    CLOSED = enum.auto()


class Session(db.Entity):
    name = orm.Required(str)
    users = orm.Set("User")
    dates = orm.Set("Date")
    rounds = orm.Set("Round")
    status = orm.Required(SessionStatus)


class User(db.Entity):
    name = orm.Required(str)
    tarot = orm.Required(int)
    sessions = orm.Set(Session)
    lefts = orm.Set("Date", reverse="left")
    rights = orm.Set("Date", reverse="right")
    recommendations = orm.Set("Recommendation", reverse="subject")
    recommended_to = orm.Set("Recommendation", reverse="object")
    similarities = orm.Set("Similarity", reverse="subject")
    similar_to = orm.Set("Similarity", reverse="object")


class Date(db.Entity):
    session = orm.Required(Session)
    tableno = orm.Required(int)
    left = orm.Required(User, reverse="lefts")
    right = orm.Required(User, reverse="rights")
    decision_left = orm.Optional(bool)
    decision_right = orm.Optional(bool)


class Round(db.Entity):
    session = orm.Set(Session)
    similarities = orm.Set("Similarity")
    recommendations = orm.Set("Recommendation")


class Similarity(db.Entity):
    round = orm.Required(Round)
    subject = orm.Required(User, reverse="similarities")
    object = orm.Required(User, reverse="similar_to")
    weight = orm.Required(float)


class Recommendation(db.Entity):
    round = orm.Required(Round)
    subject = orm.Required(User, reverse="recommendations")
    object = orm.Required(User, reverse="recommended_to")
    weight = orm.Required(float)


app = Quart(__name__)


def with_user(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        with orm.db_session:
            if not (
                (userid := request.cookies.get("userid"))
                and (user := User.get(id=int(userid)))
            ):
                tarot_card = random.choice(TAROT_CARDS)
                adjective = random.choice(ADJECTIVES)

                user = User(
                    name=f"{adjective.title()} {tarot_card.noun}",
                    tarot=tarot_card.index,
                )
                user.flush()
        response = await make_response(await func(*args, userid=user.id, **kwargs))
        response.set_cookie("userid", str(user.id))
        return response
    return wrapper


@app.template_filter('qr')
async def make_qr(data: str) -> str:
    qr = qrcode.make(
        data,
        image_factory=qrcode.image.svg.SvgPathImage,
    )
    return jinja2.filters.do_mark_safe(qr.to_string().decode())


@app.route("/")
async def index():
    with orm.db_session:
        sessions = list(Session.select())
    return await render_template("index.html", sessions=sessions)


@app.route("/sessions", methods=["GET", "POST"])
async def sessions_page():
    if request.method == "GET":
        with orm.db_session:
            sessions = list(Session.select())
        return await render_template(
            "sessions.html",
            sessions=sessions,
            refresh_url=url_for('sessions_page_events'),
        )
    elif request.method == "POST":
        with orm.db_session:
            session = Session(
                name=(await request.form)["Session name"],
                status=SessionStatus.PENDING
            )
        await sessions_notify_subscribers()
        return redirect(url_for('session_page', sessionid=session.id))


@app.route("/sessions/events")
async def sessions_page_events():
    event = asyncio.Event()
    await sessions_subscribe_to_changes(event)
    return Response(
        stream_notifications(event, sessions_unsubscribe(event)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/sessions/<int:sessionid>")
async def session_page(sessionid: int):
    with orm.db_session:
        session = Session.get(id=sessionid)
        session.load()

        users = list(session.users.order_by(User.id))
        for user in users:
            user.load()
        
        tables = {}
        for date in session.dates:
            if not (other := tables.get(date.tableno)) or date.id > other.id:
                tables[date.tableno] = date
        
        dates = sorted(tables.values(), key=lambda date: date.tableno)

        historical_dates = list(
            orm.select(
                date for date in session.dates
                if date.decision_left is not None or date.decision_right is not None
            )
        )
        decisions = {
            user.id: {
                other.id: ""
                for other in session.users
            }
            for user in session.users
        }
        for date in historical_dates:
            if date.decision_left is not None:
                decisions[date.left.id][date.right.id] = "Y" if date.decision_left else "N"
            if date.decision_right is not None:
                decisions[date.right.id][date.left.id] = "Y" if date.decision_right else "N"


    if session:
        return await render_template(
            "session.html",
            session=session,
            users=users,
            dates=dates,
            decisions=decisions,
            refresh_url=url_for("session_page_events", sessionid=sessionid),
        )
    abort(404)


@app.route("/sessions/<int:sessionid>/events")
async def session_page_events(sessionid: int):
    with orm.db_session:
        session = Session.get(id=sessionid)
    if session:
        event = asyncio.Event()
        await session_subscribe_to_changes(sessionid, event)
        return Response(
            stream_notifications(event, session_unsubscribe(sessionid, event)),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    abort(404)


@app.route("/sessions/<int:sessionid>/start", methods=["POST"])
async def session_start(sessionid: int):
    with orm.db_session:
        session = Session.get(id=sessionid)
    if session:
        session.status = SessionStatus.ACTIVE
        return redirect(url_for("session_page", sessionid=session.id))
    abort(404)


@app.route("/sessions/<int:sessionid>/end", methods=["POST"])
async def session_end(sessionid: int):
    with orm.db_session:
        session = Session.get(id=sessionid)
    if session:
        session.status = SessionStatus.CLOSED
        return redirect(url_for("session_page", sessionid=session.id))
    abort(404)


@app.route("/sessions/<int:sessionid>/matchmaker")
@with_user
async def matchmaker_page(sessionid: int, userid: int):
    with orm.db_session:
        if (
            (session := Session.get(id=sessionid))
            and (user := User.get(id=userid))
        ):
            session.load()
            user.load()
            session.users.add(user)
            if date := get_current_date(sessionid, userid):
                tableno = date.tableno
                if date.left.id == user.id:
                    other_user = date.right
                    decision = date.decision_left
                else:
                    other_user = date.left
                    decision = date.decision_right
                other_user.load()
            else:
                tableno = None
                other_user = None
                decision = None

            if round := orm.select(round for round in session.rounds).order_by(orm.desc(Round.id)).first():
                similarities = list(
                    orm.select(
                        similarity
                        for similarity in round.similarities
                        if similarity.subject == user
                    )
                    .prefetch(Similarity.object)
                    .order_by(orm.desc(Similarity.weight))
                )

                recommendations = list(
                    orm.select(
                        recommendation
                        for recommendation in round.recommendations
                        if recommendation.subject == user
                    )
                    .prefetch(Recommendation.object)
                    .order_by(orm.desc(Recommendation.weight))
                )
            else:
                similarities = None
                recommendations = None

    
    if session and user:
        await session_notify_subscribers(sessionid)
        return await render_template(
            "matchmaker.html",
            session=session,
            user=user,
            tableno=tableno,
            no_decision=decision is None,
            card=TAROT_CARDS[user.tarot],
            other_user=other_user,
            other_card=TAROT_CARDS[other_user.tarot] if other_user is not None else None,
            refresh_url=url_for('matchmaker_page_events', sessionid=sessionid),
            round=round,
            similarities=similarities,
            recommendations=recommendations,
        )
    abort(404)


@app.route("/sessions/<int:sessionid>/matchmaker/events")
@with_user
async def matchmaker_page_events(sessionid: int, userid: int):
    event = asyncio.Event()
    await user_subscribe_to_changes(userid, event)
    return Response(
        stream_notifications(event, user_unsubscribe(userid, event)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/sessions/<int:sessionid>/decide", methods=["POST"])
@with_user
async def user_decide(sessionid: int, userid: int):
    if (await request.form).get("decide_yes"):
        verdict = True
    elif (await request.form).get("decide_no"):
        verdict = False
    else:
        verdict = None
    
    with orm.db_session:
        session = Session.get(id=sessionid)
        if not session:
            abort(404)

        user = User.get(id=userid)
        if not user:
            abort(500)

        if date := orm.select(
            date for date in Date
            if date.session == session
            if date.left == user
            if date.decision_left is None
        ).first():
            date.decision_left = verdict

        elif date := orm.select(
            date for date in Date
            if date.session == session
            if date.right == user
            if date.decision_right is None
        ).first():
            date.decision_right = verdict

    await user_notify_subscribers(user.id)
    await session_notify_subscribers(sessionid)
    return redirect(request.referrer or url_for('matchmaker_page', sessionid=sessionid))


@app.route("/user")
@with_user
async def user_page(userid: int):
    with orm.db_session:
        user = User.get(id=userid)
        user.load()

    return await render_template(
        "user.html",
        user=user,
        recovery_phrase=id_to_recovery_phrase(user.id),
    )


@app.route("/user/su", methods=["POST"])
async def user_change_identity():
    if recovery_phrase := (await request.form).get("phrase"):
        try:
            tgt_id = recovery_phrase_to_id(recovery_phrase)
        except LookupError:
            return redirect(request.referrer or url_for('user_page'))

        with orm.db_session:
            user = User.get(id=tgt_id)
            if not user:
                return redirect(request.referrer or url_for('user_page'))
        
        response = redirect(request.referrer or url_for('user_page'))
        response.set_cookie("userid", str(user.id))
        return response

    return redirect(request.referrer or url_for('user_page'))


@app.route("/user/draw_tarot", methods=["POST"])
@with_user
async def user_draw_tarot(userid: int):
    tarot = random.choice(TAROT_CARDS)
    with orm.db_session:
        user = User.get(id=userid)
        if not user:
            abort(500)
        user_tarot = TAROT_CARDS[user.tarot]
        user.name = re.sub(rf"\b{user_tarot.noun}\b", tarot.noun, user.name)
        user.tarot = tarot.index
        user.sessions.load()
    await user_notify_subscribers(user.id)
    for session in user.sessions:
        await session_notify_subscribers(session.id)
    return redirect(request.referrer or url_for('index'))


@app.route("/user/draw_adjective", methods=["POST"])
@with_user
async def user_draw_adjective(userid: int):
    adjective = random.choice(ADJECTIVES)
    with orm.db_session:
        user = User.get(id=userid)
        if not user:
            abort(500)
        old_name = user.name
        for user_adjective in ADJECTIVES:
            user.name = re.sub(rf"\b{user_adjective}\b", adjective, user.name)
        user.sessions.load()
    if user.name != old_name:
        await user_notify_subscribers(user.id)
        for session in user.sessions:
            await session_notify_subscribers(session.id)
    return redirect(request.referrer or url_for('index'))


@app.route("/user", methods=["POST"])
@with_user
async def user_edit(userid: int):
    if new_name := (await request.form).get("name"):
        with orm.db_session:
            user = User.get(id=userid)
            if not user:
                abort(500)
            user.name = new_name
            user.sessions.load()
        await user_notify_subscribers(user.id)
        for session in user.sessions:
            await session_notify_subscribers(session.id)
    return redirect(request.referrer or url_for('index'))


@app.route("/tarot")
async def tarots_page():
    return await render_template("tarots.html", cards=TAROT_CARDS)


@app.route("/tarot/<int:index>")
async def tarot_page(index: int):
    return await render_template("tarot.html", card=TAROT_CARDS[index])


@app.route("/sessions/<int:sessionid>/matchmake", methods=["POST"])
async def matchmake(sessionid: int):
    with orm.db_session:
        session = Session.get(id=sessionid)
        if not session:
            abort(404)

        users = orm.select(user for user in session.users)
        all_users: list[User] = list(users)

        # remove any users with undecided dates from consideration
        eligible: set[int] = {
            user.id for user in users.filter(
                lambda user: not orm.exists(
                    date for date in Date
                    if date.session.id == sessionid
                    if date.left.id == user.id
                    if date.decision_left is None
                )
            ).filter(
                lambda user: not orm.exists(
                    date for date in Date
                    if date.session.id == sessionid
                    if date.right.id == user.id
                    if date.decision_right is None
                )
            )
        }

        n = len(all_users)
        index_map = {user.id: i for i, user in enumerate(all_users)}
        table_map = {}
        for user in all_users:
            if date := get_current_date(sessionid, user.id):
                table_map[user.id] = date.tableno

        # build the decisions matrix
        decisions = np.zeros((n, n), dtype=np.int8)
        for i, user in enumerate(all_users):
            for date in orm.select(
                date for date in Date
                if date.session.id == sessionid
                if date.left == user
            ):
                j = index_map[date.right.id]
                decisions[i, j] = (1 if date.decision_left else -1)
                decisions[j, i] = (1 if date.decision_right else -1)
        
    # calculate similarity
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

    graph = []
    for i in range(n):
        if all_users[i].id not in eligible:
            continue
        for j in range(i + 1, n):
            if all_users[j].id not in eligible:
                continue
            if decisions[i, j] == 0:
                # basic: match people who are recommended to each other
                weight = min(
                    recommendations[i, j],  # * horniness[j] / (hotness[i] + 1),
                    recommendations[j, i],  # * horniness[i] / (hotness[j] + 1),
                )
                # algo works better without floating point error
                graph.append((i, j, int(weight * 10000)))
    random.shuffle(graph)

    mate = maxWeightMatching(graph, maxcardinality=True)

    # match dates to tables to minimize movement
    def goodness(i, j, tableno):
        if (
            (i_table := table_map.get(all_users[i].id))
            and (j_table := table_map.get(all_users[j].id))
        ):
            return -(
                abs(tableno - i_table)
                + abs(tableno - j_table)
            )
        else:
            return 0
    
    dates = [(i, j) for i, j in enumerate(mate) if j > i]

    num_tables = len(dates) if not table_map else max(len(dates), *table_map.values())
    table_graph = [
        (tableno, num_tables + k, goodness(i, j, tableno))
        for k, (i, j) in enumerate(dates)
        for tableno in range(num_tables)
    ]
    table_assignments = maxWeightMatching(table_graph, maxcardinality=True)

    # record metadata
    with orm.db_session:
        round = Round(session=Session.get(id=sessionid))
        for i in range(n):
            for j in range(n):
                if i != j:
                    Similarity(
                        round=round,
                        subject=User.get(id=all_users[i].id),
                        object=User.get(id=all_users[j].id),
                        weight=S[i, j],
                    )
        
        for i, j, w in graph:
            Recommendation(
                round=round,
                subject=User.get(id=all_users[i].id),
                object=User.get(id=all_users[j].id),
                weight=w,
            )
            Recommendation(
                round=round,
                subject=User.get(id=all_users[j].id),
                object=User.get(id=all_users[i].id),
                weight=w,
            )

    # perform the dates
    changed_users = set()

    with orm.db_session:
        session = Session.get(id=sessionid)

        for tableno, k in enumerate(table_assignments[:num_tables]):
            if k < 0:
                pass

            i, j = dates[k - num_tables]
            changed_users.add(all_users[i].id)
            changed_users.add(all_users[j].id)
            left = User.get(id=all_users[i].id)
            right = User.get(id=all_users[j].id)
            date = Date(session=session, tableno=tableno, left=left, right=right)

    for userid in changed_users:
        await user_notify_subscribers(userid)
    
    await session_notify_subscribers(sessionid)

    return redirect(request.referrer or url_for("session_page", sessionid=sessionid))


def get_current_date(sessionid: int, userid: int) -> Date | None:
    return orm.select(
        date for date in Date
        if date.session.id == sessionid
        if (date.left.id == userid or date.right.id == userid)
    ).sort_by(orm.desc(Date.id)).first()


def get_other_user(sessionid: int, userid: int) -> User | None:
    if date := get_current_date(sessionid, userid):
        if date.left.id == userid:
            return date.right
        if date.right.id == userid:
            return date.left
    return None
    

user_notify_on_changes: dict[int, set[asyncio.Event]] = {}
session_notify_on_changes: dict[int, set[asyncio.Event]] = {}
sessions_notify_on_changes: set[asyncio.Event] = set()

async def user_notify_subscribers(id: int) -> None:
    for event in user_notify_on_changes.get(id, set()):
        event.set()

async def session_notify_subscribers(id: int) -> None:
    for event in session_notify_on_changes.get(id, set()):
        event.set()

async def sessions_notify_subscribers() -> None:
    for event in sessions_notify_on_changes:
        event.set()

async def user_subscribe_to_changes(id: int, event: asyncio.Event):
    user_notify_on_changes.setdefault(id, set()).add(event)

async def session_subscribe_to_changes(id: int, event: asyncio.Event):
    session_notify_on_changes.setdefault(id, set()).add(event)

async def sessions_subscribe_to_changes(event: asyncio.Event):
    sessions_notify_on_changes.add(event)

async def user_unsubscribe(id: int, event: asyncio.Event):
    user_notify_on_changes.setdefault(id, set()).discard(event)

async def session_unsubscribe(id: int, event: asyncio.Event):
    session_notify_on_changes.setdefault(id, set()).discard(event)

async def sessions_unsubscribe(event: asyncio.Event):
    sessions_notify_on_changes.discard(event)

async def wait_and_reset(event: asyncio.Event, timeout: float) -> bool:
    """Runs on the asyncio loop.
    Returns True if event fired, False on timeout."""
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        event.clear()
        return True
    except asyncio.TimeoutError:
        return False

async def stream_notifications(event: asyncio.Event, teardown: Awaitable[None]):
    try:
        while True:
            fired = await wait_and_reset(event, timeout=5.0)
            if fired:
                yield "event: update\ndata: \n\n"
            else:
                yield "event: ping\ndata: \n\n"
    finally:
        await teardown


@app.before_serving
def init_db():
    db.bind(provider="sqlite", filename="database.sqlite", create_db=True)
    db.generate_mapping(create_tables=True)


M = len(ADJECTIVES) * len(NOUNS) * len(VERBS) * len(ADJECTIVES)
KEY = b'obstreperous nightmare'


def id_to_recovery_phrase(userid: int) -> str:
    x = permute(userid, M, KEY)
    i, x = x % len(ADJECTIVES), x // len(ADJECTIVES)
    j, x = x % len(NOUNS), x // len(NOUNS)
    k, x = x % len(VERBS), x // len(VERBS)
    l, x = x % len(ADJECTIVES), x // len(ADJECTIVES)

    adjective = ADJECTIVES[i]
    noun = NOUNS[j]
    verb = VERBS[k]
    adjective2 = ADJECTIVES[l]
    if adjective2.endswith("y"):
        adverb = adjective2[:-1] + "ily"
    else:
        adverb = adjective2 + "ly"
    return f"{adjective} {noun} {verb} {adverb}".lower()


def recovery_phrase_to_id(phrase: str) -> int:
    try:
        adjective, noun, verb, adverb = phrase.split()
    except ValueError:
        raise LookupError("could not parse recovery phrase")
    
    try:
        i = ADJECTIVES.index(adjective.title())
    except ValueError:
        raise LookupError("adjective not in list")
    
    try:
        j = NOUNS.index(noun.title())
    except ValueError:
        raise LookupError("noun not in list")
    
    try:
        k = VERBS.index(verb.title())
    except ValueError:
        raise LookupError("verb not in list")

    try:
        assert adverb.endswith("ly")
        if adverb.endswith("ily"):
            adjective2 = adverb[:-3] + "y"
        else:
            adjective2 = adverb[:-2]
        l = ADJECTIVES.index(adjective2.title())
    except (AssertionError, ValueError):
        raise LookupError("adverb not in list")
    
    x = i + len(ADJECTIVES) * (j + len(NOUNS) * (k + len(VERBS) * l))
    return unpermute(x, M, KEY)


if __name__ == "__main__":
    app.run(port=8000)
