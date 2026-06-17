import asyncio
import contextlib
import concurrent.futures
import enum
import itertools
import re
import functools
import random
from collections.abc import Awaitable, Generator

import numpy as np
import qrcode
import qrcode.image.svg
import jinja2.filters
from quart import Quart, Response, abort, render_template, redirect, jsonify, url_for, make_response, request
from pony import orm

from speeddatingsim.mwmatching import maxWeightMatching
from speeddatingsim.tarot import TAROT_CARDS


db = orm.Database()


class SessionStatus(enum.IntEnum):
    PENDING = enum.auto()
    ACTIVE = enum.auto()
    CLOSED = enum.auto()


class Session(db.Entity):
    name = orm.Required(str)
    users = orm.Set("User")
    dates = orm.Set("Date")
    status = orm.Required(SessionStatus)
    freshness = orm.Required(int, default=0)


class User(db.Entity):
    name = orm.Required(str)
    tarot = orm.Required(int)
    sessions = orm.Set(Session)
    lefts = orm.Set("Date", reverse="left")
    rights = orm.Set("Date", reverse="right")
    freshness = orm.Required(int, default=0)


class Date(db.Entity):
    session = orm.Required("Session")
    left = orm.Required("User", reverse="lefts")
    right = orm.Required("User", reverse="rights")
    decision_left = orm.Optional(bool)
    decision_right = orm.Optional(bool)


app = Quart(__name__)


ADJECTIVES = [
    "Sexy",
    "Alluring",
    "Intriguing",
    "Kinky",
    "Curious",
    "Aloof",
    "Dominant",
    "Submissive",
]


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
        session.users.load()
        for user in session.users:
            user.load()

    if session:
        return await render_template(
            "session.html",
            session=session,
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
            other_right = orm.select(
                date.right for date in Date
                if date.session == session
                if date.left == user
                if date.decision_left is None
            ).first()
            other_left = orm.select(
                date.left for date in Date
                if date.session == session
                if date.right == user
                if date.decision_right is None
            ).first()
            other_user = other_left or other_right
            if other_user:
                other_user.load()
    
    if session and user:
        await session_notify_subscribers(sessionid)
        return await render_template(
            "matchmaker.html",
            session=session,
            user=user,
            other=other_user,
            card=TAROT_CARDS[user.tarot],
            refresh_url=url_for('matchmaker_page_events', sessionid=sessionid),
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
    return redirect(request.referrer or url_for('/user'))


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
    return redirect(request.referrer or url_for('/user'))


@app.route("/user", methods=["POST"])
@with_user
async def user_edit(userid: int):
    if new_name := request.form.get("name"):
        with orm.db_session:
            user = User.get(id=userid)
            if not user:
                abort(500)
            user.name = new_name
            user.sessions.load()
        await user_notify_subscribers(user.id)
        for session in user.sessions:
            await session_notify_subscribers(session.id)
    return redirect(request.referrer or url_for('/user'))


@app.route("/tarot")
async def tarots_page():
    return await render_template("tarots.html", cards=TAROT_CARDS)


@app.route("/tarot/<int:index>")
async def tarot_page(index: int):
    return await render_template("tarot.html", card=TAROT_CARDS[index])


async def matchmake(sessionid: int):
    with orm.db_session:
        # remove any users with undecided dates from consideration
        users: list[User] = orm.select(
            user for user in User
            if sessionid in user.session.id
            if all(
                date.decision_left is not None
                for date in Date
                if date.session.id == sessionid
                if date.left.id == user.id
            )
            if all(
                date.decision_right is not None
                for date in Date
                if date.session.id == sessionid
                if date.right.id == user.id
            )
        )

        n = len(users)
        index_map = {user.id: i for user in users}

        # build the decisions matrix
        decisions = np.zeros((n, n), dtype=np.int8)
        for i, user in enumerate(users):
            for date in orm.select(
                date for date in Date
                if date.left.id == user.id
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
        for j in range(i + 1, n):
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

    # perform the dates
    with orm.db_session:
        session = Session.get(id=sessionid)

        for i, j in enumerate(mate):
            if j < i:
                continue
            
            left = User.get(id=users[i].id)
            left.freshness += 1
            right = User.get(id=users[j].id)
            right.freshness += 1
            date = Date(session=session, left=left, right=right)


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


if __name__ == "__main__":
    db.bind(provider="sqlite", filename="database.sqlite", create_db=True)
    db.generate_mapping(create_tables=True)

    app.run(port=8000)
