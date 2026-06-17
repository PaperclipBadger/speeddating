import enum
import re
import functools
import random

import qrcode
import qrcode.image.svg
import jinja2.filters
from flask import Flask, abort, render_template, redirect, jsonify,  url_for, make_response, request
from pony import orm
from pony.flask import Pony

from speeddatingsim.tarot import TAROT_CARDS


db = orm.Database()


class SessionStatus(enum.IntEnum):
    PENDING = enum.auto()
    ACTIVE = enum.auto()
    CLOSED = enum.auto()


class Session(db.Entity):
    name = orm.Required(str)
    users = orm.Set("User")
    status = orm.Required(SessionStatus)


class User(db.Entity):
    name = orm.Required(str)
    tarot = orm.Required(int)
    sessions = orm.Set(Session)
    dates = orm.Set("User", reverse="dates")
    verdicts = orm.Set("Verdict")
    defenses = orm.Set("Verdict")


class Verdict(db.Entity):
    judge = orm.Required(User, reverse="verdicts")
    defendant = orm.Required(User, reverse="defenses")
    decision = orm.Required(bool)



app = Flask(__name__)
Pony(app)


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
    def wrapper(*args, **kwargs):
        if userid := request.cookies.get("userid"):
            user = User.get(id=int(userid))
        else:
            tarot_card = random.choice(TAROT_CARDS)
            adjective = random.choice(ADJECTIVES)
            user = User(
                name=f"{adjective.title()} {tarot_card.noun}",
                tarot=tarot_card.index,
            )
            user.flush()
        response = make_response(func(*args, user=user, **kwargs))
        response.set_cookie("userid", str(user.id))
        return response
    return wrapper


@app.template_filter('qr')
def make_qr(data: str) -> str:
    qr = qrcode.make(
        data,
        image_factory=qrcode.image.svg.SvgPathImage,
    )
    return jinja2.filters.do_mark_safe(qr.to_string().decode())


@app.route("/")
def index():
    return render_template("index.html", sessions=Session.select())


@app.route("/sessions", methods=["GET", "POST"])
def sessions_page():
    if request.method == "GET":
        return render_template("sessions.html", sessions=Session.select())
    elif request.method == "POST":
        session = Session(
            name=request.form["Session name"], status=SessionStatus.PENDING
        )
        session.flush()
        return redirect(url_for('session_page', sessionid=session.id))


@app.route("/sessions/<int:sessionid>")
def session_page(sessionid: int):
    if session := Session.get(id=sessionid):
        return render_template("session.html", session=session)
    abort(404)


@app.route("/sessions/<int:sessionid>/start", methods=["POST"])
def session_start(sessionid: int):
    if session := Session.get(id=sessionid):
        session.status = SessionStatus.ACTIVE
        return redirect(url_for("session_page", sessionid=session.id))
    abort(404)


@app.route("/sessions/<int:sessionid>/end", methods=["POST"])
def session_end(sessionid: int):
    if session := Session.get(id=sessionid):
        session.status = SessionStatus.CLOSED
        return redirect(url_for("session_page", sessionid=session.id))
    abort(404)


@app.route("/sessions/<int:sessionid>/matchmaker")
@with_user
def matchmaker_page(sessionid: int, user: User):
    if session := Session.get(id=sessionid):
        session.users.add(user)
        return render_template(
            "matchmaker.html",
            session=session,
            user=user,
            card=TAROT_CARDS[user.tarot],
            refresh_url=url_for('matchmaker_page_stale', sessionid=sessionid),
        )
    abort(404)


@app.route("/sessions/<int:sessionid>/matchmaker/stale")
@with_user
def matchmaker_page_stale(sessionid: int, user: User):
    return jsonify(False)


@app.route("/user/draw_tarot", methods=["POST"])
@with_user
def user_draw_tarot(user: User):
    tarot = random.choice(TAROT_CARDS)
    user_tarot = TAROT_CARDS[user.tarot]
    user.name = re.sub(rf"\b{user_tarot.noun}\b", tarot.noun, user.name)
    user.tarot = tarot.index
    return redirect(request.referrer or url_for('/user'))


@app.route("/user/draw_adjective", methods=["POST"])
@with_user
def user_draw_adjective(user: User):
    adjective = random.choice(ADJECTIVES)
    for user_adjective in ADJECTIVES:
        user.name = re.sub(rf"\b{user_adjective}\b", adjective, user.name)
    return redirect(request.referrer or url_for('/user'))


@app.route("/user", methods=["POST"])
@with_user
def user_edit(user: User):
    if new_name := request.form.get("name"):
        user.name = new_name
    return redirect(request.referrer or url_for('/user'))


@app.route("/tarot")
def tarots_page():
    return render_template("tarots.html", cards=TAROT_CARDS)


@app.route("/tarot/<int:index>")
def tarot_page(index: int):
    return render_template("tarot.html", card=TAROT_CARDS[index])


if __name__ == "__main__":
    db.bind(provider="sqlite", filename="database.sqlite", create_db=True)
    db.generate_mapping(create_tables=True)
    app.run(port=8000)
