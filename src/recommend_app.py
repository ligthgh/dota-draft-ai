from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from src.nn_predictor import NeuralDraftPredictor
from src.recommender import LastPickRecommender

HEROES_URL = "https://api.opendota.com/api/constants/heroes"
WINDOW_META = Path("data/recent_window.json")


@st.cache_data(ttl=86400)
def load_heroes():
    response = requests.get(HEROES_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    heroes = []
    for key, value in payload.items():
        hero_id = int(value.get("id", key))
        name = value.get("localized_name", value.get("name", str(hero_id)))
        heroes.append((hero_id, name))
    return sorted(heroes, key=lambda x: x[1].lower())


@st.cache_resource
def load_engine():
    return LastPickRecommender(NeuralDraftPredictor())


def recent_window_label() -> str | None:
    if not WINDOW_META.exists():
        return None

    try:
        data = json.loads(
            WINDOW_META.read_text(encoding="utf-8")
        )
        days = int(data.get("days", 30))

        if days:
            return f"{days} дн."

    except Exception:
        pass

    return None


st.set_page_config(page_title="Dota Draft AI — Last Pick", layout="wide")
st.title("Dota Draft AI — Fifth Pick")
window = recent_window_label()
if window:
    st.caption(
        f"Модель и статистика: только самые свежие матчи за последние {window} "
        "Сторона карты не учитывается."
    )
else:
    st.caption(
        "4 своих героя + 5 героев противника → лучший пятый пик. "
        "Сторона карты не учитывается."
    )

try:
    heroes = load_heroes()
    engine = load_engine()
except Exception as exc:
    st.error(str(exc))
    st.info("Сначала собери свежие матчи и обучи модель: python -m src.train_nn")
    st.stop()

hero_ids = [h[0] for h in heroes]
labels = {hero_id: f"{name} ({hero_id})" for hero_id, name in heroes}

role = st.selectbox(
    "На какую позицию нужен пятый герой?",
    [1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "1 — Carry",
        2: "2 — Mid",
        3: "3 — Offlane",
        4: "4 — Soft Support",
        5: "5 — Hard Support",
    }[x],
)

left, right = st.columns(2)
my_team = []
enemy = []

with left:
    st.subheader("Моя команда: 4 героя")
    for i in range(4):
        my_team.append(
            st.selectbox(
                f"Свой герой {i + 1}",
                hero_ids,
                format_func=lambda x: labels[x],
                key=f"own_{i}",
                index=i,
            )
        )

with right:
    st.subheader("Противник: 5 героев")
    for i in range(5):
        enemy.append(
            st.selectbox(
                f"Герой противника {i + 1}",
                hero_ids,
                format_func=lambda x: labels[x],
                key=f"enemy_{i}",
                index=min(i + 10, len(hero_ids) - 1),
            )
        )

top_k = st.slider("Сколько вариантов показать", 3, 20, 10)

if st.button("Найти лучший пятый пик", type="primary", use_container_width=True):
    try:
        recs = engine.recommend(my_team, enemy, role=role, top_k=top_k)
        rows = []
        for rank, rec in enumerate(recs, start=1):
            rows.append(
                {
                    "#": rank,
                    "Hero": labels.get(rec.hero_id, str(rec.hero_id)),
                    "Win probability": round(rec.win_probability * 100, 2),
                    "Advantage vs avg pick": round(rec.delta_vs_baseline * 100, 2),
                }
            )

        st.subheader("Рекомендации")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if recs:
            best = recs[0]
            st.success(
                f"Лучший кандидат: {labels.get(best.hero_id, str(best.hero_id))} — "
                f"{best.win_probability:.1%}"
            )
    except ValueError as exc:
        st.error(str(exc))
