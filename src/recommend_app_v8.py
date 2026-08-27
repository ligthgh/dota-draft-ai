from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

import requests
import streamlit as st

from src.nn_predictor_v8 import NeuralDraftPredictorV8
from src.recommender_v8 import LastPickRecommenderV8

HEROES_URL = "https://api.opendota.com/api/constants/heroes"


@st.cache_data(ttl=86400)
def load_heroes():
    r = requests.get(
        HEROES_URL,
        timeout=30,
    )
    r.raise_for_status()

    heroes = []

    for key, value in r.json().items():
        hero_id = int(
            value.get("id", key)
        )

        name = value.get(
            "localized_name",
            value.get(
                "name",
                str(hero_id),
            ),
        )

        heroes.append(
            (hero_id, name)
        )

    return sorted(
        heroes,
        key=lambda x: x[1].lower(),
    )


@st.cache_resource
def load_engine():
    return LastPickRecommenderV8(
        NeuralDraftPredictorV8()
    )


st.set_page_config(
    page_title="Dota Draft AI v8",
    layout="wide",
)

st.title(
    "Dota Draft AI — подбор пятого героя"
)

st.caption(
    "4 наших героя против 4 известных героев противника. "
    "Пятый герой противника считается неизвестным."
)

try:
    heroes = load_heroes()
    engine = load_engine()

except Exception as exc:
    st.error(str(exc))
    st.info(
        "Обучи V8: python -m src.train_nn_v8"
    )
    st.stop()


hero_ids = [
    h[0]
    for h in heroes
]

labels = {
    hero_id: f"{name} ({hero_id})"
    for hero_id, name in heroes
}


needed_role = st.selectbox(
    "Позиция нашего пятого героя",
    [1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "1 — Carry",
        2: "2 — Mid",
        3: "3 — Offlane",
        4: "4 — Soft Support",
        5: "5 — Hard Support",
    }[x],
)


avg_mmr = st.number_input(
    "Средний MMR",
    min_value=0,
    max_value=15000,
    value=4000,
    step=100,
)


rank_bracket = st.selectbox(
    "MMR bracket",
    list(range(0, 9)),
    format_func=lambda x: {
        0: "Неизвестно",
        1: "<1000",
        2: "1000–1999",
        3: "2000–2999",
        4: "3000–3999",
        5: "4000–4999",
        6: "5000–5999",
        7: "6000–6999",
        8: "7000+",
    }[x],
    index=5,
)


candidate_pick_order = st.slider(
    "Номер нашего следующего пика",
    1,
    10,
    9,
)


left, right = st.columns(2)

my_heroes = []
my_positions = []
my_picks = []

enemy_heroes = []
enemy_positions = []
enemy_picks = []


with left:
    st.subheader("Наша команда — 4 героя")

    for i in range(4):
        my_heroes.append(
            st.selectbox(
                f"Наш герой {i + 1}",
                hero_ids,
                format_func=lambda x: labels[x],
                key=f"my_h_{i}",
                index=i,
            )
        )

        my_positions.append(
            st.selectbox(
                f"Позиция нашего героя {i + 1}",
                [1, 2, 3, 4, 5],
                key=f"my_pos_{i}",
                index=i,
            )
        )

        my_picks.append(
            st.slider(
                f"Порядок пика нашего героя {i + 1}",
                1,
                10,
                min(2 * i + 1, 9),
                key=f"my_pick_{i}",
            )
        )


with right:
    st.subheader(
        "Противник — 4 известных героя"
    )

    for i in range(4):
        enemy_heroes.append(
            st.selectbox(
                f"Герой противника {i + 1}",
                hero_ids,
                format_func=lambda x: labels[x],
                key=f"enemy_h_{i}",
                index=min(
                    i + 10,
                    len(hero_ids) - 1,
                ),
            )
        )

        enemy_positions.append(
            st.selectbox(
                f"Позиция противника {i + 1}",
                [1, 2, 3, 4, 5],
                key=f"enemy_pos_{i}",
                index=i,
            )
        )

        enemy_picks.append(
            st.slider(
                f"Порядок пика противника {i + 1}",
                1,
                10,
                min(2 * i + 2, 10),
                key=f"enemy_pick_{i}",
            )
        )


used_enemy_positions = set(
    enemy_positions
)

missing_positions = [
    pos
    for pos in [1, 2, 3, 4, 5]
    if pos not in used_enemy_positions
]

if len(missing_positions) == 1:
    enemy_missing_role = missing_positions[0]

    st.info(
        f"Неизвестный пятый герой противника: "
        f"позиция {enemy_missing_role}"
    )

else:
    enemy_missing_role = st.selectbox(
        "Позиция неизвестного пятого героя противника",
        [1, 2, 3, 4, 5],
    )


top_k = st.slider(
    "Сколько рекомендаций показать",
    3,
    20,
    10,
)


if st.button(
    "Найти лучший пятый пик",
    type="primary",
    use_container_width=True,
):

    all_known = (
        my_heroes
        + enemy_heroes
    )

    if len(set(all_known)) != len(all_known):
        st.error(
            "В известных героях есть дубликаты."
        )

    elif len(set(my_positions)) != 4:
        st.error(
            "У четырёх наших героев должны быть разные позиции."
        )

    elif len(set(enemy_positions)) != 4:
        st.error(
            "У четырёх героев противника должны быть разные позиции."
        )

    elif needed_role in my_positions:
        st.error(
            "Выбранная позиция пятого героя уже занята нашей командой."
        )

    else:
        recs = engine.recommend(
            my_four=my_heroes,
            enemy_four=enemy_heroes,
            my_positions=my_positions,
            enemy_positions=enemy_positions,
            needed_role=needed_role,
            enemy_missing_role=enemy_missing_role,
            my_picks=my_picks,
            enemy_picks=enemy_picks,
            candidate_pick_order=candidate_pick_order,
            avg_mmr=float(avg_mmr),
            rank_bracket=int(rank_bracket),
            top_k=top_k,
        )

        rows = []

        for rank, rec in enumerate(
            recs,
            start=1,
        ):
            rows.append(
                {
                    "#": rank,
                    "Hero": labels.get(
                        rec.hero_id,
                        str(rec.hero_id),
                    ),
                    "Win probability": round(
                        rec.win_probability * 100,
                        2,
                    ),
                    "Vs average pick": round(
                        rec.delta_vs_average * 100,
                        2,
                    ),
                }
            )

        st.subheader(
            "Рекомендации"
        )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )
