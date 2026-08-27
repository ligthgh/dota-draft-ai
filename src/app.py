from __future__ import annotations

import requests
import streamlit as st

from src.predictor import DraftPredictor


HEROES_URL = "https://api.opendota.com/api/constants/heroes"


@st.cache_data(ttl=86400)
def load_heroes():
    r = requests.get(HEROES_URL, timeout=30)
    r.raise_for_status()
    payload = r.json()
    heroes = []
    for key, value in payload.items():
        hero_id = int(value.get("id", key))
        heroes.append((hero_id, value.get("localized_name", value.get("name", str(hero_id)))))
    return sorted(heroes, key=lambda x: x[1].lower())


@st.cache_resource
def load_predictor():
    return DraftPredictor()


st.set_page_config(page_title="Dota Draft AI", layout="wide")
st.title("Dota Draft AI")
st.caption("MVP: вероятность победы только по 5v5 набору героев.")

try:
    heroes = load_heroes()
    predictor = load_predictor()
except Exception as e:
    st.error(str(e))
    st.stop()

labels = {hero_id: f"{name} ({hero_id})" for hero_id, name in heroes}
hero_ids = [h[0] for h in heroes]

left, right = st.columns(2)

radiant = []
dire = []

with left:
    st.subheader("Radiant")
    for i in range(5):
        radiant.append(
            st.selectbox(
                f"Radiant {i+1}",
                hero_ids,
                format_func=lambda x: labels[x],
                key=f"r{i}",
                index=i,
            )
        )

with right:
    st.subheader("Dire")
    for i in range(5):
        dire.append(
            st.selectbox(
                f"Dire {i+1}",
                hero_ids,
                format_func=lambda x: labels[x],
                key=f"d{i}",
                index=min(i + 5, len(hero_ids) - 1),
            )
        )

if st.button("Analyze draft", type="primary", use_container_width=True):
    try:
        result = predictor.predict(radiant, dire)
    except ValueError as e:
        st.error(str(e))
    else:
        pr = result["radiant_win_probability"]
        st.metric("Radiant win probability", f"{pr:.1%}")
        st.progress(pr)
        st.write(f"Predicted winner: **{result['predicted_winner'].title()}**")

        st.subheader("Largest linear-model contributions")
        rows = []
        for item in result["hero_contributions"]:
            rows.append({
                "Hero": labels.get(item["hero_id"], str(item["hero_id"])),
                "Side": item["side"],
                "Contribution": round(item["logit_contribution"], 4),
                "Known": item["known_to_model"],
            })
        st.dataframe(rows, use_container_width=True)

        with st.expander("Model test metrics"):
            st.json(result["model_metrics"])
