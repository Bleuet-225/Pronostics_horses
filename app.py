"""Interface web : streamlit run app.py"""

from datetime import date

import pandas as pd
import streamlit as st

from pronostics import modele, service
from pronostics.api import ErreurAPI

st.set_page_config(page_title="Pronostics hippiques", layout="wide")
st.title("Pronostics hippiques – plat et obstacles")

with st.sidebar:
    jour = st.date_input("Date", value=date.today(), format="DD/MM/YYYY")
    st.caption("Modèles enregistrés")
    for t in ("plat", "obstacle"):
        try:
            p = modele.charger(t)
            m = p["metriques"]["gagnant"]
            st.write(f"**{t}** – entraîné le {p['entraine_le'][:10]} sur {p['nb_courses']} courses ; "
                     f"1er choix gagnant : {m['taux_reussite_1er_choix']:.1%} "
                     f"(hasard : {m['reference_hasard']:.1%})")
        except FileNotFoundError:
            st.warning(f"Pas de modèle « {t} ». Lancez `python -m pronostics entrainer`.")


@st.cache_data(ttl=300, show_spinner="Chargement du programme…")
def _courses(d):
    return [{"reunion": c["_reunion"], "course": c["numOrdre"], "hippodrome": c["_hippodrome"],
             "libelle": c.get("libelle", ""), "type": c["_type"],
             "discipline": c.get("discipline", ""), "distance": c.get("distance")}
            for c in service.courses_du_jour(d)]


try:
    courses = _courses(jour)
except ErreurAPI as e:
    st.error(f"Impossible de joindre l'API PMU : {e}")
    st.stop()

if not courses:
    st.info("Aucune course de plat ou d'obstacles pour cette date.")
    st.stop()

libelles = {f"R{c['reunion']}C{c['course']} – {c['hippodrome']} – {c['libelle']} "
            f"({c['type']}, {c['distance']} m)": c for c in courses}
choix = libelles[st.selectbox("Course", list(libelles))]

try:
    res = service.pronostiquer(jour, choix["reunion"], choix["course"])
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if res.empty:
    st.info("Pas de partants disponibles pour cette course.")
    st.stop()

aff = res[service.COLONNES_AFFICHAGE].copy()
aff["proba_gagnant"] *= 100
aff["proba_top3"] *= 100
aff = aff.rename(columns={"rang_prono": "Rang", "num": "N°", "nom": "Cheval", "jockey": "Jockey",
                          "entraineur": "Entraîneur", "musique": "Musique", "cote_direct": "Cote",
                          "proba_gagnant": "% victoire", "proba_top3": "% top 3",
                          "arrivee": "Arrivée"})
if aff["Arrivée"].isna().all():
    aff = aff.drop(columns="Arrivée")

st.dataframe(
    aff, hide_index=True, use_container_width=True,
    column_config={
        "% victoire": st.column_config.ProgressColumn(format="%.1f %%", min_value=0, max_value=100),
        "% top 3": st.column_config.ProgressColumn(format="%.1f %%", min_value=0, max_value=100),
        "Rang": st.column_config.NumberColumn(format="%d"),
    },
)
st.bar_chart(pd.DataFrame({"% victoire": aff["% victoire"].to_numpy()},
                          index=aff["Cheval"].astype(str)), horizontal=True)
st.caption("Probabilités estimées par le modèle ; elles ne garantissent aucun résultat. "
           "Jouer comporte des risques : endettement, dépendance… Appelez le 09 74 75 13 13 (appel non surtaxé).")
