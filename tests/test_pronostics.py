"""Tests de la mécanique du programme.

Les données utilisées ici sont SYNTHÉTIQUES : elles servent uniquement à
vérifier que le code fonctionne, pas à mesurer la qualité des pronostics.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pronostics import features as F
from pronostics import modele
from pronostics.collecte import _arrivees, lignes_course
from pronostics.musique import analyser, caracteristiques


# ------------------------------------------------------------------ musique
def test_analyser_musique():
    perfs = analyser("1p3p(25)0hTsAp")
    assert [(p.resultat, p.discipline) for p in perfs] == [
        ("1", "p"), ("3", "p"), ("0", "h"), ("T", "s"), ("A", "p")]
    assert perfs[2].place == 10
    assert perfs[3].incident and perfs[3].place is None


def test_caracteristiques_meme_discipline():
    c = caracteristiques("1p2h3p4s", "obstacle")
    assert c["mus_nb_meme"] == 2
    assert c["mus_place_moy_meme"] == 3.0
    assert c["mus_victoires"] == 1
    assert c["mus_derniere_place"] == 1


def test_musique_vide():
    c = caracteristiques(None, "plat")
    assert c["mus_nb_perfs"] == 0 and np.isnan(c["mus_place_moy"])
    assert caracteristiques(float("nan"), "plat")["mus_nb_perfs"] == 0


# ------------------------------------------------------------------ collecte
def test_arrivees_ex_aequo():
    assert _arrivees({"ordreArrivee": [[4], [2, 7], [1]]}) == {4: 1, 2: 2, 7: 2, 1: 4}


def test_lignes_course_exclut_non_partants():
    course = {"_reunion": 1, "numOrdre": 2, "_type": "plat", "ordreArrivee": [[2], [1]]}
    parts = {"participants": [
        {"numPmu": 1, "nom": "A", "statut": "PARTANT"},
        {"numPmu": 2, "nom": "B", "statut": "PARTANT"},
        {"numPmu": 3, "nom": "C", "statut": "NON_PARTANT"},
    ]}
    lignes = lignes_course(date(2026, 1, 1), course, parts)
    assert [l["nom"] for l in lignes] == ["A", "B"]
    assert [l["gagnant"] for l in lignes] == [0, 1]


# ------------------------------------------------------------------ données synthétiques
def _synthetique(nb_courses=400, seed=0):
    rng = np.random.default_rng(seed)
    jockeys = [f"J{i}" for i in range(30)]
    lignes = []
    debut = date(2025, 1, 1)
    for k in range(nb_courses):
        d = debut + timedelta(days=k // 4)
        typ = "plat" if k % 2 == 0 else "obstacle"
        n = int(rng.integers(6, 15))
        force = rng.normal(size=n)
        ordre = np.argsort(-(force + rng.normal(scale=1.0, size=n)))
        place = np.empty(n, int)
        place[ordre] = np.arange(1, n + 1)
        for i in range(n):
            nbc = int(rng.integers(0, 30))
            lignes.append({
                "date": d.isoformat(), "reunion": 1 + (k % 4) // 2, "course": k % 4 + 1,
                "hippodrome": "X", "pays": "FRA", "type": typ,
                "discipline": "PLAT" if typ == "plat" else "HAIE", "libelle": "PRIX",
                "distance": 2000.0, "montant_prix": 20000.0, "nb_declares": n,
                "terrain_mesure": 3.5, "handicap": 0, "num": i + 1, "nom": f"C{k}_{i}",
                "age": float(rng.integers(2, 9)), "sexe": "MALES", "oeilleres": None,
                "entraineur": f"E{rng.integers(20)}", "jockey": rng.choice(jockeys),
                "jockey_change": 0, "inedit": int(nbc == 0),
                "musique": "".join(f"{max(1, min(9, int(5 - 2 * force[i] + rng.normal())))}p"
                                   for _ in range(4)),
                "nb_courses": nbc, "nb_victoires": nbc // 5, "nb_places": nbc // 2,
                "gains_carriere": float(np.exp(10 + force[i])), "gains_annee": 1000.0,
                "gains_annee_prec": 1000.0, "poids": 58.0, "decharge": 0.0, "handicap_valeur": np.nan,
                "place_corde": float(i + 1), "cote_direct": float(np.exp(1.5 - force[i])),
                "cote_reference": np.nan, "arrivee": int(place[i]),
                "gagnant": int(place[i] == 1), "top3": int(place[i] <= 3),
            })
    return pd.DataFrame(lignes)


def test_stats_jockey_sans_fuite():
    h = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"],
        "type": ["plat"] * 4, "jockey": ["J"] * 4,
        "gagnant": [1, 1, 0, 0], "top3": [1, 1, 1, 0],
    })
    r = F._stats_acteur(h, h, "jockey")
    # 1er jour : aucune course antérieure ; 2e jour : 2 courses ; 3e jour : 3 courses
    assert r["jockey_nb"].tolist() == [0, 0, 2, 3]
    assert r["jockey_taux_victoire"].iloc[2] == pytest.approx((2 + 1.0) / 12)


def test_normaliser():
    df = pd.DataFrame({"date": ["d"] * 5, "reunion": [1] * 5, "course": [1] * 5})
    brut = np.array([0.9, 0.5, 0.3, 0.2, 0.1])
    g = modele.normaliser(df, brut, "gagnant")
    t = modele.normaliser(df, brut, "top3")
    assert g.sum() == pytest.approx(1.0)
    assert t.sum() == pytest.approx(3.0, abs=1e-3) and t.max() <= 1.0


def test_entrainement_et_prediction(tmp_path):
    hist = _synthetique()
    for t in ("plat", "obstacle"):
        paquet = modele.entrainer(hist, t, dossier=tmp_path)
        m = paquet["metriques"]["gagnant"]
        # Sur ces données construites avec un signal, le modèle doit battre le hasard.
        assert m["taux_reussite_1er_choix"] > m["reference_hasard"]
    partants = hist[hist["date"] == hist["date"].max()].drop(columns=["arrivee", "gagnant", "top3"])
    res = modele.predire(partants, hist, dossier=tmp_path)
    sommes = res.groupby(modele.CLE_COURSE)["proba_gagnant"].sum()
    assert np.allclose(sommes, 1.0)
    assert res["rang_prono"].min() == 1


def test_trop_peu_de_courses(tmp_path):
    with pytest.raises(ValueError, match="au moins"):
        modele.entrainer(_synthetique(nb_courses=20), "plat", dossier=tmp_path)
