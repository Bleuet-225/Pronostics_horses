"""Simulation de stratégies de paris en simple placé sur l'historique.

Les gains utilisent les rapports définitifs du PMU (`rapport_place`, gain pour
1 € misé, mise comprise ; 0 si non placé). Les courses sans pari placé
(moins de 4 partants) sont ignorées. Les paris sont joués dans l'ordre des
heures de départ.

Sélections disponibles :
  - favori      : cote finale la plus basse (pari juste avant le départ)
  - prono_pmu   : 1er du pronostic PMU (publié avant les paris)
  - presse      : cheval le plus souvent cité 1er par la presse

Gestions de mise :
  - fixe        : 1 unité à chaque course
  - fibonacci   : 1, 1, 2, 3, 5, 8… ; +1 cran après une perte, -2 crans après un gain
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLE = ["date", "reunion", "course"]


def selection(hist: pd.DataFrame, critere: str) -> pd.DataFrame:
    """Une ligne (le cheval joué) par course où un placé était proposé."""
    h = hist[(hist["place_ouvert"] == 1) & hist["rapport_place"].notna()].copy()
    if critere == "favori":
        h = h.dropna(subset=["cote_direct"]).sort_values(["cote_direct", "num"])
    elif critere == "prono_pmu":
        h = h[h["pmu_prono_rang"] == 1]
    elif critere == "presse":
        h = h[h["presse_nb_premier"] > 0].sort_values(["presse_nb_premier", "presse_nb_cite"],
                                                      ascending=False)
    else:
        raise ValueError(f"Sélection inconnue : {critere}")
    choix = h.groupby(CLE, as_index=False).head(1)
    return choix.sort_values(["date", "heure_depart", "reunion", "course"]).reset_index(drop=True)


def mises(gagne: np.ndarray, methode: str) -> np.ndarray:
    if methode == "fixe":
        return np.ones(len(gagne))
    if methode != "fibonacci":
        raise ValueError(f"Méthode inconnue : {methode}")
    fib = [1, 1]
    res = np.empty(len(gagne))
    i = 0
    for k, g in enumerate(gagne):
        while i >= len(fib):
            fib.append(fib[-1] + fib[-2])
        res[k] = fib[i]
        i = max(0, i - 2) if g else i + 1
    return res


def simuler(paris: pd.DataFrame, methode: str) -> dict:
    gagne = (paris["rapport_place"] > 0).to_numpy()
    m = mises(gagne, methode)
    gains = m * paris["rapport_place"].to_numpy() - m
    solde = np.cumsum(gains)
    # capital nécessaire : pire point bas du solde, en comptant la mise engagée
    capital = float(max(0.0, np.max(m - np.concatenate([[0.0], solde[:-1]]))))
    pertes, serie = 0, 0
    for g in gagne:
        serie = 0 if g else serie + 1
        pertes = max(pertes, serie)
    plus_haut = np.maximum.accumulate(np.concatenate([[0.0], solde]))
    return {
        "paris": int(len(m)),
        "taux_place": float(gagne.mean()) if len(m) else float("nan"),
        "total_mise": float(m.sum()),
        "resultat": float(solde[-1]) if len(m) else 0.0,
        "rendement": float(solde[-1] / m.sum()) if len(m) else float("nan"),
        "pire_baisse": float(np.max(plus_haut - np.concatenate([[0.0], solde]))),
        "plus_longue_serie_perdante": int(pertes),
        "mise_max": float(m.max()) if len(m) else 0.0,
        "capital_necessaire": capital,
        "solde": solde,
    }


def tableau(hist: pd.DataFrame, criteres=("favori",), methodes=("fixe", "fibonacci"),
            debut: str | None = None, fin: str | None = None) -> pd.DataFrame:
    h = hist
    if debut:
        h = h[h["date"] >= debut]
    if fin:
        h = h[h["date"] <= fin]
    lignes = []
    for critere in criteres:
        for typ in ("plat", "obstacle", "tous"):
            sous = h if typ == "tous" else h[h["type"] == typ]
            paris = selection(sous, critere)
            for methode in methodes:
                r = simuler(paris, methode)
                r.pop("solde")
                lignes.append({"selection": critere, "type": typ, "mise": methode, **r})
    return pd.DataFrame(lignes)
