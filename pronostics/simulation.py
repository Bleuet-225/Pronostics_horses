"""Simulation de stratégies de paris en simple placé sur l'historique.

Les gains utilisent les rapports définitifs du PMU (`rapport_place`, gain pour
1 € misé, mise comprise ; 0 si non placé). Les courses sans pari placé
(moins de 4 partants) sont ignorées. Les paris sont joués dans l'ordre des
heures de départ.

Sélections disponibles :
  - favori      : cote finale la plus basse (pari juste avant le départ)
  - favori_30min: cote de référence la plus basse (relevée ~30 min avant le
                  départ) ; seules les courses où tous les partants ont cette
                  cote sont gardées (sinon elle est peu significative)
  - prono_pmu   : 1er du pronostic PMU (publié avant les paris)
  - presse      : cheval le plus souvent cité 1er par la presse

Filtres (voir `Filtres`) : sans handicap, sans Groupe, cote maximale du
cheval joué, écart minimal = (plus basse cote des autres partants) / (cote
du cheval joué). Pour un favori, l'écart est le rapport 2e favori / favori.
La cote utilisée est celle connue au moment du choix : cote finale pour
« favori », cote de référence (~30 min) pour les autres sélections.

Gestions de mise :
  - fixe        : 1 unité à chaque course
  - fibonacci   : 1, 1, 2, 3, 5, 8… ; +1 cran après une perte, -2 crans après un gain
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CLE = ["date", "reunion", "course"]


COLONNE_COTE = {"favori": "cote_direct", "favori_30min": "cote_reference",
                "prono_pmu": "cote_reference", "presse": "cote_reference"}


@dataclass(frozen=True)
class Filtres:
    sans_handicap: bool = False
    sans_groupes: bool = False
    cote_max: float | None = None
    ecart_min: float | None = None

    def actifs(self) -> str:
        t = []
        if self.sans_handicap:
            t.append("sans handicap")
        if self.sans_groupes:
            t.append("sans Groupe")
        if self.cote_max is not None:
            t.append(f"cote < {self.cote_max:g}")
        if self.ecart_min is not None:
            t.append(f"écart ≥ {self.ecart_min:g}")
        return ", ".join(t) or "aucun"


REGLES_RECOMMANDEES = Filtres(sans_handicap=True, ecart_min=2.0)


def selection(hist: pd.DataFrame, critere: str, filtres: Filtres = Filtres()) -> pd.DataFrame:
    """Une ligne (le cheval joué) par course où un placé était proposé,
    après application des filtres."""
    if critere not in COLONNE_COTE:
        raise ValueError(f"Sélection inconnue : {critere}")
    col = COLONNE_COTE[critere]
    h = hist[(hist["place_ouvert"] == 1) & hist["rapport_place"].notna()].copy()
    if critere == "favori_30min":
        complet = h.groupby(CLE)[col].transform(lambda c: c.notna().all())
        h = h[complet]

    # Plus basse cote de la course et 2e plus basse (pour l'écart)
    cotes = h.groupby(CLE)[col]
    h["_cote1"] = cotes.transform("min")
    h["_cote2"] = cotes.transform(lambda c: c.nsmallest(2).iloc[-1] if c.notna().sum() >= 2 else np.nan)

    if critere in ("favori", "favori_30min"):
        h = h.dropna(subset=[col]).sort_values([col, "num"])
    elif critere == "prono_pmu":
        h = h[h["pmu_prono_rang"] == 1]
    else:
        h = h[h["presse_nb_premier"] > 0].sort_values(["presse_nb_premier", "presse_nb_cite"],
                                                      ascending=False)
    choix = h.groupby(CLE, as_index=False).head(1).copy()

    categorie = choix["categorie"].fillna("").astype(str).str.upper()
    if filtres.sans_handicap:
        choix = choix[~categorie.str.contains("HANDICAP")]
        categorie = categorie.loc[choix.index]
    if filtres.sans_groupes:
        choix = choix[~categorie.str.startswith("GROUPE")]
    if filtres.cote_max is not None:
        choix = choix[choix[col] < filtres.cote_max]
    if filtres.ecart_min is not None:
        # cote la plus basse parmi les AUTRES partants / cote du cheval joué
        autre = np.where(choix[col] <= choix["_cote1"], choix["_cote2"], choix["_cote1"])
        choix = choix[(autre / choix[col]) >= filtres.ecart_min]
    choix = choix.drop(columns=["_cote1", "_cote2"])
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
            debut: str | None = None, fin: str | None = None,
            filtres: Filtres = Filtres()) -> pd.DataFrame:
    h = hist
    if debut:
        h = h[h["date"] >= debut]
    if fin:
        h = h[h["date"] <= fin]
    lignes = []
    for critere in criteres:
        for typ in ("plat", "obstacle", "tous"):
            sous = h if typ == "tous" else h[h["type"] == typ]
            paris = selection(sous, critere, filtres)
            for methode in methodes:
                r = simuler(paris, methode)
                r.pop("solde")
                lignes.append({"selection": critere, "type": typ, "mise": methode, **r})
    return pd.DataFrame(lignes)
