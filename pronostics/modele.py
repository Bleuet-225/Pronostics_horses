"""Entraînement, évaluation et prédiction.

Un modèle par type de course (plat / obstacle) et par cible :
  - `gagnant` : probabilité de victoire (normalisée pour sommer à 1 par course)
  - `top3`    : probabilité de finir dans les 3 premiers (somme = min(3, partants))
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from . import features as F

log = logging.getLogger(__name__)

DOSSIER_MODELES = Path("modeles")
CLE_COURSE = ["date", "reunion", "course"]
CIBLES = ("gagnant", "top3")
MIN_COURSES = 200


def _nouveau_classifieur() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=0,
    )


def normaliser(df: pd.DataFrame, brut: np.ndarray, cible: str) -> np.ndarray:
    """Rend les probabilités cohérentes à l'intérieur de chaque course."""
    p = pd.Series(np.clip(brut, 1e-6, 1 - 1e-6), index=df.index)
    groupes = [df[c] for c in CLE_COURSE]
    if cible == "gagnant":
        return (p / p.groupby(groupes).transform("sum")).to_numpy()
    # top3 : somme visée = min(3, nb partants), chaque proba plafonnée à 1
    cible_somme = p.groupby(groupes).transform("size").clip(upper=3)
    for _ in range(10):
        p = (p * cible_somme / p.groupby(groupes).transform("sum")).clip(upper=1.0)
    return p.to_numpy()


def _metriques(df: pd.DataFrame, proba: np.ndarray, cible: str) -> dict:
    y = df[cible].to_numpy()
    tmp = df[CLE_COURSE].copy()
    tmp["p"], tmp["y"] = proba, y
    tmp["rang"] = tmp.groupby(CLE_COURSE)["p"].rank(ascending=False, method="first")
    nb_courses = tmp.groupby(CLE_COURSE).ngroups
    m = {
        "nb_courses_test": int(nb_courses),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))),
        "auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
    }
    if cible == "gagnant":
        m["taux_reussite_1er_choix"] = float(tmp.loc[tmp["rang"] == 1, "y"].mean())
        m["reference_hasard"] = float((1 / tmp.groupby(CLE_COURSE)["y"].transform("size")).groupby(
            [tmp[c] for c in CLE_COURSE]).first().mean())
        if "cote_direct" in df and df["cote_direct"].notna().any():
            tmp["cote"] = df["cote_direct"].to_numpy()
            fav = tmp.dropna(subset=["cote"]).sort_values("cote").groupby(CLE_COURSE).head(1)
            m["reference_favori_cote"] = float(fav["y"].mean())
    else:
        top = tmp[tmp["rang"] <= 3]
        m["precision_3_choix"] = float(top["y"].mean())
    return m


def entrainer(historique: pd.DataFrame, type_course: str, avec_cotes: bool = False,
              part_test: float = 0.2, dossier: Path = DOSSIER_MODELES) -> dict:
    """Entraîne les modèles pour un type de course.

    1. découpage temporel : les `part_test` dernières dates servent de test ;
    2. évaluation sur ce test ;
    3. ré-entraînement final sur toutes les données, sauvegarde.
    """
    h = historique[(historique["type"] == type_course) & historique["gagnant"].notna()].copy()
    nb_courses = h.groupby(CLE_COURSE).ngroups
    if nb_courses < MIN_COURSES:
        raise ValueError(
            f"Seulement {nb_courses} courses de type « {type_course} » dans l'historique : "
            f"il en faut au moins {MIN_COURSES}. Collectez une période plus longue.")
    h["gagnant"] = h["gagnant"].astype(int)
    h["top3"] = h["top3"].astype(int)
    if avec_cotes and h["cote_direct"].notna().mean() < 0.5:
        raise ValueError("Moins de la moitié des lignes ont une cote : impossible d'utiliser --avec-cotes.")

    x = F.construire(h, historique=historique, avec_cotes=avec_cotes)
    cols = F.colonnes_features(type_course, avec_cotes)
    vides = [c for c in cols if x[c].isna().all()]
    if vides:
        log.warning("%s : colonnes sans aucune valeur, ignorées : %s", type_course, vides)
        cols = [c for c in cols if c not in vides]

    dates = np.sort(x["date"].unique())
    coupure = dates[int(len(dates) * (1 - part_test))]
    app, test = x[x["date"] < coupure], x[x["date"] >= coupure]

    metriques = {"periode_apprentissage": [str(dates[0]), str(coupure)],
                 "periode_test": [str(coupure), str(dates[-1])]}
    modeles = {}
    for cible in CIBLES:
        clf = _nouveau_classifieur().fit(app[cols], app[cible])
        proba = normaliser(test, clf.predict_proba(test[cols])[:, 1], cible)
        metriques[cible] = _metriques(test, proba, cible)
        log.info("%s / %s : %s", type_course, cible, metriques[cible])
        modeles[cible] = _nouveau_classifieur().fit(x[cols], x[cible])

    paquet = {
        "type": type_course, "colonnes": cols, "avec_cotes": avec_cotes,
        "modeles": modeles, "metriques": metriques,
        "entraine_le": datetime.now().isoformat(timespec="seconds"),
        "nb_courses": int(nb_courses),
    }
    dossier.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquet, dossier / f"{type_course}.joblib")
    return paquet


def charger(type_course: str, dossier: Path = DOSSIER_MODELES) -> dict:
    fichier = dossier / f"{type_course}.joblib"
    if not fichier.exists():
        raise FileNotFoundError(
            f"Aucun modèle « {type_course} » ({fichier}). Lancez d'abord la commande « entrainer ».")
    return joblib.load(fichier)


def predire(partants: pd.DataFrame, historique: pd.DataFrame | None,
            dossier: Path = DOSSIER_MODELES) -> pd.DataFrame:
    """Probabilités de victoire et de top 3 pour des partants (toutes courses)."""
    sorties = []
    for type_course, groupe in partants.groupby("type"):
        paquet = charger(type_course, dossier)
        x = F.construire(groupe, historique=historique, avec_cotes=paquet["avec_cotes"])
        for cible in CIBLES:
            brut = paquet["modeles"][cible].predict_proba(x[paquet["colonnes"]])[:, 1]
            x[f"proba_{cible}"] = normaliser(x, brut, cible)
        sorties.append(x)
    if not sorties:
        return pd.DataFrame()
    res = pd.concat(sorties, ignore_index=True)
    res["rang_prono"] = res.groupby(CLE_COURSE)["proba_gagnant"].rank(ascending=False, method="first")
    return res.sort_values(CLE_COURSE + ["rang_prono"]).reset_index(drop=True)
