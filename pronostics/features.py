"""Construction des variables explicatives.

Toutes les statistiques jockey / entraîneur sont calculées uniquement à
partir des courses de dates STRICTEMENT antérieures à la course concernée,
pour éviter toute fuite d'information (entraînement comme prédiction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .musique import caracteristiques

ACTEURS = ("jockey", "entraineur")
LISSAGE = 10.0          # poids de l'a priori dans les taux lissés
PRIOR_VICTOIRE = 0.10
PRIOR_TOP3 = 0.30

# Variables comparées à la moyenne de la course (écart et rang).
VARIABLES_RELATIVES = [
    "gains_par_course", "taux_victoire", "taux_place", "mus_place_moy_meme",
    "mus_taux_top3_meme", "jockey_taux_victoire", "entraineur_taux_victoire",
    "poids", "age", "handicap_valeur",
]


def _stats_acteur(historique: pd.DataFrame, cible: pd.DataFrame, acteur: str) -> pd.DataFrame:
    """Ajoute à `cible` les stats cumulées de `acteur` (par type de course)
    calculées sur les dates < date de la ligne."""
    cols = [f"{acteur}_nb", f"{acteur}_taux_victoire", f"{acteur}_taux_top3"]
    cible = cible.copy()
    if historique is None or historique.empty or acteur not in historique:
        for c in cols:
            cible[c] = np.nan
        return cible

    h = historique.dropna(subset=["gagnant", acteur]).copy()
    h["_d"] = pd.to_datetime(h["date"])
    jour = (h.groupby([acteur, "type", "_d"], as_index=False)
              .agg(_n=("gagnant", "size"), _v=("gagnant", "sum"), _t=("top3", "sum"))
              .sort_values("_d"))
    jour[["_cn", "_cv", "_ct"]] = jour.groupby([acteur, "type"])[["_n", "_v", "_t"]].cumsum()

    cible["_d"] = pd.to_datetime(cible["date"])
    cible["_ordre"] = np.arange(len(cible))
    gauche = cible.sort_values("_d")
    fusion = pd.merge_asof(
        gauche, jour[[acteur, "type", "_d", "_cn", "_cv", "_ct"]],
        on="_d", by=[acteur, "type"], allow_exact_matches=False, direction="backward",
    ).sort_values("_ordre")
    n = fusion["_cn"].fillna(0.0)
    cible[cols[0]] = n.to_numpy()
    cible[cols[1]] = ((fusion["_cv"].fillna(0) + LISSAGE * PRIOR_VICTOIRE) / (n + LISSAGE)).to_numpy()
    cible[cols[2]] = ((fusion["_ct"].fillna(0) + LISSAGE * PRIOR_TOP3) / (n + LISSAGE)).to_numpy()
    return cible.drop(columns=["_d", "_ordre"])


def _texte(serie: pd.Series) -> pd.Series:
    return serie.fillna("").astype(str).str.upper()


def construire(df: pd.DataFrame, historique: pd.DataFrame | None = None,
               avec_cotes: bool = False) -> pd.DataFrame:
    """Retourne `df` enrichi des variables ; utilisez `colonnes_features()`
    pour obtenir la liste des colonnes à donner au modèle.

    `historique` sert aux stats jockey/entraîneur. Pour l'entraînement,
    passer le même DataFrame que `df` (le calcul reste antérieur à chaque date).
    """
    x = df.copy()

    # Musique (forme récente)
    mus = [caracteristiques(m, t) for m, t in zip(x["musique"], x["type"])]
    x = pd.concat([x.reset_index(drop=True), pd.DataFrame(mus)], axis=1)

    # Carrière
    nb = x["nb_courses"].replace(0, np.nan)
    x["taux_victoire"] = x["nb_victoires"] / nb
    x["taux_place"] = x["nb_places"] / nb
    x["gains_par_course"] = x["gains_carriere"] / nb
    x["log_gains_carriere"] = np.log1p(x["gains_carriere"].clip(lower=0))
    x["log_gains_annee"] = np.log1p(x["gains_annee"].clip(lower=0))
    x["log_montant_prix"] = np.log1p(x["montant_prix"].clip(lower=0))

    # Variables catégorielles converties en indicateurs
    sexe = _texte(x["sexe"])
    x["sexe_femelle"] = sexe.str.startswith("F").astype(int)
    x["sexe_hongre"] = sexe.str.startswith("H").astype(int)
    oeil = _texte(x["oeilleres"])
    x["oeilleres_oui"] = ((oeil != "") & ~oeil.str.contains("SANS")).astype(int)
    disc = _texte(x["discipline"])
    x["disc_haie"] = disc.str.contains("HAIE").astype(int)
    x["disc_steeple"] = disc.str.contains("STEEPLE").astype(int)
    x["disc_cross"] = disc.str.contains("CROSS").astype(int)

    # Taille du champ réellement partant
    cle = ["date", "reunion", "course"]
    x["nb_partants"] = x.groupby(cle)["num"].transform("size")

    # Jockey / entraîneur
    for acteur in ACTEURS:
        x = _stats_acteur(historique if historique is not None else x, x, acteur)

    # Cotes (optionnel) : probabilité implicite normalisée dans la course
    if avec_cotes:
        inv = 1.0 / x["cote_direct"].where(x["cote_direct"] > 1)
        x["proba_cote"] = inv / inv.groupby([x[c] for c in cle]).transform("sum")
        x["rang_cote"] = x.groupby(cle)["cote_direct"].rank(method="average")

    # Variables relatives à la course
    groupes = x.groupby(cle)
    for v in VARIABLES_RELATIVES:
        x[f"{v}_ecart"] = x[v] - groupes[v].transform("mean")
        x[f"{v}_rang"] = groupes[v].rank(method="average", pct=True)
    return x


def colonnes_features(type_course: str, avec_cotes: bool = False) -> list[str]:
    cols = [
        "age", "distance", "log_montant_prix", "nb_partants", "terrain_mesure",
        "handicap", "inedit", "jockey_change", "nb_courses", "taux_victoire",
        "taux_place", "gains_par_course", "log_gains_carriere", "log_gains_annee",
        "poids", "decharge", "handicap_valeur", "sexe_femelle", "sexe_hongre", "oeilleres_oui",
        "mus_nb_perfs", "mus_nb", "mus_place_moy", "mus_victoires", "mus_top3",
        "mus_incidents", "mus_taux_top3", "mus_nb_meme", "mus_place_moy_meme",
        "mus_victoires_meme", "mus_top3_meme", "mus_incidents_meme",
        "mus_taux_top3_meme", "mus_derniere_place", "mus_derniere_incident",
    ]
    for a in ACTEURS:
        cols += [f"{a}_nb", f"{a}_taux_victoire", f"{a}_taux_top3"]
    for v in VARIABLES_RELATIVES:
        cols += [f"{v}_ecart", f"{v}_rang"]
    if type_course == "plat":
        cols.append("place_corde")
    else:
        cols += ["disc_haie", "disc_steeple", "disc_cross"]
    if avec_cotes:
        cols += ["proba_cote", "rang_cote"]
    return cols
