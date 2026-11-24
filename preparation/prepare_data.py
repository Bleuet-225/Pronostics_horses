"""Préparation des données pour l'identification des facteurs du « placé ».

Usage (depuis la racine du projet) :
    python preparation/prepare_data.py
    python preparation/prepare_data.py --debut 2024-11-24 --fin 2025-11-23 --part-test 0.2

Le fichier source n'est jamais modifié : le script lit une copie.
Toutes les décisions appliquées ici ont été validées une à une ; elles sont
décrites dans preparation/README_donnees.md.

Ordre des opérations (le découpage précède tout calcul de paramètre, pour
qu'aucune information du jeu de test ne serve à préparer l'entraînement) :
  1. période, lignes sans pari placé retirées, correction du poids aberrant
  2. variable y et facteurs dérivés (musique, taux jockey/entraîneur antérieurs)
  3. découpage 80/20 par date (courses jamais coupées en deux)
  4. imputation des manquants (médianes du jeu d'entraînement) + indicateurs
  5. encodage one-hot avec modalité de référence (modalités du jeu d'entraînement)
  6. centrage-réduction de toutes les colonnes (paramètres du jeu d'entraînement)
  7. sauvegarde de X_train, X_test, y_train, y_test et des paramètres
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from pronostics.features import _stats_acteur  # noqa: E402
from pronostics.musique import caracteristiques  # noqa: E402

CLE = ["date", "reunion", "course", "num"]

# Colonnes exclues des facteurs (validé à l'étape 1)
EXCLUES = [
    # identifiants
    "date", "reunion", "course", "num", "nom", "libelle", "heure_depart",
    # constante
    "pays",
    # connues seulement après la course (contiennent la réponse)
    "arrivee", "gagnant", "top3", "rapport_place", "rapport_gagnant", "place_ouvert",
    # cotes (exclues sur décision)
    "cote_direct", "cote_reference",
    # pronostics : vides avant juillet 2026
    "pmu_prono_rang", "pmu_cote_prob", "presse_nb_avis", "presse_synthese",
    "presse_nb_cite", "presse_nb_premier", "presse_rang_moy", "presse_points",
    # redondante avec les modalités « handicap » de `categorie` (validé à l'étape 3)
    "handicap",
    # remplacées par des facteurs dérivés
    "musique", "jockey", "entraineur",
]

# Correction validée : poids enregistré en kg au lieu d'hectogrammes par l'API
# (handicapPoids brut = 530, poidsConditionMonte brut = 52).
CORRECTIONS_POIDS = [
    {"date": "2025-11-08", "nom": "MIA L'HOMBRE", "poids": 52.0, "decharge": 53.0 - 52.0},
]

# Encodage one-hot : modalité de référence (non créée en colonne)
REFERENCES = {
    "discipline": "PLAT",
    "categorie": "COURSE_A_CONDITIONS",
    "sexe": "MALES",
    "oeilleres": "SANS_OEILLERES",
    "hippodrome": None,        # None = la plus fréquente du jeu d'entraînement
    "terrain_libelle": None,
}
SEUIL_RARE = {"categorie": 50}  # modalités < 50 lignes (entraînement) → « AUTRE »


def charger(source: Path, copie: Path, debut: str, fin: str) -> pd.DataFrame:
    copie.parent.mkdir(parents=True, exist_ok=True)
    if not copie.exists():
        shutil.copyfile(source, copie)
    df = pd.read_csv(copie, low_memory=False)
    return df[(df["date"] >= debut) & (df["date"] <= fin)].copy()


def nettoyer(df: pd.DataFrame, journal: list[str]) -> pd.DataFrame:
    n = len(df)
    df = df[df["place_ouvert"] == 1].copy()
    journal.append(f"Lignes de courses sans pari placé retirées : {n - len(df)}")
    for c in CORRECTIONS_POIDS:
        m = (df["date"] == c["date"]) & (df["nom"] == c["nom"])
        if m.sum() != 1:
            raise ValueError(f"Correction de poids introuvable ou ambiguë : {c}")
        df.loc[m, ["poids", "decharge"]] = [c["poids"], c["decharge"]]
        journal.append(f"Poids corrigé : {c['nom']} le {c['date']} → {c['poids']} kg, "
                       f"décharge {c['decharge']} kg")
    df["jockey"] = df["jockey"].fillna("INCONNU")
    df["terrain_libelle"] = df["terrain_libelle"].fillna("INCONNU")
    return df


def deriver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["y_place"] = (df["rapport_place"] > 0).astype(int)

    # Indicateurs de forme issus de la musique (même discipline = plat ou obstacle)
    mus = pd.DataFrame([caracteristiques(m, t) for m, t in zip(df["musique"], df["type"])])
    df = pd.concat([df, mus], axis=1)
    df["musique_absente"] = (~df["musique"].apply(lambda m: isinstance(m, str) and m != "")).astype(int)

    # Taux jockey / entraîneur : uniquement sur les dates strictement antérieures,
    # par type de course, lissés vers un a priori (voir pronostics/features.py)
    for acteur in ("jockey", "entraineur"):
        df = _stats_acteur(df, df, acteur)

    df["obstacle"] = (df["type"] == "obstacle").astype(int)
    df["piste_psf"] = df["terrain_libelle"].str.upper().str.startswith("PSF").astype(int)
    df["handicap_valeur_absente"] = df["handicap_valeur"].isna().astype(int)
    df["gains_absents"] = df["gains_carriere"].isna().astype(int)
    df["place_corde"] = df["place_corde"].fillna(0.0)  # 0 = pas de corde (obstacle)
    return df.drop(columns=["type"])


def decouper(df: pd.DataFrame, part_test: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    dates = np.sort(df["date"].unique())
    coupure = str(dates[int(len(dates) * (1 - part_test))])
    return df[df["date"] < coupure].copy(), df[df["date"] >= coupure].copy(), coupure


# Décision validée : les lignes de test dont la modalité n'existe pas dans
# l'entraînement sont retirées (variables de niveau course : courses entières).
RETIRER_SI_INCONNUE = ("hippodrome", "terrain_libelle")


def retirer_inconnues(app: pd.DataFrame, test: pd.DataFrame, journal: list[str]) -> pd.DataFrame:
    for col in RETIRER_SI_INCONNUE:
        inconnues = sorted(set(test[col].unique()) - set(app[col].unique()))
        if inconnues:
            m = test[col].isin(inconnues)
            nb_courses = test[m].groupby(["date", "reunion", "course"]).ngroups
            test = test[~m]
            journal.append(f"Test : {int(m.sum())} lignes ({nb_courses} courses) retirées, `{col}` "
                           f"absent de l'entraînement : {inconnues}")
    return test.copy()


def imputer(app: pd.DataFrame, test: pd.DataFrame, journal: list[str]) -> dict:
    """Médianes calculées sur l'entraînement uniquement."""
    medianes = {}
    # terrain : médiane des mesures hors PSF
    medianes["terrain_mesure"] = float(app.loc[app["piste_psf"] == 0, "terrain_mesure"].median())
    for c in ("handicap_valeur", "gains_carriere", "gains_annee", "gains_annee_prec"):
        medianes[c] = float(app[c].median())
    # facteurs musique non définis (aucune course, ou aucune dans la discipline)
    mus_cols = [c for c in app.columns if c.startswith("mus_")]
    for c in mus_cols:
        if app[c].isna().any() or test[c].isna().any():
            medianes[c] = float(app[c].median())
    for d in (app, test):
        d["mus_meme_absente"] = d["mus_place_moy_meme"].isna().astype(int)
        for c, v in medianes.items():
            d[c] = d[c].fillna(v)
    for c, v in medianes.items():
        journal.append(f"Manquants de `{c}` remplacés par la médiane d'entraînement {v:.4g}")
    return medianes


def encoder(app: pd.DataFrame, test: pd.DataFrame, journal: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    infos = {}
    for col, ref in REFERENCES.items():
        freq = app[col].value_counts()
        if col in SEUIL_RARE:
            rares = freq[freq < SEUIL_RARE[col]].index.tolist()
            if rares:
                for d in (app, test):
                    d[col] = d[col].where(~d[col].isin(rares), "AUTRE")
                journal.append(f"`{col}` : modalités rares regroupées en AUTRE : {rares}")
                freq = app[col].value_counts()
        ref = ref if ref is not None else freq.index[0]
        modalites = [m for m in freq.index if m != ref]
        inconnues = sorted(set(test[col].unique()) - set(freq.index))
        if inconnues:
            raise ValueError(
                f"`{col}` : modalités présentes dans le test mais absentes de l'entraînement : "
                f"{inconnues}. Décision nécessaire avant de continuer.")
        cols = {}
        for nom, d in (("app", app), ("test", test)):
            bloc = pd.DataFrame({f"{col}__{m}": (d[col] == m).astype(int) for m in modalites},
                                index=d.index)
            cols[nom] = pd.concat([d.drop(columns=[col]), bloc], axis=1)
        app, test = cols["app"], cols["test"]
        infos[col] = {"reference": ref, "modalites": modalites}
        journal.append(f"`{col}` : one-hot, {len(modalites)} colonnes, référence = {ref}")
    return app, test, infos


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=RACINE / "data/historique.csv.gz")
    p.add_argument("--sortie", type=Path, default=RACINE / "preparation/donnees")
    p.add_argument("--debut", default="2024-11-24")
    p.add_argument("--fin", default="2025-11-23")
    p.add_argument("--part-test", type=float, default=0.2)
    a = p.parse_args(argv)

    journal: list[str] = []
    df = charger(a.source, a.sortie / "historique_copie.csv.gz", a.debut, a.fin)
    journal.append(f"Période {a.debut} → {a.fin} : {len(df)} lignes")
    df = nettoyer(df, journal)
    df = deriver(df)

    app, test, coupure = decouper(df, a.part_test)
    journal.append(f"Découpage par date : entraînement < {coupure} ({len(app)} lignes, "
                   f"{app['date'].nunique()} jours), test ≥ {coupure} ({len(test)} lignes, "
                   f"{test['date'].nunique()} jours)")

    test = retirer_inconnues(app, test, journal)
    medianes = imputer(app, test, journal)
    app, test, encodage = encoder(app, test, journal)

    ids_cols = [c for c in CLE]
    ids_app, ids_test = app[ids_cols], test[ids_cols]
    y_app, y_test = app["y_place"], test["y_place"]
    a_retirer = [c for c in EXCLUES if c in app.columns] + ["y_place"]
    X_app, X_test = app.drop(columns=a_retirer), test.drop(columns=a_retirer)
    X_test = X_test[X_app.columns]

    restants = X_app.columns[X_app.isna().any()].tolist() + X_test.columns[X_test.isna().any()].tolist()
    if restants:
        raise ValueError(f"Valeurs manquantes non traitées : {sorted(set(restants))}")
    constantes = X_app.columns[X_app.std() == 0].tolist()
    if constantes:
        journal.append(f"Colonnes constantes dans l'entraînement (écart-type nul) : {constantes}")

    scaler = StandardScaler().fit(X_app)
    X_app = pd.DataFrame(scaler.transform(X_app), columns=X_app.columns, index=X_app.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    journal.append(f"Centrage-réduction de {X_app.shape[1]} colonnes (paramètres d'entraînement)")

    a.sortie.mkdir(parents=True, exist_ok=True)
    X_app.to_csv(a.sortie / "X_train.csv", index=False)
    X_test.to_csv(a.sortie / "X_test.csv", index=False)
    y_app.to_csv(a.sortie / "y_train.csv", index=False)
    y_test.to_csv(a.sortie / "y_test.csv", index=False)
    ids_app.to_csv(a.sortie / "ids_train.csv", index=False)
    ids_test.to_csv(a.sortie / "ids_test.csv", index=False)
    params = {
        "periode": [a.debut, a.fin], "coupure_test": coupure, "part_test": a.part_test,
        "colonnes": list(X_app.columns), "medianes": medianes, "encodage": encodage,
        "moyennes": dict(zip(X_app.columns, scaler.mean_.tolist())),
        "ecarts_types": dict(zip(X_app.columns, scaler.scale_.tolist())),
        "journal": journal,
    }
    (a.sortie / "parametres.json").write_text(json.dumps(params, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print("\n".join(journal))
    print(f"\nX_train {X_app.shape} · X_test {X_test.shape} · y_train placé {y_app.mean():.1%} · "
          f"y_test placé {y_test.mean():.1%}")


if __name__ == "__main__":
    main()
