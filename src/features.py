"""Construit la table partant × course avec des variables connues AVANT le départ.

Toutes les statistiques historiques (jockey, entraîneur, cheval) sont calculées
uniquement à partir des jours précédents la course (pas de fuite d'information).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "dataset.parquet"

MUS_RE = re.compile(r"(\d|[A-Z])([a-z])")


def parse_musique(m, n=5):
    """Dernières places en plat (lettre 'p') ; 0 = au-delà du 9e, lettre = incident."""
    vals = []
    for pos, disc in MUS_RE.findall(m or ""):
        if pos.isdigit():
            v = int(pos)
            vals.append(10 if v == 0 else v)
        else:
            vals.append(11)  # disqualifié, tombé, arrêté...
    vals = vals[:n]
    out = {f"mus_{i+1}": (vals[i] if i < len(vals) else np.nan) for i in range(n)}
    out["mus_mean"] = np.mean(vals) if vals else np.nan
    out["mus_wins"] = sum(v == 1 for v in vals)
    out["mus_top3"] = sum(v <= 3 for v in vals)
    out["mus_len"] = len(vals)
    return out


def load_rows():
    rows = []
    for f in sorted(RAW.glob("*.json")):
        for r in json.loads(f.read_text()):
            race_id = f"{r['date']}_R{r['reunion']}C{r['numOrdre']}"
            for p in r["participants"]:
                if p.get("statut") != "PARTANT":
                    continue
                g = p.get("gainsParticipant") or {}
                rd = p.get("dernierRapportDirect") or {}
                rr = p.get("dernierRapportReference") or {}
                row = dict(
                    race_id=race_id, date=r["date"], heure=r["heureDepart"],
                    pays=r["pays"], hippodrome=r["hippodrome"],
                    distance=r.get("distance"), montantPrix=r.get("montantPrix"),
                    corde=r.get("corde"), penetrometre=(r.get("penetrometre") or {}).get("valeurMesure"),
                    categorie=r.get("categorieParticularite"),
                    nom=p.get("nom"), numPmu=p.get("numPmu"), age=p.get("age"),
                    sexe=p.get("sexe"), placeCorde=p.get("placeCorde"),
                    oeilleres=p.get("oeilleres"), entraineur=p.get("entraineur"),
                    jockey=p.get("driver"), driverChange=p.get("driverChange"),
                    inedit=p.get("indicateurInedit"),
                    nbCourses=p.get("nombreCourses"), nbVictoires=p.get("nombreVictoires"),
                    nbPlaces=p.get("nombrePlaces"),
                    gainsCarriere=g.get("gainsCarriere"), gainsAnnee=g.get("gainsAnneeEnCours"),
                    gainsAnneePrec=g.get("gainsAnneePrecedente"),
                    handicapValeur=p.get("handicapValeur"), handicapPoids=p.get("handicapPoids"),
                    poids=p.get("poidsConditionMonte"), supplement=p.get("supplement"),
                    paysEntrainement=p.get("paysEntrainement"),
                    cote=rd.get("rapport"), coteRef=rr.get("rapport"),
                    coteDate=rd.get("dateRapport"),
                    ordreArrivee=p.get("ordreArrivee"),
                )
                row.update(parse_musique(p.get("musique")))
                rows.append(row)
    return pd.DataFrame(rows)


def add_history(df, key, prefix, prior=0.08, k=20):
    """Taux de victoire lissé de `key` calculé sur les jours strictement antérieurs."""
    daily = (df.groupby([key, "date"]).agg(n=("win", "size"), w=("win", "sum"))
             .reset_index().sort_values([key, "date"]))
    daily[f"{prefix}_n"] = daily.groupby(key)["n"].cumsum() - daily["n"]
    daily[f"{prefix}_w"] = daily.groupby(key)["w"].cumsum() - daily["w"]
    daily[f"{prefix}_rate"] = (daily[f"{prefix}_w"] + prior * k) / (daily[f"{prefix}_n"] + k)
    return df.merge(daily[[key, "date", f"{prefix}_n", f"{prefix}_rate"]],
                    on=[key, "date"], how="left")


def build():
    df = load_rows()
    df = df[df["ordreArrivee"].notna()].copy()  # course courue avec arrivée connue
    df["win"] = (df["ordreArrivee"] == 1).astype(int)
    # ne garder que les courses ayant exactement un gagnant identifié
    nwin = df.groupby("race_id")["win"].transform("sum")
    df = df[nwin == 1].copy()

    df["penetrometre"] = pd.to_numeric(df["penetrometre"].str.replace(",", "."), errors="coerce")
    df["partants"] = df.groupby("race_id")["race_id"].transform("size")
    df["win_rate"] = df["nbVictoires"] / df["nbCourses"].replace(0, np.nan)
    df["place_rate"] = df["nbPlaces"] / df["nbCourses"].replace(0, np.nan)
    df["gains_par_course"] = df["gainsCarriere"] / df["nbCourses"].replace(0, np.nan)
    df["corde_rel"] = df["placeCorde"] / df["partants"]

    for key, prefix in [("jockey", "jk"), ("entraineur", "tr"), ("nom", "hs")]:
        df = add_history(df, key, prefix)

    # cotes : probabilité implicite normalisée dans la course
    df["p_cote"] = 1 / df["cote"]
    df["p_cote"] = df["p_cote"] / df.groupby("race_id")["p_cote"].transform("sum")
    df["p_coteRef"] = 1 / df["coteRef"]
    df["p_coteRef"] = df["p_coteRef"] / df.groupby("race_id")["p_coteRef"].transform("sum")

    # rangs relatifs dans la course (1 = meilleur)
    rel = {"gainsCarriere": False, "gainsAnnee": False, "gains_par_course": False,
           "win_rate": False, "place_rate": False, "mus_mean": True, "handicapValeur": False,
           "poids": False, "jk_rate": False, "tr_rate": False, "hs_rate": False,
           "cote": True, "coteRef": True}
    for c, asc in rel.items():
        df[f"rk_{c}"] = df.groupby("race_id")[c].rank(ascending=asc, method="min")
        mean = df.groupby("race_id")[c].transform("mean")
        std = df.groupby("race_id")[c].transform("std").replace(0, np.nan)
        df[f"z_{c}"] = (df[c] - mean) / std

    for c in ["sexe", "oeilleres", "corde", "categorie", "pays", "hippodrome", "paysEntrainement"]:
        df[c] = df[c].astype("category")
    for c in ["driverChange", "inedit"]:
        df[c] = df[c].astype(float)
    df.to_parquet(OUT)
    print(df.shape, df["race_id"].nunique(), "courses")


if __name__ == "__main__":
    build()
