"""Collecte des courses (plat + obstacles) et mise à plat en tableau.

Une ligne = un cheval partant dans une course. Les colonnes `arrivee`,
`gagnant` et `top3` ne sont remplies que pour les courses courues.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .api import ClientPMU, courses_plat_obstacle

log = logging.getLogger(__name__)

FICHIER_HISTORIQUE = Path("data/historique.csv.gz")


def _nombre(v) -> float:
    if v is None or v == "":
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", ".").strip())
    except ValueError:
        return float("nan")


def _arrivees(course: dict) -> dict[int, int]:
    """numPmu → place, à partir de `ordreArrivee` de la course (ex-aequo gérés)."""
    places: dict[int, int] = {}
    ordre = course.get("ordreArrivee") or []
    rang = 1
    for groupe in ordre:
        groupe = groupe if isinstance(groupe, list) else [groupe]
        for num in groupe:
            places[int(num)] = rang
        rang += len(groupe)
    return places


def lignes_course(d: date, course: dict, participants: dict | None) -> list[dict]:
    if not participants:
        return []
    arrivees = _arrivees(course)
    penetro = course.get("penetrometre") or {}
    categorie = str(course.get("categorieParticularite") or "")
    lignes = []
    partants = [p for p in participants.get("participants", [])
                if str(p.get("statut", "PARTANT")).upper() != "NON_PARTANT"]
    for p in partants:
        num = p.get("numPmu")
        gains = p.get("gainsParticipant") or {}
        place = arrivees.get(int(num)) if num is not None and arrivees else None
        if place is None and arrivees:
            # Le cheval a couru mais n'est pas dans l'ordre d'arrivée publié
            # (non classé, tombé, arrêté…) : on le considère non placé.
            place_p = p.get("ordreArrivee")
            place = int(place_p) if isinstance(place_p, (int, float)) else 99
        poids = _nombre(p.get("handicapPoids"))
        if pd.isna(poids):
            poids = _nombre(p.get("poidsConditionMonte"))
        lignes.append({
            "date": d.isoformat(),
            "reunion": course.get("_reunion"),
            "course": course.get("numOrdre"),
            "hippodrome": course.get("_hippodrome"),
            "pays": course.get("_pays"),
            "type": course.get("_type"),
            "discipline": course.get("discipline"),
            "libelle": course.get("libelle"),
            "heure_depart": course.get("heureDepart"),
            "distance": _nombre(course.get("distance")),
            "montant_prix": _nombre(course.get("montantPrix")),
            "nb_declares": _nombre(course.get("nombreDeclaresPartants")),
            "terrain_mesure": _nombre(penetro.get("valeurMesure")),
            "terrain_libelle": penetro.get("intitule"),
            "handicap": int("HANDICAP" in categorie.upper()),
            "categorie": categorie,
            "num": num,
            "nom": p.get("nom"),
            "age": _nombre(p.get("age")),
            "sexe": p.get("sexe"),
            "oeilleres": p.get("oeilleres"),
            "entraineur": p.get("entraineur"),
            "jockey": p.get("driver"),
            "jockey_change": int(bool(p.get("driverChange"))),
            "musique": p.get("musique"),
            "inedit": int(bool(p.get("indicateurInedit"))),
            "nb_courses": _nombre(p.get("nombreCourses")),
            "nb_victoires": _nombre(p.get("nombreVictoires")),
            "nb_places": _nombre(p.get("nombrePlaces")),
            "gains_carriere": _nombre(gains.get("gainsCarriere")),
            "gains_annee": _nombre(gains.get("gainsAnneeEnCours")),
            "gains_annee_prec": _nombre(gains.get("gainsAnneePrecedente")),
            "poids": poids,
            "handicap_valeur": _nombre(p.get("handicapValeur")),
            "place_corde": _nombre(p.get("placeCorde")),
            "cote_direct": _nombre((p.get("dernierRapportDirect") or {}).get("rapport")),
            "cote_reference": _nombre((p.get("dernierRapportReference") or {}).get("rapport")),
            "arrivee": place,
            "gagnant": None if place is None else int(place == 1),
            "top3": None if place is None else int(place <= 3),
        })
    return lignes


def collecter_jour(client: ClientPMU, d: date, seulement_courues: bool = False) -> list[dict]:
    prog = client.programme(d)
    if not prog:
        return []
    lignes: list[dict] = []
    for course in courses_plat_obstacle(prog):
        if seulement_courues and not course.get("ordreArrivee"):
            continue
        parts = client.participants(d, course["_reunion"], course["numOrdre"])
        lignes.extend(lignes_course(d, course, parts))
    return lignes


def collecter_periode(debut: date, fin: date, client: ClientPMU | None = None,
                      fichier: Path = FICHIER_HISTORIQUE) -> pd.DataFrame:
    """Collecte toutes les courses courues entre `debut` et `fin` (inclus)
    et les ajoute au fichier historique (sans doublons)."""
    client = client or ClientPMU()
    lignes: list[dict] = []
    d = debut
    while d <= fin:
        try:
            jour = collecter_jour(client, d, seulement_courues=True)
            log.info("%s : %d lignes", d.isoformat(), len(jour))
            lignes.extend(jour)
        except Exception as e:  # on continue la collecte malgré un jour en erreur
            log.warning("%s : erreur %s", d.isoformat(), e)
        d += timedelta(days=1)

    nouveau = pd.DataFrame(lignes)
    if fichier.exists():
        ancien = pd.read_csv(fichier)
        nouveau = pd.concat([ancien, nouveau], ignore_index=True)
    if not nouveau.empty:
        nouveau = nouveau.drop_duplicates(subset=["date", "reunion", "course", "num"], keep="last")
        nouveau = nouveau.sort_values(["date", "reunion", "course", "num"])
        fichier.parent.mkdir(parents=True, exist_ok=True)
        nouveau.to_csv(fichier, index=False)
    return nouveau
