"""Collecte des courses (plat + obstacles) et mise à plat en tableau.

Une ligne = un cheval partant dans une course. Les colonnes `arrivee`,
`gagnant` et `top3` ne sont remplies que pour les courses courues.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .api import PRONOSTICS_DISPONIBLES_DEPUIS, ClientPMU, courses_plat_obstacle

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


def _dividendes(rapports: list | dict | None, type_pari: str) -> dict[int, float]:
    """numPmu → gain pour 1 € misé (l'API donne des centimes pour 1 €)."""
    res: dict[int, float] = {}
    blocs = rapports if isinstance(rapports, list) else [rapports or {}]
    for bloc in blocs:
        if bloc.get("typePari") != type_pari:
            continue
        for r in bloc.get("rapports", []):
            try:
                res[int(r["combinaison"])] = float(r["dividendePourUnEuro"]) / 100
            except (KeyError, TypeError, ValueError):
                continue
    return res


def _cote_fractionnaire(v) -> float:
    """« 3/1 » → 4.0 (cote décimale = a/b + 1)."""
    try:
        a, b = str(v).split("/")
        return float(a) / float(b) + 1
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _variables_pronostics(prono: dict | None, detail: dict | None) -> dict[int, dict]:
    """numPmu → variables issues du pronostic PMU et de la synthèse presse.
    Dictionnaire vide si les pronostics n'ont pas été récupérés."""
    res: dict[int, dict] = {}
    if prono is None and detail is None:
        return res

    def ligne(num):
        return res.setdefault(int(num), {})

    for sel in (prono or {}).get("selection", []):
        if sel.get("num_partant") is None:
            continue
        l = ligne(sel["num_partant"])
        l["pmu_prono_rang"] = _nombre(sel.get("rang"))
        l["pmu_cote_prob"] = _cote_fractionnaire(sel.get("cote_prob"))

    avis = [a.get("pronostics", []) for a in (detail or {}).get("avis", []) if a.get("pronostics")]
    cites: dict[int, list[int]] = {}
    for liste in avis:
        for pos, x in enumerate(liste, 1):
            if x.get("numPmu") is not None:
                cites.setdefault(int(x["numPmu"]), []).append(pos)
    for num, positions in cites.items():
        l = ligne(num)
        l["presse_nb_cite"] = float(len(positions))
        l["presse_nb_premier"] = float(sum(1 for p in positions if p == 1))
        l["presse_rang_moy"] = sum(positions) / len(positions)
    for synth in (detail or {}).get("syntheses", []):
        for x in synth.get("classement", []):
            if x.get("numPmu") is not None:
                ligne(x["numPmu"])["presse_points"] = _nombre(x.get("nbPoints"))
    res[-1] = {"presse_nb_avis": float(len(avis)),
               "presse_synthese": int(bool((detail or {}).get("syntheses")))}
    return res


COLONNES_PRONOS = ["pmu_prono_rang", "pmu_cote_prob", "presse_nb_avis", "presse_synthese",
                   "presse_nb_cite", "presse_nb_premier", "presse_rang_moy", "presse_points"]


def lignes_course(d: date, course: dict, participants: dict | None,
                  rapports: list | None = None, pronos: dict[int, dict] | None = None) -> list[dict]:
    if not participants:
        return []
    pronos = pronos or {}
    general = pronos.get(-1)
    div_place = _dividendes(rapports, "SIMPLE_PLACE")
    div_gagnant = _dividendes(rapports, "SIMPLE_GAGNANT")
    place_ouvert = int(bool(div_place)) if rapports is not None else None
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
        # Poids en hectogrammes dans l'API (580 = 58 kg). `poidsConditionMonte`
        # (quand il est renseigné et > 0) correspond au poids après décharge du jockey.
        poids_theorique = _nombre(p.get("handicapPoids")) / 10
        poids_monte = _nombre(p.get("poidsConditionMonte")) / 10
        poids = poids_monte if poids_monte > 0 else poids_theorique
        decharge = (poids_theorique - poids_monte) if poids_monte > 0 else 0.0
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
            "decharge": decharge,
            "handicap_valeur": _nombre(p.get("handicapValeur")),
            "place_corde": _nombre(p.get("placeCorde")),
            "cote_direct": _nombre((p.get("dernierRapportDirect") or {}).get("rapport")),
            "cote_reference": _nombre((p.get("dernierRapportReference") or {}).get("rapport")),
            "arrivee": place,
            "gagnant": None if place is None else int(place == 1),
            "top3": None if place is None else int(place <= 3),
            # Rapports définitifs (None si non récupérés) ; 0 = non payé.
            "place_ouvert": place_ouvert,
            "rapport_place": None if place_ouvert is None else div_place.get(int(num), 0.0),
            "rapport_gagnant": None if place_ouvert is None else div_gagnant.get(int(num), 0.0),
            **_colonnes_prono(general, pronos.get(int(num)) if num is not None else None),
        })
    return lignes


def _colonnes_prono(general: dict | None, cheval: dict | None) -> dict:
    """Pronostics non récupérés → NaN partout ; récupérés mais cheval non
    cité → 0 citation (rang/cote/points restent NaN)."""
    v = {c: float("nan") for c in COLONNES_PRONOS}
    if general is not None:
        v.update(general)
        v["presse_nb_cite"] = 0.0
        v["presse_nb_premier"] = 0.0
    v.update(cheval or {})
    return v


def collecter_jour(client: ClientPMU, d: date, seulement_courues: bool = False) -> list[dict]:
    prog = client.programme(d)
    if not prog:
        return []
    lignes: list[dict] = []
    for course in courses_plat_obstacle(prog):
        if seulement_courues and not course.get("ordreArrivee"):
            continue
        parts = client.participants(d, course["_reunion"], course["numOrdre"])
        rapports = None
        if course.get("ordreArrivee"):
            rapports = client.rapports(d, course["_reunion"], course["numOrdre"]) or []
        pronos = None
        if d >= PRONOSTICS_DISPONIBLES_DEPUIS:
            pronos = _variables_pronostics(
                client.pronostics(d, course["_reunion"], course["numOrdre"]),
                client.pronostics_detailles(d, course["_reunion"], course["numOrdre"]))
        lignes.extend(lignes_course(d, course, parts, rapports, pronos))
    return lignes


def collecter_periode(debut: date, fin: date, client: ClientPMU | None = None,
                      fichier: Path = FICHIER_HISTORIQUE, paralleles: int = 8) -> pd.DataFrame:
    """Collecte toutes les courses courues entre `debut` et `fin` (inclus)
    et les ajoute au fichier historique (sans doublons).

    Les jours sont téléchargés en parallèle (`paralleles` connexions).
    """
    client = client or ClientPMU(pause=0.05)
    jours = [debut + timedelta(days=i) for i in range((fin - debut).days + 1)]
    lignes: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, paralleles)) as pool:
        taches = {pool.submit(collecter_jour, client, d, True): d for d in jours}
        for i, t in enumerate(as_completed(taches), 1):
            d = taches[t]
            try:
                jour = t.result()
                lignes.extend(jour)
                log.info("[%d/%d] %s : %d lignes", i, len(jours), d.isoformat(), len(jour))
            except Exception as e:  # on continue la collecte malgré un jour en erreur
                log.warning("[%d/%d] %s : erreur %s", i, len(jours), d.isoformat(), e)

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
