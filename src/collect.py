"""Collecte des courses de PLAT (API PMU turfinfo) entre deux dates.

Écrit un fichier JSON par jour dans data/raw/AAAA-MM-JJ.json contenant, pour
chaque course de plat terminée : métadonnées de course + partants.

Usage : python src/collect.py 2023-01-01 2024-12-31
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/1/programme"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"

COURSE_KEYS = ["numOrdre", "libelle", "discipline", "specialite", "distance",
               "corde", "heureDepart", "montantPrix", "nombreDeclaresPartants",
               "conditionAge", "conditionSexe", "categorieParticularite",
               "penetrometre", "parcours", "ordreArrivee", "arriveeDefinitive"]
PART_KEYS = ["nom", "numPmu", "age", "sexe", "race", "statut", "placeCorde",
             "oeilleres", "entraineur", "driver", "driverChange", "proprietaire",
             "indicateurInedit", "musique", "nombreCourses", "nombreVictoires",
             "nombrePlaces", "nombrePlacesSecond", "nombrePlacesTroisieme",
             "gainsParticipant", "handicapValeur", "handicapPoids",
             "poidsConditionMonte", "nomPere", "nomMere", "ordreArrivee",
             "incident", "dernierRapportDirect", "dernierRapportReference",
             "paysEntrainement", "jumentPleine", "supplement", "engagement"]

session = requests.Session()


def get(url, tries=5):
    for i in range(tries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (204, 404):
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** i)
    return None


def collect_day(d: date):
    path = OUT / f"{d.isoformat()}.json"
    if path.exists():
        return d, "skip"
    ds = d.strftime("%d%m%Y")
    prog = get(f"{BASE}/{ds}")
    if prog is None:
        return d, "no-programme"
    races = []
    for r in prog["programme"]["reunions"]:
        for c in r["courses"]:
            if c.get("discipline") != "PLAT" or not c.get("arriveeDefinitive"):
                continue
            parts = get(f"{BASE}/{ds}/R{r['numOfficiel']}/C{c['numOrdre']}/participants")
            if parts is None:
                continue
            race = {k: c.get(k) for k in COURSE_KEYS}
            race.update(date=d.isoformat(), reunion=r["numOfficiel"],
                        pays=r["pays"]["code"], hippodrome=r["hippodrome"]["code"])
            race["participants"] = [{k: p.get(k) for k in PART_KEYS}
                                    for p in parts["participants"]]
            races.append(race)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(races, ensure_ascii=False))
    tmp.rename(path)
    return d, len(races)


def main(start, end):
    OUT.mkdir(parents=True, exist_ok=True)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    days = [d0 + timedelta(i) for i in range((d1 - d0).days + 1)]
    with ThreadPoolExecutor(8) as ex:
        for n, (d, res) in enumerate(ex.map(collect_day, days), 1):
            if n % 20 == 0 or res == "no-programme":
                print(f"{n}/{len(days)} {d} -> {res}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
