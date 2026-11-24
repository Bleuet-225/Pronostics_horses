"""Client pour l'API JSON publique (non officielle) du PMU.

Points d'entrée utilisés :
  - programme du jour :      /programme/{JJMMAAAA}
  - partants d'une course :  /programme/{JJMMAAAA}/R{r}/C{c}/participants

ATTENTION : cette API n'est pas documentée officiellement par le PMU.
Les noms de champs utilisés dans ce projet n'ont pas pu être vérifiés sur
des réponses réelles lors de l'écriture du code (hôte bloqué par le réseau
de l'environnement de développement). Utilisez `python -m pronostics brut`
pour inspecter une réponse et signaler toute différence.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
DOSSIER_CACHE = Path("data/brut")

# Spécialités conservées : plat et obstacles (le trot est exclu).
SPECIALITES = {"PLAT": "plat", "OBSTACLE": "obstacle"}


class ErreurAPI(RuntimeError):
    pass


def format_date(d: date) -> str:
    return d.strftime("%d%m%Y")


class ClientPMU:
    def __init__(self, cache: Path | None = DOSSIER_CACHE, pause: float = 0.3,
                 timeout: float = 20.0):
        self.cache = cache
        self.pause = pause
        self.timeout = timeout
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        """Une session HTTP par thread (requests.Session n'est pas thread-safe)."""
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = "pronostics-horses/0.1"
            adaptateur = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
            s.mount("https://", adaptateur)
            self._local.session = s
        return s

    # ------------------------------------------------------------------ bas niveau
    def _get(self, chemin: str, utiliser_cache: bool = True) -> dict | None:
        fichier = None
        if self.cache is not None:
            fichier = self.cache / (chemin.strip("/").replace("/", "_") + ".json")
            if utiliser_cache and fichier.exists():
                return json.loads(fichier.read_text(encoding="utf-8"))

        url = f"{BASE_URL}/{chemin.lstrip('/')}"
        for essai in range(4):
            try:
                rep = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as e:
                if essai == 3:
                    raise ErreurAPI(f"Échec réseau sur {url} : {e}") from e
                time.sleep(2 ** essai)
                continue
            if rep.status_code == 204 or rep.status_code == 404:
                return None
            if (rep.status_code == 429 or rep.status_code >= 500) and essai < 3:
                attente = rep.headers.get("Retry-After")
                time.sleep(float(attente) if attente and attente.isdigit() else 2 ** (essai + 1))
                continue
            if rep.status_code != 200:
                raise ErreurAPI(f"HTTP {rep.status_code} sur {url}")
            donnees = rep.json()
            if fichier is not None:
                fichier.parent.mkdir(parents=True, exist_ok=True)
                fichier.write_text(json.dumps(donnees, ensure_ascii=False), encoding="utf-8")
            time.sleep(self.pause)
            return donnees
        return None

    # ------------------------------------------------------------------ haut niveau
    def programme(self, d: date, utiliser_cache: bool = True) -> dict | None:
        return self._get(f"programme/{format_date(d)}", utiliser_cache)

    def participants(self, d: date, reunion: int, course: int,
                     utiliser_cache: bool = True) -> dict | None:
        return self._get(f"programme/{format_date(d)}/R{reunion}/C{course}/participants",
                         utiliser_cache)


def courses_plat_obstacle(programme: dict, pays: str | None = "FRA") -> list[dict]:
    """Extrait du programme les courses de plat et d'obstacles
    (par défaut uniquement en France ; `pays=None` pour tous les pays).

    Retourne une liste de dict « course » enrichis de `_reunion`,
    `_hippodrome` et `_type` ('plat' ou 'obstacle').
    """
    resultat = []
    reunions = (programme or {}).get("programme", {}).get("reunions", [])
    for reunion in reunions:
        num_r = reunion.get("numOfficiel")
        hippo = (reunion.get("hippodrome") or {}).get("libelleCourt", "")
        code_pays = (reunion.get("pays") or {}).get("code", "")
        if pays is not None and code_pays != pays:
            continue
        for course in reunion.get("courses", []):
            type_course = SPECIALITES.get(course.get("specialite", ""))
            if type_course is None:
                continue
            c = dict(course)
            c["_reunion"] = num_r
            c["_hippodrome"] = hippo
            c["_pays"] = code_pays
            c["_type"] = type_course
            resultat.append(c)
    return resultat
