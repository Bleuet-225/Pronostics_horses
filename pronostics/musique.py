"""Analyse de la « musique » d'un cheval (ex. « 1p3p(25)5h0hTsAp »).

Chaque performance = un caractère de résultat + une lettre de discipline :
  résultat  : 1..9 = place, 0 = au-delà de la 9e place,
              lettres = incident (A arrêté, T tombé, D disqualifié, R refus...).
  discipline: p plat, h haies, s steeple-chase, c cross, a attelé, m monté.
Les marqueurs « (25) » indiquent un changement d'année et sont ignorés.
La performance la plus récente est à gauche.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DISCIPLINES_OBSTACLE = set("hsc")
DISCIPLINES_PLAT = set("p")

_MOTIF = re.compile(r"\(\d+\)|([0-9A-Za-z])([phscamPHSCAM])")


@dataclass(frozen=True)
class Perf:
    resultat: str     # '1'..'9', '0' ou une lettre d'incident (majuscule)
    discipline: str   # 'p', 'h', 's', 'c', 'a', 'm'

    @property
    def place(self) -> int | None:
        """Place numérique ; 0 → 10 (non placé) ; None si incident."""
        if self.resultat.isdigit():
            n = int(self.resultat)
            return 10 if n == 0 else n
        return None

    @property
    def incident(self) -> bool:
        return not self.resultat.isdigit()


def analyser(musique: str | None) -> list[Perf]:
    if not isinstance(musique, str) or not musique:
        return []  # None, NaN (valeur manquante relue d'un CSV) ou chaîne vide
    perfs = []
    for m in _MOTIF.finditer(musique):
        if m.group(1) is None:
            continue  # marqueur d'année
        perfs.append(Perf(m.group(1).upper(), m.group(2).lower()))
    return perfs


def _filtrer(perfs: list[Perf], type_course: str | None) -> list[Perf]:
    if type_course == "plat":
        return [p for p in perfs if p.discipline in DISCIPLINES_PLAT]
    if type_course == "obstacle":
        return [p for p in perfs if p.discipline in DISCIPLINES_OBSTACLE]
    return perfs


def caracteristiques(musique: str | None, type_course: str | None = None,
                     n: int = 5) -> dict[str, float]:
    """Indicateurs de forme à partir des n dernières performances.

    Si `type_course` est donné, seuls les résultats de la même famille
    (plat ou obstacle) sont comptés pour les indicateurs « _meme ».
    """
    tout = analyser(musique)
    meme = _filtrer(tout, type_course)
    res: dict[str, float] = {"mus_nb_perfs": float(len(tout))}
    for suffixe, perfs in (("", tout[:n]), ("_meme", meme[:n])):
        places = [p.place for p in perfs if p.place is not None]
        k = len(perfs)
        res[f"mus_nb{suffixe}"] = float(k)
        res[f"mus_place_moy{suffixe}"] = (sum(places) / len(places)) if places else float("nan")
        res[f"mus_victoires{suffixe}"] = float(sum(1 for p in perfs if p.place == 1))
        res[f"mus_top3{suffixe}"] = float(sum(1 for p in perfs if p.place is not None and p.place <= 3))
        res[f"mus_incidents{suffixe}"] = float(sum(1 for p in perfs if p.incident))
        res[f"mus_taux_top3{suffixe}"] = (res[f"mus_top3{suffixe}"] / k) if k else float("nan")
    derniere = tout[0].place if tout else None
    res["mus_derniere_place"] = float(derniere) if derniere is not None else float("nan")
    res["mus_derniere_incident"] = float(tout[0].incident) if tout else float("nan")
    return res
