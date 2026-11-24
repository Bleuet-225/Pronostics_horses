"""Fonctions communes à la ligne de commande et à l'application web."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from . import modele
from .api import ClientPMU, courses_plat_obstacle
from .collecte import FICHIER_HISTORIQUE, _variables_pronostics, lignes_course


def charger_historique(fichier: Path = FICHIER_HISTORIQUE) -> pd.DataFrame | None:
    if not fichier.exists():
        return None
    return pd.read_csv(fichier)


def courses_du_jour(d: date, client: ClientPMU | None = None) -> list[dict]:
    client = client or ClientPMU()
    frais = d >= date.today()  # programme du jour/futur : pas de cache
    return courses_plat_obstacle(client.programme(d, utiliser_cache=not frais))


def partants(d: date, reunion: int | None = None, course: int | None = None,
             client: ClientPMU | None = None) -> pd.DataFrame:
    client = client or ClientPMU()
    frais = d >= date.today()
    lignes = []
    for c in courses_du_jour(d, client):
        if reunion is not None and c["_reunion"] != reunion:
            continue
        if course is not None and c["numOrdre"] != course:
            continue
        r, n = c["_reunion"], c["numOrdre"]
        p = client.participants(d, r, n, utiliser_cache=not frais)
        pronos = _variables_pronostics(client.pronostics(d, r, n, utiliser_cache=not frais),
                                       client.pronostics_detailles(d, r, n, utiliser_cache=not frais))
        lignes.extend(lignes_course(d, c, p, pronos=pronos))
    return pd.DataFrame(lignes)


def pronostiquer(d: date, reunion: int | None = None, course: int | None = None,
                 client: ClientPMU | None = None) -> pd.DataFrame:
    df = partants(d, reunion, course, client)
    if df.empty:
        return df
    return modele.predire(df, charger_historique())


COLONNES_AFFICHAGE = ["rang_prono", "num", "nom", "jockey", "entraineur", "musique",
                      "cote_direct", "proba_gagnant", "proba_top3", "arrivee"]
