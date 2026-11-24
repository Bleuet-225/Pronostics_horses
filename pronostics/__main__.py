"""Ligne de commande : python -m pronostics <commande> ...

Commandes :
  collecter  --debut JJ/MM/AAAA --fin JJ/MM/AAAA
  entrainer  [--type plat|obstacle|tous] [--avec-cotes]
  evaluer    : affiche les métriques des modèles enregistrés
  predire    --date JJ/MM/AAAA [--reunion N] [--course N]
  simuler    [--selection favori favori_30min prono_pmu presse] [--regles]
             [--sans-handicap] [--sans-groupes] [--cote-max X] [--ecart-min X]
             [--mise EUR] [--debut ..] [--fin ..]
  brut       --date JJ/MM/AAAA [--reunion N --course N] : réponse JSON brute de l'API
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime

import pandas as pd

from . import modele, service
from .api import ClientPMU, ErreurAPI
from .collecte import collecter_periode


def _date(s: str) -> date:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d%m%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Date invalide : {s} (attendu JJ/MM/AAAA)")


def cmd_collecter(a):
    df = collecter_periode(a.debut, a.fin, paralleles=a.paralleles)
    if df.empty:
        print("Aucune donnée collectée.")
        return 1
    nb = df.groupby(["date", "reunion", "course"]).ngroups
    print(f"Historique : {len(df)} lignes, {nb} courses "
          f"({df['date'].min()} → {df['date'].max()}).")
    print(df.groupby("type").apply(lambda g: g.groupby(["date", "reunion", "course"]).ngroups,
                                   include_groups=False).rename("courses").to_string())
    return 0


def cmd_entrainer(a):
    hist = service.charger_historique()
    if hist is None:
        print("Pas d'historique : lancez d'abord « collecter ».")
        return 1
    types = ["plat", "obstacle"] if a.type == "tous" else [a.type]
    code = 0
    for t in types:
        try:
            paquet = modele.entrainer(hist, t, avec_cotes=a.avec_cotes, avec_pronos=a.avec_pronos)
            _afficher_metriques(paquet)
        except ValueError as e:
            print(f"[{t}] {e}")
            code = 1
    return code


def _afficher_metriques(paquet):
    m = paquet["metriques"]
    print(f"\n=== Modèle {paquet['type']} ({paquet['nb_courses']} courses, "
          f"cotes {'utilisées' if paquet['avec_cotes'] else 'non utilisées'}, "
          f"pronostics {'utilisés' if paquet.get('avec_pronos') else 'non utilisés'}) ===")
    print(f"Apprentissage : {m['periode_apprentissage'][0]} → {m['periode_apprentissage'][1]} (exclu)")
    print(f"Test          : {m['periode_test'][0]} → {m['periode_test'][1]}")
    for cible in modele.CIBLES:
        print(f"  [{cible}] " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in m[cible].items()))


def cmd_evaluer(a):
    code = 0
    for t in ("plat", "obstacle"):
        try:
            _afficher_metriques(modele.charger(t))
        except FileNotFoundError as e:
            print(e)
            code = 1
    return code


def cmd_predire(a):
    res = service.pronostiquer(a.date, a.reunion, a.course)
    if res.empty:
        print("Aucune course de plat ou d'obstacles trouvée pour ces critères.")
        return 1
    pd.set_option("display.width", 200)
    for (d, r, c), g in res.groupby(modele.CLE_COURSE, sort=False):
        e = g.iloc[0]
        print(f"\nR{r}C{c} – {e['hippodrome']} – {e['libelle']} "
              f"({e['type']}, {e['discipline']}, {e['distance']:.0f} m, {len(g)} partants)")
        aff = g[service.COLONNES_AFFICHAGE].copy()
        aff["proba_gagnant"] = (aff["proba_gagnant"] * 100).round(1)
        aff["proba_top3"] = (aff["proba_top3"] * 100).round(1)
        aff = aff.rename(columns={"proba_gagnant": "% victoire", "proba_top3": "% top 3"})
        if aff["arrivee"].isna().all():
            aff = aff.drop(columns="arrivee")
        print(aff.to_string(index=False))
    return 0


def cmd_simuler(a):
    from .simulation import REGLES_RECOMMANDEES, Filtres, tableau
    hist = service.charger_historique()
    if hist is None or "rapport_place" not in hist:
        print("Pas de rapports dans l'historique : relancez « collecter ».")
        return 1
    if a.regles:
        f = REGLES_RECOMMANDEES
        filtres = Filtres(sans_handicap=True, sans_groupes=a.sans_groupes,
                          cote_max=a.cote_max, ecart_min=a.ecart_min or f.ecart_min)
    else:
        filtres = Filtres(sans_handicap=a.sans_handicap, sans_groupes=a.sans_groupes,
                          cote_max=a.cote_max, ecart_min=a.ecart_min)
    debut = a.debut.isoformat() if a.debut else None
    fin = a.fin.isoformat() if a.fin else None
    t = tableau(hist, criteres=a.selection, debut=debut, fin=fin, filtres=filtres)
    t["taux_place"] = (t["taux_place"] * 100).round(1)
    t["rendement"] = (t["rendement"] * 100).round(1)
    for c in ("total_mise", "resultat", "pire_baisse", "mise_max", "capital_necessaire"):
        t[c] = t[c] * a.mise
    pd.set_option("display.width", 220)
    unite = f"{a.mise:g} € par mise de base" if a.mise != 1 else "unités de mise (1 = mise de base)"
    print(f"Filtres : {filtres.actifs()} | montants en {unite} | rendement = résultat / total misé (%)")
    print(t.round(1).to_string(index=False))
    return 0


def cmd_brut(a):
    client = ClientPMU(cache=None)
    if a.reunion and a.course:
        donnees = client.participants(a.date, a.reunion, a.course)
    else:
        donnees = client.programme(a.date)
    print(json.dumps(donnees, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pronostics", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sp = p.add_subparsers(dest="commande", required=True)

    s = sp.add_parser("collecter", help="récupère l'historique des courses")
    s.add_argument("--debut", type=_date, required=True)
    s.add_argument("--fin", type=_date, required=True)
    s.add_argument("--paralleles", type=int, default=8,
                   help="nombre de jours téléchargés en parallèle (défaut : 8)")
    s.set_defaults(f=cmd_collecter)

    s = sp.add_parser("entrainer", help="entraîne les modèles")
    s.add_argument("--type", choices=["plat", "obstacle", "tous"], default="tous")
    s.add_argument("--avec-cotes", action="store_true",
                   help="utilise la cote PMU comme variable (disponible seulement près du départ)")
    s.add_argument("--avec-pronos", action="store_true",
                   help="utilise le pronostic PMU et la synthèse presse (publiés avant les paris)")
    s.set_defaults(f=cmd_entrainer)

    s = sp.add_parser("evaluer", help="affiche les métriques des modèles")
    s.set_defaults(f=cmd_evaluer)

    s = sp.add_parser("simuler", help="simule une stratégie en simple placé sur l'historique")
    s.add_argument("--selection", nargs="+", default=["favori"],
                   choices=["favori", "favori_30min", "prono_pmu", "presse"])
    s.add_argument("--debut", type=_date)
    s.add_argument("--fin", type=_date)
    s.add_argument("--regles", action="store_true",
                   help="règles recommandées : sans handicap et écart ≥ 2 avec le 2e favori")
    s.add_argument("--sans-handicap", action="store_true", help="exclut tous les handicaps")
    s.add_argument("--sans-groupes", action="store_true", help="exclut les Groupes I, II et III")
    s.add_argument("--cote-max", type=float, help="ne joue que si la cote du cheval est inférieure")
    s.add_argument("--ecart-min", type=float,
                   help="ne joue que si (plus basse cote des autres) / (cote du cheval) ≥ cette valeur")
    s.add_argument("--mise", type=float, default=1.0, help="mise de base en euros (défaut : 1)")
    s.set_defaults(f=cmd_simuler)

    for nom, f, aide in (("predire", cmd_predire, "pronostics pour une date"),
                         ("brut", cmd_brut, "réponse brute de l'API (diagnostic)")):
        s = sp.add_parser(nom, help=aide)
        s.add_argument("--date", type=_date, default=date.today())
        s.add_argument("--reunion", type=int)
        s.add_argument("--course", type=int)
        s.set_defaults(f=f)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    try:
        return a.f(a)
    except ErreurAPI as e:
        print(f"Erreur API PMU : {e}")
        return 2
    except FileNotFoundError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
