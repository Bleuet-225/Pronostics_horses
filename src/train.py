"""Entraîne sur 2023, évalue sur 2024 la prédiction du GAGNANT (top 1).

Deux modèles sont comparés :
  - SANS cotes : forme, carrière, poids, corde, jockey/entraîneur...
  - AVEC cotes : mêmes variables + cotes PMU (référence et clôture).
Référence (baseline) : choisir le favori des cotes PMU.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT / "data" / "dataset.parquet").sort_values(["date", "race_id"])

ODDS = {"cote", "coteRef", "p_cote", "p_coteRef", "rk_cote", "rk_coteRef", "z_cote", "z_coteRef"}
EXCLUDE = {"race_id", "date", "heure", "nom", "numPmu", "entraineur", "jockey",
           "ordreArrivee", "win", "coteDate", "hippodrome"}
BASE_FEATS = [c for c in df.columns if c not in EXCLUDE and c not in ODDS]
ODDS_FEATS = BASE_FEATS + sorted(ODDS)

train = df[df["date"] < "2024-01-01"]
test = df[df["date"] >= "2024-01-01"]
# validation pour l'arrêt précoce : 4 derniers mois de 2023 (toujours dans 2023)
fit, val = train[train["date"] < "2023-09-01"], train[train["date"] >= "2023-09-01"]


def groups(d):
    return d.groupby("race_id", sort=False).size().to_numpy()


def topk_acc(d, score, k=1):
    d = d.assign(s=score)
    d["r"] = d.groupby("race_id")["s"].rank(ascending=False, method="first")
    return d.loc[d["win"] == 1, "r"].le(k).mean()


def run(feats, name):
    params = dict(objective="lambdarank", metric="ndcg", eval_at=[1], learning_rate=0.03,
                  num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=3,
                  verbose=-1, seed=0)
    dfit = lgb.Dataset(fit[feats], fit["win"], group=groups(fit))
    dval = lgb.Dataset(val[feats], val["win"], group=groups(val))
    m = lgb.train(params, dfit, 3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    best = m.best_iteration
    # ré-entraînement sur tout 2023 avec le nombre d'itérations retenu
    m = lgb.train(params, lgb.Dataset(train[feats], train["win"], group=groups(train)), best)
    s = m.predict(test[feats])
    imp = pd.Series(m.feature_importance("gain"), feats).sort_values(ascending=False)
    return s, imp, best


def report(d, s):
    return {"top1": topk_acc(d, s, 1), "top3": topk_acc(d, s, 3)}


results = {}
fav = -test["cote"].fillna(999).to_numpy()
segments = {"tout": test.index == test.index, "France": (test["pays"] == "FRA").to_numpy()}
scores = {"Favori PMU (baseline)": fav}
importances = {}
for feats, name in [(BASE_FEATS, "Modèle SANS cotes"), (ODDS_FEATS, "Modèle AVEC cotes")]:
    s, imp, best = run(feats, name)
    scores[name] = s
    importances[name] = imp.head(15).round(0).to_dict()
    print(name, "itérations:", best)

rng = np.random.default_rng(0)
scores["Hasard"] = rng.random(len(test))
for name, s in scores.items():
    results[name] = {seg: report(test[mask], s[mask]) for seg, mask in segments.items()}

info = {"courses_train_2023": int(train["race_id"].nunique()),
        "courses_test_2024": int(test["race_id"].nunique()),
        "courses_test_2024_France": int(test.loc[test["pays"] == "FRA", "race_id"].nunique()),
        "partants_moyen_test": float(test.groupby("race_id").size().mean())}
print(json.dumps(info, indent=1))
for name, r in results.items():
    print(f"{name:25s} " + "  ".join(f"{seg}: top1={v['top1']:.1%} top3={v['top3']:.1%}" for seg, v in r.items()))
for name, imp in importances.items():
    print(name, list(imp)[:10])
json.dump({"info": info, "resultats": results, "importances": importances},
          open(ROOT / "results.json", "w"), indent=1, ensure_ascii=False, default=float)
