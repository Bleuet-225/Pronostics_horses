# Pronostics hippiques – plat et obstacles

Programme Python qui estime, pour chaque cheval d'une course française de **plat**
ou d'**obstacles** (haies, steeple-chase, cross) :

- la **probabilité de victoire** (somme = 100 % par course) ;
- la **probabilité de finir dans les 3 premiers**.

Les données proviennent de l'API JSON publique du PMU
(`online.turfinfo.api.pmu.fr`). Cette API **n'est pas documentée officiellement** :
elle peut changer sans préavis.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Collecter l'historique

```bash
python -m pronostics -v collecter --debut 01/01/2025 --fin 23/11/2026
```

Les jours sont téléchargés en parallèle (`--paralleles 8` par défaut) :
3 mois se collectent en environ 1 minute. Seules les courses en France
sont retenues. Les réponses brutes sont mises en cache dans `data/brut/` et l'historique mis à
plat est écrit dans `data/historique.csv.gz` (une ligne par partant). Relancer la
commande sur une autre période complète l'historique sans doublons.

### 2. Entraîner les modèles

```bash
python -m pronostics entrainer               # plat et obstacle
python -m pronostics entrainer --type plat
python -m pronostics entrainer --avec-cotes  # ajoute la cote PMU comme variable
```

Il faut au moins 200 courses du type concerné. Les 20 % de dates les plus
récentes servent de période de test ; les métriques sont affichées (log-loss,
AUC, taux de réussite du 1er choix comparé au hasard et au favori des cotes),
puis le modèle final est ré-entraîné sur tout l'historique et enregistré dans
`modeles/`. `python -m pronostics evaluer` réaffiche ces métriques.

Avec `--avec-cotes`, les prédictions dépendent de la cote au moment de l'appel :
elles ne sont fiables que près du départ. Dans l'historique, la cote enregistrée
est la dernière cote connue (cote finale), ce qui rend les métriques
d'entraînement avec cotes plus favorables que l'usage réel avant la clôture.

### 3. Pronostiquer

En ligne de commande :

```bash
python -m pronostics predire --date 24/11/2026
python -m pronostics predire --date 24/11/2026 --reunion 1 --course 3
```

Interface web :

```bash
streamlit run app.py
```

### 4. Simuler une stratégie en simple placé

Rejoue une stratégie sur l'historique avec les rapports définitifs du PMU,
en mise fixe et en progression de Fibonacci :

```bash
python -m pronostics simuler --selection favori_30min                 # favori à ~30 min
python -m pronostics simuler --selection favori_30min --regles --mise 2
python -m pronostics simuler --selection favori --sans-handicap --cote-max 2 --ecart-min 2
```

- Sélections : `favori` (cote finale), `favori_30min` (cote de référence,
  ~30 min avant le départ), `prono_pmu`, `presse`.
- Filtres : `--sans-handicap`, `--sans-groupes`, `--cote-max X`,
  `--ecart-min X` (plus basse cote des autres partants / cote du cheval joué).
- `--regles` = règles recommandées : sans handicap et écart ≥ 2.
  Validées sur avril → novembre 2026 (courses non utilisées pour les choisir) :
  rendement −5,6 % contre −10,1 % sans règle. Elles réduisent la perte,
  aucune règle testée n'a rendu le placé sur le favori rentable.

### Diagnostic

```bash
python -m pronostics brut --date 24/11/2026                       # programme
python -m pronostics brut --date 24/11/2026 --reunion 1 --course 3  # partants
```

## Fonctionnement

- **Variables** (`pronostics/features.py`) : âge, sexe, poids, corde (plat),
  type d'obstacle, distance, terrain, allocation, statistiques de carrière,
  forme récente extraite de la musique (globale et même discipline),
  réussite du jockey et de l'entraîneur, et écarts/rangs de ces valeurs par
  rapport aux autres partants de la course.
- **Sans fuite temporelle** : les statistiques jockey/entraîneur d'une course
  n'utilisent que les courses de dates strictement antérieures.
- **Modèle** (`pronostics/modele.py`) : un modèle par type (plat / obstacle)
  et par cible, puis normalisation des probabilités à l'intérieur de chaque
  course.
  - victoire : moyenne d'un gradient boosting et d'un réseau de neurones
    (scikit-learn) ;
  - top 3 : gradient boosting seul (le mélange n'y apportait pas de gain
    significatif).

## Tests

```bash
python -m pytest
```

Les tests utilisent des données synthétiques qui vérifient la mécanique du
code, pas la qualité des pronostics.

## Avertissement

Les probabilités sont des estimations statistiques et ne garantissent aucun
résultat. Jouer comporte des risques : endettement, dépendance… Appelez le
09 74 75 13 13 (appel non surtaxé).
