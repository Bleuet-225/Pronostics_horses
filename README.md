# Pronostics_horses

Prédiction du **gagnant** des courses de **plat** PMU : entraînement sur 2023, test sur 2024.

## Utilisation

```bash
pip install -r requirements.txt
python src/collect.py 2023-01-01 2024-12-31   # ~45 min, API PMU turfinfo -> data/raw/
python src/features.py                        # -> data/dataset.parquet
python src/train.py                           # -> results.json
```

## Données

- 15 579 courses de plat (2023 : 7 585 ; 2024 : 7 994 dont 4 107 en France), ~10 partants/course.
- Toutes les variables sont connues avant le départ (musique et stats carrière vérifiées « avant course »).
  Les taux jockey / entraîneur / cheval sont calculés uniquement sur les jours antérieurs.
- Limites : le 07/11/2024 est indisponible dans l'API (réponse vide). Beaucoup de courses étrangères
  (Chili, USA, Émirats…) n'ont pas de cotes PMU.
- Les « cotes finales » (`dernierRapportDirect`) sont horodatées en moyenne ~4 min après l'heure
  de départ prévue ; elles sont traitées comme les cotes à la clôture des paris.

## Résultats sur 2024 (modèle jamais exposé à 2024)

| Méthode | Top 1, toutes courses | Top 1, France | Gagnant dans top 3, France |
|---|---|---|---|
| Hasard | 10,7 % | 10,7 % | 31,1 % |
| Modèle SANS cotes | 23,5 % | 22,4 % | 52,4 % |
| Favori PMU (référence) | 24,0 % | **29,6 %** | 61,1 % |
| Modèle AVEC cotes | **27,8 %** | 29,2 % | 60,6 % |

**L'objectif de 80 % de gagnants exacts n'est pas atteint et ne l'est par aucune méthode honnête** :
le marché des paris lui-même ne désigne le vainqueur que ~30 % du temps. En France, le modèle avec
cotes égale le favori sans le battre ; sur l'ensemble (courses étrangères sans cotes incluses), il
le dépasse de ~4 points. Un score de 80 % ne s'obtiendrait qu'en entraînant sur les courses de 2024
elles-mêmes (fuite de données), ce qui n'a aucune valeur prédictive.
