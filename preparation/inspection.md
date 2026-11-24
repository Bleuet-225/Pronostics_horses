# Étape 1 — Inspection

Source : copie de `data/historique.csv.gz` (`preparation/donnees/historique_copie.csv.gz`),
filtrée sur le 24/11/2024 → 23/11/2025.

- Lignes : 58992 (une par partant) · Colonnes : 52 · Courses : 5494
- Doublons : 0 ligne identique ; 0 doublon sur (date, réunion, course, numéro)
- y = placé (rapport_place > 0) : 1 → 15640 (26.5%), 0 → 43352

| Colonne | Type | Manquants | Valeurs distinctes | Résumé |
|---|---|---|---|---|
| `date` | str | 0 (0.0%) | 364 | 364 valeurs uniques ; ex. 2025-03-22 (391), 2025-11-16 (364), 2025-03-01 (350) |
| `reunion` | int64 | 0 (0.0%) | 16 | min 1 · max 16 · moyenne 3.161 |
| `course` | int64 | 0 (0.0%) | 12 | min 1 · max 12 · moyenne 4.702 |
| `hippodrome` | str | 0 (0.0%) | 57 | 57 valeurs uniques ; ex. CHANTILLY (5334), DEAUVILLE (4990), AUTEUIL (3181) |
| `pays` | str | 0 (0.0%) | 1 | 1 valeurs uniques ; ex. FRA (58992) |
| `type` | str | 0 (0.0%) | 2 | 2 valeurs uniques ; ex. plat (45408), obstacle (13584) |
| `discipline` | str | 0 (0.0%) | 4 | 4 valeurs uniques ; ex. PLAT (45408), HAIE (8409), STEEPLECHASE (4515) |
| `libelle` | str | 0 (0.0%) | 5111 | 5111 valeurs uniques ; ex. PRIX CLAUDE ROUGET (94), PRIX FRANCE GALOP (73), PRIX DES LANDES (61) |
| `heure_depart` | int64 | 0 (0.0%) | 5460 | min 1.732e+12 · max 1.764e+12 · moyenne 1.749e+12 |
| `distance` | float64 | 0 (0.0%) | 72 | min 800 · max 7300 · moyenne 2336 |
| `montant_prix` | float64 | 0 (0.0%) | 209 | min 6000 · max 5e+06 · moyenne 3.312e+04 |
| `nb_declares` | float64 | 0 (0.0%) | 17 | min 4 · max 20 · moyenne 12.06 |
| `terrain_mesure` | float64 | 13374 (22.7%) | 31 | min 1.1 · max 5.6 · moyenne 3.784 |
| `terrain_libelle` | str | 22 (0.0%) | 12 | 12 valeurs uniques ; ex. PSF STANDARD (12706), Très souple (11648), Souple (9771) |
| `handicap` | int64 | 0 (0.0%) | 2 | min 0 · max 1 · moyenne 0.4648 |
| `categorie` | str | 0 (0.0%) | 15 | 15 valeurs uniques ; ex. COURSE_A_CONDITIONS (21817), HANDICAP_DIVISE (14744), A_RECLAMER (4918) |
| `num` | int64 | 0 (0.0%) | 20 | min 1 · max 20 · moyenne 6.516 |
| `nom` | str | 0 (0.0%) | 12325 | 12325 valeurs uniques ; ex. LA FERRETCAPIENNE (27), GOOD QUESTION (27), ARMORIGENE (26) |
| `age` | float64 | 0 (0.0%) | 13 | min 2 · max 14 · moyenne 4.512 |
| `sexe` | str | 0 (0.0%) | 3 | 3 valeurs uniques ; ex. HONGRES (26595), FEMELLES (25039), MALES (7358) |
| `oeilleres` | str | 0 (0.0%) | 3 | 3 valeurs uniques ; ex. SANS_OEILLERES (44695), OEILLERES_AUSTRALIENNES (8803), OEILLERES_CLASSIQUE (5494) |
| `entraineur` | str | 0 (0.0%) | 1021 | 1021 valeurs uniques ; ex. M.SEROR (S) (858), HA.PANTALL (S) (747), C.ESCUDER (714) |
| `jockey` | str | 2 (0.0%) | 965 | 965 valeurs uniques ; ex. C.DEMURO (1402), M.GUYON (1361), A.MADAMET (1189) |
| `jockey_change` | int64 | 0 (0.0%) | 2 | min 0 · max 1 · moyenne 0.0159 |
| `musique` | str | 3881 (6.6%) | 48297 | 48297 valeurs uniques ; ex. 0p (388), 6p (260), 5p (248) |
| `inedit` | int64 | 0 (0.0%) | 2 | min 0 · max 1 · moyenne 0.05155 |
| `nb_courses` | float64 | 0 (0.0%) | 190 | min 0 · max 210 · moyenne 17.99 |
| `nb_victoires` | float64 | 0 (0.0%) | 21 | min 0 · max 20 · moyenne 1.962 |
| `nb_places` | float64 | 0 (0.0%) | 77 | min 0 · max 76 · moyenne 8.405 |
| `gains_carriere` | float64 | 1254 (2.1%) | 20760 | min 0 · max 5.859e+08 · moyenne 4.969e+06 |
| `gains_annee` | float64 | 1254 (2.1%) | 11184 | min 0 · max 2.585e+08 · moyenne 1.035e+06 |
| `gains_annee_prec` | float64 | 1254 (2.1%) | 4173 | min 0 · max 3.429e+08 · moyenne 1.595e+06 |
| `poids` | float64 | 0 (0.0%) | 59 | min 5.2 · max 76.5 · moyenne 59.31 |
| `decharge` | float64 | 0 (0.0%) | 15 | min -2.5 · max 47.8 · moyenne 0.5269 |
| `handicap_valeur` | float64 | 17077 (28.9%) | 130 | min 2 · max 81 · moyenne 34.5 |
| `place_corde` | float64 | 13584 (23.0%) | 20 | min 1 · max 20 · moyenne 6.658 |
| `cote_direct` | float64 | 92 (0.2%) | 284 | min 1.1 · max 326 · moyenne 21.06 |
| `cote_reference` | float64 | 2176 (3.7%) | 216 | min 1.1 · max 177 · moyenne 15.7 |
| `arrivee` | int64 | 0 (0.0%) | 19 | min 1 · max 99 · moyenne 11.03 |
| `gagnant` | int64 | 0 (0.0%) | 2 | min 0 · max 1 · moyenne 0.0934 |
| `top3` | int64 | 0 (0.0%) | 2 | min 0 · max 1 · moyenne 0.2797 |
| `place_ouvert` | float64 | 0 (0.0%) | 2 | min 0 · max 1 · moyenne 0.9997 |
| `rapport_place` | float64 | 0 (0.0%) | 185 | min 0 · max 54.9 · moyenne 0.7704 |
| `rapport_gagnant` | float64 | 0 (0.0%) | 393 | min 0 · max 138.6 · moyenne 0.7795 |
| `pmu_prono_rang` | float64 | 58992 (100.0%) | 0 | vide |
| `pmu_cote_prob` | float64 | 58992 (100.0%) | 0 | vide |
| `presse_nb_avis` | float64 | 58992 (100.0%) | 0 | vide |
| `presse_synthese` | float64 | 58992 (100.0%) | 0 | vide |
| `presse_nb_cite` | float64 | 58992 (100.0%) | 0 | vide |
| `presse_nb_premier` | float64 | 58992 (100.0%) | 0 | vide |
| `presse_rang_moy` | float64 | 58992 (100.0%) | 0 | vide |
| `presse_points` | float64 | 58992 (100.0%) | 0 | vide |
