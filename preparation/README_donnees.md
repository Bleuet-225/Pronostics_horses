# Préparation des données — facteurs du « placé »

Préparation uniquement (aucun entraînement). Script : `preparation/prepare_data.py`
(relançable : `python preparation/prepare_data.py`). Paramètres et journal
complets : `preparation/donnees/parametres.json`.

## Données

| | |
|---|---|
| Source | `data/historique.csv.gz` (non modifié ; copie de travail `preparation/donnees/historique_copie.csv.gz`) |
| Période | 2024-11-24 → 2025-11-23 (première année de l'historique) |
| Unité statistique | un cheval partant dans une course de plat ou d'obstacle en France |
| y | `y_place` = 1 si le cheval a été payé en simple placé (`rapport_place` > 0), sinon 0 — catégorie binaire |
| Découpage | 80/20 **par date** : entraînement < 2025-09-12, test ≥ 2025-09-12 ; aucune course coupée en deux. Pas de tirage aléatoire, donc pas de graine nécessaire. |

## Fichiers produits (`preparation/donnees/`, non versionnés)

| Fichier | Contenu |
|---|---|
| `X_train.csv` / `X_test.csv` | matrices des facteurs centrés-réduits (45 105 × 127 / 13 539 × 127) |
| `y_train.csv` / `y_test.csv` | `y_place` (26,5 % / 26,7 % de 1) |
| `ids_train.csv` / `ids_test.csv` | date, réunion, course, numéro : identification de chaque ligne (même ordre que X) |
| `parametres.json` | colonnes, médianes, encodage, moyennes et écarts-types d'entraînement, journal |

## Transformations, dans l'ordre

1. Filtre de période.
2. Retrait des lignes de courses sans pari placé (moins de 4 partants) : y n'y a pas de sens.
3. Correction validée : poids de MIA L'HOMBRE (08/11/2025) enregistré par l'API en kg (`52`) au lieu
   d'hectogrammes → 52 kg ; décharge recalculée = 53 − 52 = 1 kg (poids théorique brut 530 hg).
4. `jockey` et `terrain_libelle` manquants → « INCONNU » (2 et 22 lignes).
5. Facteurs dérivés :
   - musique → indicateurs de forme sur les 5 dernières performances (toutes, et même famille plat/obstacle) ;
   - jockey et entraîneur → nombre de courses, taux de victoire et de top 3, calculés **uniquement sur les
     dates strictement antérieures**, par type de course, lissés vers un a priori (10 courses fictives à
     10 % de victoires et 30 % de top 3). Les premières semaines n'ont donc presque pas d'historique ;
   - indicateurs : `obstacle`, `piste_psf`, `handicap_valeur_absente`, `gains_absents`, `musique_absente`,
     `mus_meme_absente` ;
   - `place_corde` manquante (obstacle) → 0.
6. Découpage par date.
7. Retrait du test des courses dont l'hippodrome ou l'état de terrain n'existe pas dans l'entraînement.
8. Valeurs manquantes restantes → médiane **du jeu d'entraînement** (+ indicateur ci-dessus).
9. Encodage one-hot avec une modalité de référence par variable (modalités définies sur l'entraînement) ;
   catégories de course de moins de 50 lignes regroupées en AUTRE.
10. Colonnes exclues : identifiants, `pays` (constant), résultats et rapports (connus après la course),
    `place_ouvert`, cotes (décision), pronostics (vides avant juillet 2026), `handicap` (redondant avec
    `categorie`), `musique`, `jockey`, `entraineur` (remplacés par les facteurs dérivés).
11. Centrage-réduction de **toutes** les colonnes, y compris les 0/1 (décision), avec les moyennes et
    écarts-types **du jeu d'entraînement**.

### Journal d'exécution

- Période 2024-11-24 → 2025-11-23 : 58992 lignes
- Lignes de courses sans pari placé retirées : 15
- Poids corrigé : MIA L'HOMBRE le 2025-11-08 → 52.0 kg, décharge 1.0 kg
- Découpage par date : entraînement < 2025-09-12 (45105 lignes, 291 jours), test ≥ 2025-09-12 (13872 lignes, 73 jours)
- Test : 319 lignes (29 courses) retirées, `hippodrome` absent de l'entraînement : ['AMIENS', 'NIMES', 'PRUNELLI']
- Test : 14 lignes (2 courses) retirées, `terrain_libelle` absent de l'entraînement : ['Très léger']
- Manquants de `terrain_mesure` remplacés par la médiane d'entraînement 3.6
- Manquants de `handicap_valeur` remplacés par la médiane d'entraînement 31.5
- Manquants de `gains_carriere` remplacés par la médiane d'entraînement 2.5e+06
- Manquants de `gains_annee` remplacés par la médiane d'entraînement 2.7e+05
- Manquants de `gains_annee_prec` remplacés par la médiane d'entraînement 7.7e+05
- Manquants de `mus_place_moy` remplacés par la médiane d'entraînement 5.25
- Manquants de `mus_taux_top3` remplacés par la médiane d'entraînement 0.2
- Manquants de `mus_place_moy_meme` remplacés par la médiane d'entraînement 5.2
- Manquants de `mus_taux_top3_meme` remplacés par la médiane d'entraînement 0.2
- Manquants de `mus_derniere_place` remplacés par la médiane d'entraînement 5
- Manquants de `mus_derniere_incident` remplacés par la médiane d'entraînement 0
- `discipline` : one-hot, 3 colonnes, référence = PLAT
- `categorie` : modalités rares regroupées en AUTRE : ['QUALIFICATION_ACCAF', 'COURSE_INTERNATIONALE', 'FINALE_REGIONALE_ACCAF']
- `categorie` : one-hot, 12 colonnes, référence = COURSE_A_CONDITIONS
- `sexe` : one-hot, 2 colonnes, référence = MALES
- `oeilleres` : one-hot, 2 colonnes, référence = SANS_OEILLERES
- `hippodrome` : one-hot, 53 colonnes, référence = DEAUVILLE
- `terrain_libelle` : one-hot, 11 colonnes, référence = PSF STANDARD
- Centrage-réduction de 127 colonnes (paramètres d'entraînement)

## Colonnes finales (127)

Moyenne et écart-type = valeurs **d'entraînement avant centrage-réduction**, dans l'unité d'origine.
Pour revenir à l'unité d'origine : valeur = valeur réduite × écart-type + moyenne. Pour une colonne 0/1,
la moyenne est la proportion de 1.

| Colonne | Signification | Unité d'origine | Moyenne | Écart-type |
|---|---|---|---|---|
| `distance` | Distance de la course | mètres | 2327 | 910.3 |
| `montant_prix` | Allocation totale de la course | euros | 3.121e+04 | 5.286e+04 |
| `nb_declares` | Nombre de partants déclarés | nombre | 12.11 | 3.226 |
| `terrain_mesure` | Mesure au pénétromètre (PSF : médiane d'entraînement hors PSF) | indice pénétromètre, sans unité | 3.696 | 0.4134 |
| `age` | Âge du cheval | années | 4.598 | 1.862 |
| `jockey_change` | Changement de jockey annoncé | 0/1 | 0.01632 | 0.1267 |
| `inedit` | Cheval n'ayant jamais couru | 0/1 | 0.05392 | 0.2259 |
| `nb_courses` | Courses courues (semble limité à la même spécialité, constaté sur un échantillon) | nombre | 18.05 | 21.3 |
| `nb_victoires` | Victoires | nombre | 1.959 | 2.574 |
| `nb_places` | Places | nombre | 8.356 | 9.64 |
| `gains_carriere` | Gains en carrière | centimes d'euro (supposé, non confirmé) | 4.822e+06 | 8.711e+06 |
| `gains_annee` | Gains de l'année en cours | centimes d'euro (supposé) | 8.27e+05 | 2.516e+06 |
| `gains_annee_prec` | Gains de l'année précédente | centimes d'euro (supposé) | 1.598e+06 | 4.089e+06 |
| `poids` | Poids porté (après décharge) | kg | 59.27 | 4.92 |
| `decharge` | Poids théorique − poids porté (négatif sur 6 555 lignes, sens incertain) | kg | 0.5142 | 1.093 |
| `handicap_valeur` | Valeur de handicap (manquante : médiane d'entraînement) | valeur de handicap, unité non documentée par l'API | 33.47 | 10.24 |
| `place_corde` | Numéro de corde au départ (0 = obstacle, pas de corde) | numéro | 5.168 | 4.486 |
| `mus_nb_perfs` | Nombre de performances lisibles dans la musique | nombre | 6.6 | 3.236 |
| `mus_nb` | Performances retenues (5 dernières max) | nombre | 4.115 | 1.609 |
| `mus_place_moy` | Place moyenne sur les 5 dernières (0 = 10e et au-delà → compté 10) | place | 5.351 | 1.904 |
| `mus_victoires` | Victoires sur les 5 dernières | nombre | 0.4485 | 0.7066 |
| `mus_top3` | Top 3 sur les 5 dernières | nombre | 1.329 | 1.236 |
| `mus_incidents` | Incidents (arrêté, tombé, disqualifié…) sur les 5 dernières | nombre | 0.1322 | 0.4298 |
| `mus_taux_top3` | Part de top 3 sur les 5 dernières | proportion | 0.3145 | 0.2735 |
| `mus_nb_meme` | Idem, même famille (plat ou obstacle) | nombre | 3.992 | 1.698 |
| `mus_place_moy_meme` | Place moyenne, même famille | place | 5.335 | 1.903 |
| `mus_victoires_meme` | Victoires, même famille | nombre | 0.4395 | 0.7018 |
| `mus_top3_meme` | Top 3, même famille | nombre | 1.299 | 1.237 |
| `mus_incidents_meme` | Incidents, même famille | nombre | 0.1231 | 0.4182 |
| `mus_taux_top3_meme` | Part de top 3, même famille | proportion | 0.3147 | 0.2755 |
| `mus_derniere_place` | Place à la dernière course | place | 5.419 | 2.903 |
| `mus_derniere_incident` | Incident à la dernière course | 0/1 | 0.03237 | 0.177 |
| `musique_absente` | Musique absente | 0/1 | 0.06691 | 0.2499 |
| `jockey_nb` | Courses du jockey avant la date (même type) | nombre | 171.8 | 197.4 |
| `jockey_taux_victoire` | Taux de victoire du jockey avant la date, lissé | proportion | 0.09804 | 0.04525 |
| `jockey_taux_top3` | Taux de top 3 du jockey avant la date, lissé | proportion | 0.288 | 0.08148 |
| `entraineur_nb` | Courses de l'entraîneur avant la date (même type) | nombre | 81.64 | 93.45 |
| `entraineur_taux_victoire` | Taux de victoire de l'entraîneur avant la date, lissé | proportion | 0.0984 | 0.04736 |
| `entraineur_taux_top3` | Taux de top 3 de l'entraîneur avant la date, lissé | proportion | 0.292 | 0.08936 |
| `obstacle` | Course d'obstacle (référence : plat) | 0/1 | 0.2256 | 0.418 |
| `piste_psf` | Piste en sable fibré (PSF) | 0/1 | 0.239 | 0.4265 |
| `handicap_valeur_absente` | Valeur de handicap manquante | 0/1 | 0.2926 | 0.4549 |
| `gains_absents` | Gains manquants | 0/1 | 0.0278 | 0.1644 |
| `mus_meme_absente` | Aucune performance dans la même famille | 0/1 | 0.08531 | 0.2793 |
| `discipline__HAIE` | `discipline` = HAIE (référence : PLAT) | 0/1 | 0.1393 | 0.3462 |
| `discipline__STEEPLECHASE` | `discipline` = STEEPLECHASE (référence : PLAT) | 0/1 | 0.07372 | 0.2613 |
| `discipline__CROSS` | `discipline` = CROSS (référence : PLAT) | 0/1 | 0.01262 | 0.1116 |
| `categorie__HANDICAP_DIVISE` | `categorie` = HANDICAP_DIVISE (référence : COURSE_A_CONDITIONS) | 0/1 | 0.2527 | 0.4345 |
| `categorie__HANDICAP_CATEGORIE_DIVISE` | `categorie` = HANDICAP_CATEGORIE_DIVISE (référence : COURSE_A_CONDITIONS) | 0/1 | 0.08839 | 0.2839 |
| `categorie__A_RECLAMER` | `categorie` = A_RECLAMER (référence : COURSE_A_CONDITIONS) | 0/1 | 0.08261 | 0.2753 |
| `categorie__HANDICAP` | `categorie` = HANDICAP (référence : COURSE_A_CONDITIONS) | 0/1 | 0.07928 | 0.2702 |
| `categorie__HANDICAP_DE_CATEGORIE` | `categorie` = HANDICAP_DE_CATEGORIE (référence : COURSE_A_CONDITIONS) | 0/1 | 0.04669 | 0.211 |
| `categorie__COURSE_A_CONDITION_QUALIF_HP` | `categorie` = COURSE_A_CONDITION_QUALIF_HP (référence : COURSE_A_CONDITIONS) | 0/1 | 0.03691 | 0.1886 |
| `categorie__INCONNU` | `categorie` = INCONNU (référence : COURSE_A_CONDITIONS) | 0/1 | 0.01918 | 0.1371 |
| `categorie__GROUPE_III` | `categorie` = GROUPE_III (référence : COURSE_A_CONDITIONS) | 0/1 | 0.0123 | 0.1102 |
| `categorie__GROUPE_I` | `categorie` = GROUPE_I (référence : COURSE_A_CONDITIONS) | 0/1 | 0.004833 | 0.06935 |
| `categorie__GROUPE_II` | `categorie` = GROUPE_II (référence : COURSE_A_CONDITIONS) | 0/1 | 0.004013 | 0.06322 |
| `categorie__HANDICAP_A_RECLAMER` | `categorie` = HANDICAP_A_RECLAMER (référence : COURSE_A_CONDITIONS) | 0/1 | 0.00266 | 0.05151 |
| `categorie__AUTRE` | `categorie` = AUTRE (référence : COURSE_A_CONDITIONS) | 0/1 | 0.001818 | 0.0426 |
| `sexe__HONGRES` | `sexe` = HONGRES (référence : MALES) | 0/1 | 0.4523 | 0.4977 |
| `sexe__FEMELLES` | `sexe` = FEMELLES (référence : MALES) | 0/1 | 0.4223 | 0.4939 |
| `oeilleres__OEILLERES_AUSTRALIENNES` | `oeilleres` = OEILLERES_AUSTRALIENNES (référence : SANS_OEILLERES) | 0/1 | 0.1493 | 0.3564 |
| `oeilleres__OEILLERES_CLASSIQUE` | `oeilleres` = OEILLERES_CLASSIQUE (référence : SANS_OEILLERES) | 0/1 | 0.09345 | 0.2911 |
| `hippodrome__CHANTILLY` | `hippodrome` = CHANTILLY (référence : DEAUVILLE) | 0/1 | 0.08533 | 0.2794 |
| `hippodrome__CAGNES/MER` | `hippodrome` = CAGNES/MER (référence : DEAUVILLE) | 0/1 | 0.05505 | 0.2281 |
| `hippodrome__PAU` | `hippodrome` = PAU (référence : DEAUVILLE) | 0/1 | 0.05044 | 0.2188 |
| `hippodrome__ParisLongchamp` | `hippodrome` = ParisLongchamp (référence : DEAUVILLE) | 0/1 | 0.04844 | 0.2147 |
| `hippodrome__COMPIEGNE` | `hippodrome` = COMPIEGNE (référence : DEAUVILLE) | 0/1 | 0.04725 | 0.2122 |
| `hippodrome__SAINT-CLOUD` | `hippodrome` = SAINT-CLOUD (référence : DEAUVILLE) | 0/1 | 0.04576 | 0.209 |
| `hippodrome__AUTEUIL` | `hippodrome` = AUTEUIL (référence : DEAUVILLE) | 0/1 | 0.04323 | 0.2034 |
| `hippodrome__CLAIREFONTAINE` | `hippodrome` = CLAIREFONTAINE (référence : DEAUVILLE) | 0/1 | 0.03388 | 0.1809 |
| `hippodrome__LA TESTE` | `hippodrome` = LA TESTE (référence : DEAUVILLE) | 0/1 | 0.03035 | 0.1716 |
| `hippodrome__FONTAINEBLEAU` | `hippodrome` = FONTAINEBLEAU (référence : DEAUVILLE) | 0/1 | 0.02844 | 0.1662 |
| `hippodrome__PORNICHET` | `hippodrome` = PORNICHET (référence : DEAUVILLE) | 0/1 | 0.02811 | 0.1653 |
| `hippodrome__DIEPPE` | `hippodrome` = DIEPPE (référence : DEAUVILLE) | 0/1 | 0.02785 | 0.1645 |
| `hippodrome__VICHY` | `hippodrome` = VICHY (référence : DEAUVILLE) | 0/1 | 0.0272 | 0.1627 |
| `hippodrome__LYON-PARILLY` | `hippodrome` = LYON-PARILLY (référence : DEAUVILLE) | 0/1 | 0.02481 | 0.1555 |
| `hippodrome__MONT DE MARSAN` | `hippodrome` = MONT DE MARSAN (référence : DEAUVILLE) | 0/1 | 0.02135 | 0.1445 |
| `hippodrome__LYON LA SOIE` | `hippodrome` = LYON LA SOIE (référence : DEAUVILLE) | 0/1 | 0.02111 | 0.1437 |
| `hippodrome__LA CEPIERE` | `hippodrome` = LA CEPIERE (référence : DEAUVILLE) | 0/1 | 0.02091 | 0.1431 |
| `hippodrome__LE LION D'ANGERS` | `hippodrome` = LE LION D'ANGERS (référence : DEAUVILLE) | 0/1 | 0.02009 | 0.1403 |
| `hippodrome__PONT DE VIVAUX` | `hippodrome` = PONT DE VIVAUX (référence : DEAUVILLE) | 0/1 | 0.01929 | 0.1375 |
| `hippodrome__BORELY` | `hippodrome` = BORELY (référence : DEAUVILLE) | 0/1 | 0.0174 | 0.1308 |
| `hippodrome__LE BOUSCAT` | `hippodrome` = LE BOUSCAT (référence : DEAUVILLE) | 0/1 | 0.01536 | 0.123 |
| `hippodrome__ANGERS` | `hippodrome` = ANGERS (référence : DEAUVILLE) | 0/1 | 0.01503 | 0.1217 |
| `hippodrome__MOULINS` | `hippodrome` = MOULINS (référence : DEAUVILLE) | 0/1 | 0.01293 | 0.113 |
| `hippodrome__DAX` | `hippodrome` = DAX (référence : DEAUVILLE) | 0/1 | 0.01224 | 0.1099 |
| `hippodrome__STRASBOURG` | `hippodrome` = STRASBOURG (référence : DEAUVILLE) | 0/1 | 0.01086 | 0.1037 |
| `hippodrome__NANTES` | `hippodrome` = NANTES (référence : DEAUVILLE) | 0/1 | 0.01053 | 0.1021 |
| `hippodrome__CHATEAUBRIANT` | `hippodrome` = CHATEAUBRIANT (référence : DEAUVILLE) | 0/1 | 0.01007 | 0.09982 |
| `hippodrome__SENONNES POUANCE` | `hippodrome` = SENONNES POUANCE (référence : DEAUVILLE) | 0/1 | 0.009356 | 0.09627 |
| `hippodrome__VITTEL` | `hippodrome` = VITTEL (référence : DEAUVILLE) | 0/1 | 0.008602 | 0.09235 |
| `hippodrome__CRAON` | `hippodrome` = CRAON (référence : DEAUVILLE) | 0/1 | 0.008447 | 0.09152 |
| `hippodrome__TARBES` | `hippodrome` = TARBES (référence : DEAUVILLE) | 0/1 | 0.007627 | 0.087 |
| `hippodrome__NANCY-BRABOIS` | `hippodrome` = NANCY-BRABOIS (référence : DEAUVILLE) | 0/1 | 0.007228 | 0.08471 |
| `hippodrome__ARGENTAN` | `hippodrome` = ARGENTAN (référence : DEAUVILLE) | 0/1 | 0.007161 | 0.08432 |
| `hippodrome__SALON DE PROVENCE` | `hippodrome` = SALON DE PROVENCE (référence : DEAUVILLE) | 0/1 | 0.00705 | 0.08367 |
| `hippodrome__SAINT MALO` | `hippodrome` = SAINT MALO (référence : DEAUVILLE) | 0/1 | 0.006673 | 0.08142 |
| `hippodrome__LE MANS` | `hippodrome` = LE MANS (référence : DEAUVILLE) | 0/1 | 0.006274 | 0.07896 |
| `hippodrome__LE CROISE LAROCHE` | `hippodrome` = LE CROISE LAROCHE (référence : DEAUVILLE) | 0/1 | 0.006208 | 0.07854 |
| `hippodrome__AIX LES BAINS` | `hippodrome` = AIX LES BAINS (référence : DEAUVILLE) | 0/1 | 0.005343 | 0.0729 |
| `hippodrome__AGEN LA GARENNE` | `hippodrome` = AGEN LA GARENNE (référence : DEAUVILLE) | 0/1 | 0.004456 | 0.06661 |
| `hippodrome__LIGNIERES` | `hippodrome` = LIGNIERES (référence : DEAUVILLE) | 0/1 | 0.004257 | 0.0651 |
| `hippodrome__EVREUX` | `hippodrome` = EVREUX (référence : DEAUVILLE) | 0/1 | 0.003791 | 0.06146 |
| `hippodrome__SAINT BRIEUC` | `hippodrome` = SAINT BRIEUC (référence : DEAUVILLE) | 0/1 | 0.003636 | 0.06019 |
| `hippodrome__CHOLET` | `hippodrome` = CHOLET (référence : DEAUVILLE) | 0/1 | 0.003481 | 0.0589 |
| `hippodrome__MACHECOUL` | `hippodrome` = MACHECOUL (référence : DEAUVILLE) | 0/1 | 0.003436 | 0.05852 |
| `hippodrome__LE TOUQUET` | `hippodrome` = LE TOUQUET (référence : DEAUVILLE) | 0/1 | 0.003392 | 0.05814 |
| `hippodrome__LES SABLES D OLONNE` | `hippodrome` = LES SABLES D OLONNE (référence : DEAUVILLE) | 0/1 | 0.003148 | 0.05602 |
| `hippodrome__CAVAILLON` | `hippodrome` = CAVAILLON (référence : DEAUVILLE) | 0/1 | 0.003037 | 0.05503 |
| `hippodrome__HYERES` | `hippodrome` = HYERES (référence : DEAUVILLE) | 0/1 | 0.002417 | 0.0491 |
| `hippodrome__MAURE DE BRETAGNE` | `hippodrome` = MAURE DE BRETAGNE (référence : DEAUVILLE) | 0/1 | 0.001995 | 0.04462 |
| `hippodrome__PONTCHATEAU` | `hippodrome` = PONTCHATEAU (référence : DEAUVILLE) | 0/1 | 0.001818 | 0.0426 |
| `hippodrome__MARTINIQUE` | `hippodrome` = MARTINIQUE (référence : DEAUVILLE) | 0/1 | 0.001618 | 0.0402 |
| `hippodrome__MESLAY DU MAINE` | `hippodrome` = MESLAY DU MAINE (référence : DEAUVILLE) | 0/1 | 0.0006429 | 0.02535 |
| `hippodrome__MACON-CLUNY` | `hippodrome` = MACON-CLUNY (référence : DEAUVILLE) | 0/1 | 0.0002217 | 0.01489 |
| `terrain_libelle__Bon souple` | `terrain_libelle` = Bon souple (référence : PSF STANDARD) | 0/1 | 0.1868 | 0.3897 |
| `terrain_libelle__Très souple` | `terrain_libelle` = Très souple (référence : PSF STANDARD) | 0/1 | 0.1803 | 0.3844 |
| `terrain_libelle__Souple` | `terrain_libelle` = Souple (référence : PSF STANDARD) | 0/1 | 0.1649 | 0.3711 |
| `terrain_libelle__Bon` | `terrain_libelle` = Bon (référence : PSF STANDARD) | 0/1 | 0.08788 | 0.2831 |
| `terrain_libelle__Collant` | `terrain_libelle` = Collant (référence : PSF STANDARD) | 0/1 | 0.08008 | 0.2714 |
| `terrain_libelle__Lourd` | `terrain_libelle` = Lourd (référence : PSF STANDARD) | 0/1 | 0.05132 | 0.2207 |
| `terrain_libelle__PSF LENTE` | `terrain_libelle` = PSF LENTE (référence : PSF STANDARD) | 0/1 | 0.01432 | 0.1188 |
| `terrain_libelle__Très lourd` | `terrain_libelle` = Très lourd (référence : PSF STANDARD) | 0/1 | 0.006363 | 0.07951 |
| `terrain_libelle__Bon léger` | `terrain_libelle` = Bon léger (référence : PSF STANDARD) | 0/1 | 0.002638 | 0.0513 |
| `terrain_libelle__INCONNU` | `terrain_libelle` = INCONNU (référence : PSF STANDARD) | 0/1 | 0.0004878 | 0.02208 |
| `terrain_libelle__Léger` | `terrain_libelle` = Léger (référence : PSF STANDARD) | 0/1 | 0.0003104 | 0.01762 |

## Points restés incertains

1. **Unité des gains** (`gains_*`) : supposée en centimes d'euro (cohérent avec les montants observés),
   non confirmée par une documentation de l'API. Sans effet sur la préparation (centrage-réduction).
2. **Décharge négative** (6 555 lignes, jusqu'à −2,5 kg) : conservée telle quelle ; sa signification
   (surcharge ?) n'est pas établie.
3. **`handicap_valeur`** : unité non documentée ; renseignée aussi pour environ la moitié des courses non
   handicap, sans règle identifiée.
4. **`nb_courses`, `nb_victoires`, `nb_places`** : semblent ne compter que la même spécialité (constaté sur
   quelques chevaux, pas vérifié systématiquement).
5. **Taux jockey/entraîneur au début de la période** : sans historique avant le 24/11/2024, ils valent
   surtout l'a priori pendant les premières semaines.
6. **Colonnes 0/1 centrées-réduites** : leurs poids s'expriment « par écart-type », comme les autres ;
   pour une modalité rare, un écart-type correspond à un grand saut 0 → 1.
7. **Non traité, à décider à l'étape d'entraînement** : les variables très asymétriques (gains,
   allocation : quelques valeurs très élevées, ex. Arc de Triomphe à 5 M€) n'ont pas été transformées
   (pas de logarithme), faute de validation ; elles peuvent peser fortement dans un modèle linéaire.
8. **Corrélations entre facteurs** (ex. `nb_courses` / `nb_places`, `mus_*` / `mus_*_meme`, `obstacle` /
   disciplines) : non traitées ; elles peuvent rendre les poids individuels instables dans un modèle
   linéaire.
9. **Colinéarités exactes (en attente de décision)** : `obstacle` = somme des colonnes `discipline__*`, et
   `piste_psf` = 1 − somme des colonnes `terrain_libelle__*` non PSF. La matrice d'entraînement est de rang
   126 au lieu de 128 (avec la constante) : un modèle linéaire ne peut pas attribuer de poids unique à ces
   facteurs. Correction proposée : retirer `obstacle` et `piste_psf`.
