# Détection de structures radioélectriques sur imagerie aérienne

**FR Hack! 2026 — Challenge 4 (ANFR × ISEP)**

Chaîne complète : référentiel ANFR → imagerie aérienne → annotations faibles →
détection YOLOv8 → géolocalisation des détections → tableau de bord cartographique.

---

## En une minute

Le référentiel ANFR donne, pour chaque site radioélectrique, une **coordonnée** et une
**hauteur de support** — jamais une boîte englobante. Impossible d'entraîner un détecteur
sans fabriquer d'abord cette supervision.

Notre approche : **la boîte est centrée sur le crop, et l'incertitude est portée par sa
taille, pas par sa position.** Chaque demi-côté est la somme de termes physiques
identifiés (erreur de recalage, résidu de géoréférencement, emprise du support,
déplacement dû au relief).

Le verrou réel du projet n'était pas le modèle mais **l'échelle des images**, absente des
métadonnées. Notre première hypothèse était fausse d'un facteur 3. Nous l'avons **mesurée**
par corrélation de phase entre crops voisins, et cette correction a fait passer le modèle
du bruit pur à un détecteur fonctionnel.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install ultralytics pandas numpy pillow streamlit folium streamlit-folium plotly
```

Testé sur Python 3.12, CPU uniquement (Intel Celeron N4120 / Intel i5-7500).

---

## Reproduire le projet de bout en bout

```bash
# 1 — MESURER l'emprise au sol des crops (à faire en premier, toujours)
python calibrate_echelle.py \
    --csv donnees_antennes_hackathon.csv \
    --images "photos hackathon"

# 2 — construire le dataset YOLO à l'emprise mesurée
python prepare_dataset.py build \
    --csv donnees_antennes_hackathon.csv \
    --images "photos hackathon" --out data2 \
    --extent-m 70.1 --shadow-mode none --max-offset-m 20

# 3 — entraîner
python starter_antennes.py train \
    --data data2 --epochs 40 --imgsz 416 --batch 8

# 4 — annoter manuellement (tableau de bord)
streamlit run app.py

# 5 — évaluer sur la vérité terrain humaine
python build_val_manuel.py --data-root data2 --eval --weights best_70m_40ep.pt
python eval_localisation.py --weights best_70m_40ep.pt

# 6 — chercher des structures sans correspondance au référentiel
python detect_candidats.py --weights best_70m_40ep.pt \
    --csv donnees_antennes_hackathon.csv \
    --images data2/images/test --conf 0.58 --seuil-m 25

# contrôle — doit afficher « TOUS LES TESTS PASSENT »
python test_geoloc.py
python test_app_logic.py
```

> ⚠️ **Toute valeur d'emprise autre que 70,1 m est un bug.** Le dossier `data/` est
> l'ancien dataset construit à 204,8 m, conservé pour mémoire. Le dataset de référence
> est `data2/`.

---

## Arborescence

```
projet/
├── photos hackathon/              554 crops JPG 1024×1024 (source, ne pas modifier)
├── donnees_antennes_hackathon.csv référentiel ANFR, 554 sites
│
├── data2/                         ✅ dataset courant (70,1 m)
│   ├── images/{train,val,test}/
│   ├── labels/{train,val}/
│   ├── images/labels/val_manuel/  jeu annoté à la main
│   ├── antennes.yaml              config YOLO
│   ├── dataset_manifest.csv       traçabilité : 554 lignes, une par site
│   ├── test_douteux.csv           les 200 sites écartés + motif
│   ├── val_manual_todo.csv        liste des sites à annoter à la main
│   ├── METHODE_ANNOTATION.md      spécification de la méthode d'annotation
│   └── qc/                        planches de contrôle visuel
│
├── data/                          ❌ ancien dataset erroné (204,8 m) — ne pas utiliser
├── best_70m_40ep.pt               poids du modèle entraîné
├── coordonnees_corrigees.csv      annotations manuelles produites par app.py
├── runs/                          sorties d'entraînement et de validation
└── *.py                           les scripts (ci-dessous)
```

---

## Les scripts

### `geoloc.py` — moteur géodésique

Conversion pixel ↔ WGS84 sur plan tangent local à l'ellipsoïde GRS80. Aucune dépendance
hors numpy. Utilisé par `app.py`, `build_val_manuel.py`, `eval_localisation.py` et
`detect_candidats.py`.

| fonction | rôle |
|---|---|
| `pixels_to_gps(x, y, lat0, lon0, …)` | un pixel du crop → coordonnée WGS84 |
| `gps_to_pixels(lat, lon, lat0, lon0, …)` | la réciproque exacte |
| `ground_extent_from_mercator(span, lat)` | corrige une emprise demandée en Web Mercator (facteur 1/cos φ ≈ 1,55 ici) |
| `mercator_scale(lat)` | unités projetées EPSG:3857 par mètre au sol |
| `lambert93_convergence(lon)` | écart entre nord de grille Lambert et nord géographique |

**La méthode.** La Terre a deux rayons de courbure différents selon la direction :
méridien *M* vers le nord, transverse *N* vers l'est — 18 km d'écart à notre latitude.
On convertit donc les mètres en degrés avec le bon rayon :

```
Δφ = d_nord / M(φ)          Δλ = d_est / (N(φ)·cos φ)
```

Le rayon transverse est évalué à la latitude **médiane** du déplacement, ce qui annule le
terme d'erreur du premier ordre et rend l'aller-retour exact.

Validé par `test_geoloc.py` : aller-retour exact à 10⁻⁹ px, distances contrôlées contre
une projection Lambert-93 implémentée indépendamment, écart nul à **0,01 cm près sur
145 m**.

---

### `calibrate_echelle.py` — mesure de l'emprise au sol

Le script qui a trouvé les 70,1 m. Quinze paires de sites ANFR sont distantes de moins de
90 m : leurs crops se recouvrent. La distance au sol entre deux centres est connue par le
référentiel ; le décalage en pixels se mesure par **corrélation de phase** (FFT). Le
rapport des deux donne la taille du pixel.

| fonction | rôle |
|---|---|
| `enu_offset(latA, lonA, latB, lonB)` | déplacement A→B en mètres (Est, Nord) |
| `prep(path, size)` | niveaux de gris + fenêtre de Hann, prêt pour la FFT |
| `phase_correlate(a, b)` | décalage en pixels entre deux images, et netteté du pic |

Mesure X et Y **séparément** : un écart entre les deux révélerait des crops étirés,
symptôme d'une bbox demandée en degrés. Ici les deux axes concordent.

Rejet robuste des aberrantes par écart absolu médian (MAD).

---

### `prepare_dataset.py` — construction du dataset

Trois sous-commandes : `build`, `calibrate`, `estimate-azimuth`.

| fonction | rôle |
|---|---|
| `isotropic_radius_m(h, offset)` | demi-extension de la boîte, en mètres |
| `compute_box(h, offset, cfg)` | boîte YOLO normalisée (xc, yc, w, h) |
| `load_sites(csv)` | lit le référentiel, marque les sites non recalés |
| `index_images(dir)` | indexe les images par `sup_id` extrait du nom de fichier |
| `spatial_groups(df, rayon)` | union-find : regroupe les sites dont les crops se recouvrent |
| `assign_split(…)` | split 80/20 **par groupe spatial**, pas par site |
| `pick_manual_val(…)` | tire un échantillon stratifié par hauteur à annoter à la main |
| `contact_sheet(…)` | planches de contrôle visuel dans `qc/` |
| `_shadow_score_by_angle(…)` | estimation de l'azimut solaire (implémentée, non retenue) |

**Deux choix de conception.**

*La boîte est toujours centrée en (0,5 ; 0,5).* Le crop étant centré sur la coordonnée
ANFR, c'est l'estimateur non biaisé de la position du support. Déplacer la boîte au jugé
introduirait un biais ; l'agrandir n'introduit que du flou.

*Le split est spatial.* Deux sites distants de moins de 70 m produisent des crops qui se
recouvrent ; les séparer entre train et val ferait passer les mêmes pixels des deux côtés
et surévaluerait la mAP. 9 groupes concernés.

---

### `app.py` — tableau de bord Streamlit

Carte Folium des 554 sites + interface de correction manuelle.

| fonction | rôle |
|---|---|
| `click_grid(w, h, step)` | grille de points **invisibles** superposée à l'image |
| `parse_selection(event)` | extrait le pixel cliqué de l'événement Plotly |
| `geometry_for_site(lat, lon, w)` | renvoie (emprise, m/px, convergence) selon le CRS choisi |
| `load_image_b64(path)` | encode l'image en data-URI |

**Le point non évident.** Plotly ne renvoie dans un événement de sélection que les points
appartenant à une *trace*. Une image n'en contient aucune : cliquer dessus produit une
sélection vide. D'où la grille de points à 1 % d'opacité qui sert de surface cliquable.
L'image est transmise en data-URI plutôt qu'en tableau de pixels, faute de quoi le
navigateur devrait sérialiser 3,1 millions de nombres à chaque rafraîchissement.

⚠️ Dans la barre latérale, laisser **« Résolution mesurée » = 0.0684**.

---

### `build_val_manuel.py` — annotations manuelles → métriques

`app.py` produit un CSV de positions ; YOLO a besoin de fichiers `.txt`. Ce script fait la
conversion **et** calcule la mAP.

Une fois la position pointée à la main, l'incertitude `offset_m` tombe à zéro : la boîte
se réduit au support seul (7–20 m au lieu de 28 m). Bien plus serré, donc bien plus
exigeant — et c'est précisément la vérité terrain demandée par le barème.

Le script transforme les verdicts `rien_de_visible` en **exemples négatifs** et refuse
d'évaluer sur un site du train (fuite).

> La hauteur du support est récupérée depuis `dataset_manifest.csv`. Sans elle, toutes les
> boîtes de référence auraient la même taille et la mAP ne voudrait rien dire.

---

### `eval_localisation.py` — la métrique opérationnelle

Mesure **à combien de mètres** le modèle place le support par rapport au point cliqué à la
main, via la chaîne géodésique.

Pourquoi ce script existe : la mAP@50 exige 50 % de recouvrement. Le modèle a appris sur
des boîtes faibles de 28 m ; la vérité humaine fait 10 m. Même parfaitement centrée, une
boîte de 28 m recouvre une boîte de 10 m à IoU = 0,13. **La mAP@50 est donc écrasée par
construction géométrique**, sans rien dire de la capacité à trouver le site.

La question opérationnelle de l'ANFR n'est pas « le rectangle est-il de la bonne taille »
mais « où est le site ». C'est ce que mesure ce script.

---

### `detect_candidats.py` — recherche de structures non déclarées

| fonction | rôle |
|---|---|
| `enu_matrix(lat)` | facteurs mètres-par-degré au voisinage d'une latitude |

**L'erreur à ne pas commettre.** Scanner `data2/images/test/` en annonçant chercher des
sites non déclarés est une erreur de catégorie : ces 200 images sont **toutes centrées sur
un site déclaré à l'ANFR**. Une détection au centre n'est pas une anomalie, c'est le site
lui-même.

Ce que le script fait à la place : pour chaque détection, il convertit le centre de la
boîte en WGS84, cherche le site le plus proche **parmi les 554 du référentiel** (pas
seulement celui du crop), et ne retient que les détections qu'aucune déclaration connue
n'explique à moins de 25 m.

Un candidat **n'est pas** une preuve. Causes bénignes fréquentes : faux positif, support
sans émetteur (pylône électrique, éolienne, silo), décalage temporel ortho/référentiel,
site hors du CSV fourni.

---

### `test_geoloc.py` et `test_app_logic.py`

Tests de validation. `test_app_logic.py` extrait les fonctions réelles d'`app.py` par AST,
sans avoir besoin d'installer Streamlit. Les deux doivent afficher
« TOUS LES TESTS PASSENT ».

---

## Le dataset en chiffres

| Lot | Effectif | Critère | Destination |
|---|---|---|---|
| Exploitables | **354** | `height_m` renseignée et `offset_m` ≤ 20 m | 283 train / 71 val |
| Non recalés | **101** | `height_m` absente et `offset_m` = 0 | `test/`, sans label |
| Offset > 20 m | **99** | la boîte saturerait le crop | `test/`, sans label |
| **Total** | **554** | | |

Boîte médiane : **28,1 m au sol** (0,401 × 0,401 normalisé, 16 % du crop). Étendue
12,5 m → 63,1 m. **Aucune boîte saturée.**

Le seuil de 20 m n'est pas arbitraire : le demi-crop mesure 35 m, donc au-delà de 20 m
d'offset la boîte couvre l'essentiel du champ et n'enseigne plus rien au réseau.

Les 200 sites écartés ne sont pas perdus : lot d'inspection et réservoir de candidats,
motif tracé dans `dataset_manifest.csv`.

---

## Le modèle

YOLOv8n, 40 epochs, `imgsz=416`, `batch=8`, CPU, **30 minutes**.

> **Pourquoi 416 et pas 1024.** La BD ORTHO est native à 20 cm/px. À 6,84 cm/px les crops
> sont suréchantillonnés ×2,9 — l'information optique réelle tient dans ~350×350 px.
> Entraîner à 1024 ferait travailler le réseau sur des pixels interpolés : coût triplé,
> information nulle ajoutée. Ce choix découle directement de la calibration.

| Métrique (validation faible) | Valeur |
|---|---|
| mAP@50 | **0,482** |
| mAP@50-95 | 0,276 |
| Précision | 0,387 |
| Rappel | 0,690 |
| Matrice | 48 VP · 75 FP · 23 FN |
| F1 max | 0,49 à conf. 0,247 |
| Précision = 1,00 | à conf. 0,580 |

**Ce que ce chiffre dit — et ne dit pas.** 0,482 mesure la fidélité du modèle à notre
propre formule géométrique, pas sa capacité à détecter des antennes. La vérité terrain de
ce calcul est elle-même une annotation faible. C'est exactement pourquoi le jeu annoté
manuellement existe.

---

## Limites assumées

| Limite | Conséquence | Correction |
|---|---|---|
| **Aucun exemple négatif** | Les 354 images d'entraînement contiennent toutes exactement un site. Le modèle n'a jamais vu d'image vide, donc sa confiance est mal calibrée : 75 faux positifs en validation, et zéro détection hors distribution. | Échantillonner ~100 crops sans site déclaré et les inclure sans label |
| **Vérité terrain principalement faible** | La mAP de 0,482 mesure une formule, pas une détection | Étendre le jeu manuel à 200–300 sites |
| **Jeu manuel petit et biaisé** | Les sites annotés sont tous en verdict `ok` : les cas où rien n'était visible ont été passés. Le rappel est donc optimiste et la précision non mesurable sur scène vide. | Annoter aussi les cas négatifs et ambigus |
| **Désaccord d'échelle des boîtes** | Boîtes apprises à 28 m contre vérité humaine à 10 m → la mAP@50 est écrasée mécaniquement | Ré-entraîner sur les boîtes serrées, ou évaluer à IoU plus permissif |
| **Azimut solaire non estimé** | Les ombres, souvent plus visibles que le support, ne sont pas exploitées | Métadonnées de prise de vue, ou estimateur appris |
| **Classes de support inexploitées** | Mono-classe : impossible de distinguer un pylône radio d'un pylône électrique | Le référentiel contient la nature du support : détection multi-classes |
| **Boîtes non orientées** | Un mât et son ombre forment une structure oblique mal décrite par un rectangle droit | YOLO-OBB |

---

## Pour reprendre le projet

Par ordre de rapport effort/gain :

1. **Les exemples négatifs.** Correction unique qui traite à la fois les faux positifs et
   l'absence de détection hors distribution. Une centaine de crops sans site suffirait
   probablement à recalibrer les confiances.
2. **Ré-entraîner sur des boîtes serrées** issues du jeu manuel élargi, pour aligner
   l'échelle de prédiction sur celle de la vérité terrain.
3. **Porter le jeu manuel à 200–300 sites**, pour une métrique de détection avec un
   intervalle de confiance utilisable.
4. **Passer en multi-classes** sur la nature du support.
5. **Itérer sur la supervision** : utiliser le modèle pour resserrer les boîtes faibles,
   puis réentraîner — auto-distillation classique en supervision bruitée.

Le matériel est le facteur limitant évident : tout l'entraînement a été fait sur CPU en
30 minutes. Sur GPU, les points 1 à 3 tiennent dans une journée.

---

## Pièges connus

| Symptôme | Cause | Remède |
|---|---|---|
| Résultats incohérents | travail sur `data/` au lieu de `data2/` | `data/` est l'ancien dataset erroné |
| `app.py` affiche n'importe quoi | mauvais mode d'échelle en barre latérale | « Résolution mesurée » = 0.0684 |
| YOLO ne trouve pas les images | `data2/antennes.yaml` contient un chemin absolu | corriger la ligne `path:` |
| Le clic sur l'image ne réagit pas | variante de version Plotly | cocher « sélection par rectangle » |
| Un entraînement écrase les poids | `exist_ok=True` dans `starter_antennes.py` | `best_70m_40ep.pt` est la sauvegarde |
| Avertissements NNPACK au démarrage | CPU non supporté, repli sur le calcul standard | sans effet, ne pas interrompre |

---

## Licence et crédits

Projet réalisé dans le cadre du **FR Hack! 2026**, challenge 4 proposé par l'**ANFR** en
partenariat avec l'**ISEP**. Équipe de 5 — développement et coordination : Omar.

`starter_antennes.py` est fourni par l'ANFR (non modifié, hors ajout de
`detect_anomalies`, désormais remplacé par `detect_candidats.py`).

Données : référentiel ANFR et imagerie aérienne fournis par les organisateurs.
