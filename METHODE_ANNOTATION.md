# Methode de generation des annotations faibles

FR Hack! 2026 — Challenge 4 (ANFR x ISEP). Genere par `prepare_dataset.py`.

## 1. Le probleme

Une coordonnee ANFR **n'est pas une verite terrain visuelle**. Elle indique qu'un site
radioelectrique se trouve a proximite, sans dire quels pixels correspondent au support.
Transformer cette coordonnee en boite englobante est donc une modelisation d'incertitude,
pas une simple conversion de reperes.

## 2. Tri prealable des sites

| Lot | Effectif | Critere | Usage |
|---|---|---|---|
| Exploitables | 453 | `height_m` renseignee **et** `offset_m` > 0 | train + val |
| Douteux | 101 | `height_m` absente **et** `offset_m` == 0 | `test/`, sans label |

Les 101 sites ecartes n'ont pas de recalage : leur `offset_m` vaut 0 par defaut
(valeur non mesuree, pas une precision parfaite) et leurs coordonnees sont souvent arrondies
au degre-minute-seconde. Les entrainer reviendrait a apprendre du bruit. Ils sont conserves
comme lot d'inspection qualitative et comme reservoir de candidats pour le Niveau 3.

## 3. Position de la boite

Chaque crop est centre sur la coordonnee ANFR. La boite est donc centree en
**(0.5, 0.5)** : c'est l'estimateur non biaise de la position du support, puisque
l'erreur de recalage n'a pas de direction privilegiee connue.

**L'incertitude est portee par la taille de la boite, pas par sa position.**

## 4. Taille de la boite

Demi-extension isotrope, en metres :

```
r_iso = offset_m                          # incertitude de recalage fournie par l'ANFR
      + 2.0                             # residu de georeferencement de l'ortho
      + clip(0.1 x height_m, 1.5, 10.0)   # empreinte au sol du support
      + 0.1 x height_m                    # deplacement radial du sommet dans l'ortho
      + 2.0                             # marge
```

Le terme de **deplacement radial** vient du fait qu'une orthophoto est redressee sur le
modele de terrain : un objet vertical de hauteur h vu a distance du nadir est deplace
d'environ h x (d / H). Un pylone de 40 m peut ainsi apparaitre a une dizaine de metres
de sa position cadastrale, dans une direction inconnue.

### Ombre portee

L = height_m / tan(55.0 deg) = **0.70 x height_m**, dont on retient
60% (`--shadow-frac`).

Mode actif : **directional**.

- `symmetric` — les deux axes sont elargis de `shadow_frac x L`. Sans hypothese sur
  l'azimut solaire, mais la boite contient beaucoup de fond.
- `directional` — le centre est decale d'une demi-longueur d'ombre le long de l'azimut
  (340.0 deg) et seul cet axe est allonge. Meme couverture de l'ombre pour environ
  moitie moins de surface de fond. A n'utiliser que si l'azimut a ete verifie sur les
  images (sur l'echantillon inspecte, les ombres pointent vers le NNO).

L'ombre est retenue parce qu'en vue verticale, **le support lui-meme est souvent moins
lisible que son ombre** : un pylone treillis se resume a quelques pixels au nadir, alors
que son ombre dessine une structure allongee nette sur le sol.

### Resultat

Cote de boite au sol : p05 = 17.2 m, mediane = 38.2 m, p95 = 82.3 m.
La taille varie donc d'un facteur ~4.8 entre un mat de toiture et un grand pylone,
au lieu d'une boite fixe identique pour tous.

## 5. Classes

`--classes single` -> ['site_radio']

En mode `support`, les classes sont derivees de `height_m` par seuils
(8 / 20 / 45 m). **C'est un proxy assume** : la hauteur ne distingue
pas un chateau d'eau d'un pylone treillis. Ces classes servent de pre-annotation a
corriger manuellement, pas de verite terrain.

## 6. Split train / val

Mode **group**, 20% en validation.

En mode `group`, les sites distants de moins de 205 m sont places dans
le meme split. Sans cette precaution, deux sites voisins produisent des crops qui se
recouvrent : les memes pixels se retrouveraient en entrainement et en validation, et la
mAP@50 serait surevaluee. Dans ce jeu, 29 groupes contiennent plusieurs sites.

## 7. Limite a garder en tete

Le jeu de validation produit ici est annote **faiblement**, comme le jeu d'entrainement.
Une mAP@50 mesuree dessus evalue la capacite du modele a reproduire l'heuristique, pas a
detecter reellement les supports. Le fichier `val_manual_todo.csv` liste
60 sites stratifies a annoter a la main : c'est ce jeu-la qui doit servir de reference
pour les 25 points de performance du bareme.
