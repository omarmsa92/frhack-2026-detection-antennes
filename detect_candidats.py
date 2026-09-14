#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_candidats.py — FR Hack! 2026, Challenge 4 (Niveau 3)

Recherche de structures détectées SANS correspondance dans le référentiel ANFR.

Pourquoi l'approche naïve ne marche pas
---------------------------------------
Scanner `data2/images/test/` en annonçant chercher des "sites non déclarés" est
une erreur de catégorie : ces 200 images sont TOUTES centrées sur un site
déclaré à l'ANFR. Par construction, une détection au centre n'est pas une
anomalie, c'est le site lui-même.

Ce que ce script fait à la place
--------------------------------
Il exploite le fait qu'un crop de 70 m couvre du sol AUTOUR du site déclaré.
Pour chaque détection :

    1. le centre de la boîte est converti en coordonnée WGS84 (geoloc.py) ;
    2. on cherche le site ANFR le PLUS PROCHE parmi les 554 du référentiel
       (pas seulement celui du crop : un voisin déclaré explique aussi bien) ;
    3. si cette distance dépasse `--seuil-m`, la structure détectée n'est
       expliquée par AUCUNE déclaration connue -> candidat à examiner.

Un candidat n'est PAS une preuve de site non déclaré. Les causes bénignes sont
nombreuses : faux positif du modèle, support sans émetteur (pylône électrique,
éolienne, silo), décalage temporel entre l'ortho et le référentiel, site déclaré
hors du périmètre du CSV fourni. La sortie est une liste À EXAMINER, classée.

Usage
-----
    python detect_candidats.py --weights best_70m_40ep.pt \
        --csv "données antennes hackathon.csv" --images data2/images/test \
        --conf 0.58 --seuil-m 25

Le seuil de confiance 0,58 n'est pas arbitraire : c'est le point où la courbe
précision-confiance du modèle atteint 1,00 sur le jeu de validation.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from geoloc import pixels_to_gps

_A, _F = 6378137.0, 1.0 / 298.257222101
_E2 = 2 * _F - _F * _F


def enu_matrix(lat_ref: float):
    """Facteurs mètres-par-degré au voisinage de lat_ref (plan tangent local)."""
    la = math.radians(lat_ref)
    w = math.sqrt(1 - _E2 * math.sin(la) ** 2)
    m_per_deg_lat = math.radians(1) * _A * (1 - _E2) / w ** 3
    m_per_deg_lon = math.radians(1) * (_A / w) * math.cos(la)
    return m_per_deg_lat, m_per_deg_lon


def main():
    p = argparse.ArgumentParser(description="Candidats sans correspondance ANFR")
    p.add_argument("--weights", required=True)
    p.add_argument("--csv", required=True, help="référentiel ANFR complet (554 sites)")
    p.add_argument("--images", default="data2/images/test")
    p.add_argument("--conf", type=float, default=0.58,
                   help="seuil de confiance (0.58 = précision 1.00 sur le val)")
    p.add_argument("--seuil-m", type=float, default=25.0,
                   help="distance au site ANFR le plus proche au-delà de laquelle "
                        "une détection n'est expliquée par aucune déclaration")
    p.add_argument("--extent-m", type=float, default=70.1)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--top", type=int, default=3, help="candidats retenus au final")
    p.add_argument("--out", default="candidats_a_examiner.csv")
    a = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERREUR] ultralytics requis")

    ref = pd.read_csv(a.csv)
    ref = ref.dropna(subset=["lat", "lon"])
    lat_ref = float(ref.lat.mean())
    MLAT, MLON = enu_matrix(lat_ref)
    ref_y = ref.lat.to_numpy() * MLAT          # mètres, repère local plan
    ref_x = ref.lon.to_numpy() * MLON
    ref_sup = ref.sup_id.to_numpy()

    img_dir = Path(a.images)
    images = sorted([q for q in img_dir.iterdir()
                     if q.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        sys.exit(f"[ERREUR] aucune image dans {img_dir}")

    # coordonnée ANFR du centre de chaque crop, par sup_id extrait du nom
    centre = {int(r.sup_id): (float(r.lat), float(r.lon)) for r in ref.itertuples()}

    print("=" * 74)
    print(f"  Recherche de candidats — {len(images)} images, conf >= {a.conf}")
    print(f"  Référentiel : {len(ref)} sites déclarés")
    print(f"  Seuil de non-correspondance : {a.seuil_m:.0f} m")
    print("=" * 74)

    model = YOLO(a.weights)
    rows, n_det, n_sans_img = [], 0, 0

    for q in images:
        nums = re.findall(r"\d+", q.stem)
        if not nums:
            continue
        sup = int(max(nums, key=len))
        if sup not in centre:
            n_sans_img += 1
            continue
        lat0, lon0 = centre[sup]

        res = model.predict(source=str(q), conf=a.conf, imgsz=a.imgsz,
                            save=False, verbose=False)
        for r in res:
            if r.boxes is None:
                continue
            W = r.orig_shape[1]
            for b in r.boxes:
                conf = float(b.conf.item())
                x1, y1, x2, y2 = (float(v) for v in b.xyxy.squeeze().tolist())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                lat, lon = pixels_to_gps(cx, cy, lat0, lon0,
                                         image_size_px=W, extent_m=a.extent_m)
                n_det += 1

                d = np.hypot((lon * MLON) - ref_x, (lat * MLAT) - ref_y)
                k = int(np.argmin(d))
                d_min = float(d[k])
                d_centre = math.hypot((cx - W / 2), (cy - W / 2)) * (a.extent_m / W)

                rows.append(dict(
                    image=q.name, sup_id_crop=sup, confiance=round(conf, 3),
                    lat=round(lat, 7), lon=round(lon, 7),
                    dist_site_du_crop_m=round(d_centre, 1),
                    anfr_plus_proche=int(ref_sup[k]),
                    dist_anfr_plus_proche_m=round(d_min, 1),
                    sans_correspondance=d_min > a.seuil_m,
                    box_px=f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}",
                    geoportail=f"https://www.geoportail.gouv.fr/carte?c={lon:.7f},{lat:.7f}&z=19",
                ))

    if not rows:
        print(f"\n  {len(images)} image(s) parcourue(s).")
        if n_sans_img:
            print(f"  [!] {n_sans_img} image(s) sans sup_id correspondant dans le CSV "
                  f"-> NON analysées.")
        print(f"\n  Aucune détection au-dessus de {a.conf}.")
        print(f"  C'est un résultat recevable : le brief prévoit explicitement de")
        print(f"  conclure qu'aucun candidat n'est suffisamment fiable.")
        pd.DataFrame(columns=["image", "confiance", "lat", "lon"]).to_csv(a.out, index=False)
        return

    df = pd.DataFrame(rows).sort_values(
        ["sans_correspondance", "dist_anfr_plus_proche_m", "confiance"],
        ascending=[False, False, False])
    df.to_csv(a.out, index=False)

    cand = df[df.sans_correspondance]
    print(f"\n  {n_det} détection(s) au-dessus du seuil, sur {len(images)} images")
    if n_sans_img:
        print(f"  [!] {n_sans_img} image(s) sans sup_id correspondant dans le CSV")
    print(f"  {len(df) - len(cand)} expliquée(s) par un site déclaré à moins de "
          f"{a.seuil_m:.0f} m")
    print(f"  {len(cand)} SANS correspondance -> candidat(s)")

    if len(cand):
        print(f"\n  Top {min(a.top, len(cand))} à examiner :\n")
        for i, t in enumerate(cand.head(a.top).itertuples(), 1):
            print(f"   {i}. {t.image}  conf {t.confiance:.2f}")
            print(f"      position       : {t.lat:.6f}, {t.lon:.6f}")
            print(f"      ANFR le + proche: {t.anfr_plus_proche} à "
                  f"{t.dist_anfr_plus_proche_m:.0f} m")
            print(f"      {t.geoportail}\n")
        print(f"  >>> VÉRIFIE CES {min(a.top, len(cand))} SUR GÉOPORTAIL AVANT DE LES CITER.")
        print(f"      Causes bénignes fréquentes : pylône électrique, éolienne, silo,")
        print(f"      château d'eau sans émetteur, site déclaré hors du CSV fourni.")
    else:
        print(f"\n  Aucune détection sans correspondance : toutes s'expliquent par un")
        print(f"  site déclaré. Conclusion recevable et à documenter comme telle.")

    print(f"\n  Détail complet : {a.out}\n")


if __name__ == "__main__":
    main()
