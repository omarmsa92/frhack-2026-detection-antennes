#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_localisation.py — FR Hack! 2026, Challenge 4

Mesure ce que la mAP ne mesure pas : A QUELLE DISTANCE, EN METRES, le modele
place-t-il le support par rapport a la position pointee a la main ?

Pourquoi cette metrique
-----------------------
La mAP@50 exige un recouvrement de 50 % entre la boite predite et la boite de
reference. Or le modele a ete entraine sur des annotations FAIBLES dont la
boite mediane fait 28 m de cote, tandis que la verite terrain humaine fait 7 a
20 m. Meme parfaitement centree, une boite de 28 m recouvre une boite de 10 m
a IoU = 100/784 = 0.13 : la mAP@50 est donc quasi nulle PAR CONSTRUCTION, sans
que cela dise quoi que ce soit sur la capacite du modele a trouver le support.

La question operationnelle de l'ANFR n'est pas "le rectangle est-il de la bonne
taille" mais "ou est le site". C'est exactement ce que mesure ce script :
l'erreur de localisation en metres, via la chaine geodesique de geoloc.py.

Usage
-----
    python eval_localisation.py --weights best_70m_40ep.pt
    python eval_localisation.py --weights best_70m_40ep.pt --conf 0.05
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from geoloc import pixels_to_gps

_A, _F = 6378137.0, 1.0 / 298.257222101
_E2 = 2 * _F - _F * _F


def metres_par_degre(lat: float):
    la = math.radians(lat)
    w = math.sqrt(1 - _E2 * math.sin(la) ** 2)
    return (math.radians(1) * _A * (1 - _E2) / w ** 3,      # latitude
            math.radians(1) * (_A / w) * math.cos(la))       # longitude


def main():
    p = argparse.ArgumentParser(
        description="Erreur de localisation en metres sur le jeu annote a la main")
    p.add_argument("--weights", required=True)
    p.add_argument("--corrections", default="coordonnees_corrigees.csv")
    p.add_argument("--data-root", default="data2")
    p.add_argument("--conf", type=float, default=0.10,
                   help="seuil bas volontairement : on veut la meilleure boite, "
                        "pas une decision de detection")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--out", default="localisation_manuel.csv")
    a = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERREUR] ultralytics requis")

    c = pd.read_csv(a.corrections)

    # --- garde-fou : jamais de site vu a l'entrainement --------------------- #
    man_path = Path(a.data_root) / "dataset_manifest.csv"
    if man_path.exists():
        man = pd.read_csv(man_path)
        splits = dict(zip(man["id"].astype(str), man["split"]))
        c["split_origine"] = c["id"].astype(str).map(splits)
        n_fuite = int((c["split_origine"] == "train").sum())
        if n_fuite:
            print(f"  [!] {n_fuite} site(s) du TRAIN ecarte(s) (fuite)")
            c = c[c["split_origine"] != "train"].copy()

    c = c[c["verdict"].fillna("ok").isin(["ok", "boite_a_corriger"])].copy()
    if c.empty:
        sys.exit("[ERREUR] aucun site exploitable (verdict ok / boite_a_corriger)")

    print("=" * 74)
    print(f"  ERREUR DE LOCALISATION — {len(c)} site(s) annote(s) a la main")
    print(f"  seuil de confiance : {a.conf}  (boite la plus sure retenue par image)")
    print("=" * 74)

    model = YOLO(a.weights)
    lignes = []

    for r in c.itertuples():
        img = Path(str(r.image_path))
        if not img.exists():
            print(f"  [WARN] image absente : {img}")
            continue

        res = model.predict(source=str(img), conf=a.conf, imgsz=a.imgsz,
                            save=False, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            lignes.append(dict(id=r.id, sup_id=r.sup_id, detecte=False,
                               confiance=np.nan, erreur_m=np.nan,
                               cote_predit_m=np.nan, n_boites=0))
            continue

        confs = res.boxes.conf.cpu().numpy()
        k = int(np.argmax(confs))
        x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[k].tolist())
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        W = float(r.image_width)
        extent = float(r.extent_m)
        gsd = float(r.pixel_size_m)

        lat_p, lon_p = pixels_to_gps(cx, cy, float(r.lat_anfr), float(r.lon_anfr),
                                     image_size_px=W, extent_m=extent)

        mlat, mlon = metres_par_degre(float(r.lat_anfr))
        err = math.hypot((lat_p - float(r.lat_corr)) * mlat,
                         (lon_p - float(r.lon_corr)) * mlon)

        lignes.append(dict(
            id=r.id, sup_id=r.sup_id, detecte=True,
            confiance=round(float(confs[k]), 3),
            erreur_m=round(err, 1),
            cote_predit_m=round((x2 - x1) * gsd, 1),
            n_boites=int(len(confs)),
            lat_pred=round(lat_p, 7), lon_pred=round(lon_p, 7),
            lat_humain=r.lat_corr, lon_humain=r.lon_corr,
            image=img.name,
        ))

    d = pd.DataFrame(lignes)
    d.to_csv(a.out, index=False)

    det = d[d.detecte]
    n, nd = len(d), len(det)
    print(f"\n  {nd}/{n} site(s) avec au moins une detection "
          f"({nd / max(n, 1):.0%})")

    if nd == 0:
        print("\n  Aucune detection : rien a mesurer.")
        return

    e = det.erreur_m.to_numpy(dtype=float)
    print(f"\n  ERREUR DE LOCALISATION (distance au point pointe a la main)")
    print(f"    mediane   : {np.median(e):6.1f} m")
    print(f"    moyenne   : {e.mean():6.1f} m")
    print(f"    p90       : {np.percentile(e, 90):6.1f} m")
    print(f"    max       : {e.max():6.1f} m")

    print(f"\n  TAUX DE LOCALISATION CORRECTE (sur les {n} sites annotes)")
    for seuil in (5, 10, 15, 25):
        k = int((e <= seuil).sum())
        print(f"    a moins de {seuil:2d} m : {k:3d}/{n}  ({k / n:5.1%})")

    print(f"\n  TAILLE DES BOITES (ce qui explique la mAP@50)")
    print(f"    cote predit par le modele : mediane {det.cote_predit_m.median():.1f} m")
    print(f"    cote de la verite humaine : voir build_val_manuel.py")
    print(f"    -> une boite trop large reste PENALISEE par la mAP meme bien centree")

    print(f"\n  Detail par site : {a.out}\n")


if __name__ == "__main__":
    main()
