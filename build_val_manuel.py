#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_val_manuel.py — FR Hack! 2026, Challenge 4

Transforme les corrections humaines de `coordonnees_corrigees.csv` (produites par
app.py) en un VRAI jeu de validation YOLO, puis calcule la mAP dessus.

Pourquoi ce jeu est different du val automatique
------------------------------------------------
Le val genere par prepare_dataset.py porte les memes annotations FAIBLES que le
train : sa mAP mesure la capacite du modele a reproduire une formule, pas a
detecter des antennes. Le bareme demande explicitement une mAP@50 sur un jeu
"annote manuellement".

Une fois la position pointee a la main, le terme `offset_m` de l'incertitude
tombe a zero : on sait ou est le support. La boite se reduit alors a l'objet
lui-meme :

        demi-boite = empreinte_support + deplacement_relief + marge
                   = clip(0.10 h, 1.5, 10) + 0.10 h + 1.5      (metres)

C'est beaucoup plus serre que l'annotation faible, donc beaucoup plus exigeant :
une mAP@50 mesuree la-dessus est une vraie mesure de detection.

Verdicts pris en compte
-----------------------
    ok / boite_a_corriger  -> boite autour du point clique
    rien_de_visible        -> image conservee SANS objet : exemple negatif, que
                              le jeu d'entrainement ne contient pas du tout
    mauvais_support        -> ecarte (le site existe mais l'annotation est douteuse)

Usage
-----
    python build_val_manuel.py --data-root data2
    python build_val_manuel.py --data-root data2 --eval --weights runs/.../best.pt
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

K_FOOTPRINT, FOOT_MIN, FOOT_MAX = 0.10, 1.5, 10.0
K_RELIEF, MARGE = 0.10, 1.5
MIN_BOX_M, MAX_BOX_FRAC = 4.0, 0.95


def demi_boite_m(height_m: float) -> float:
    """Demi-extension du support seul, sans l'incertitude de position."""
    if not np.isfinite(height_m):
        height_m = 15.0                      # hauteur mediane, faute de mieux
    empreinte = float(np.clip(K_FOOTPRINT * height_m, FOOT_MIN, FOOT_MAX))
    return empreinte + K_RELIEF * height_m + MARGE


def main():
    p = argparse.ArgumentParser(
        description="Construit le jeu de validation annote a la main + evalue")
    p.add_argument("--corrections", default="coordonnees_corrigees.csv")
    p.add_argument("--data-root", default="data2")
    p.add_argument("--name", default="val_manuel")
    p.add_argument("--link", action="store_true",
                   help="liens symboliques au lieu de copies")
    p.add_argument("--eval", action="store_true", help="lance la validation YOLO")
    p.add_argument("--weights", default=None)
    p.add_argument("--imgsz", type=int, default=416)
    a = p.parse_args()

    root = Path(a.data_root)
    corr_path = Path(a.corrections)
    if not corr_path.exists():
        sys.exit(f"[ERREUR] {corr_path} introuvable. Annote d'abord avec app.py.")

    c = pd.read_csv(corr_path)
    print("=" * 70)
    print(f"  Jeu de validation annote a la main — {len(c)} correction(s)")
    print("=" * 70)

    if "verdict" not in c.columns:
        c["verdict"] = "ok"
    for v, n in c["verdict"].fillna("ok").value_counts().items():
        print(f"    {v:20s} {n:4d}")

    # --- garde-fou : ne jamais evaluer sur des images vues a l'entrainement --- #
    man_path = root / "dataset_manifest.csv"
    if man_path.exists():
        man = pd.read_csv(man_path)
        splits = dict(zip(man["id"].astype(str), man["split"]))
        c["split_origine"] = c["id"].astype(str).map(splits)

        # --- hauteur du support : recuperee depuis le manifeste --------------
        # coordonnees_corrigees.csv ne contient PAS height_m (app.py ne l'ecrit
        # pas). Sans elle, demi_boite_m() retombe sur sa valeur par defaut de
        # 15 m et TOUTES les boites de verite terrain font exactement 9,0 m,
        # que le support mesure 5 m ou 42 m. La mAP mesuree ainsi ne veut rien
        # dire : on la calculerait contre une boite arbitraire.
        haut = dict(zip(man["id"].astype(str),
                        pd.to_numeric(man["height_m"], errors="coerce")))
        c["height_m"] = c["id"].astype(str).map(haut)
        n_sans_h = int(c["height_m"].isna().sum())
        if n_sans_h:
            print(f"\n  [!] {n_sans_h} site(s) sans height_m dans le manifeste : "
                  f"boite calculee avec la hauteur mediane (15 m).")

        fuite = c["split_origine"] == "train"
        if fuite.any():
            print(f"\n  [!] {int(fuite.sum())} site(s) annote(s) appartiennent au TRAIN.")
            print(f"      Evaluer dessus serait de la fuite : ils sont ecartes.")
            c = c[~fuite].copy()

    img_dir = root / "images" / a.name
    lbl_dir = root / "labels" / a.name
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()

    n_obj = n_neg = n_skip = 0
    tailles = []
    for r in c.itertuples():
        verdict = str(getattr(r, "verdict", "ok") or "ok")
        if verdict == "mauvais_support":
            n_skip += 1
            continue

        src = Path(str(r.image_path))
        if not src.exists():
            print(f"  [WARN] image absente : {src}")
            n_skip += 1
            continue

        dst = img_dir / src.name
        if a.link:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

        lbl = lbl_dir / f"{src.stem}.txt"
        if verdict == "rien_de_visible":
            lbl.write_text("", encoding="utf-8")      # image de fond, sans objet
            n_neg += 1
            continue

        W, H = float(r.image_width), float(r.image_height)
        gsd = float(r.pixel_size_m)

        box_px = str(getattr(r, "box_px", "") or "")
        if box_px.count(",") == 3:                    # rectangle trace a la main
            x0, y0, x1, y1 = (float(v) for v in box_px.split(","))
            xc, yc = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            w, h = abs(x1 - x0) / W, abs(y1 - y0) / H
        else:                                         # point clique -> boite calculee
            h_site = getattr(r, "height_m", np.nan)
            demi = demi_boite_m(float(h_site) if pd.notna(h_site) else np.nan)
            cote_px = max(2 * demi / gsd, MIN_BOX_M / gsd)
            xc, yc = float(r.pixel_x) / W, float(r.pixel_y) / H
            w = h = cote_px / W

        w, h = min(w, MAX_BOX_FRAC), min(h, MAX_BOX_FRAC)
        xc = float(np.clip(xc, w / 2, 1 - w / 2))
        yc = float(np.clip(yc, h / 2, 1 - h / 2))
        lbl.write_text(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
        tailles.append(w * W * gsd)
        n_obj += 1

    print(f"\n  {n_obj} image(s) avec objet | {n_neg} negative(s) | {n_skip} ecartee(s)")
    if tailles:
        t = np.array(tailles)
        print(f"  cote de boite : min {t.min():.1f} m | mediane {np.median(t):.1f} m "
              f"| max {t.max():.1f} m")
        print(f"  (rappel : annotation faible mediane = 28.1 m)")
    if n_neg:
        print(f"  {n_neg} exemple(s) negatif(s) : le jeu d'entrainement n'en contient AUCUN.")

    yaml_path = root / f"{a.name}.yaml"
    yaml_path.write_text("\n".join([
        f"# Jeu de validation annote manuellement — genere par build_val_manuel.py",
        f"path: {root.resolve()}",
        f"train: images/{a.name}",
        f"val: images/{a.name}",
        "", "nc: 1", "names: ['site_radio']", "",
    ]), encoding="utf-8")
    print(f"\n  ecrit : {img_dir}/  {lbl_dir}/  {yaml_path}")

    if n_obj + n_neg == 0:
        sys.exit("\n[ERREUR] aucune annotation exploitable.")

    if a.eval:
        if not a.weights:
            sys.exit("\n[ERREUR] --eval requiert --weights")
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("\n[ERREUR] ultralytics requis pour --eval")
        print("\n" + "=" * 70)
        print("  EVALUATION SUR ANNOTATIONS HUMAINES")
        print("=" * 70)
        m = YOLO(a.weights).val(data=str(yaml_path), imgsz=a.imgsz, verbose=False)
        print(f"\n  mAP@50     : {m.box.map50:.3f}")
        print(f"  mAP@50-95  : {m.box.map:.3f}")
        print(f"  Precision  : {m.box.mp:.3f}")
        print(f"  Rappel     : {m.box.mr:.3f}")
        print(f"\n  >>> C'est CE chiffre qui va dans le rapport, pas celui du val"
              f" automatique.\n")
    else:
        w = a.weights or "runs/detect/runs/detect/antennes/weights/best.pt"
        print(f"\n  Evaluer :\n    python build_val_manuel.py --data-root {a.data_root} "
              f"--eval --weights {w}\n")


if __name__ == "__main__":
    main()
