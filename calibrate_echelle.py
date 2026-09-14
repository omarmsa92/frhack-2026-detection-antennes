#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_echelle.py — FR Hack! 2026, Challenge 4

MESURE l'emprise au sol reelle des crops, sans oeil humain et sans metadonnees.

Principe
--------
Certains sites ANFR sont distants de quelques dizaines de metres seulement :
leurs crops se recouvrent et montrent le MEME sol. On connait exactement la
distance au sol entre les deux centres (depuis les coordonnees du CSV). Il
suffit de mesurer le decalage EN PIXELS entre les deux images, par correlation
de phase, pour obtenir :

        taille du pixel (m/px) = distance au sol (m) / decalage (px)

C'est une mesure directe, indiscutable, et qui donne l'echelle en X et en Y
SEPAREMENT : si les crops ont ete demandes avec une bbox en degres, l'image est
etiree et gsd_x != gsd_y. Ce test le detecte.

Usage
-----
    python calibrate_echelle.py --csv "donnees antennes hackathon.csv" --images ./crops

Dependances : numpy, pandas, Pillow.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_A, _F = 6378137.0, 1.0 / 298.257222101
_E2 = 2 * _F - _F * _F
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def enu_offset(lat_a, lon_a, lat_b, lon_b):
    """Deplacement A -> B en metres dans le repere local (Est, Nord)."""
    la = math.radians(lat_a)
    w = math.sqrt(1 - _E2 * math.sin(la) ** 2)
    M = _A * (1 - _E2) / w ** 3
    N = _A / w
    d_north = math.radians(lat_b - lat_a) * M
    d_east = math.radians(lon_b - lon_a) * N * math.cos(la)
    return d_east, d_north


def index_images(d: Path):
    idx = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            nums = re.findall(r"\d+", p.stem)
            if nums:
                idx.setdefault(int(max(nums, key=len)), p)
    return idx


def prep(path: Path, size: int):
    """Niveaux de gris, passe-haut leger, fenetre de Hann."""
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size))
    a = np.asarray(im, dtype=np.float64)
    a = a - a.mean()
    win = np.hanning(size)
    return a * np.outer(win, win)


def phase_correlate(a, b):
    """
    Decalage (sx, sy) tel qu'un point au pixel (x, y) de B se retrouve
    en (x + sx, y + sy) dans A. Renvoie aussi la nettete du pic.
    (Convention verifiee par test sur crops synthetiques a echelle connue.)
    """
    F, G = np.fft.fft2(a), np.fft.fft2(b)
    R = F * np.conj(G)
    R /= np.abs(R) + 1e-12
    r = np.fft.ifft2(R).real
    h, w = r.shape
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    peak = r[iy, ix]
    sy = iy - h if iy > h // 2 else iy
    sx = ix - w if ix > w // 2 else ix
    sharp = (peak - r.mean()) / (r.std() + 1e-12)
    return float(sx), float(sy), float(sharp)


def main():
    p = argparse.ArgumentParser(description="Mesure l'emprise au sol reelle des crops")
    p.add_argument("--csv", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--size", type=int, default=1024, help="cote des crops en px")
    p.add_argument("--dmin", type=float, default=8.0, help="distance min entre sites (m)")
    p.add_argument("--dmax", type=float, default=90.0, help="distance max entre sites (m)")
    p.add_argument("--min-shift", type=float, default=12.0,
                   help="decalage minimal en px pour que la mesure soit fiable")
    p.add_argument("--min-sharp", type=float, default=6.0,
                   help="nettete minimale du pic de correlation")
    p.add_argument("--max-pairs", type=int, default=60)
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    idx = index_images(Path(a.images))
    print(f"[INFO] {len(df)} sites, {len(idx)} images indexees\n")

    # --- paires de sites assez proches pour que les crops se recouvrent ------
    if {"x_recal", "y_recal"}.issubset(df.columns):
        xy = df[["x_recal", "y_recal"]].to_numpy(float)
    else:
        sys.exit("[ERREUR] colonnes x_recal / y_recal absentes")

    pairs = []
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1])
            if a.dmin <= d <= a.dmax:
                pairs.append((i, j, d))
    pairs.sort(key=lambda t: t[2])
    print(f"[INFO] {len(pairs)} paires de sites distantes de {a.dmin:.0f} a {a.dmax:.0f} m")
    if not pairs:
        sys.exit("[ERREUR] aucune paire exploitable : elargis --dmax")

    rows = []
    for i, j, d in pairs[: a.max_pairs]:
        ka, kb = int(df.sup_id[i]), int(df.sup_id[j])
        if ka not in idx or kb not in idx:
            continue
        try:
            A, B = prep(idx[ka], a.size), prep(idx[kb], a.size)
        except Exception as e:
            print(f"[WARN] {ka}/{kb} illisible : {e}")
            continue

        sx, sy, sharp = phase_correlate(A, B)
        dE, dN = enu_offset(df.lat[i], df.lon[i], df.lat[j], df.lon[j])

        # phase_correlate renvoie le decalage B -> A, d'ou :
        #     sx = (E_B - E_A)/gsd_x       sy = -(N_B - N_A)/gsd_y
        gsd_x = dE / sx if abs(sx) >= a.min_shift else np.nan
        gsd_y = -dN / sy if abs(sy) >= a.min_shift else np.nan
        ok = sharp >= a.min_sharp and (not np.isnan(gsd_x) or not np.isnan(gsd_y))
        rows.append(dict(sup_a=ka, sup_b=kb, dist_m=d, dE=dE, dN=dN,
                         sx=sx, sy=sy, sharp=sharp,
                         gsd_x=gsd_x, gsd_y=gsd_y, retenu=ok))

    r = pd.DataFrame(rows)
    if r.empty:
        sys.exit("[ERREUR] aucune paire mesurable (images manquantes ?)")

    keep = r[r.retenu].copy()
    print(f"[INFO] {len(r)} paires testees, {len(keep)} retenues "
          f"(pic net et decalage > {a.min_shift:.0f} px)\n")

    print(f"{'site A':>9} {'site B':>9} {'dist':>7} {'decal px':>16} {'nettete':>8} "
          f"{'gsd_x':>8} {'gsd_y':>8}")
    for t in r.head(25).itertuples():
        flag = "" if t.retenu else "  (ecarte)"
        gx = f"{t.gsd_x:.4f}" if not np.isnan(t.gsd_x) else "   -  "
        gy = f"{t.gsd_y:.4f}" if not np.isnan(t.gsd_y) else "   -  "
        print(f"{t.sup_a:>9} {t.sup_b:>9} {t.dist_m:6.1f}m "
              f"({t.sx:+6.0f},{t.sy:+6.0f}) {t.sharp:8.1f} {gx:>8} {gy:>8}{flag}")

    # --- rejet robuste des aberrantes (MAD) ------------------------------- #
    def robuste(v: pd.Series, nom: str) -> pd.Series:
        v = v.dropna()
        if len(v) < 3:
            return v
        m = v.median()
        mad = (v - m).abs().median()
        tol = max(3 * 1.4826 * mad, 0.02 * abs(m))
        garde = v[(v - m).abs() <= tol]
        rejet = len(v) - len(garde)
        if rejet:
            print(f"[INFO] {nom} : {rejet} mesure(s) aberrante(s) ecartee(s) "
                  f"(recouvrement trop faible ou correlation trompeuse)")
        return garde

    print()
    gx = robuste(keep.gsd_x, "axe X")
    gy = robuste(keep.gsd_y, "axe Y")
    allg = pd.concat([gx, gy])
    if allg.empty:
        sys.exit("[ERREUR] aucune mesure exploitable")

    print("\n" + "=" * 70)
    med = float(allg.median())
    etire = False
    if len(gx) and len(gy):
        mx, my = float(gx.median()), float(gy.median())
        etire = abs(my / mx - 1) > 0.05
        print(f"  TAILLE DU PIXEL MESUREE")
        print(f"    axe X : {mx*100:6.2f} cm/px  (n={len(gx)}, "
              f"etendue {gx.min()*100:.2f}-{gx.max()*100:.2f})")
        print(f"    axe Y : {my*100:6.2f} cm/px  (n={len(gy)}, "
              f"etendue {gy.min()*100:.2f}-{gy.max()*100:.2f})")
        if etire:
            print(f"\n    [!] X et Y different d'un facteur {my/mx:.3f} : les crops sont")
            print(f"        ETIRES. Une bbox carree demandee en DEGRES donne exactement")
            print(f"        ce symptome (1 deg de longitude != 1 deg de latitude).")
            print(f"\n  >>> EMPRISE AU SOL : {mx*a.size:.1f} m en X, {my*a.size:.1f} m en Y")
            print("=" * 70)
            print(f"\n  prepare_dataset.py et geoloc.py supposent un crop CARRE au sol.")
            print(f"  Deux options :")
            print(f"    - reechantillonner les crops pour les rendre isotropes, ou")
            print(f"    - utiliser --extent-m {my*a.size:.1f} (axe Y, le plus grand) en")
            print(f"      acceptant une erreur de {abs(my/mx-1)*100:.0f} % sur l'axe X.")
            print(f"  Dis-le-moi si tu tombes dans ce cas, la correction est courte.\n")
        else:
            print(f"\n  >>> EMPRISE AU SOL : {med*a.size:.1f} m pour {a.size} px "
                  f"({med*100:.2f} cm/px)")
            print("=" * 70)
            print(f"\n  A utiliser : prepare_dataset.py build --extent-m {med*a.size:.1f}")
            print(f"               app.py -> 'Resolution manuelle' = {med:.4f} m/px\n")
    else:
        print(f"  TAILLE DU PIXEL MESUREE : {med*100:.2f} cm/px")
        print(f"\n  >>> EMPRISE AU SOL : {med*a.size:.1f} m pour {a.size} px")
        print("=" * 70)
        print(f"\n  A utiliser : prepare_dataset.py build --extent-m {med*a.size:.1f}\n")

    r.to_csv("calibration_echelle.csv", index=False)
    print("[INFO] detail par paire : calibration_echelle.csv")


if __name__ == "__main__":
    main()
