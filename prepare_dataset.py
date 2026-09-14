#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py — FR Hack! 2026, Challenge 4 (ANFR x ISEP)
=============================================================

Construit un dataset YOLO a partir :
  - du referentiel ANFR (CSV : id, sup_id, x_recal, y_recal, lat, lon, height_m, offset_m)
  - d'un dossier d'imagerie deja telechargee (1 crop par site, centre sur la coordonnee ANFR)

Principe des annotations faibles
--------------------------------
Le crop etant centre sur la coordonnee ANFR, la boite est centree en (0.5, 0.5).
L'INCERTITUDE N'EST PAS DANS LA POSITION, ELLE EST DANS LA TAILLE : la demi-extension
de la boite est la somme des sources d'erreur, toutes en metres, puis convertie en
pixels via l'emprise au sol du crop.

    demi_extension = offset_m          (incertitude de recalage, fournie par l'ANFR)
                   + residu_georef     (georeferencement de l'ortho, ~2 m)
                   + empreinte_support (~10 % de la hauteur, borne 1.5-10 m)
                   + deplacement_relief(~10 % de la hauteur : un objet vertical est
                                        deplace radialement dans une orthophoto)
                   + marge
                   + fraction d'ombre portee (axe Y, ou axe azimut en mode directionnel)

L'ombre portee vaut L = height_m / tan(elevation_solaire). C'est souvent l'indice le
plus visible d'un pylone en vue verticale (cf. vignette ANFR_105438).

Sorties
-------
    <out>/images/{train,val,test}/     images (copie, ou lien symbolique avec --link)
    <out>/labels/{train,val}/          annotations faibles YOLO
    <out>/antennes.yaml                config dataset (compatible starter_antennes.py)
    <out>/dataset_manifest.csv         tracabilite complete site -> split -> boite
    <out>/test_douteux.csv             les sites non recales, isoles
    <out>/val_manual_todo.csv          echantillon a annoter A LA MAIN (obligatoire, cf. jury)
    <out>/METHODE_ANNOTATION.md        la methode, redigee (livrable exige par le brief)
    <out>/qc/                          planches de controle avec boites dessinees

Usage
-----
    # 0. verifier l'echelle des images (IMPORTANT, a faire en premier)
    python prepare_dataset.py calibrate --images ./crops --extent-m 204.8

    # 1. construire le dataset
    python prepare_dataset.py build \
        --csv "donnees antennes hackathon.csv" \
        --images ./crops \
        --out ./data \
        --extent-m 204.8

    # variantes utiles
        --classes support        # multi-classes derivees de la hauteur (Niveau 2)
        --shadow-mode directional --shadow-azimuth 340
        --split-mode random      # au lieu du groupement spatial
        --link                   # liens symboliques au lieu de copies (rapide, peu d'espace)

Dependances : pandas, numpy (+ Pillow pour les planches de controle)
"""

from __future__ import annotations

import argparse
import math
import random
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None


# --------------------------------------------------------------------------- #
# Constantes du modele d'incertitude  (toutes en metres, sauf mention)
# --------------------------------------------------------------------------- #

GEOREF_RESIDUAL_M = 2.0   # erreur de georeferencement residuelle de la BD ORTHO
MARGIN_M          = 2.0   # marge de securite
K_FOOTPRINT       = 0.10  # empreinte au sol du support ~ 10 % de sa hauteur
FOOTPRINT_MIN_M   = 1.5
FOOTPRINT_MAX_M   = 10.0
K_RELIEF          = 0.10  # deplacement radial du sommet dans l'ortho ~ 10 % de la hauteur
MIN_BOX_M         = 6.0   # une boite ne descend jamais sous 6 m de cote
MAX_BOX_FRAC      = 0.95  # ni au-dessus de 95 % du crop

# Classes multi-supports derivees de la hauteur.
# ATTENTION : c'est un PROXY. La hauteur ne distingue pas un chateau d'eau d'un pylone.
# A remplacer par les vraies classes apres relecture manuelle (Niveau 2).
SUPPORT_BINS  = [0.0, 8.0, 20.0, 45.0, math.inf]
SUPPORT_NAMES = ["support_bas", "support_moyen", "support_haut", "support_tres_haut"]

SINGLE_CLASS_NAMES = ["site_radio"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class BoxConfig:
    extent_m: float            # emprise au sol du crop (cote, en metres)
    sun_elevation_deg: float   # elevation solaire a la prise de vue
    shadow_mode: str           # none | symmetric | directional
    shadow_frac: float         # fraction de l'ombre incluse dans la boite
    shadow_azimuth_deg: float  # direction VERS LAQUELLE pointe l'ombre (0 = Nord, 90 = Est)

    @property
    def shadow_ratio(self) -> float:
        """Longueur d'ombre par metre de hauteur."""
        if self.shadow_mode == "none":
            return 0.0
        return 1.0 / math.tan(math.radians(self.sun_elevation_deg))


# --------------------------------------------------------------------------- #
# Geometrie des boites
# --------------------------------------------------------------------------- #

def isotropic_radius_m(height_m: float, offset_m: float) -> float:
    """Demi-extension independante de l'ombre : ou peut se trouver le support."""
    footprint = float(np.clip(K_FOOTPRINT * height_m, FOOTPRINT_MIN_M, FOOTPRINT_MAX_M))
    relief    = K_RELIEF * height_m
    return offset_m + GEOREF_RESIDUAL_M + footprint + relief + MARGIN_M


def compute_box(height_m: float, offset_m: float, cfg: BoxConfig) -> dict:
    """
    Renvoie la boite YOLO normalisee (xc, yc, w, h) + les intermediaires en metres,
    pour un crop carre d'emprise cfg.extent_m centre sur la coordonnee ANFR.
    """
    r_iso    = isotropic_radius_m(height_m, offset_m)
    shadow_m = cfg.shadow_ratio * height_m * cfg.shadow_frac

    if cfg.shadow_mode == "directional" and shadow_m > 0:
        # L'ombre part de la base et s'etend dans une seule direction : on decale le
        # centre de la boite d'une demi-longueur d'ombre et on n'allonge que cet axe.
        az = math.radians(cfg.shadow_azimuth_deg)
        dx_m = math.sin(az) * shadow_m / 2.0          # Est positif
        dy_m = -math.cos(az) * shadow_m / 2.0         # Nord = -y en coordonnees image
        half_w_m = r_iso + abs(math.sin(az)) * shadow_m / 2.0
        half_h_m = r_iso + abs(math.cos(az)) * shadow_m / 2.0
    else:
        # Mode symetrique : l'azimut est inconnu, on elargit les deux axes.
        dx_m = dy_m = 0.0
        half_w_m = r_iso + shadow_m
        half_h_m = r_iso + shadow_m

    w_m = max(2.0 * half_w_m, MIN_BOX_M)
    h_m = max(2.0 * half_h_m, MIN_BOX_M)

    xc = 0.5 + dx_m / cfg.extent_m
    yc = 0.5 + dy_m / cfg.extent_m
    w  = w_m / cfg.extent_m
    h  = h_m / cfg.extent_m

    # La boite doit rester entierement dans l'image : on rogne symetriquement.
    w = min(w, MAX_BOX_FRAC)
    h = min(h, MAX_BOX_FRAC)
    xc = float(np.clip(xc, w / 2.0, 1.0 - w / 2.0))
    yc = float(np.clip(yc, h / 2.0, 1.0 - h / 2.0))

    return {
        "xc": round(xc, 6), "yc": round(yc, 6),
        "w": round(w, 6), "h": round(h, 6),
        "r_iso_m": round(r_iso, 2),
        "shadow_m": round(shadow_m, 2),
        "box_w_m": round(w * cfg.extent_m, 2),
        "box_h_m": round(h * cfg.extent_m, 2),
    }


def support_class(height_m: float) -> tuple[int, str]:
    for i in range(len(SUPPORT_NAMES)):
        if SUPPORT_BINS[i] <= height_m < SUPPORT_BINS[i + 1]:
            return i, SUPPORT_NAMES[i]
    return len(SUPPORT_NAMES) - 1, SUPPORT_NAMES[-1]


# --------------------------------------------------------------------------- #
# Chargement & appariement CSV <-> images
# --------------------------------------------------------------------------- #

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def load_sites(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"id", "sup_id", "lat", "lon", "height_m", "offset_m"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERREUR] colonnes manquantes dans le CSV : {sorted(missing)}")

    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    df["offset_m"] = pd.to_numeric(df["offset_m"], errors="coerce").fillna(0.0)

    # Un site est "douteux" si sa coordonnee n'a pas ete recalee : la hauteur du
    # support est absente ET l'offset est exactement nul (valeur par defaut, pas mesuree).
    df["no_height"]  = df["height_m"].isna()
    df["zero_offset"] = df["offset_m"] == 0.0
    df["is_douteux"] = df["no_height"] | df["zero_offset"]
    return df


def index_images(images_dir: Path) -> dict[int, Path]:
    """
    Indexe les images par identifiant numerique extrait du nom de fichier.
    Gere ANFR_105408.jpg, 105408.png, site-105408_crop.jpg, ...
    Si plusieurs nombres sont presents, on retient le plus long (le sup_id).
    """
    idx: dict[int, list[Path]] = {}
    for p in sorted(images_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        nums = re.findall(r"\d+", p.stem)
        if not nums:
            continue
        key = int(max(nums, key=len))
        idx.setdefault(key, []).append(p)

    out = {}
    for k, paths in idx.items():
        if len(paths) > 1:
            print(f"[WARN] identifiant {k} : {len(paths)} images, je garde {paths[0].name}")
        out[k] = paths[0]
    return out


def match_images(df: pd.DataFrame, idx: dict[int, Path]) -> pd.DataFrame:
    """Apparie d'abord sur sup_id, puis sur id en secours."""
    paths, keys = [], []
    for _, row in df.iterrows():
        for col in ("sup_id", "id"):
            k = int(row[col])
            if k in idx:
                paths.append(idx[k]); keys.append(k); break
        else:
            paths.append(None); keys.append(None)
    df = df.copy()
    df["image_path"] = paths
    df["image_key"]  = keys
    return df


# --------------------------------------------------------------------------- #
# Split train / val
# --------------------------------------------------------------------------- #

def spatial_groups(df: pd.DataFrame, radius_m: float) -> np.ndarray:
    """
    Union-find sur les sites distants de moins de radius_m.
    Deux sites proches ont des crops qui se recouvrent : les separer entre train et val
    ferait fuiter les memes pixels dans les deux jeux et gonflerait artificiellement la mAP.
    """
    if not {"x_recal", "y_recal"}.issubset(df.columns):
        return np.arange(len(df))

    xy = df[["x_recal", "y_recal"]].to_numpy(dtype=float)
    n = len(xy)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Tri par x puis balayage : evite la matrice n x n.
    order = np.argsort(xy[:, 0])
    for ii in range(n):
        i = order[ii]
        for jj in range(ii + 1, n):
            j = order[jj]
            if xy[j, 0] - xy[i, 0] > radius_m:
                break
            if abs(xy[j, 1] - xy[i, 1]) <= radius_m:
                if math.hypot(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]) <= radius_m:
                    union(i, j)
    return np.array([find(i) for i in range(n)])


def assign_split(df: pd.DataFrame, val_ratio: float, mode: str,
                 radius_m: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = random.Random(seed)

    if mode == "random":
        df["group"] = np.arange(len(df))
    else:
        df["group"] = spatial_groups(df, radius_m)

    # Groupes melanges puis empiles dans val jusqu'a atteindre le quota.
    groups = df.groupby("group").size().sort_values(ascending=False)
    gids = list(groups.index)
    rng.shuffle(gids)

    target_val = int(round(val_ratio * len(df)))
    val_groups, n_val = set(), 0
    for g in gids:
        if n_val >= target_val:
            break
        val_groups.add(g)
        n_val += int(groups[g])

    df["split"] = np.where(df["group"].isin(val_groups), "val", "train")
    return df


def pick_manual_val(df_val: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Echantillon stratifie par classe de hauteur, a annoter a la main."""
    if df_val.empty or n <= 0:
        return df_val.head(0)
    n = min(n, len(df_val))
    classes = list(df_val["class_name"].unique())
    per = max(1, n // max(1, len(classes)))
    parts = []
    for c in classes:
        g = df_val[df_val["class_name"] == c]
        parts.append(g.sample(min(len(g), per), random_state=seed))
    picked = pd.concat(parts) if parts else df_val.head(0)
    if len(picked) < n:
        rest = df_val.loc[~df_val.index.isin(picked.index)]
        if len(rest):
            picked = pd.concat([picked,
                                rest.sample(min(n - len(picked), len(rest)),
                                            random_state=seed)])
    return picked.head(n)


# --------------------------------------------------------------------------- #
# Ecriture du dataset
# --------------------------------------------------------------------------- #

def place_image(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def write_yaml(out: Path, names: list[str]) -> None:
    lines = [
        "# Genere par prepare_dataset.py - FR Hack! 2026 Challenge 4",
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(names)}",
        f"names: {names}",
        "",
    ]
    (out / "antennes.yaml").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Controle qualite
# --------------------------------------------------------------------------- #

def draw_box(im, box, color=(255, 60, 60), width=3):
    d = ImageDraw.Draw(im)
    W, H = im.size
    x1 = (box["xc"] - box["w"] / 2) * W
    y1 = (box["yc"] - box["h"] / 2) * H
    x2 = (box["xc"] + box["w"] / 2) * W
    y2 = (box["yc"] + box["h"] / 2) * H
    d.rectangle([x1, y1, x2, y2], outline=color, width=width)
    d.line([W / 2 - 10, H / 2, W / 2 + 10, H / 2], fill=(60, 255, 60), width=2)
    d.line([W / 2, H / 2 - 10, W / 2, H / 2 + 10], fill=(60, 255, 60), width=2)
    return im


def contact_sheet(rows, out_path, cols=5, thumb=320, title=""):
    out_path = Path(out_path)
    if Image is None:
        print("[WARN] Pillow absent : planches de controle ignorees")
        return
    rows = list(rows)
    if not rows:
        return
    n = len(rows)
    r = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * thumb, r * thumb), (18, 18, 20))
    for i, (path, box, label) in enumerate(rows):
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] {path} illisible : {e}")
            continue
        draw_box(im, box, width=max(2, im.size[0] // 200))
        im = im.resize((thumb, thumb))
        d = ImageDraw.Draw(im)
        d.rectangle([0, thumb - 18, thumb, thumb], fill=(0, 0, 0))
        d.text((4, thumb - 15), label[:46], fill=(255, 255, 255))
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=88)
    print(f"[OK] planche de controle : {out_path}  ({n} vignettes) {title}")


def cmd_calibrate(args):
    """Dessine une regle graduee de 50 m pour verifier l'emprise au sol declaree."""
    if Image is None:
        sys.exit("[ERREUR] Pillow requis : pip install pillow")
    idx = index_images(Path(args.images))
    if not idx:
        sys.exit(f"[ERREUR] aucune image trouvee dans {args.images}")
    keys = sorted(idx)[: args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in keys:
        im = Image.open(idx[k]).convert("RGB")
        W, H = im.size
        px_per_m = W / args.extent_m
        d = ImageDraw.Draw(im)
        y = H - 40
        d.rectangle([18, y - 26, 18 + 50 * px_per_m + 14, y + 16], fill=(0, 0, 0))
        d.line([25, y, 25 + 50 * px_per_m, y], fill=(255, 220, 0), width=4)
        for m in range(0, 51, 10):
            x = 25 + m * px_per_m
            d.line([x, y - 9, x, y + 9], fill=(255, 220, 0), width=3)
        d.text((25, y - 22), f"50 m  ({args.extent_m:.1f} m d'emprise, "
                             f"{args.extent_m / W * 100:.1f} cm/px)", fill=(255, 220, 0))
        p = out / f"calib_{k}.jpg"
        im.save(p, quality=92)
        print(f"[OK] {p}")
    print("\n>>> Ouvre ces images. Une voiture doit mesurer ~4,5 m et une chaussee "
          "~6-7 m sur la regle. Sinon, corrige --extent-m et relance.")



# --------------------------------------------------------------------------- #
# Estimation empirique de l'azimut des ombres
# --------------------------------------------------------------------------- #

def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Moyenne glissante separable via sommes cumulees (pas de dependance scipy)."""
    if r < 1:
        return a
    pad = np.pad(a, r, mode="edge")
    c = np.cumsum(pad, axis=0)
    c = np.vstack([np.zeros((1, c.shape[1]), c.dtype), c])
    a1 = (c[2 * r + 1:] - c[:-(2 * r + 1)]) / (2 * r + 1)
    c = np.cumsum(a1, axis=1)
    c = np.hstack([np.zeros((c.shape[0], 1), c.dtype), c])
    a2 = (c[:, 2 * r + 1:] - c[:, :-(2 * r + 1)]) / (2 * r + 1)
    return a2


def _shadow_score_by_angle(gray: np.ndarray, dists_px, n_angles=72, hp_radius=None):
    """
    Une ombre portee est une zone sombre situee a cote d'un objet clair, toujours
    dans la meme direction. On decale le masque des objets clairs de d pixels dans
    la direction theta et on mesure son recouvrement avec le masque des zones sombres.

    Deux precautions, sans lesquelles on mesure l'occupation du sol et non les ombres :

    1. FILTRAGE PASSE-HAUT. Sans lui, "champ clair a cote d'un bois sombre" domine
       completement le signal. On retranche une moyenne glissante large pour ne
       garder que le contraste local.
    2. SCORE ANTISYMETRIQUE score(theta) - score(theta+180). Toute correlation
       symetrique (texture, lisiere, parcellaire) s'annule ; seul subsiste le
       decalage oriente, qui est la signature d'une ombre portee.
    """
    g = gray.astype(np.float32)
    r = hp_radius if hp_radius is not None else max(4, gray.shape[0] // 12)
    g = g - _box_blur(g, r)

    lo, hi = np.percentile(g, 10), np.percentile(g, 90)
    shadow = (g <= lo).astype(np.float32)
    struct = (g >= hi).astype(np.float32)
    if shadow.sum() < 50 or struct.sum() < 50:
        return None
    shadow /= shadow.sum()

    n_half = n_angles // 2
    angles = np.arange(0, 360, 360 / n_angles)
    raw = np.zeros(len(angles), dtype=np.float64)
    for k, a in enumerate(angles):
        ar = math.radians(a)
        # azimut 0 = Nord = haut de l'image (-y) ; 90 = Est = +x
        ux, uy = math.sin(ar), -math.cos(ar)
        tot = 0.0
        for d in dists_px:
            sh = np.roll(np.roll(struct, int(round(uy * d)), axis=0),
                         int(round(ux * d)), axis=1)
            tot += float((sh * shadow).sum())
        raw[k] = tot / len(dists_px)

    # antisymetrisation : on ne garde que la composante orientee
    scores = raw - np.roll(raw, n_half)
    return angles, scores, raw


def cmd_estimate_azimuth(args):
    """Mesure l'azimut dominant des ombres sur un echantillon d'images."""
    if Image is None:
        sys.exit("[ERREUR] Pillow requis : pip install pillow")
    idx = index_images(Path(args.images))
    if not idx:
        sys.exit(f"[ERREUR] aucune image dans {args.images}")

    keys = sorted(idx)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    keys = keys[: args.n]

    dists_px = [int(round(d / args.extent_m * args.work_size))
                for d in (8.0, 14.0, 22.0, 32.0)]
    dists_px = [d for d in dists_px if d >= 1] or [3, 6, 9]

    vecs, per_image = [], []
    for k in keys:
        try:
            im = Image.open(idx[k]).convert("L").resize((args.work_size, args.work_size))
        except Exception as e:
            print(f"[WARN] {idx[k].name} illisible : {e}")
            continue
        res = _shadow_score_by_angle(np.asarray(im), dists_px, args.n_angles)
        if res is None:
            continue
        angles, scores, raw = res
        if raw.mean() <= 0 or scores.max() <= 0:
            continue
        # nettete = amplitude du pic oriente rapportee au niveau moyen de correlation
        contrast = float(scores.max() / raw.mean())
        best = float(angles[int(np.argmax(scores))])
        per_image.append((k, best, contrast))
        # vecteur pondere par la nettete du pic -> moyenne circulaire robuste
        vecs.append((contrast * math.sin(math.radians(best)),
                     contrast * math.cos(math.radians(best))))

    if not vecs:
        sys.exit("[ERREUR] estimation impossible (images trop uniformes ?)")

    sx = sum(v[0] for v in vecs); sy = sum(v[1] for v in vecs)
    mean_az = math.degrees(math.atan2(sx, sy)) % 360.0
    norm = math.hypot(sx, sy) / sum(abs(v[0]) + abs(v[1]) or 1e-9 for v in vecs)
    # concentration : 1 = toutes les images d'accord, 0 = aucune coherence
    R = math.hypot(sx, sy) / max(sum(c for _, _, c in per_image), 1e-9)

    print("=" * 66)
    print(f"  Azimut des ombres estime sur {len(per_image)} images")
    print("=" * 66)
    for k, a, c in sorted(per_image, key=lambda t: -t[2])[: args.show]:
        print(f"    {k:>9}  azimut {a:6.1f} deg   nettete du pic {c:5.2f}")
    if len(per_image) > args.show:
        print(f"    ... ({len(per_image) - args.show} autres)")
    print()
    print(f"  >>> AZIMUT MOYEN : {mean_az:.1f} deg   (coherence R = {R:.2f})")
    print()
    if R >= 0.55:
        print(f"  Coherence bonne : utilise --shadow-mode directional "
              f"--shadow-azimuth {mean_az:.0f}")
    elif R >= 0.3:
        print(f"  Coherence moyenne. L'ortho est une mosaique de plusieurs vols :")
        print(f"  l'azimut n'est probablement pas constant sur tout le jeu.")
        print(f"  Le mode directional reste jouable a {mean_az:.0f} deg, mais compare-le a")
        print(f"  --shadow-mode symmetric avant de trancher.")
    else:
        print(f"  Coherence faible : pas d'azimut dominant fiable.")
        print(f"  Reste en --shadow-mode symmetric.")
    print()


# --------------------------------------------------------------------------- #
# Commande principale
# --------------------------------------------------------------------------- #

def cmd_build(args):
    csv_path   = Path(args.csv)
    images_dir = Path(args.images)
    out        = Path(args.out)

    cfg = BoxConfig(
        extent_m=args.extent_m,
        sun_elevation_deg=args.sun_elevation,
        shadow_mode=args.shadow_mode,
        shadow_frac=args.shadow_frac,
        shadow_azimuth_deg=args.shadow_azimuth,
    )

    print("=" * 74)
    print("  FR Hack! 2026 - Challenge 4 : construction du dataset")
    print("=" * 74)
    print(f"  emprise au sol      : {cfg.extent_m} m par crop")
    print(f"  ombre               : mode={cfg.shadow_mode} frac={cfg.shadow_frac} "
          f"elev={cfg.sun_elevation_deg} deg -> L = {cfg.shadow_ratio:.2f} x hauteur")
    print(f"  classes             : {args.classes}")
    print(f"  split               : {args.split_mode}, val = {args.val_ratio:.0%}")
    print()

    # ---- 1. chargement --------------------------------------------------- #
    df = load_sites(csv_path)
    print(f"[1] {len(df)} sites charges depuis {csv_path.name}")
    n_nh, n_zo = int(df.no_height.sum()), int(df.zero_offset.sum())
    n_both = int((df.no_height & df.zero_offset).sum())
    print(f"    height_m absente : {n_nh}   |   offset_m == 0 : {n_zo}   |   les deux : {n_both}")
    if n_both != n_nh or n_both != n_zo:
        print("    [!] les deux criteres ne coincident pas parfaitement -> union retenue")
    print(f"    => {int(df.is_douteux.sum())} sites DOUTEUX (non recales) isoles en test/")

    # ---- 2. appariement images ------------------------------------------- #
    idx = index_images(images_dir)
    print(f"\n[2] {len(idx)} images indexees dans {images_dir}")
    df = match_images(df, idx)
    n_ok = int(df.image_path.notna().sum())
    print(f"    {n_ok}/{len(df)} sites apparies a une image")
    if n_ok < len(df):
        manquants = df.loc[df.image_path.isna(), "sup_id"].tolist()
        print(f"    [!] {len(manquants)} sans image, ex. sup_id : {manquants[:8]}")
    orphelines = set(idx) - set(df.image_key.dropna().astype(int))
    if orphelines:
        print(f"    [!] {len(orphelines)} images sans site correspondant (ignorees)")

    df = df[df.image_path.notna()].copy()
    if df.empty:
        sys.exit("[ERREUR] aucune image appariee. Verifie le nommage des fichiers.")

    # ---- 3. separation douteux / exploitables ----------------------------- #
    df_test  = df[df.is_douteux].copy()
    df_test["motif_exclusion"] = "non_recale"
    df_train = df[~df.is_douteux].copy()
    print(f"\n[3] {len(df_train)} sites exploitables  |  {len(df_test)} sites douteux -> test/")

    # Sur un crop etroit, un offset ANFR proche du demi-crop donne une boite qui
    # couvre presque toute l'image : l'annotation ne porte plus d'information de
    # localisation et degrade l'entrainement. Ces sites sont ecartes, pas perdus.
    if args.max_offset_m is not None:
        trop = df_train.offset_m > args.max_offset_m
        if trop.any():
            ecartes = df_train[trop].copy()
            ecartes["motif_exclusion"] = f"offset_sup_{args.max_offset_m:g}m"
            df_test = pd.concat([df_test, ecartes], ignore_index=True)
            df_train = df_train[~trop].copy()
            print(f"    [filtre] {int(trop.sum())} sites avec offset_m > "
                  f"{args.max_offset_m:g} m ecartes vers test/ "
                  f"(demi-crop = {cfg.extent_m/2:.1f} m)")
            print(f"    => {len(df_train)} sites conserves pour train + val")

    # ---- 4. boites + classes --------------------------------------------- #
    boxes = [compute_box(float(r.height_m), float(r.offset_m), cfg)
             for r in df_train.itertuples()]
    df_train = pd.concat([df_train.reset_index(drop=True),
                          pd.DataFrame(boxes)], axis=1)

    if args.classes == "support":
        cls = df_train.height_m.apply(support_class)
        df_train["class_id"]   = [c[0] for c in cls]
        df_train["class_name"] = [c[1] for c in cls]
        names = SUPPORT_NAMES
    else:
        df_train["class_id"]   = 0
        df_train["class_name"] = SINGLE_CLASS_NAMES[0]
        names = SINGLE_CLASS_NAMES

    print(f"\n[4] boites calculees. Taille au sol (cote) :")
    print(f"    min {df_train.box_w_m.min():6.1f} m | mediane {df_train.box_w_m.median():6.1f} m "
          f"| max {df_train.box_h_m.max():6.1f} m")
    print(f"    surface normalisee : mediane {(df_train.w * df_train.h).median():.4f} "
          f"({(df_train.w * df_train.h).median() * 100:.2f} % du crop)")
    # Invariant de securite du mode directional : le decalage du centre doit rester
    # inferieur a la demi-extension isotrope, sinon un azimut faux ferait sortir la
    # base du support de la boite.
    if cfg.shadow_mode == "directional":
        shift = df_train.shadow_m / 2.0
        bad = int((shift >= df_train.r_iso_m).sum())
        ratio = float((shift / df_train.r_iso_m).max())
        if bad:
            print(f"    [!] {bad} sites ou le decalage depasse le rayon isotrope : "
                  f"un azimut errone sortirait le support de la boite")
        else:
            print(f"    [OK] invariant directional verifie sur {len(df_train)} sites : "
                  f"decalage/rayon max = {ratio:.2f} < 1")
            print(f"         -> meme avec un azimut faux a 180 deg, la base du support "
                  f"reste dans la boite")
    clipped = int(((df_train.w >= MAX_BOX_FRAC) | (df_train.h >= MAX_BOX_FRAC)).sum())
    if clipped:
        print(f"    [!] {clipped} boites rognees a {MAX_BOX_FRAC:.0%} du crop "
              f"(crop trop petit pour ces supports)")
    if args.classes == "support":
        for nm, c in df_train.class_name.value_counts().items():
            print(f"      {nm:20s} {c:4d}")

    # ---- 5. split -------------------------------------------------------- #
    radius = args.group_radius if args.group_radius is not None else cfg.extent_m
    df_train = assign_split(df_train, args.val_ratio, args.split_mode, radius, args.seed)
    n_tr = int((df_train.split == "train").sum())
    n_va = int((df_train.split == "val").sum())
    print(f"\n[5] split {args.split_mode} : train {n_tr} ({n_tr/len(df_train):.0%}) "
          f"| val {n_va} ({n_va/len(df_train):.0%})")
    if args.split_mode == "group":
        ngroups = df_train.group.nunique()
        print(f"    {ngroups} groupes spatiaux (rayon {radius:.0f} m) -> "
              f"{len(df_train) - ngroups} sites regroupes pour eviter la fuite train/val")

    # ---- 6. ecriture ----------------------------------------------------- #
    for sub in ("images/train", "images/val", "images/test", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for r in df_train.itertuples():
        stem = Path(r.image_path).stem
        place_image(Path(r.image_path),
                    out / "images" / r.split / Path(r.image_path).name, args.link)
        (out / "labels" / r.split / f"{stem}.txt").write_text(
            f"{r.class_id} {r.xc:.6f} {r.yc:.6f} {r.w:.6f} {r.h:.6f}\n", encoding="utf-8")

    for r in df_test.itertuples():
        place_image(Path(r.image_path),
                    out / "images" / "test" / Path(r.image_path).name, args.link)

    write_yaml(out, names)
    print(f"\n[6] dataset ecrit dans {out.resolve()}")
    print(f"    images/train {n_tr} | images/val {n_va} | images/test {len(df_test)} "
          f"(sans label : lot a inspecter)")

    # ---- 7. tracabilite & livrables -------------------------------------- #
    keep = ["id", "sup_id", "lat", "lon", "height_m", "offset_m", "split",
            "class_id", "class_name", "xc", "yc", "w", "h",
            "r_iso_m", "shadow_m", "box_w_m", "box_h_m", "group"]
    man = df_train[keep].copy()
    man["image"] = [Path(p).name for p in df_train.image_path]
    man["source"] = "annotation_faible"

    t = df_test[["id", "sup_id", "lat", "lon", "height_m", "offset_m",
                 "motif_exclusion"]].copy()
    t["split"] = "test"; t["source"] = t["motif_exclusion"]
    t["image"] = [Path(p).name for p in df_test.image_path]

    pd.concat([man, t], ignore_index=True).to_csv(out / "dataset_manifest.csv", index=False)
    t.to_csv(out / "test_douteux.csv", index=False)

    todo = pick_manual_val(df_train[df_train.split == "val"], args.manual_val, args.seed)
    todo_cols = ["sup_id", "lat", "lon", "height_m", "offset_m", "class_name"]
    todo_out = todo[todo_cols].copy()
    todo_out["image"] = [Path(p).name for p in todo.image_path]
    todo_out["annote_par"] = ""
    todo_out["verdict"] = ""   # ok | boite_a_corriger | rien_de_visible | mauvais_support
    todo_out.to_csv(out / "val_manual_todo.csv", index=False)
    print(f"    dataset_manifest.csv ({len(man) + len(t)} lignes), test_douteux.csv, "
          f"val_manual_todo.csv ({len(todo_out)} sites)")

    (out / "METHODE_ANNOTATION.md").write_text(methode_md(cfg, args, df_train, df_test, names),
                                               encoding="utf-8")
    print(f"    METHODE_ANNOTATION.md")

    # ---- 8. planches de controle ----------------------------------------- #
    if not args.no_qc:
        qc = out / "qc"
        s = df_train.sort_values("height_m")
        sel = pd.concat([s.head(10), s.iloc[len(s)//2 - 5: len(s)//2 + 5], s.tail(10)])
        contact_sheet(
            [(r.image_path, {"xc": r.xc, "yc": r.yc, "w": r.w, "h": r.h},
              f"{r.sup_id} h={r.height_m:.0f}m off={r.offset_m:.0f}m")
             for r in sel.itertuples()],
            qc / "qc_par_hauteur.jpg", title="(10 plus bas / 10 medians / 10 plus hauts)")
        big = df_train[df_train.offset_m > 25].head(20)
        if len(big):
            contact_sheet(
                [(r.image_path, {"xc": r.xc, "yc": r.yc, "w": r.w, "h": r.h},
                  f"{r.sup_id} off={r.offset_m:.0f}m")
                 for r in big.itertuples()],
                qc / "qc_offset_eleve.jpg", title="(offset > 25 m)")

    print("\n" + "=" * 74)
    print("  ETAPE SUIVANTE")
    print("=" * 74)
    print(f"  1. Ouvre {out/'qc'}/ et verifie que les supports sont DANS les boites.")
    print(f"     Si elles sont systematiquement trop grandes/petites -> --extent-m est faux.")
    print(f"  2. Annote a la main les {len(todo_out)} sites de val_manual_todo.csv.")
    print(f"  3. Entrainement :")
    print(f"     python starter_antennes.py train --data {out} --epochs 80 --imgsz 1024 --batch 8")
    print()


# --------------------------------------------------------------------------- #
# Livrable : la methode, redigee
# --------------------------------------------------------------------------- #

def methode_md(cfg, args, df_train, df_test, names) -> str:
    q = df_train.box_w_m.quantile([0.05, 0.5, 0.95]).round(1)
    return f"""# Methode de generation des annotations faibles

FR Hack! 2026 — Challenge 4 (ANFR x ISEP). Genere par `prepare_dataset.py`.

## 1. Le probleme

Une coordonnee ANFR **n'est pas une verite terrain visuelle**. Elle indique qu'un site
radioelectrique se trouve a proximite, sans dire quels pixels correspondent au support.
Transformer cette coordonnee en boite englobante est donc une modelisation d'incertitude,
pas une simple conversion de reperes.

## 2. Tri prealable des sites

| Lot | Effectif | Critere | Usage |
|---|---|---|---|
| Exploitables | {len(df_train)} | `height_m` renseignee **et** `offset_m` > 0 | train + val |
| Douteux | {len(df_test)} | `height_m` absente **et** `offset_m` == 0 | `test/`, sans label |

Les {len(df_test)} sites ecartes n'ont pas de recalage : leur `offset_m` vaut 0 par defaut
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
      + {GEOREF_RESIDUAL_M}                             # residu de georeferencement de l'ortho
      + clip({K_FOOTPRINT} x height_m, {FOOTPRINT_MIN_M}, {FOOTPRINT_MAX_M})   # empreinte au sol du support
      + {K_RELIEF} x height_m                    # deplacement radial du sommet dans l'ortho
      + {MARGIN_M}                             # marge
```

Le terme de **deplacement radial** vient du fait qu'une orthophoto est redressee sur le
modele de terrain : un objet vertical de hauteur h vu a distance du nadir est deplace
d'environ h x (d / H). Un pylone de 40 m peut ainsi apparaitre a une dizaine de metres
de sa position cadastrale, dans une direction inconnue.

### Ombre portee

L = height_m / tan({cfg.sun_elevation_deg} deg) = **{cfg.shadow_ratio:.2f} x height_m**, dont on retient
{cfg.shadow_frac:.0%} (`--shadow-frac`).

Mode actif : **{cfg.shadow_mode}**.

- `symmetric` — les deux axes sont elargis de `shadow_frac x L`. Sans hypothese sur
  l'azimut solaire, mais la boite contient beaucoup de fond.
- `directional` — le centre est decale d'une demi-longueur d'ombre le long de l'azimut
  ({cfg.shadow_azimuth_deg} deg) et seul cet axe est allonge. Meme couverture de l'ombre pour environ
  moitie moins de surface de fond. A n'utiliser que si l'azimut a ete verifie sur les
  images (sur l'echantillon inspecte, les ombres pointent vers le NNO).

L'ombre est retenue parce qu'en vue verticale, **le support lui-meme est souvent moins
lisible que son ombre** : un pylone treillis se resume a quelques pixels au nadir, alors
que son ombre dessine une structure allongee nette sur le sol.

### Resultat

Cote de boite au sol : p05 = {q[0.05]} m, mediane = {q[0.5]} m, p95 = {q[0.95]} m.
La taille varie donc d'un facteur ~{q[0.95]/max(q[0.05],1e-9):.1f} entre un mat de toiture et un grand pylone,
au lieu d'une boite fixe identique pour tous.

## 5. Classes

`--classes {args.classes}` -> {names}

En mode `support`, les classes sont derivees de `height_m` par seuils
({SUPPORT_BINS[1]:.0f} / {SUPPORT_BINS[2]:.0f} / {SUPPORT_BINS[3]:.0f} m). **C'est un proxy assume** : la hauteur ne distingue
pas un chateau d'eau d'un pylone treillis. Ces classes servent de pre-annotation a
corriger manuellement, pas de verite terrain.

## 6. Split train / val

Mode **{args.split_mode}**, {args.val_ratio:.0%} en validation.

En mode `group`, les sites distants de moins de {(args.group_radius or cfg.extent_m):.0f} m sont places dans
le meme split. Sans cette precaution, deux sites voisins produisent des crops qui se
recouvrent : les memes pixels se retrouveraient en entrainement et en validation, et la
mAP@50 serait surevaluee. Dans ce jeu, {int((df_train.groupby('group').size() > 1).sum())} groupes contiennent plusieurs sites.

## 7. Limite a garder en tete

Le jeu de validation produit ici est annote **faiblement**, comme le jeu d'entrainement.
Une mAP@50 mesuree dessus evalue la capacite du modele a reproduire l'heuristique, pas a
detecter reellement les supports. Le fichier `val_manual_todo.csv` liste
{args.manual_val} sites stratifies a annoter a la main : c'est ce jeu-la qui doit servir de reference
pour les {25} points de performance du bareme.
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        description="FR Hack! 2026 Challenge 4 - annotations faibles et dataset YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="construire le dataset YOLO")
    b.add_argument("--csv", required=True, help="CSV ANFR")
    b.add_argument("--images", required=True, help="dossier des crops")
    b.add_argument("--out", default="data", help="dossier de sortie")
    b.add_argument("--extent-m", type=float, default=204.8,
                   help="emprise au sol d'un crop, en metres (1024 px a 20 cm = 204.8). "
                        "VERIFIE-LA avec la commande calibrate")
    b.add_argument("--classes", choices=["single", "support"], default="single")
    b.add_argument("--shadow-mode", choices=["none", "symmetric", "directional"],
                   default="directional")
    b.add_argument("--shadow-frac", type=float, default=0.6)
    b.add_argument("--shadow-azimuth", type=float, default=340.0,
                   help="direction vers laquelle pointe l'ombre (deg, 0=Nord)")
    b.add_argument("--sun-elevation", type=float, default=55.0)
    b.add_argument("--val-ratio", type=float, default=0.20)
    b.add_argument("--split-mode", choices=["group", "random"], default="group")
    b.add_argument("--group-radius", type=float, default=None,
                   help="rayon de regroupement spatial en m (defaut : emprise du crop)")
    b.add_argument("--max-offset-m", type=float, default=None,
                   help="ecarte les sites dont offset_m depasse ce seuil : sur un crop "
                        "etroit leur boite sature et n'apprend plus rien. Ils partent "
                        "en test/ avec le motif documente.")
    b.add_argument("--manual-val", type=int, default=60,
                   help="nombre de sites a annoter a la main")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--link", action="store_true",
                   help="liens symboliques au lieu de copies")
    b.add_argument("--no-qc", action="store_true")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("calibrate", help="verifier l'emprise au sol des crops")
    c.add_argument("--images", required=True)
    c.add_argument("--extent-m", type=float, default=204.8)
    c.add_argument("--n", type=int, default=6)
    c.add_argument("--out", default="calibration")
    c.set_defaults(func=cmd_calibrate)

    e = sub.add_parser("estimate-azimuth",
                       help="mesurer l'azimut des ombres sur un echantillon")
    e.add_argument("--images", required=True)
    e.add_argument("--extent-m", type=float, default=204.8)
    e.add_argument("--n", type=int, default=60, help="nombre d'images echantillonnees")
    e.add_argument("--work-size", type=int, default=384)
    e.add_argument("--n-angles", type=int, default=72)
    e.add_argument("--show", type=int, default=12)
    e.add_argument("--seed", type=int, default=42)
    e.set_defaults(func=cmd_estimate_azimuth)

    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    a.func(a)
