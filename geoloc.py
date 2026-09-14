#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geoloc.py — FR Hack! 2026, Challenge 4 (ANFR x ISEP)
=====================================================

Conversion pixel <-> GPS pour les crops centres sur une coordonnee ANFR.
Complement de prepare_dataset.py : sert a transformer un clic de correction
manuelle en coordonnee WGS84 exacte (Niveau 2 du brief).

Convention d'image
------------------
    - crop carre de `image_size_px` pixels de cote (1024 par defaut)
    - emprise au sol `extent_m` metres de cote (204.8 m par defaut -> 20 cm/px)
    - le centre geometrique du crop porte la coordonnee ANFR (lat0, lon0)
    - x croit vers l'EST, y croit vers le SUD (convention image)
    - image supposee orientee nord (aucune rotation) : c'est le cas d'une dalle
      BD ORTHO ou d'un GetMap WMS standard

Dependances : numpy uniquement. pyproj / geopy ne sont pas requis (voir la note
sur geopy en bas de fichier).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Ellipsoide GRS80 (= WGS84 a 0.1 mm pres sur les demi-axes)
# --------------------------------------------------------------------------- #

_A  = 6378137.0                    # demi-grand axe (m)
_F  = 1.0 / 298.257222101          # aplatissement GRS80
_E2 = 2.0 * _F - _F * _F           # premiere excentricite au carre


def _radii(lat_rad):
    """
    Rayons de courbure locaux de l'ellipsoide a la latitude donnee.

    M = rayon de courbure meridien       -> gouverne le deplacement NORD/SUD
    N = rayon de courbure transverse     -> gouverne le deplacement EST/OUEST
        (aussi appele grande normale)
    """
    s2 = np.sin(lat_rad) ** 2
    w  = np.sqrt(1.0 - _E2 * s2)
    M = _A * (1.0 - _E2) / (w ** 3)
    N = _A / w
    return M, N


# --------------------------------------------------------------------------- #
# Pixel -> GPS
# --------------------------------------------------------------------------- #

def pixels_to_gps(x_px, y_px, lat0, lon0,
                  image_size_px: float = 1024.0,
                  extent_m: float = 204.8,
                  grid_convergence_deg: float = 0.0):
    """
    Convertit un clic en pixels sur le crop en coordonnee WGS84.

    Parameters
    ----------
    x_px, y_px : float ou array
        Position du clic dans l'image. Origine en haut a gauche, x vers la
        droite (est), y vers le bas (sud). Coordonnees continues : le centre
        geometrique vaut image_size_px / 2.
    lat0, lon0 : float
        Coordonnee ANFR portee par le centre du crop, en degres WGS84.
    image_size_px : float
        Cote du crop en pixels.
    extent_m : float
        Cote du crop en metres AU SOL. Attention : si le crop a ete decoupe en
        EPSG:3857, ce n'est PAS la largeur de la bbox demandee (cf.
        `ground_extent_from_mercator`).
    grid_convergence_deg : float
        Angle entre le haut de l'image et le NORD GEOGRAPHIQUE, positif vers
        l'est. Vaut 0 pour un crop EPSG:3857 (les meridiens y sont verticaux).
        Pour un crop EPSG:2154, passer `lambert93_convergence(lon0)` : le haut
        de l'image suit alors le nord de la GRILLE Lambert, qui s'ecarte du nord
        geographique de ~1.2 deg a l'ouest de la Somme (soit 2 m au coin du crop).

    Returns
    -------
    (lat, lon) en degres WGS84, meme forme que les entrees.
    """
    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)

    gsd = extent_m / image_size_px          # taille du pixel au sol (m/px)
    c   = image_size_px / 2.0               # centre geometrique du crop

    # 1) pixels -> deplacement metrique dans le plan tangent local
    d_east  = (x_px - c) * gsd              # vers l'est  : x croissant
    d_north = (c - y_px) * gsd              # vers le nord : y DECROISSANT

    # 1bis) rotation du repere image vers le repere nord geographique
    if grid_convergence_deg:
        g = np.radians(grid_convergence_deg)
        d_east, d_north = (d_east * np.cos(g) + d_north * np.sin(g),
                           -d_east * np.sin(g) + d_north * np.cos(g))

    lat0_rad = np.radians(lat0)
    M, N = _radii(lat0_rad)

    # 2) metres -> degres
    #    dlat = d_north / M          (arc de meridien)
    #    dlon = d_east / (N cos lat) (arc de parallele)
    dlat = d_north / M

    #    Raffinement : le rayon du parallele est evalue a la latitude MOYENNE
    #    du trajet, et non a celle du depart. Sur 100 m l'effet est
    #    millimetrique, mais c'est gratuit et rend la formule symetrique
    #    (aller-retour exact).
    lat_mid = lat0_rad + dlat / 2.0
    _, N_mid = _radii(lat_mid)
    dlon = d_east / (N_mid * np.cos(lat_mid))

    lat = np.degrees(lat0_rad + dlat)
    lon = np.degrees(np.radians(lon0) + dlon)

    if lat.ndim == 0:
        return float(lat), float(lon)
    return lat, lon


# --------------------------------------------------------------------------- #
# GPS -> pixel  (reciproque exacte ; indispensable pour l'affichage)
# --------------------------------------------------------------------------- #

def gps_to_pixels(lat, lon, lat0, lon0,
                  image_size_px: float = 1024.0,
                  extent_m: float = 204.8,
                  grid_convergence_deg: float = 0.0):
    """
    Reciproque de pixels_to_gps : place une coordonnee WGS84 sur le crop.
    Sert a superposer les detections du modele et les points ANFR voisins.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    gsd = extent_m / image_size_px
    c   = image_size_px / 2.0

    lat0_rad = np.radians(lat0)
    dlat = np.radians(lat) - lat0_rad
    dlon = np.radians(lon) - np.radians(lon0)

    M, _ = _radii(lat0_rad)
    lat_mid = lat0_rad + dlat / 2.0
    _, N_mid = _radii(lat_mid)

    d_north = dlat * M
    d_east  = dlon * N_mid * np.cos(lat_mid)

    if grid_convergence_deg:          # repere nord geographique -> repere image
        g = np.radians(grid_convergence_deg)
        d_east, d_north = (d_east * np.cos(g) - d_north * np.sin(g),
                           d_east * np.sin(g) + d_north * np.cos(g))

    x_px = c + d_east / gsd
    y_px = c - d_north / gsd

    if x_px.ndim == 0:
        return float(x_px), float(y_px)
    return x_px, y_px


# --------------------------------------------------------------------------- #
# Garde-fou : quelle est la VRAIE emprise au sol du crop ?
# --------------------------------------------------------------------------- #

def ground_extent_from_mercator(bbox_span_units: float, lat: float) -> float:
    """
    Si le crop a ete demande a un WMS en EPSG:3857 (Web Mercator, le CRS par
    defaut de data.geopf.fr), la largeur de bbox n'est PAS une distance au sol.

    Web Mercator dilate les distances d'un facteur 1 / cos(lat). A la latitude
    de la Somme (~50 deg) ce facteur vaut ~1.55 : une bbox de 204.8 unites ne
    couvre que ~132 m au sol, soit une gsd de 12.9 cm/px et non 20 cm/px.

    Ignorer cette correction produit une erreur de position d'environ 36 %,
    soit jusqu'a 37 m au coin du crop.
    """
    return bbox_span_units * float(np.cos(np.radians(lat)))


def mercator_scale(lat: float) -> float:
    """Unites projetees EPSG:3857 par metre au sol a cette latitude."""
    return 1.0 / float(np.cos(np.radians(lat)))


# Constante conique du Lambert-93 : n = sin(latitude standard equivalente).
# Derivee des deux paralleles d'automecoicite 44 deg et 49 deg.
_LAMBERT93_N = 0.7256077650532695


def lambert93_convergence(lon: float, lat: float | None = None) -> float:
    """
    Convergence des meridiens du Lambert-93, en degres, positive vers l'est :
    angle entre le nord de la GRILLE et le nord GEOGRAPHIQUE.

        gamma = n * (lon - lon0)      avec n = 0.72561 et lon0 = 3 deg

    A passer comme `grid_convergence_deg` quand les crops ont ete decoupes en
    EPSG:2154. Sur l'emprise du jeu (lon 1.40 a 3.19) gamma va de -1.16 a
    +0.14 deg : ignore, cela deplace un coin de crop de pres de 2 m.
    """
    return _LAMBERT93_N * (float(lon) - 3.0)


# --------------------------------------------------------------------------- #
# Note sur geopy
# --------------------------------------------------------------------------- #
#
# geopy.distance.distance(metres).destination(Point(lat0, lon0), bearing)
# resout le meme probleme via la geodesique de Karney (geographiclib) et donne
# une precision sub-millimetrique. Deux raisons de ne pas l'utiliser ici :
#
#   1. Il faut d'abord convertir (d_east, d_north) en (distance, azimut), donc
#      un atan2 + un hypot, puis geopy refait le chemin inverse en interne.
#   2. C'est environ deux ordres de grandeur plus lent et non vectorise : pour
#      reprojeter des milliers de detections dans le dashboard, cela se voit.
#
# Sur une portee de +/- 145 m (la demi-diagonale du crop), l'ecart entre la
# formule ci-dessus et la geodesique exacte est inferieur au millimetre, soit
# 200 000 fois plus petit que l'incertitude `offset_m` du referentiel ANFR.
# Utiliser geopy ici, c'est mesurer au micrometre une planche coupee a la hache.
