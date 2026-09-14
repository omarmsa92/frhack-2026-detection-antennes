# -*- coding: utf-8 -*-
"""
app.py — FR Hack! 2026, Challenge 4 (ANFR x ISEP)
Dashboard de correction manuelle des coordonnées ANFR.

Pourquoi une grille de points invisibles
----------------------------------------
`st.plotly_chart(on_select="rerun", selection_mode=["points"])` ne renvoie QUE
des points appartenant à une trace. Une image (`go.Image`) n'en contient aucun :
cliquer dessus produit une sélection vide — c'est pour ça que rien ne se
déclenchait.

On superpose donc à l'image une trace `Scattergl` de points quasi transparents,
espacés de GRID_STEP pixels. Plotly renvoie le point le plus proche du clic,
soit le pixel à ±GRID_STEP/2. Les deux champs sous l'image permettent ensuite
de descendre au pixel exact.

Le mode rectangle (drag) renvoie en plus des coordonnées continues exactes ET
une emprise — c'est le format dont a besoin le jeu de validation annoté à la main.

Prérequis : streamlit >= 1.35 (pour `on_select`), plotly >= 5.20.
"""

import base64
import io
import os

import numpy as np
import pandas as pd
import streamlit as st
import folium
import plotly.graph_objects as go
from streamlit_folium import st_folium
from PIL import Image

from geoloc import pixels_to_gps, lambert93_convergence, ground_extent_from_mercator

st.set_page_config(page_title="FrHack! 2026 - Radar ANFR", layout="wide")
st.title("🛰️ FrHack! 2026 — Correction manuelle des coordonnées ANFR")

CORRECTIONS_PATH = "coordonnees_corrigees.csv"
DATA_ROOT        = "data2"                       # dataset reconstruit a 70.1 m
MANIFEST_PATH    = f"{DATA_ROOT}/dataset_manifest.csv"
VERDICTS = ["ok", "boite_a_corriger", "rien_de_visible", "mauvais_support"]

# Emprise mesuree par calibrate_echelle.py (correlation de phase sur 5 paires
# de sites voisins, dispersion 0,05 %). Ce n'est PAS une supposition.
GSD_MESUREE = 0.0684


# --------------------------------------------------------------------------- #
# Sidebar — géoréférencement
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Géoréférencement")

    crs_mode = st.radio(
        "Échelle des crops",
        ["Résolution mesurée", "EPSG:3857 (Web Mercator)", "EPSG:2154 (Lambert-93)"],
        index=0,
        help="« Résolution mesurée » = valeur obtenue par calibrate_echelle.py. "
             "Les deux autres modes ne servent qu'à rejouer les hypothèses écartées.",
    )

    if crs_mode.startswith("Résolution"):
        pixel_size_manual = st.number_input(
            "Résolution (m / pixel)", 0.01, 5.0, GSD_MESUREE, 0.0001, format="%.4f",
        )
        st.success(f"Emprise **{pixel_size_manual * 1024:.1f} m** pour 1024 px.\n\n"
                   f"Mesurée par corrélation de phase entre sites voisins "
                   f"(5 paires, dispersion 0,05 %).", icon="📐")
        bbox_span, use_convergence = None, False
    elif crs_mode.startswith("EPSG:3857"):
        bbox_span = st.number_input(
            "Largeur de bbox demandée (unités projetées)",
            min_value=1.0, max_value=5000.0, value=204.8, step=0.1,
            help="La valeur passée au WMS. En Web Mercator ce ne sont PAS des mètres.",
        )
        st.warning("Web Mercator dilate les distances de 1/cos(lat) ≈ **1,55** ici. "
                   "L'emprise réelle est recalculée site par site.", icon="⚠️")
        pixel_size_manual, use_convergence = None, False
    elif crs_mode.startswith("EPSG:2154"):
        bbox_span = st.number_input("Emprise au sol (m)", 1.0, 5000.0, 204.8, 0.1)
        st.info("Le haut de l'image suit le nord de la **grille** Lambert. "
                "La convergence des méridiens est appliquée automatiquement.", icon="🧭")
        pixel_size_manual, use_convergence = None, True

    GRID_STEP = st.select_slider("Pas de la grille cliquable (px)",
                                 options=[2, 4, 8, 16], value=4,
                                 help="Plus fin = clic plus précis, mais page plus lourde.")
    allow_box = st.checkbox("Sélection par rectangle (drag)", value=False,
                            help="Coordonnées exactes + emprise, pour annoter une boîte.")

    st.caption("Géodésie : plan tangent local sur ellipsoïde GRS80 (`geoloc.py`).")
    st.markdown("---")

    st.header("📁 Corrections")
    if os.path.exists(CORRECTIONS_PATH):
        corr_df = pd.read_csv(CORRECTIONS_PATH)
        st.metric("Sites corrigés", len(corr_df))
        st.dataframe(corr_df.tail(10), use_container_width=True)
        with open(CORRECTIONS_PATH, "rb") as f:
            st.download_button("⬇️ Télécharger le CSV", f, CORRECTIONS_PATH, "text/csv")
    else:
        st.info("Aucune correction enregistrée pour l'instant.")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_manifest() -> pd.DataFrame:
    if os.path.exists(MANIFEST_PATH):
        return pd.read_csv(MANIFEST_PATH)
    st.warning(f"Fichier {MANIFEST_PATH} introuvable.")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_image_b64(path: str):
    """
    Encode l'image en data-URI. Indispensable : `go.Image(z=array)` sérialiserait
    1024x1024x3 = 3,1 millions de nombres en JSON à chaque rerun.
    """
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return uri, img.size


@st.cache_data(show_spinner=False)
def click_grid(w: int, h: int, step: int):
    """Points cliquables couvrant l'image, centrés dans leur cellule."""
    xs = np.arange(step / 2.0, w, step)
    ys = np.arange(step / 2.0, h, step)
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel()


def geometry_for_site(lat0: float, lon0: float, img_w: int):
    """(emprise_m, gsd_m_par_px, convergence_deg) selon le CRS choisi."""
    if pixel_size_manual is not None:
        return img_w * pixel_size_manual, pixel_size_manual, 0.0
    if use_convergence:                                   # Lambert-93
        return bbox_span, bbox_span / img_w, lambert93_convergence(lon0)
    extent = ground_extent_from_mercator(bbox_span, lat0)  # Web Mercator
    return extent, extent / img_w, 0.0


def resolve_image_path(row):
    for p in (f"{DATA_ROOT}/images/{row.get('split', 'train')}/{row.get('image', '')}",
              row.get("image_path", ""),
              f"photos hackathon/{row.get('image', '')}"):
        if isinstance(p, str) and p and os.path.exists(p):
            return p
    return None


def parse_selection(event, grid_curve: int = 1):
    """
    Extrait (x, y, box) de l'événement Streamlit/Plotly.
    `box` vaut None si l'utilisateur a cliqué un point au lieu de tracer un rectangle.
    """
    if not event:
        return None, None, None
    sel = event.get("selection") if isinstance(event, dict) \
        else getattr(event, "selection", None)
    if not sel:
        return None, None, None

    for b in sel.get("box") or []:
        bx, by = b.get("x") or [], b.get("y") or []
        if len(bx) >= 2 and len(by) >= 2:
            x0, x1 = sorted(float(v) for v in bx[:2])
            y0, y1 = sorted(float(v) for v in by[:2])
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0, (x0, y0, x1, y1)

    for p in sel.get("points") or []:
        if p.get("curve_number") in (None, grid_curve):
            return float(p["x"]), float(p["y"]), None
    return None, None, None


def set_pixel(px, py, box=None):
    """Source de vérité unique pour la position corrigée."""
    st.session_state.pixel = (float(px), float(py))
    st.session_state.box = box


# --------------------------------------------------------------------------- #
# État
# --------------------------------------------------------------------------- #

df = load_manifest()
for k, v in (("selected_id", None), ("pixel", None), ("box", None),
             ("last_sel", None), ("pixel_src", None), ("annotateur", "")):
    st.session_state.setdefault(k, v)

if df.empty:
    st.stop()


# --------------------------------------------------------------------------- #
# Vue d'ensemble + carte
# --------------------------------------------------------------------------- #

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📊 Dataset")
    splits = df["split"].value_counts()
    st.metric("Entraînement", int(splits.get("train", 0)))
    st.metric("Validation", int(splits.get("val", 0)))
    st.metric("Écartés (test/)", int(splits.get("test", 0)))
    if "source" in df.columns:
        for motif, n in df[df.split == "test"]["source"].value_counts().items():
            st.caption(f"· {motif} : {n}")
    if os.path.exists(CORRECTIONS_PATH):
        done = len(pd.read_csv(CORRECTIONS_PATH))
        target = 60      # val_manual_todo.csv
        st.progress(min(done / target, 1.0),
                    text=f"{done} / {target} annotés à la main")
    st.markdown("---")
    st.write("**Classes générées**")
    st.dataframe(df["class_name"].value_counts(), use_container_width=True)

with col2:
    st.subheader("🗺️ Répartition — cliquez sur un point")
    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()],
                   zoom_start=9, tiles="CartoDB positron")
    colors = {"train": "#2563eb", "val": "#16a34a", "test": "#dc2626"}
    for r in df.itertuples():
        folium.CircleMarker(
            location=[r.lat, r.lon], radius=4,
            popup=f"ID {r.id} — {r.height_m} m — {r.split}",
            color=colors.get(r.split, "gray"),
            fill=True, fill_opacity=0.75, weight=1,
        ).add_to(m)
    map_data = st_folium(m, width=900, height=500,
                         returned_objects=["last_object_clicked"])

clicked = (map_data or {}).get("last_object_clicked")
if clicked and clicked.get("lat") is not None:
    d2 = (df["lat"] - clicked["lat"]) ** 2 + (df["lon"] - clicked["lng"]) ** 2
    new_id = str(df.loc[d2.idxmin(), "id"])
    if st.session_state.selected_id != new_id:
        st.session_state.selected_id = new_id
        st.session_state.pixel = None
        st.session_state.box = None
        st.session_state.last_sel = None
        st.session_state.pixel_src = None
        st.rerun()


# --------------------------------------------------------------------------- #
# Panneau de correction
# --------------------------------------------------------------------------- #

if st.session_state.selected_id is None:
    st.info("Sélectionnez un site sur la carte pour démarrer la correction.")
    st.stop()

row = df[df["id"].astype(str) == st.session_state.selected_id].iloc[0]
img_path = resolve_image_path(row)

st.markdown("---")
st.subheader(f"📍 Site {row['id']} — split `{row['split']}`")
st.write(f"ANFR : `{row['lat']:.6f}, {row['lon']:.6f}` · hauteur {row['height_m']} m "
         f"· offset déclaré {row['offset_m']:.1f} m")

if img_path is None:
    st.warning(f"Image introuvable pour le site {row['id']} (`{row.get('image', '?')}`).")
    st.stop()

img_uri, (img_w, img_h) = load_image_b64(img_path)
extent_m, gsd, conv = geometry_for_site(float(row["lat"]), float(row["lon"]), img_w)

c_img, c_info = st.columns([2, 1])

# ------------------------------- image ------------------------------------- #
with c_img:
    st.write("**Cliquez sur l'antenne dans l'image.**")

    gx, gy = click_grid(img_w, img_h, GRID_STEP)

    fig = go.Figure()
    fig.add_trace(go.Image(source=img_uri, hoverinfo="skip"))          # curve 0
    fig.add_trace(go.Scattergl(                                        # curve 1
        x=gx, y=gy, mode="markers",
        marker=dict(size=GRID_STEP, color="rgba(0,0,0,0.01)", line=dict(width=0)),
        hovertemplate="x=%{x:.0f} · y=%{y:.0f}<extra></extra>",
        showlegend=False, name="grille",
    ))
    if st.session_state.pixel:
        px, py = st.session_state.pixel
        fig.add_trace(go.Scattergl(                                    # curve 2
            x=[px], y=[py], mode="markers",
            marker=dict(symbol="x-thin", size=22,
                        line=dict(color="#ff2b2b", width=4)),
            hoverinfo="skip", showlegend=False, name="correction",
        ))
    if st.session_state.box:
        x0, y0, x1, y1 = st.session_state.box
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="#ff2b2b", width=2), fillcolor="rgba(0,0,0,0)")

    fig.update_xaxes(constrain="domain", title="x (px)", showgrid=False)
    fig.update_yaxes(scaleanchor="x", autorange="reversed", title="y (px)", showgrid=False)
    fig.update_layout(
        dragmode="select" if allow_box else "pan",
        clickmode="event+select",
        margin=dict(l=45, r=10, t=10, b=45),
        height=640,
    )

    event = st.plotly_chart(
        fig, use_container_width=True,
        key=f"plot_{row['id']}_{GRID_STEP}_{int(allow_box)}",
        on_select="rerun",
        selection_mode=["points", "box"] if allow_box else ["points"],
    )

    nx, ny, nbox = parse_selection(event)
    if nx is not None:
        # La sélection est renvoyée à CHAQUE rerun. Sans cette garde elle
        # écraserait le réglage fin et boucherait sur st.rerun().
        sel_key = (round(nx, 2), round(ny, 2), nbox)
        if st.session_state.last_sel != sel_key:
            st.session_state.last_sel = sel_key
            set_pixel(round(nx, 2), round(ny, 2), nbox)
            st.rerun()

    if st.session_state.pixel:
        st.caption(f"Réglage fin — grille au pas de {GRID_STEP} px "
                   f"({GRID_STEP * gsd:.2f} m au sol)")
        kx, ky = f"fx_{row['id']}", f"fy_{row['id']}"

        # Un nouveau clic doit réinitialiser les deux champs : on écrit dans
        # session_state AVANT d'instancier les widgets (seule façon supportée).
        if st.session_state.pixel_src != st.session_state.pixel:
            st.session_state[kx] = float(st.session_state.pixel[0])
            st.session_state[ky] = float(st.session_state.pixel[1])
            st.session_state.pixel_src = st.session_state.pixel

        def _sync_fine():
            set_pixel(st.session_state[kx], st.session_state[ky],
                      st.session_state.box)
            st.session_state.pixel_src = st.session_state.pixel

        f1, f2 = st.columns(2)
        f1.number_input("x (px)", 0.0, float(img_w), step=1.0,
                        key=kx, on_change=_sync_fine)
        f2.number_input("y (px)", 0.0, float(img_h), step=1.0,
                        key=ky, on_change=_sync_fine)
        st.caption("Si un simple clic ne réagit pas, cochez « sélection par "
                   "rectangle » et encadrez l'antenne.")

# ------------------------------ panneau droit ------------------------------ #
with c_info:
    if not st.session_state.pixel:
        st.info("Cliquez sur l'image : le calcul s'affiche ici.")
    else:
        px_x, px_y = st.session_state.pixel
        cx, cy = img_w / 2.0, img_h / 2.0

        new_lat, new_lon = pixels_to_gps(
            px_x, px_y, float(row["lat"]), float(row["lon"]),
            image_size_px=img_w, extent_m=extent_m, grid_convergence_deg=conv,
        )

        d_east   = (px_x - cx) * gsd
        d_north  = (cy - px_y) * gsd
        offset_m = float(np.hypot(d_east, d_north))

        st.metric("Pixel cliqué", f"({px_x:.1f}, {px_y:.1f})")
        st.metric("Décalage vs ANFR", f"{offset_m:.1f} m",
                  delta=f"{offset_m - float(row['offset_m']):+.1f} m vs offset déclaré")
        st.caption(f"Est {d_east:+.1f} m · Nord {d_north:+.1f} m")
        st.caption(f"Emprise {extent_m:.1f} m · {gsd * 100:.2f} cm/px"
                   + (f" · convergence {conv:+.2f}°" if conv else ""))

        st.markdown("**Position corrigée**")
        st.code(f"{new_lat:.7f}, {new_lon:.7f}", language=None)
        st.link_button(
            "🌍 Vérifier sur Géoportail",
            f"https://www.geoportail.gouv.fr/carte?c={new_lon:.7f},{new_lat:.7f}&z=19",
            use_container_width=True,
        )

        verdict = st.selectbox("Verdict", VERDICTS, index=0,
                               help="Alimente le jeu de validation annoté à la main.")
        annotateur = st.text_input("Annoté par", value=st.session_state.annotateur)
        st.session_state.annotateur = annotateur

        if st.button("💾 Sauvegarder", type="primary", use_container_width=True):
            record = {
                "id": row["id"], "sup_id": row.get("sup_id"),
                "lat_anfr": row["lat"], "lon_anfr": row["lon"],
                # height_m est indispensable a build_val_manuel.py : sans elle
                # toutes les boites de verite terrain auraient la meme taille.
                "height_m": row.get("height_m"),
                "pixel_x": px_x, "pixel_y": px_y,
                "image_width": img_w, "image_height": img_h,
                "extent_m": round(extent_m, 3), "pixel_size_m": round(gsd, 5),
                "crs_mode": crs_mode, "convergence_deg": round(float(conv), 4),
                "lat_corr": round(float(new_lat), 7),
                "lon_corr": round(float(new_lon), 7),
                "offset_mesure_m": round(offset_m, 2),
                "offset_anfr_m": round(float(row["offset_m"]), 2),
                "verdict": verdict, "annote_par": annotateur,
                "box_px": ",".join(f"{v:.1f}" for v in st.session_state.box)
                          if st.session_state.box else "",
                "image_path": img_path,
            }
            out = pd.DataFrame([record])
            if os.path.exists(CORRECTIONS_PATH):
                prev = pd.read_csv(CORRECTIONS_PATH)
                prev = prev[prev["id"].astype(str) != str(row["id"])]
                out = pd.concat([prev, out], ignore_index=True)
            out.to_csv(CORRECTIONS_PATH, index=False)
            st.success(f"✅ Site {row['id']} enregistré — {len(out)} corrections.")
            st.session_state.pixel = None
            st.session_state.box = None
            st.session_state.last_sel = None
            st.session_state.pixel_src = None
            st.rerun()