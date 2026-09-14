#!/usr/bin/env python3
"""
starter_antennes.py
====================

FrHack! 2026 — Challenge #4 : Détection d'antennes sur images satellites/aériennes
ANFR x ISEP

Objectif
--------
Ce script fournit une base de travail pour détecter des antennes (pylônes,
mâts, antennes relais...) sur des images aériennes/satellites à très haute
résolution (ex: IGN BDORTHO), à l'aide d'un modèle de détection d'objets
YOLOv8 (Ultralytics).

Il couvre 3 modes :
    1. train      -> entraîner (ou fine-tuner) un modèle YOLOv8 sur le dataset
    2. predict     -> détecter des antennes sur une image ou un dossier d'images
    3. eval        -> évaluer un modèle sur un jeu de validation annoté
    4. anomalies   -> détecter les sites non déclarés (Niveau 3)

Structure de données attendue
------------------------------
data/
├── images/
│   ├── train/*.jpg (ou .png/.tif)
│   ├── val/*.jpg
│   └── test/*.jpg
├── labels/
│   ├── train/*.txt   (format YOLO: class x_center y_center width height, normalisés [0,1])
│   └── val/*.txt
└── antennes.yaml      (fichier de config du dataset, généré automatiquement si absent)

Installation rapide
--------------------
    pip install ultralytics opencv-python pillow numpy matplotlib

Exemples d'utilisation
-----------------------
    # Entraîner un modèle à partir d'un modèle pré-entraîné YOLOv8 nano
    python starter_antennes.py train --data data --epochs 50 --imgsz 640

    # Détecter les antennes sur une image
    python starter_antennes.py predict --weights runs/detect/train/weights/best.pt --source image.jpg

    # Évaluer les performances (mAP, précision, rappel)
    python starter_antennes.py eval --weights best.pt --data data
    
    # Scanner les anomalies et exporter un CSV (Niveau 3)
    python starter_antennes.py anomalies --weights best.pt --source data/images/test
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CLASS_NAMES = ["antenne"]  # ajoutez d'autres classes si besoin: "pylone", "parabole", ...

DEFAULT_YAML_TEMPLATE = """\
# Fichier de configuration du dataset - généré par starter_antennes.py
path: {data_root}
train: images/train
val: images/val
test: images/test

nc: {nc}
names: {names}
"""


# --------------------------------------------------------------------------- #
# Préparation des données
# --------------------------------------------------------------------------- #

def ensure_dataset_yaml(data_root: Path) -> Path:
    """Crée (si absent) le fichier antennes.yaml requis par YOLOv8."""
    yaml_path = data_root / "antennes.yaml"
    if not yaml_path.exists():
        content = DEFAULT_YAML_TEMPLATE.format(
            data_root=str(data_root.resolve()),
            nc=len(CLASS_NAMES),
            names=CLASS_NAMES,
        )
        yaml_path.write_text(content, encoding="utf-8")
        print(f"[INFO] Fichier de config créé : {yaml_path}")
    return yaml_path


def tile_image(image_path: Path, tile_size: int = 1024, overlap: int = 128, out_dir: Path = None):
    """
    Découpe une grande image satellite en tuiles avec recouvrement.
    Retourne la liste des chemins de tuiles générées.
    """
    if cv2 is None:
        raise ImportError("opencv-python est requis pour tile_image (pip install opencv-python)")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    h, w = img.shape[:2]
    out_dir = out_dir or image_path.parent / f"{image_path.stem}_tiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    step = tile_size - overlap
    tiles_info = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = img[y:y_end, x:x_end]
            if tile.shape[0] < 10 or tile.shape[1] < 10:
                continue
            tile_path = out_dir / f"{image_path.stem}_x{x}_y{y}.jpg"
            cv2.imwrite(str(tile_path), tile)
            tiles_info.append({"path": tile_path, "x_offset": x, "y_offset": y})

    print(f"[INFO] {len(tiles_info)} tuiles générées dans {out_dir}")
    return tiles_info


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #

def train(args):
    if YOLO is None:
        raise ImportError("ultralytics est requis (pip install ultralytics)")

    data_root = Path(args.data)
    yaml_path = ensure_dataset_yaml(data_root)

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project="runs/detect",
        name="antennes",
        exist_ok=True,
    )
    print("[INFO] Entraînement terminé. Poids sauvegardés dans runs/detect/antennes/weights/")


# --------------------------------------------------------------------------- #
# Prédiction / Détection
# --------------------------------------------------------------------------- #

def predict(args):
    if YOLO is None:
        raise ImportError("ultralytics est requis (pip install ultralytics)")

    model = YOLO(args.weights)
    source = Path(args.source)

    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project="runs/detect",
        name="predict_antennes",
        exist_ok=True,
    )

    detections_summary = []
    for r in results:
        img_path = r.path
        boxes = r.boxes
        detections = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy.squeeze().tolist()
                detections.append({
                    "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                    "confidence": round(conf, 3),
                    "bbox_xyxy": [round(c, 1) for c in xyxy],
                })
        detections_summary.append({"image": img_path, "detections": detections})
        print(f"[INFO] {img_path} -> {len(detections)} antenne(s) détectée(s)")

    out_json = Path("runs/detect/predict_antennes/detections.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(detections_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Résultats détaillés exportés dans {out_json}")

    return detections_summary


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #

def evaluate(args):
    if YOLO is None:
        raise ImportError("ultralytics est requis (pip install ultralytics)")

    data_root = Path(args.data)
    yaml_path = ensure_dataset_yaml(data_root)

    model = YOLO(args.weights)
    metrics = model.val(data=str(yaml_path), imgsz=args.imgsz)

    print("[INFO] Résultats d'évaluation :")
    print(f"  mAP50    : {metrics.box.map50:.3f}")
    print(f"  mAP50-95 : {metrics.box.map:.3f}")
    print(f"  Précision moyenne : {metrics.box.mp:.3f}")
    print(f"  Rappel moyen      : {metrics.box.mr:.3f}")

    return metrics


# --------------------------------------------------------------------------- #
# Détection d'anomalies (Niveau 3) — candidats non déclarés
# --------------------------------------------------------------------------- #

def detect_anomalies(weights_path: str, test_images_dir: str, conf_threshold: float = 0.70,
                     imgsz: int = 1280, output_csv: str = "alertes_illegales.csv"):
    """
    Scanne un dossier d'images (typiquement data/images/test/) avec le modèle YOLO
    entraîné et exporte dans un CSV les détections dont la confiance dépasse
    `conf_threshold`. Ces détections constituent des CANDIDATS à examiner :
    une alerte n'est PAS une preuve d'antenne illégale (faux positifs, décalage
    temporel, support sans radio, etc.).
    """
    if YOLO is None:
        raise ImportError("ultralytics est requis (pip install ultralytics)")

    import csv

    weights_path = Path(weights_path)
    test_dir = Path(test_images_dir)

    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Dossier test introuvable : {test_dir}")

    print(f"[INFO] Chargement du modèle : {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[INFO] Inférence sur : {test_dir} (conf >= {conf_threshold})")
    results = model.predict(
        source=str(test_dir),
        conf=conf_threshold,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )

    rows = []
    for r in results:
        img_name = Path(r.path).name
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            conf = float(box.conf.item())
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls.item())
            xyxy = box.xyxy.squeeze().tolist()
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            rows.append({
                "image": img_name,
                "classe": cls_name,
                "confiance": round(conf, 3),
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1),
            })

    # Tri par confiance décroissante : les alertes les plus fiables en tête
    rows.sort(key=lambda r: r["confiance"], reverse=True)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "classe", "confiance", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] {len(rows)} alerte(s) exportée(s) dans {out_path.resolve()}")
    if not rows:
        print("[INFO] Aucune détection au-dessus du seuil : aucun candidat à examiner.")

    return rows


# --------------------------------------------------------------------------- #
# Référence ANFR (optionnel, pour croiser/valider les détections)
# --------------------------------------------------------------------------- #

def load_anfr_reference(geojson_or_csv_path: Path):
    import csv

    path = Path(geojson_or_csv_path)
    if path.suffix.lower() == ".geojson" or path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        points = [
            feature["geometry"]["coordinates"]
            for feature in data.get("features", [])
            if feature.get("geometry", {}).get("type") == "Point"
        ]
        print(f"[INFO] {len(points)} stations de référence chargées depuis {path.name}")
        return points
    elif path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"[INFO] {len(rows)} stations de référence chargées depuis {path.name}")
        return rows
    else:
        raise ValueError(f"Format non supporté : {path.suffix}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(
        description="FrHack! 2026 - Challenge #4 : détection d'antennes sur images satellites"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Entraîner le modèle YOLOv8")
    p_train.add_argument("--data", type=str, default="data", help="Dossier racine du dataset")
    p_train.add_argument("--model", type=str, default="yolov8n.pt", help="Modèle de base (yolov8n/s/m.pt)")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--patience", type=int, default=15, help="Early stopping")
    p_train.set_defaults(func=train)

    # predict
    p_pred = subparsers.add_parser("predict", help="Détecter des antennes sur une image / un dossier")
    p_pred.add_argument("--weights", type=str, required=True, help="Chemin vers les poids entraînés (.pt)")
    p_pred.add_argument("--source", type=str, required=True, help="Image ou dossier d'images")
    p_pred.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance")
    p_pred.add_argument("--imgsz", type=int, default=1280, help="Taille d'entrée (élevée pour petits objets)")
    p_pred.set_defaults(func=predict)

    # eval
    p_eval = subparsers.add_parser("eval", help="Évaluer le modèle sur le jeu de validation")
    p_eval.add_argument("--weights", type=str, required=True)
    p_eval.add_argument("--data", type=str, default="data")
    p_eval.add_argument("--imgsz", type=int, default=640)
    p_eval.set_defaults(func=evaluate)

    # anomalies (Niveau 3)
    p_ano = subparsers.add_parser(
        "anomalies",
        help="Scanner un dossier test et exporter les candidats (conf > seuil) dans un CSV"
    )
    p_ano.add_argument("--weights", type=str, required=True, help="Chemin vers les poids entraînés (.pt)")
    p_ano.add_argument("--source", type=str, default="data/images/test",
                       help="Dossier d'images à scanner (zones douteuses)")
    p_ano.add_argument("--conf", type=float, default=0.70,
                       help="Seuil de confiance pour déclencher une alerte")
    p_ano.add_argument("--imgsz", type=int, default=1280)
    p_ano.add_argument("--output", type=str, default="alertes_illegales.csv",
                       help="Chemin du CSV de sortie")
    p_ano.set_defaults(func=lambda a: detect_anomalies(a.weights, a.source, a.conf, a.imgsz, a.output))

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()