from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COLLECTION_DIR = ROOT / "collection"
EXPORT_DIR = ROOT / "kpms_inputs_2d"

BODYPOINTS = [
    "nose",
    "headcentre",
    "earl",
    "earr",
    "neck",
    "bcl",
    "bcr",
    "bodycentre",
    "hipl",
    "hipr",
    "tailbase",
    "tailcentre",
    "tailtip",
]

EXPORT_DIR.mkdir(exist_ok=True)

def find_col(columns, suffix: str):
    matches = [c for c in columns if str(c).endswith(suffix)]
    if len(matches) == 0:
        raise KeyError(f"No column ends with '{suffix}'")
    return matches[0]

def find_optional_conf_col(columns, bp: str):
    for suffix in [f"{bp}.likelihood", f"{bp}.confidence", f"{bp}.score", f"{bp}.conf"]:
        matches = [c for c in columns if str(c).endswith(suffix)]
        if matches:
            return matches[0]
    return None

csv_files = sorted(COLLECTION_DIR.rglob("left.csv"))
print(f"Found {len(csv_files)} left-camera CSV files")

with open(EXPORT_DIR / "bodyparts.json", "w", encoding="utf-8") as f:
    json.dump(BODYPOINTS, f, indent=2)

for csv_file in csv_files:
    print(f"\nProcessing: {csv_file}")

    df = pd.read_csv(csv_file)
    df.columns = [str(col).replace(".conf", ".likelihood") for col in df.columns]

    coords_per_bp = []
    confs_per_bp = []

    for bp in BODYPOINTS:
        x_col = find_col(df.columns, f"{bp}.x")
        y_col = find_col(df.columns, f"{bp}.y")
        xy = df[[x_col, y_col]].to_numpy(dtype=float)
        coords_per_bp.append(xy)

        conf_col = find_optional_conf_col(df.columns, bp)
        if conf_col is None:
            conf = np.ones(len(df), dtype=float)
        else:
            conf = df[conf_col].to_numpy(dtype=float)

        confs_per_bp.append(conf)

    coordinates = np.stack(coords_per_bp, axis=1)   # (T, K, 2)
    confidences = np.stack(confs_per_bp, axis=1)    # (T, K)

    video_id = csv_file.parent.name

    out_dir = EXPORT_DIR / video_id
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "coordinates.npy", coordinates)
    np.save(out_dir / "confidences.npy", confidences)

    print(f"Exported {video_id}: {coordinates.shape}")