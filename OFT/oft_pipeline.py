import json
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3r.behaviour as p3b

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

warnings.filterwarnings("ignore", message="distance has not been calibrated")
warnings.filterwarnings("ignore", message="tracking data have not been smoothed")

# ==================================================
# Settings
# ==================================================
SKIP_HEAVY_VIZ = os.environ.get("CI", "").lower() in ("true", "1", "yes")
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "tracking"
TAGS_CSV = ROOT_DIR / "tags.csv"
OUT_DIR = Path(os.environ.get("NB_OUT_DIR", ROOT_DIR / "_artifacts"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FPS = 30
N_CLUSTERS = 25

# ==================================================
# Helpers
# ==================================================
def print_section(title):
    print("=" * 60)
    print(title)
    print("=" * 60)

# ==================================================
# LOAD TRACKING
# ==================================================
print_section("LOAD TRACKING")
tracking_dict = {}

csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stem)

for csv_path in csv_files:
    video_handle = csv_path.stem   # filename without .csv

    tracking_dict[video_handle] = p3b.Tracking.from_yolo3r(
        filepath=str(csv_path),
        handle=video_handle,
        fps=FPS,
    )

    rename_map = {}
    valid_suffixes = {"x", "y", "conf", "likelihood"}

    for col in tracking_dict[video_handle].data.columns:
        if col in {"frame_index", "max_dim.x", "max_dim.y"}:
            continue

        parts = col.split(".")
        if len(parts) >= 2 and parts[-1] in valid_suffixes:
            suffix = "likelihood" if parts[-1] == "conf" else parts[-1]
            rename_map[col] = f"{parts[-2]}.{suffix}"

    if rename_map:
        tracking_dict[video_handle].data = tracking_dict[video_handle].data.rename(columns=rename_map)

    if not isinstance(getattr(tracking_dict[video_handle], "meta", None), dict):
        tracking_dict[video_handle].meta = {}
    tracking_dict[video_handle].meta.setdefault("fps", FPS)

tracking_collection = p3b.TrackingCollection(tracking_dict)
tc = tracking_collection
print(f"Initial videos loaded: {len(tracking_dict)}")
print(tc)

if len(tc) == 0:
    raise ValueError(
        f"No CSV files were loaded from {DATA_DIR}. "
        f"Expected files like session1.csv, mouseA.csv, test_03.csv."
    )

# ==================================================
# PREPROCESS
# ==================================================
print_section("PREPROCESS")
tc.each.filter_likelihood(threshold=0.9)
tc.each.interpolate(limit=5)
tc.each.smooth_all(window=3, method="mean")
tc.each.rescale_by_known_distance(
    point1="tl",
    point2="br",
    distance_in_metres=0.64,
)

# ==================================================
# QC PLOTS
# ==================================================
print_section("QC PLOTS")
trajectories = ["bodycentre"]
static = ["tl", "tr", "bl", "br"]
lines = [
    ("tr", "tl"),
    ("tl", "bl"),
    ("bl", "br"),
    ("br", "tr"),
]
first_key = list(tc.keys())[0]
tc[first_key].plot(
    trajectories=trajectories,
    static=static,
    lines=lines,
    show=True,
)

# ==================================================
# FEATURE COLLECTION
# ==================================================
print_section("FEATURE COLLECTION")
fc = p3b.FeaturesCollection.from_tracking_collection(tc)
ordered_oft_corners = ["tl", "tr", "br", "bl"]

# ==================================================
# BOUNDARY FEATURES
# ==================================================
print_section("BOUNDARY FEATURES")
centre_boundary = fc.each.define_static_boundary(
    ordered_oft_corners,
    scale_dim1=0.5,
    scale_dim2=0.5,
    name="centre",
)
in_centre = fc.each.within_boundary(point="bodycentre", boundary=centre_boundary)
in_centre_by_name = fc.each.within_boundary(point="bodycentre", boundary="centre")

for handle in fc.keys():
    assert in_centre[handle].equals(in_centre_by_name[handle])

in_centre.store()

_ = fc.each.define_static_boundary(
    ordered_oft_corners,
    scale_dim1=0.8,
    scale_dim2=0.8,
    name="not_periphery",
)

_ = fc.each.define_static_boundary(
    ordered_oft_corners,
    name="oft",
)

(
    fc.each.within_boundary("bodycentre", "oft")
    & (~fc.each.within_boundary("bodycentre", "not_periphery"))
).store("in_periphery")

in_corners = {}
for c in ordered_oft_corners:
    _ = fc.each.define_static_boundary(
        ordered_oft_corners,
        scale_dim1=0.2,
        scale_dim2=0.2,
        name=f"{c}_corner",
        anchor=c,
    )
    in_corners[c] = fc.each.within_boundary("bodycentre", boundary=f"{c}_corner")

(in_corners["tl"] | in_corners["tr"] | in_corners["bl"] | in_corners["br"]).store("in_corner")
fc.each.compose_state_from_booleans(in_corners).store("corner_state")

first_fc_key = list(fc.keys())[0]
non_bfa_feats = fc[first_fc_key].data.columns

dist_change = fc.each.distance_change("bodycentre")
dist_change_in_centre = in_centre.astype("Int64") * dist_change
dist_change_in_centre.store(name="dist_change_bodycentre_in_centre")

# ==================================================
# KINEMATIC FEATURES
# ==================================================
print_section("KINEMATIC FEATURES")
for pt in ["nose", "neck", "earr", "earl", "bodycentre", "hipl", "hipr", "tailbase"]:
    fc.each.speed(pt).store()
for basepoint, pointdirection1, pointdirection2 in [
    ("tailbase", "hipr", "hipl"),
    ("bodycentre", "tailbase", "neck"),
    ("neck", "bodycentre", "headcentre"),
    ("headcentre", "earr", "earl"),
]:
    fc.each.azimuth_deviation(basepoint, pointdirection1, pointdirection2).store()
for p1, p2 in [
    ("nose", "headcentre"),
    ("neck", "headcentre"),
    ("neck", "bodycentre"),
    ("bcr", "bodycentre"),
    ("bcl", "bodycentre"),
    ("tailbase", "bodycentre"),
    ("tailbase", "hipr"),
    ("tailbase", "hipl"),
    ("bcr", "hipr"),
    ("bcl", "hipl"),
    ("bcl", "earl"),
    ("bcr", "earr"),
    ("nose", "earr"),
    ("nose", "earl"),
]:
    fc.each.distance_between(p1, p2).store()

# ==================================================
# DYNAMIC BODY FEATURES
# ==================================================
print_section("DYNAMIC BODY FEATURES")
DYNAMIC_BODY_BOUNDARIES = [
    ("mouse_rear", ["tailbase", "hipr", "hipl"]),
    ("mouse_mid", ["hipr", "hipl", "bcl", "bcr"]),
    ("mouse_front", ["bcr", "earr", "earl", "bcl"]),
    ("mouse_face", ["earr", "nose", "earl"]),
]
for boundary_name, boundary_points in DYNAMIC_BODY_BOUNDARIES:
    fc.each.define_dynamic_boundary(boundary_points, name=boundary_name)
    fc.each.area_of_boundary(boundary_name).store()
for pt in ["nose", "neck", "bodycentre", "tailbase"]:
    fc.each.distance_to_boundary(pt, "oft").store()

# ==================================================
# CLUSTERING
# ==================================================
print_section("CLUSTERING")
cluster_features = list(set(fc[first_fc_key].data.columns) - set(non_bfa_feats))
offset = list(np.arange(-15, 16, 1))
embedding_dict = {f: offset for f in cluster_features}
cluster_labels, centroids, _ = fc.cluster_embedding_stream(
    embedding_dict=embedding_dict,
    n_clusters=N_CLUSTERS,
)
cluster_labels.store(f"kmeans_{N_CLUSTERS}", overwrite=True)
fc.save(f"{OUT_DIR}/features", data_format="csv", overwrite=True)

# ==================================================
# SUMMARY
# ==================================================
print_section("SUMMARY")
sc = p3b.SummaryCollection.from_features_collection(fc)
sc.each.total_distance("bodycentre").store()
sc.each.time_true("within_boundary_static_bodycentre_in_centre").store("time_in_centre")
sc.each.sum_column("dist_change_bodycentre_in_centre").store(name="distance_moved_in_centre")
sc.each.by_state(
    "corner_state",
    all_states=ordered_oft_corners,
).mean_column("speed_of_bodycentre_in_xy").store("mean_speed_corners")
sc.each.by_state(
    f"kmeans_{N_CLUSTERS}",
    all_states=list(range(min(10, N_CLUSTERS))),
).mean_column("speed_of_bodycentre_in_xy").store("mean_speed_bodycentre_by_kmeans")

summary_df, series_dfs = sc.to_df(include_tags=True, series="separate")
summary_df.to_csv(OUT_DIR / "OFT_results.csv", index=False)
display(summary_df.head())
for key, val in series_dfs.items():
    print(key)
    display(val.head())

# ==================================================
# VISUALISATION
# ==================================================
print_section("VISUALISATION")
sc.each.time_in_state(f"kmeans_{N_CLUSTERS}").store("time_in_cluster")
sc.snsstrip(
    "time_in_cluster",
    random_state=42,
    show=True,
    savedir=OUT_DIR,
)
sc.snsbar(
    "time_in_cluster",
    show=True,
    savedir=OUT_DIR,
)
sc.snssuperplot(
    "time_in_cluster",
    random_state=42,
    show=True,
    savedir=OUT_DIR,
)

# ==================================================
# DONE
# ==================================================
print_section("DONE")
print(f"Outputs saved to: {OUT_DIR}")