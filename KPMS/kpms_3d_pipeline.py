from pathlib import Path
import numpy as np
import keypoint_moseq as kpms


# =========================================================
# PATHS
# =========================================================
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "kpms_inputs_3d"
PROJECT_DIR = ROOT / "moseq_project"

MODEL_NAME = "mouse_model_3d"
NUM_ITERS = 300
config = kpms.load_config(PROJECT_DIR)

coord_bundle = INPUT_DIR / "coordinates_3d.npz"
conf_bundle = INPUT_DIR / "confidences_3d.npz"

if not coord_bundle.exists():
    raise RuntimeError(f"Missing bundled coordinates file: {coord_bundle}")

if not conf_bundle.exists():
    raise RuntimeError(f"Missing bundled confidences file: {conf_bundle}")

coord_data = np.load(coord_bundle, allow_pickle=True)
conf_data = np.load(conf_bundle, allow_pickle=True)

coordinates = {key: coord_data[key] for key in coord_data.files}
confidences = {key: conf_data[key] for key in conf_data.files}

if len(coordinates) == 0:
    raise RuntimeError("No recordings found in bundled coordinates file.")

print(f"Loaded {len(coordinates)} recordings")

for video_id in sorted(coordinates.keys()):
    coords = coordinates[video_id]

    if video_id in confidences:
        conf = confidences[video_id]
    else:
        conf = np.ones(coords.shape[:2], dtype=float)
        confidences[video_id] = conf

    print(
        f"Loaded {video_id}: "
        f"coordinates {coords.shape}, "
        f"confidences {conf.shape}"
    )


# =========================================================
# FORMAT DATA
# =========================================================
data, metadata = kpms.format_data(
    coordinates,
    confidences,
    **config
)


# =========================================================
# FIT PCA
# =========================================================
pca = kpms.fit_pca(**data, **config)
kpms.save_pca(pca, PROJECT_DIR)


# =========================================================
# INIT + FIT MODEL
# =========================================================
model = kpms.init_model(data, pca=pca, **config)

fit_output = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir=PROJECT_DIR,
    model_name=MODEL_NAME,
    num_iters=NUM_ITERS,
    **config
)

if isinstance(fit_output, tuple):
    model = fit_output[0]
else:
    model = fit_output


# =========================================================
# EXTRACT RESULTS
# =========================================================
results = kpms.extract_results(
    model,
    metadata,
    project_dir=PROJECT_DIR,
    model_name=MODEL_NAME,
    save_results=True,
)

kpms.save_results_as_csv(
    results,
    project_dir=PROJECT_DIR,
    model_name=MODEL_NAME,
)

print("\nDone.")