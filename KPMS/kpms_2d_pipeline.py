from pathlib import Path
import numpy as np
import keypoint_moseq as kpms


# =========================================================
# PATHS
# =========================================================
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "kpms_inputs_2d"
PROJECT_DIR = ROOT / "moseq_project"

MODEL_NAME = "mouse_model_2d"
NUM_ITERS = 300


# =========================================================
# LOAD CONFIG FROM config.yml
# =========================================================
if not PROJECT_DIR.exists():
    raise RuntimeError(
        f"Project directory does not exist: {PROJECT_DIR}\n"
        f"Create it first with kpms.setup_project(...) or use your existing project."
    )

config = kpms.load_config(PROJECT_DIR)


# =========================================================
# LOAD EXPORTED INPUTS
# =========================================================
coordinates = {}
confidences = {}

recording_dirs = sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()])

if not recording_dirs:
    raise RuntimeError(f"No recording folders found in {INPUT_DIR}")

print(f"Found {len(recording_dirs)} recordings in {INPUT_DIR}")

for rec_dir in recording_dirs:
    video_id = rec_dir.name

    coord_path = rec_dir / "coordinates.npy"
    conf_path = rec_dir / "confidences.npy"

    if not coord_path.exists():
        print(f"Skipping {video_id}: missing coordinates.npy")
        continue

    coords = np.load(coord_path)
    coordinates[video_id] = coords

    if conf_path.exists():
        conf = np.load(conf_path)
    else:
        conf = np.ones(coords.shape[:2], dtype=float)

    confidences[video_id] = conf

    print(
        f"Loaded {video_id}: "
        f"coordinates {coords.shape}, "
        f"confidences {conf.shape}"
    )

if len(coordinates) == 0:
    raise RuntimeError("No recordings loaded into coordinates dictionary.")


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
