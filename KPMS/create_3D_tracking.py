from pathlib import Path
from py3r.behaviour.tracking.tracking_mv import TrackingMV

script_dir = Path(__file__).parent
collection_dir = script_dir / "collection"
output_dir = script_dir / "triangulated_3d"

fps = 30.0
invert_z = True

output_dir.mkdir(exist_ok=True)

folders = sorted([f for f in collection_dir.iterdir() if f.is_dir()], key=lambda x: x.name)

for folder in folders:
    left_file = folder / "left.csv"
    right_file = folder / "right.csv"
    calib_file = folder / "calibration.json"

    if not (left_file.exists() and right_file.exists() and calib_file.exists()):
        print(f"[SKIP] {folder.name}: missing required files")
        continue

    print(f"Processing folder {folder.name}")

    mv = TrackingMV.from_yolo3r(
        folder_path=folder,
        handle=folder.name,
        fps=fps
    )

    tracking_3d = mv.stereo_triangulate(invert_z=invert_z)

    print(f"3D shape for folder {folder.name}: {tracking_3d.data.shape}")

    out_file = output_dir / f"{folder.name}.csv"
    tracking_3d.data.reset_index().to_csv(out_file, index=False)

    print(f"Saved {out_file}")

print("Triangulation finished")

