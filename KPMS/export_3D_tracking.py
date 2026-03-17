from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# paths
# =========================
script_dir = Path(__file__).parent
triangulated_dir = script_dir / "triangulated_3d"
output_dir = script_dir / "kpms_inputs_3d"
output_dir.mkdir(exist_ok=True)

# expected keypoints in desired order
bodyparts = [
    "nose", "headcentre", "earl", "earr", "neck",
    "bcl", "bcr", "bodycentre", "hipl", "hipr",
    "tailbase", "tailcentre", "tailtip"
]

csv_files = sorted(triangulated_dir.glob("*.csv"), key=lambda p: p.stem)

print(f"Found {len(csv_files)} triangulated CSV files")

coordinates = {}
confidences = {}
exported = 0
skipped = 0

for csv_file in csv_files:
    recording_name = csv_file.stem
    print(f"\nProcessing: {csv_file.name}")

    df = pd.read_csv(csv_file)

    # remove accidental extra unnamed index column if present
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # find all bodypart prefixes that end in .x/.y/.z
    xyz_prefixes = []
    for col in df.columns:
        if col.endswith(".x") or col.endswith(".y") or col.endswith(".z"):
            xyz_prefixes.append(".".join(col.split(".")[:-1]))

    xyz_prefixes = sorted(set(xyz_prefixes))

    part_to_prefix = {}
    for prefix in xyz_prefixes:
        short_name = prefix.split(".")[-1]
        if short_name in bodyparts:
            part_to_prefix[short_name] = prefix

    missing = [bp for bp in bodyparts if bp not in part_to_prefix]
    if missing:
        print(f"  Skipping {recording_name}: missing {missing}")
        skipped += 1
        continue

    coords = np.stack([
        df[
            [
                f"{part_to_prefix[bp]}.x",
                f"{part_to_prefix[bp]}.y",
                f"{part_to_prefix[bp]}.z",
            ]
        ].to_numpy(dtype=float)
        for bp in bodyparts
    ], axis=1)

    conf = np.stack([
        df[f"{part_to_prefix[bp]}.likelihood"].to_numpy(dtype=float)
        for bp in bodyparts
    ], axis=1)

    coordinates[recording_name] = coords
    confidences[recording_name] = conf

    print(f"  OK: coords {coords.shape}, conf {conf.shape}")
    exported += 1

np.savez(output_dir / "coordinates_3d.npz", **coordinates)
np.savez(output_dir / "confidences_3d.npz", **confidences)
np.save(output_dir / "bodyparts.npy", np.array(bodyparts, dtype=object))

print("\n======================================")
print("Finished")
print(f"Exported: {exported}")
print(f"Skipped: {skipped}")
print(f"Output: {output_dir}")
print("======================================")