from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
ROOT = Path(__file__).resolve().parent

MOSEQ_RESULTS_DIR = ROOT / "moseq_project" / "mouse_model_2d" / "results"
LABELS_DIR = ROOT / "labels"
OUTPUT_DIR = ROOT / "label_comparison_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MOSEQ_SYLLABLE_CANDIDATES = ["syllable", "syllables", "state", "label"]
LABEL_COLUMN_CANDIDATES = ["label", "behaviour", "behavior", "class", "annotation"]
FRAME_COLUMN_CANDIDATES = ["frame", "Frame", "frames"]

NORMALIZE_LABEL_STRINGS = True
EXCLUDED_RECORDINGS = {"5"}  # skip difficult recording 5


# =========================================================
# HELPERS
# =========================================================
def find_column(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"Could not find any of {candidates}. Available columns: {list(df.columns)}")


def normalize_labels(series: pd.Series) -> pd.Series:
    if not NORMALIZE_LABEL_STRINGS:
        return series
    return series.astype(str).str.strip().str.lower()


def load_moseq_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    syll_col = find_column(df, MOSEQ_SYLLABLE_CANDIDATES)

    try:
        frame_col = find_column(df, FRAME_COLUMN_CANDIDATES)
        out = df[[frame_col, syll_col]].copy()
        out.columns = ["frame", "syllable"]
    except KeyError:
        out = pd.DataFrame({
            "frame": np.arange(len(df)),
            "syllable": df[syll_col].to_numpy()
        })

    out["frame"] = out["frame"].astype(int)
    out["syllable"] = out["syllable"].astype(int)
    return out


def load_label_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # -----------------------------------------------------
    # Case 1: label file has a single label column
    # -----------------------------------------------------
    try:
        label_col = find_column(df, LABEL_COLUMN_CANDIDATES)

        try:
            frame_col = find_column(df, FRAME_COLUMN_CANDIDATES)
            out = df[[frame_col, label_col]].copy()
            out.columns = ["frame", "true_label"]
        except KeyError:
            out = pd.DataFrame({
                "frame": np.arange(len(df)),
                "true_label": df[label_col].to_numpy()
            })

        out["frame"] = out["frame"].astype(int)
        out["true_label"] = normalize_labels(out["true_label"])
        return out

    except KeyError:
        pass

    # -----------------------------------------------------
    # Case 2: one-hot encoded columns
    # Example:
    # Unnamed: 0, background, supportedrear, unsupportedrear, grooming
    # -----------------------------------------------------
    candidate_df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")].copy()

    frame_col = None
    for cand in FRAME_COLUMN_CANDIDATES:
        matches = [c for c in candidate_df.columns if c.lower() == cand.lower()]
        if matches:
            frame_col = matches[0]
            break

    if frame_col is not None:
        behaviour_df = candidate_df.drop(columns=[frame_col])
        frames = candidate_df[frame_col].to_numpy()
    else:
        behaviour_df = candidate_df
        frames = np.arange(len(candidate_df))

    # Convert row-wise one-hot to a single label by argmax
    true_label = behaviour_df.idxmax(axis=1)

    out = pd.DataFrame({
        "frame": frames,
        "true_label": normalize_labels(true_label)
    })

    out["frame"] = out["frame"].astype(int)
    return out


def match_result_and_label_files():
    moseq_files = sorted(MOSEQ_RESULTS_DIR.glob("*.csv"))
    if not moseq_files:
        raise FileNotFoundError(f"No MoSeq result CSVs found in {MOSEQ_RESULTS_DIR}")

    pairs = []
    missing = []

    for mf in moseq_files:
        recording_id = mf.stem

        if recording_id in EXCLUDED_RECORDINGS:
            print(f"Skipping {mf.name} by request.")
            continue

        label_file = LABELS_DIR / mf.name
        if label_file.exists():
            pairs.append((mf, label_file))
        else:
            missing.append(mf.name)

    print(f"Found {len(moseq_files)} MoSeq result files")
    print(f"Matched {len(pairs)} label files")

    if missing:
        print("Missing label files for:")
        for m in missing:
            print(" ", m)

    if not pairs:
        raise FileNotFoundError("No matching MoSeq/label file pairs found.")

    return pairs


def build_joined_frame_table(pairs):
    joined_all = []

    for mf, lf in pairs:
        rec = mf.stem
        moseq_df = load_moseq_csv(mf)
        label_df = load_label_csv(lf)

        merged = pd.merge(moseq_df, label_df, on="frame", how="inner")
        merged["recording"] = rec

        print(
            f"{rec}: MoSeq frames={len(moseq_df)}, "
            f"Label frames={len(label_df)}, "
            f"Merged={len(merged)}"
        )

        joined_all.append(merged)

    if not joined_all:
        raise RuntimeError("No merged frame tables created.")

    joined = pd.concat(joined_all, ignore_index=True)
    return joined


def compute_syllable_label_table(joined: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(joined["syllable"], joined["true_label"])


def best_mapping_from_crosstab(crosstab: pd.DataFrame) -> pd.DataFrame:
    mapping_rows = []

    for syllable in crosstab.index:
        row = crosstab.loc[syllable]
        best_label = row.idxmax()
        best_count = row.max()
        total = row.sum()
        purity = best_count / total if total > 0 else np.nan

        mapping_rows.append({
            "syllable": int(syllable),
            "mapped_label": best_label,
            "count_for_mapped_label": int(best_count),
            "total_frames_for_syllable": int(total),
            "purity": float(purity),
        })

    mapping_df = pd.DataFrame(mapping_rows).sort_values(
        ["purity", "total_frames_for_syllable"],
        ascending=[False, False]
    )
    return mapping_df


def apply_mapping(joined: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    map_dict = dict(zip(mapping_df["syllable"], mapping_df["mapped_label"]))
    joined = joined.copy()
    joined["predicted_label_from_moseq"] = joined["syllable"].map(map_dict)
    return joined


def save_confusion_matrix(cm_df: pd.DataFrame, path: Path, title: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_df.values, aspect="auto")
    plt.xticks(range(len(cm_df.columns)), cm_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(cm_df.index)), cm_df.index)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, int(cm_df.iloc[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_enrichment(joined: pd.DataFrame) -> pd.DataFrame:
    counts = (
        joined.groupby(["true_label", "syllable"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        counts.groupby("true_label")["count"]
        .sum()
        .reset_index(name="label_total")
    )

    enrich = counts.merge(totals, on="true_label", how="left")
    enrich["percent_within_label"] = 100 * enrich["count"] / enrich["label_total"]
    enrich = enrich.sort_values(["true_label", "percent_within_label"], ascending=[True, False])
    return enrich


# =========================================================
# MAIN
# =========================================================
def main():
    pairs = match_result_and_label_files()
    joined = build_joined_frame_table(pairs)

    joined_path = OUTPUT_DIR / "joined_frames.csv"
    joined.to_csv(joined_path, index=False)

    # 1) syllable vs true behavior table
    ctab = compute_syllable_label_table(joined)
    ctab_path = OUTPUT_DIR / "syllable_vs_true_label_crosstab.csv"
    ctab.to_csv(ctab_path)

    # 2) best mapping from syllable -> label
    mapping_df = best_mapping_from_crosstab(ctab)
    mapping_path = OUTPUT_DIR / "best_syllable_to_label_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)

    # 3) apply mapping to every frame
    joined_mapped = apply_mapping(joined, mapping_df)
    joined_mapped_path = OUTPUT_DIR / "joined_frames_with_predicted_labels.csv"
    joined_mapped.to_csv(joined_mapped_path, index=False)

    y_true = joined_mapped["true_label"]
    y_pred = joined_mapped["predicted_label_from_moseq"]

    labels_sorted = sorted(y_true.dropna().unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

    cm_path = OUTPUT_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    fig_path = OUTPUT_DIR / "confusion_matrix.png"
    save_confusion_matrix(cm_df, fig_path, "MoSeq-derived label vs True label")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = OUTPUT_DIR / "classification_report.csv"
    report_df.to_csv(report_path)

    # 4) enrichment table: which syllables dominate each true behavior
    enrich_df = compute_enrichment(joined)
    enrich_path = OUTPUT_DIR / "label_to_syllable_enrichment.csv"
    enrich_df.to_csv(enrich_path, index=False)

    print("\n======================================")
    print("Saved outputs:")
    print(f"  {joined_path}")
    print(f"  {ctab_path}")
    print(f"  {mapping_path}")
    print(f"  {joined_mapped_path}")
    print(f"  {cm_path}")
    print(f"  {fig_path}")
    print(f"  {report_path}")
    print(f"  {enrich_path}")
    print("======================================")

    print("\nTop syllable → label mappings:")
    print(mapping_df.head(15).to_string(index=False))

    print("\nClassification report:")
    print(report_df)

    print("\nConfusion matrix:")
    print(cm_df)


if __name__ == "__main__":
    main()