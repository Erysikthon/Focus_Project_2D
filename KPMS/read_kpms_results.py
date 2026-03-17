from pathlib import Path
import pandas as pd
import numpy as np


# -----------------------------
# PATHS
# -----------------------------
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "moseq_project" / "mouse_model_3d" / "results"
OUTPUT_DIR = ROOT / "analysis_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# HELPERS
# -----------------------------
def find_syllable_column(df: pd.DataFrame) -> str:
    """
    Try to find the syllable column in a flexible way.
    """
    candidates = ["syllable", "syllables", "state", "label"]
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]

    raise KeyError(
        f"Could not find a syllable column. Available columns: {list(df.columns)}"
    )


def compute_run_lengths(sequence: np.ndarray):
    """
    Consecutive duration of identical syllables.
    """
    sequence = np.asarray(sequence).ravel()

    if len(sequence) == 0:
        return []

    durations = []
    current = sequence[0]
    count = 1

    for s in sequence[1:]:
        if s == current:
            count += 1
        else:
            durations.append(count)
            current = s
            count = 1

    durations.append(count)
    return durations


def summarize_recording(csv_file: Path):
    df = pd.read_csv(csv_file)

    syll_col = find_syllable_column(df)
    syllables = df[syll_col].to_numpy()

    n_frames = len(syllables)
    unique, counts = np.unique(syllables, return_counts=True)
    n_unique = len(unique)

    durations = compute_run_lengths(syllables)

    summary = {
        "recording": csv_file.stem,
        "n_frames": n_frames,
        "n_unique_syllables": n_unique,
        "mean_duration_frames": float(np.mean(durations)) if durations else np.nan,
        "median_duration_frames": float(np.median(durations)) if durations else np.nan,
        "max_duration_frames": int(np.max(durations)) if durations else np.nan,
    }

    # frequency table for this recording
    freq_df = pd.DataFrame({
        "recording": csv_file.stem,
        "syllable": unique,
        "count": counts,
    })
    freq_df["percent"] = 100 * freq_df["count"] / n_frames
    freq_df = freq_df.sort_values("count", ascending=False).reset_index(drop=True)

    return summary, freq_df, df


# -----------------------------
# MAIN
# -----------------------------
def main():
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No result CSV files found in {RESULTS_DIR}")

    print(f"Found {len(csv_files)} result CSV files.\n")

    all_summaries = []
    all_freqs = []

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}")

        summary, freq_df, df = summarize_recording(csv_file)

        all_summaries.append(summary)
        all_freqs.append(freq_df)

        print(f"  Frames: {summary['n_frames']}")
        print(f"  Unique syllables: {summary['n_unique_syllables']}")
        print(f"  Mean duration: {summary['mean_duration_frames']:.2f} frames")
        print("  Top 10 syllables:")
        for _, row in freq_df.head(10).iterrows():
            print(
                f"    syllable {int(row['syllable'])}: "
                f"{int(row['count'])} frames ({row['percent']:.2f}%)"
            )
        print()

    summary_df = pd.DataFrame(all_summaries).sort_values("recording")
    freq_all_df = pd.concat(all_freqs, ignore_index=True)

    # pivot table: recordings x syllables
    freq_pivot = freq_all_df.pivot_table(
        index="recording",
        columns="syllable",
        values="percent",
        fill_value=0
    )

    summary_path = OUTPUT_DIR / "recording_summary.csv"
    freq_long_path = OUTPUT_DIR / "syllable_frequencies_long.csv"
    freq_wide_path = OUTPUT_DIR / "syllable_frequencies_wide.csv"

    summary_df.to_csv(summary_path, index=False)
    freq_all_df.to_csv(freq_long_path, index=False)
    freq_pivot.to_csv(freq_wide_path)

    print("======================================")
    print("Saved outputs:")
    print(f"  {summary_path}")
    print(f"  {freq_long_path}")
    print(f"  {freq_wide_path}")
    print("======================================")

    print("\nRecording summary preview:")
    print(summary_df.head())


if __name__ == "__main__":
    main()