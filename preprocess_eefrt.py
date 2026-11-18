"""
Trial-level preprocessing pipeline for the Modified EEfRT paradigm.
Extracts effort choices, motor persistence metrics, and trial-by-trial
outcome carryover variables for mixed-effects modeling in SPSS.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def preprocess_eefrt_log(
    filepath: Path,
    min_choice_rt: float = 0.200,
    max_choice_rt: float = 5.000,
) -> pd.DataFrame:
    """Clean a participant's PsychoPy EEfRT log and compute lag-1 carryover predictors."""
    df = pd.read_csv(filepath)

    # Exclude practice trials if flagged in the log
    if "is_practice" in df.columns:
        df = df[df["is_practice"] == 0].copy()

    # Keep valid decision-making trials within RT bounds
    df = df[df["choice_rt"].between(min_choice_rt, max_choice_rt)].copy()

    # Binary effort choice (1 = High-Effort, 0 = Low-Effort)
    df["high_effort_choice"] = (df["effort_choice"].astype(str).str.lower() == "high").astype(int)

    # Motor persistence metrics (button press count and rate)
    if {"button_presses", "task_duration"}.issubset(df.columns):
        df["press_rate"] = df["button_presses"] / df["task_duration"].replace(0, np.nan)

    # Generate trial-by-trial carryover predictors (t - 1) within participant/block
    group_cols = ["participant_id", "block"] if "block" in df.columns else ["participant_id"]
    df["prev_outcome"] = df.groupby(group_cols)["trial_reward_won"].shift(1)
    df["prev_effort_choice"] = df.groupby(group_cols)["high_effort_choice"].shift(1)
    df["prev_reward_magnitude"] = df.groupby(group_cols)["reward_magnitude"].shift(1)

    # Drop first trial of each block/session where lag-1 carryover is undefined
    df = df.dropna(subset=["prev_outcome"]).copy()
    return df


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    output_file = Path("data/processed/eefrt_trial_level_spss.csv")

    if raw_dir.exists():
        processed_logs = [
            preprocess_eefrt_log(csv_path)
            for csv_path in sorted(raw_dir.glob("*.csv"))
        ]
        if processed_logs:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(processed_logs, ignore_index=True).to_csv(output_file, index=False)
