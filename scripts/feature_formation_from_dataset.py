#!/usr/bin/env python
"""Release feature formation from DatasetCognitiveLoad.csv.

The released CSV is already de-identified and state-labeled. This script starts
from that processed sample-level file and creates the window-level features used
for representation analysis: scalar index features, pupil wavelet coefficients,
and raw windowed pupil sequences.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pywt


STATE_TO_LOADS = {
    0: (0, 0),
    1: (1, 0),
    2: (2, 0),
    3: (3, 0),
    4: (0, 1),
    5: (0, 2),
    6: (0, 3),
    7: (1, 1),
    8: (1, 2),
    9: (2, 1),
    10: (2, 2),
}


REQUIRED_COLUMNS = [
    "Participant name",
    "State",
    "Pupil diameter filtered",
    "Pupil diameter left",
    "Pupil diameter right",
    "Validity left",
    "Validity right",
    "Gaze point X",
    "Gaze point Y",
    "Eye movement type",
    "Gaze event duration",
    "Eye movement type index",
]


def participant_sort_key(participant: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(participant))
    number = int(match.group(1)) if match else 10_000
    return number, str(participant)


def participant_numeric_id(participant: str) -> int:
    match = re.search(r"(\d+)", str(participant))
    if not match:
        raise ValueError(f"Cannot extract numeric participant id from {participant!r}")
    return int(match.group(1))


def load_dataset(path: Path, states: list[int] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")

    df = df.copy()
    if states is not None:
        df = df[df["State"].isin(states)].copy()

    df["source_row"] = np.arange(len(df))
    df = df.sort_values(
        by=["Participant name", "source_row"],
        key=lambda col: col.map(participant_sort_key) if col.name == "Participant name" else col,
    ).reset_index(drop=True)
    return df


def expand_boolean_mask(mask: pd.Series, radius: int) -> np.ndarray:
    values = mask.fillna(False).to_numpy(dtype=bool)
    if radius <= 0 or not values.any():
        return values
    kernel = np.ones(radius * 2 + 1, dtype=int)
    return np.convolve(values.astype(int), kernel, mode="same") > 0


def prepare_participant_signal(
    participant_df: pd.DataFrame,
    sample_rate: float,
    invalid_window_ms: float,
    interpolation: str,
) -> pd.DataFrame:
    """Reconstruct blink/artifact handling available from the released CSV.

    The private raw export used `Computer timestamp` to blank a +/-200 ms window
    around invalid samples. The release CSV does not include that raw timestamp,
    so the public script applies the same idea in sample space.
    """

    df = participant_df.copy().reset_index(drop=True)
    invalid = (
        df["Validity left"].ne("Valid")
        | df["Validity right"].ne("Valid")
        | df["Eye movement type"].eq("EyesNotFound")
        | df["Pupil diameter filtered"].isna()
    )
    radius = int(round((invalid_window_ms / 1000.0) * sample_rate))
    df["isblink"] = expand_boolean_mask(invalid, radius).astype(int)

    pupil = pd.to_numeric(df["Pupil diameter filtered"], errors="coerce")
    pupil = pupil.mask(df["isblink"].eq(1), np.nan)
    pupil = pupil.interpolate(method=interpolation, limit_direction="both")
    pupil = pupil.bfill().ffill()
    if pupil.isna().all():
        pupil = pd.Series(np.zeros(len(df)), index=df.index)
    else:
        pupil = pupil.fillna(float(pupil.mean()))
    df["Pupil diameter filtered"] = pupil

    for column in ["Pupil diameter left", "Pupil diameter right", "Gaze point X", "Gaze point Y"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def segment_windows(num_rows: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")
    step = window_size - overlap
    return [(start, start + window_size) for start in range(0, num_rows - window_size + 1, step)]


def modmax(values: np.ndarray) -> np.ndarray:
    magnitudes = np.abs(values)
    output = np.zeros_like(magnitudes, dtype=float)
    for idx in range(len(values)):
        left = magnitudes[idx - 1] if idx >= 1 else magnitudes[idx]
        center = magnitudes[idx]
        right = magnitudes[idx + 1] if idx < len(values) - 2 else magnitudes[idx]
        if (left <= center and center >= right) and (left < center or center > right):
            output[idx] = math.sqrt(values[idx] ** 2)
    return output


def calc_ipa(signal: np.ndarray, wavelet: str, duration: float) -> float:
    try:
        c_a2, c_d2, c_d1 = pywt.wavedec(signal, wavelet, "per", level=2)
    except ValueError:
        return np.nan

    c_a2 = c_a2 / math.sqrt(4.0)
    c_d1 = c_d1 / math.sqrt(2.0)
    c_d2 = c_d2 / math.sqrt(4.0)
    maxima = modmax(c_d2)
    threshold = np.std(maxima) * math.sqrt(2.0 * np.log2(len(maxima)))
    thresholded = pywt.threshold(maxima, threshold, mode="hard")
    return float(np.count_nonzero(np.abs(thresholded) > 0)) / duration


def calc_lhipa(signal: np.ndarray, wavelet: str, duration: float) -> float:
    wave = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(signal), filter_len=wave.dec_len)
    high_frequency_level = 1
    low_frequency_level = max(1, int(max_level / 2))

    c_d_high = pywt.downcoef("d", signal, wavelet, "per", level=high_frequency_level)
    c_d_low = pywt.downcoef("d", signal, wavelet, "per", level=low_frequency_level)
    c_d_high = c_d_high / math.sqrt(2**high_frequency_level)
    c_d_low = c_d_low / math.sqrt(2**low_frequency_level)

    ratio = np.zeros_like(c_d_low, dtype=float)
    scale = int((2**low_frequency_level) / (2**high_frequency_level))
    for idx, value in enumerate(c_d_low):
        denominator_index = min(scale * idx, len(c_d_high) - 1)
        denominator = c_d_high[denominator_index]
        ratio[idx] = 0.0 if value == 0 or denominator == 0 else value / denominator

    maxima = modmax(ratio)
    threshold = np.std(maxima) * math.sqrt(2.0 * np.log2(len(maxima)))
    thresholded = pywt.threshold(maxima, threshold, mode="less")
    return float(np.count_nonzero(np.abs(thresholded) > 0)) / duration


def saccade_velocity_features(window: pd.DataFrame, sample_rate: float) -> tuple[float, float]:
    saccades = window[window["Eye movement type"].eq("Saccade")].copy()
    if saccades.empty:
        return 0.0, 0.0

    dx = saccades["Gaze point X"].diff()
    dy = saccades["Gaze point Y"].diff()
    amplitude = np.sqrt(dx * dx + dy * dy).fillna(0)
    velocity = amplitude * sample_rate
    velocity = velocity.replace([np.inf, -np.inf], np.nan).fillna(0)
    return float(velocity.mean()), float(velocity.max())


def make_feature_row(
    window: pd.DataFrame,
    participant: str,
    start: int,
    end: int,
    window_index: int,
    sample_rate: float,
    wavelet: str,
    wavelet_level: int,
    include_wavelet: bool,
) -> dict[str, object] | None:
    if window["State"].nunique() != 1:
        return None

    state = int(window["State"].iloc[0])
    if state not in STATE_TO_LOADS:
        return None

    #signal = window["Pupil diameter filtered"].to_numpy(dtype=float)
    # Create a writable memory copy of the Pandas data slice for PyWavelets
    signal = window["Pupil diameter filtered"].to_numpy(dtype=float, copy=True)
    if not np.isfinite(signal).all():
        signal = pd.Series(signal).interpolate(limit_direction="both").bfill().ffill().fillna(0).to_numpy(dtype=float)

    coeffs = pywt.wavedec(signal, wavelet, "per", level=wavelet_level)
    vm_load, va_load = STATE_TO_LOADS[state]
    fixation = window[window["Eye movement type"].eq("Fixation")]
    fixation_duration = pd.to_numeric(fixation["Gaze event duration"], errors="coerce")
    saccade_mean, saccade_peak = saccade_velocity_features(window, sample_rate)

    row: dict[str, object] = {
        "participant": participant,
        "id": participant_numeric_id(participant),
        "window_index": window_index,
        "sample_start": start,
        "sample_end": end - 1,
        "label": state,
        "vm_load": vm_load,
        "va_load": va_load,
        "load_detection_label": int(state != 0),
        "load_type_label": "" if state == 0 else int(state >= 4),
        "ipas": calc_ipa(signal, wavelet, duration=len(signal)),
        "lhipas": calc_lhipa(signal, wavelet, duration=len(signal)),
        "fixation_nums": int(len(fixation)),
        "fixation_durations": float(fixation_duration.mean()) if not fixation_duration.empty else 0.0,
        "blink_rate": float(window["isblink"].mean()),
        "saccade_speeds": saccade_mean,
        "saccade_peak_speeds": saccade_peak,
        "diameter": signal,
    }

    if include_wavelet:
        row["freq_features"] = coeffs[0]
        for level, coeff in zip(range(wavelet_level, 0, -1), coeffs[1:]):
            row[f"freqcD{level}"] = coeff
    return row


def build_features(
    df: pd.DataFrame,
    window_size: int,
    overlap: int,
    sample_rate: float,
    invalid_window_ms: float,
    interpolation: str,
    wavelet: str,
    wavelet_level: int,
    include_wavelet: bool,
    max_windows_per_participant: int | None,
) -> pd.DataFrame:
    rows = []
    for participant, group in df.groupby("Participant name", sort=False):
        print(f"--> Processing participant: {participant}...", flush=True)
        prepared = prepare_participant_signal(group, sample_rate, invalid_window_ms, interpolation)
        windows = segment_windows(len(prepared), window_size, overlap)
        if max_windows_per_participant is not None:
            windows = windows[:max_windows_per_participant]
        for local_window_index, (start, end) in enumerate(windows):
            row = make_feature_row(
                window=prepared.iloc[start:end],
                participant=participant,
                start=start,
                end=end,
                window_index=local_window_index,
                sample_rate=sample_rate,
                wavelet=wavelet,
                wavelet_level=wavelet_level,
                include_wavelet=include_wavelet,
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def flatten_array_columns(df: pd.DataFrame) -> pd.DataFrame:
    scalar = df.copy()
    array_columns = [column for column in scalar.columns if scalar[column].apply(lambda x: isinstance(x, np.ndarray)).any()]
    for column in array_columns:
        expanded = pd.DataFrame(scalar[column].tolist()).add_prefix(f"{column}_")
        scalar = pd.concat([scalar.drop(columns=[column]).reset_index(drop=True), expanded.reset_index(drop=True)], axis=1)
    return scalar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(r"C:\Users\HP\Documents\Uni_Stuttgart\CAI\caip2\data\DatasetCognitiveLoad.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=511)
    parser.add_argument("--sample-rate", type=float, default=60.0)
    parser.add_argument("--invalid-window-ms", type=float, default=200.0)
    parser.add_argument("--interpolation", default="linear")
    parser.add_argument("--wavelet", default="db8")
    parser.add_argument("--wavelet-level", type=int, default=2)
    parser.add_argument("--states", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--include-wavelet", action="store_true", help="Store wavelet coefficient arrays in the pickle output.")
    parser.add_argument("--flat-csv", action="store_true", help="Also write a flattened CSV file.")
    parser.add_argument(
        "--max-windows-per-participant",
        type=int,
        default=None,
        help="Optional cap for quick checks; omit for the full feature table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input, states=args.states)
    features = build_features(
        df=df,
        window_size=args.window_size,
        overlap=args.overlap,
        sample_rate=args.sample_rate,
        invalid_window_ms=args.invalid_window_ms,
        interpolation=args.interpolation,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        include_wavelet=args.include_wavelet,
        max_windows_per_participant=args.max_windows_per_participant,
    )

    suffix = f"{args.window_size}_{args.overlap}"
    output_pkl = args.output_dir / f"features_{suffix}.pkl"
    features.to_pickle(output_pkl)
    print(f"Wrote {output_pkl} with shape {features.shape}")

    if args.flat_csv:
        output_csv = args.output_dir / f"features_{suffix}_flat.csv.gz"
        flatten_array_columns(features).to_csv(output_csv, index=False, compression="gzip")
        print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
