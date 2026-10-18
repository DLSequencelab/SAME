"""Convert official UEA TRAIN/TEST .ts files to SAME's standalone NPZ.

This is a data-format adapter only. It performs no PCA or class-dependent
preprocessing. Per-sample normalization is done by the model at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _impute_inside(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if valid.all():
        return values
    if not valid.any():
        return np.zeros_like(values)
    positions = np.arange(len(values))
    return np.interp(positions, positions[valid], values[valid]).astype(np.float32)


def _read_ts(path: Path):
    try:
        from sktime.datasets import load_from_tsfile_to_dataframe
    except ImportError as error:
        raise RuntimeError("Install sktime to convert UEA .ts files") from error
    frame, labels = load_from_tsfile_to_dataframe(
        str(path), return_separate_X_and_y=True,
        replace_missing_vals_with="NaN",
    )
    rows = []
    for sample in range(len(frame)):
        channels = [np.asarray(frame.iloc[sample, channel], dtype=np.float32)
                    for channel in range(frame.shape[1])]
        # A single time mask is shared by all channels: use their common prefix.
        length = min(map(len, channels))
        if length < 1:
            raise ValueError(f"Empty channel in sample {sample}: {path}")
        rows.append(np.stack([_impute_inside(channel[:length])
                              for channel in channels], axis=-1))
    return rows, np.asarray(labels, dtype=str)


def _pad(rows, length):
    output = np.zeros((len(rows), length, rows[0].shape[-1]), dtype=np.float32)
    mask = np.zeros((len(rows), length), dtype=bool)
    for index, sample in enumerate(rows):
        output[index, :len(sample)] = sample
        mask[index, :len(sample)] = True
    return output, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Folder containing Dataset_TRAIN.ts and Dataset_TEST.ts")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    train_path = args.data_root / f"{args.dataset}_TRAIN.ts"
    test_path = args.data_root / f"{args.dataset}_TEST.ts"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_rows, train_y = _read_ts(train_path)
    test_rows, test_y = _read_ts(test_path)
    if train_rows[0].shape[-1] != test_rows[0].shape[-1]:
        raise ValueError("TRAIN and TEST channel counts differ")
    length = max(max(map(len, train_rows)), max(map(len, test_rows)))
    train_x, train_mask = _pad(train_rows, length)
    test_x, test_mask = _pad(test_rows, length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, train_x=train_x, train_y=train_y,
                        train_mask=train_mask, test_x=test_x, test_y=test_y,
                        test_mask=test_mask)
    print(f"Saved {args.output}: train={len(train_y)}, test={len(test_y)}, "
          f"T={length}, C={train_x.shape[-1]}")


if __name__ == "__main__":
    main()
