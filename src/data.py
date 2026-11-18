"""Dataset loading (with a local cache), preprocessing and splitting.

All paths are resolved relative to this file, never the working directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_FILE = DATA_DIR / "california_housing.csv.gz"

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET = "MedHouseVal"
LOG_COLS = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
DATASET_NAME = "California Housing (sklearn.datasets.fetch_california_housing)"

SPLIT_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10


def load_dataset(verbose: bool = True) -> pd.DataFrame:
    """Return the full dataset as a DataFrame (8 features + target).

    The first call downloads through scikit-learn and writes a local CSV cache;
    later calls read the cache and need no network.
    """
    if CACHE_FILE.exists():
        if verbose:
            print("[data] using cached dataset: data/%s" % CACHE_FILE.name)
        df = pd.read_csv(CACHE_FILE)
    else:
        if verbose:
            print("[data] cache not found, downloading California Housing via scikit-learn ...")
        try:
            from sklearn.datasets import fetch_california_housing

            bunch = fetch_california_housing(
                data_home=str(DATA_DIR / "sklearn_cache"), as_frame=True
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                "Could not download the California Housing dataset. "
                "Connect to the internet and run `python train.py` again. "
                "Original error: %r" % (exc,)
            ) from exc
        df = bunch.frame[FEATURES + [TARGET]].copy()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_FILE, index=False)
        if verbose:
            print("[data] cached to data/%s" % CACHE_FILE.name)
    df = df[FEATURES + [TARGET]].astype(float).reset_index(drop=True)
    return df


def make_synthetic(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Small synthetic sample with the same schema (used by selfcheck.py, offline)."""
    rng = np.random.default_rng(seed)
    med_inc = rng.lognormal(1.2, 0.45, n)
    age = rng.uniform(1, 52, n)
    rooms = rng.lognormal(1.65, 0.25, n)
    bedrms = rng.lognormal(0.07, 0.12, n)
    pop = rng.lognormal(7.0, 0.7, n)
    occup = rng.lognormal(1.05, 0.25, n)
    lat = rng.uniform(32.5, 42.0, n)
    lon = rng.uniform(-124.3, -114.3, n)
    y = 0.42 * med_inc + 0.008 * age - 0.25 * np.log(occup) - 0.04 * (lat - 36) \
        + 0.3 * np.sin(lon) + rng.normal(0, 0.3, n)
    y = np.clip(y, 0.15, 5.0)
    return pd.DataFrame(
        np.column_stack([med_inc, age, rooms, bedrms, pop, occup, lat, lon, y]),
        columns=FEATURES + [TARGET],
    )


def split_indices(n: int, seed: int = SPLIT_SEED):
    """Shuffle once with a fixed seed and cut into 70 / 10 / 20."""
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def validate_raw(X) -> np.ndarray:
    """Check raw feature rows and return a float array of shape (k, 8)."""
    if isinstance(X, pd.DataFrame):
        missing = [c for c in FEATURES if c not in X.columns]
        if missing:
            raise ValueError("missing feature columns: %s" % ", ".join(missing))
        X = X[FEATURES].to_numpy()
    try:
        arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be numeric") from exc
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != len(FEATURES):
        raise ValueError(
            "expected %d features per row (%s), got shape %s"
            % (len(FEATURES), ", ".join(FEATURES), arr.shape)
        )
    if arr.shape[0] == 0:
        raise ValueError("no rows given")
    if not np.all(np.isfinite(arr)):
        raise ValueError("features contain NaN or infinite values")
    log_idx = [FEATURES.index(c) for c in LOG_COLS]
    if np.any(arr[:, log_idx] < 0):
        raise ValueError("%s must be non-negative" % ", ".join(LOG_COLS))
    return arr


@dataclass
class Preprocessor:
    """log1p on skewed columns, then standardise with training statistics only."""

    x_mean: np.ndarray = None
    x_scale: np.ndarray = None
    y_mean: float = 0.0
    y_std: float = 1.0

    @staticmethod
    def _log(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        for c in LOG_COLS:
            j = FEATURES.index(c)
            out[:, j] = np.log1p(out[:, j])
        return out

    def fit(self, X_raw, y) -> "Preprocessor":
        arr = self._log(validate_raw(X_raw))
        self.x_mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        self.x_scale = np.where(scale > 1e-12, scale, 1.0)
        y = np.asarray(y, dtype=float)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if float(y.std()) > 1e-12 else 1.0
        return self

    def transform(self, X_raw) -> np.ndarray:
        arr = self._log(validate_raw(X_raw))
        return (arr - self.x_mean) / self.x_scale

    def y_to_z(self, y) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.y_mean) / self.y_std

    def z_to_y(self, z) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.y_std + self.y_mean

    def to_state(self) -> dict:
        return {
            "x_mean": np.asarray(self.x_mean), "x_scale": np.asarray(self.x_scale),
            "y_mean": float(self.y_mean), "y_std": float(self.y_std),
        }

    @classmethod
    def from_state(cls, state: dict) -> "Preprocessor":
        return cls(np.asarray(state["x_mean"]), np.asarray(state["x_scale"]),
                   float(state["y_mean"]), float(state["y_std"]))


def prepare(df: pd.DataFrame, seed: int = SPLIT_SEED) -> dict:
    """Split and preprocess. Returns raw and standardised arrays for every split."""
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    tr, va, te = split_indices(len(df), seed)
    pre = Preprocessor().fit(X[tr], y[tr])
    out = {"pre": pre, "idx": {"train": tr, "val": va, "test": te}}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        out["X_raw_" + name] = X[idx]
        out["X_" + name] = pre.transform(X[idx])
        out["y_" + name] = y[idx]
    return out
