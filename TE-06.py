# -*- coding: utf-8 -*-
"""
SEISMIC TOPOLOGICAL DEEP MANIFOLDS FOR EARLY PRECURSOR DETECTION
(Autoencoders + Delay Embeddings + Persistent Homology + Stability Curves)

FAST + FULL-ARTIFACTS VERSION (MODIFIED)
----------------------------------------
This modified version accelerates runtime while preserving the planned outputs:
    - figures/  (plots, including persistence diagrams and Betti curves when available)
    - reports/  (Excel + CSV + JSON + README + extended sheets)
    - models/   (saved AE models if enabled)

Key speed-ups:
    1) Batch inference for AE-1D/AE-2D:
       - Instead of calling predict() per-window, concatenate all window point clouds
         (normalized) into one array, run predict once, then aggregate per-window.
    2) Optional FAST preset at entry point:
       - Reduces expensive components (TDA frequency/subsample, epochs, etc.)
         without removing any report/figure exports.

Notes:
    - Persistent homology is still the main bottleneck; this version keeps the same
      safeguards and adds a "fast preset" that increases stride and reduces subsample.
    - All artifacts remain generated; when TDA packages are missing, topology computations
      are skipped gracefully, and placeholders remain in outputs.

"""

# =============================================================================
# 0. IMPORTS, OPTIONAL INSTALLATION, AND GLOBAL SETTINGS
# =============================================================================

import os
import sys
import gc
import json
import time
import importlib
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def ensure_package(pkg_name: str) -> bool:
    """
    Try to import a package; if it fails, attempt to install it via pip.

    Parameters
    ----------
    pkg_name : str
        Package import name (e.g., 'ripser', 'persim', 'tensorflow').

    Returns
    -------
    bool
        True if import succeeds (before or after installation), False otherwise.
    """
    try:
        importlib.import_module(pkg_name)
        return True
    except Exception:
        try:
            print(f"[INFO] Installing package: {pkg_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            importlib.import_module(pkg_name)
            return True
        except Exception as e:
            print(f"[WARNING] Could not install '{pkg_name}': {e}")
            return False


# Ensure specialized packages:
HAS_RIPSER = ensure_package("ripser")
HAS_PERSIM = ensure_package("persim")
ensure_package("tensorflow")
HAS_OBSPY = ensure_package("obspy")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

if HAS_RIPSER:
    from ripser import ripser
if HAS_PERSIM:
    import persim
    from persim import plot_diagrams
if HAS_OBSPY:
    from obspy import read as obspy_read

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # ------------------------------
    # Reproducibility
    # ------------------------------
    SEED: int = 42

    # ------------------------------
    # Data mode
    # ------------------------------
    USE_REAL_DATA: bool = False
    REAL_DATA_PATH: str = ""

    # ------------------------------
    # Synthetic data
    # ------------------------------
    FS: float = 50.0
    DURATION_SEC: int = 3 * 3600
    PRE_EVENT_START_FRAC: float = 0.65
    EVENT_TIME_FRAC: float = 0.85
    NOISE_STD: float = 0.8
    MICRO_TREMOR_LEVEL: float = 1.0

    # ------------------------------
    # Preprocessing
    # ------------------------------
    APPLY_BANDPASS: bool = True
    BP_LOW: float = 0.5
    BP_HIGH: float = 8.0
    ROBUST_SCALE: bool = True
    DETREND: bool = True

    # ------------------------------
    # Sliding windows
    # ------------------------------
    WINDOW_SEC: float = 30.0
    HOP_SEC: float = 5.0

    # ------------------------------
    # Delay embedding
    # ------------------------------
    EMBED_DIM: int = 3
    EMBED_DELAY: int = 3

    # ------------------------------
    # Autoencoder training
    # ------------------------------
    BASELINE_FRACTION_FOR_TRAIN: float = 0.35
    EPOCHS: int = 80
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 1e-3
    AE_HIDDEN: int = 64
    LATENT_1D: int = 1
    LATENT_2D: int = 2

    # ------------------------------
    # Topological analysis (TDA)
    # ------------------------------
    DO_TDA: bool = True
    TDA_MAXDIM: int = 1
    TDA_SUBSAMPLE: int = 800
    TDA_STRIDE: int = 2
    BETTI_NGRID: int = 120

    # ------------------------------
    # Early-warning scoring
    # ------------------------------
    SMOOTH_WIN: int = 9
    ALERT_Z: float = 2.5
    USE_CHANGEPOINT: bool = True
    CHANGEPOINT_MIN_DISTANCE: int = 10

    # ------------------------------
    # Output
    # ------------------------------
    OUT_DIR: str = "outputs_seismic_topo_deep"
    SAVE_MODELS: bool = True

    # ------------------------------
    # Speed controls (new, but does not remove outputs)
    # ------------------------------
    FAST_PRESET: bool = True
    # If FAST_PRESET=True, these act as safe defaults:
    FAST_DURATION_SEC: int = 90 * 60         # 90 minutes synthetic (instead of 3 hours)
    FAST_EPOCHS: int = 35                    # fewer epochs (still enough to learn baseline)
    FAST_TDA_STRIDE: int = 4                 # compute TDA every k windows
    FAST_TDA_SUBSAMPLE: int = 500            # fewer points in point cloud for TDA
    FAST_BETTI_NGRID: int = 90               # lower resolution Betti curves


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def apply_fast_preset(cfg: Config) -> Config:
    """
    Apply a runtime-friendly preset WITHOUT removing any planned artifacts.

    The preset only reduces compute-heavy knobs, while keeping:
      - all figures
      - Excel/CSV/JSON exports
      - extended sheets
      - optional TDA (still enabled, but strided and subsampled)
    """
    if not cfg.FAST_PRESET:
        return cfg

    # Only apply in synthetic mode unless user explicitly sets otherwise
    if not cfg.USE_REAL_DATA:
        cfg.DURATION_SEC = int(cfg.FAST_DURATION_SEC)

    cfg.EPOCHS = int(cfg.FAST_EPOCHS)
    cfg.TDA_STRIDE = int(cfg.FAST_TDA_STRIDE)
    cfg.TDA_SUBSAMPLE = int(cfg.FAST_TDA_SUBSAMPLE)
    cfg.BETTI_NGRID = int(cfg.FAST_BETTI_NGRID)

    # Conservative guards
    cfg.TDA_MAXDIM = int(min(cfg.TDA_MAXDIM, 1))  # keep H0/H1 for speed by default
    cfg.BATCH_SIZE = int(max(cfg.BATCH_SIZE, 128))
    cfg.EMBED_DIM = int(min(cfg.EMBED_DIM, 3))    # 3D embedding by default for speed

    return cfg


# =============================================================================
# 2. SIGNAL GENERATION / LOADING
# =============================================================================

def synthetic_seismic_like_signal(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    fs = cfg.FS
    n = int(cfg.DURATION_SEC * fs)
    t = np.arange(n, dtype=np.float32) / fs

    pre_event_start = int(cfg.PRE_EVENT_START_FRAC * n)
    event_time = int(cfg.EVENT_TIME_FRAC * n)

    freqs = np.array([0.8, 1.7, 3.2, 5.1], dtype=np.float32)
    amps = cfg.MICRO_TREMOR_LEVEL * np.array([1.0, 0.7, 0.35, 0.25], dtype=np.float32)
    phases0 = np.random.uniform(0, 2*np.pi, size=len(freqs)).astype(np.float32)

    x = np.zeros_like(t, dtype=np.float32)
    noise = np.random.normal(0, cfg.NOISE_STD, size=n).astype(np.float32)

    def smoothstep(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, 0.0, 1.0)
        return z*z*(3.0 - 2.0*z)

    ramp = np.zeros(n, dtype=np.float32)
    if event_time > pre_event_start + 1:
        z = (np.arange(n) - pre_event_start) / float(event_time - pre_event_start)
        ramp = smoothstep(z.astype(np.float32))

    ramp_col = ramp[:, None]
    phase_jitter = (0.03 + 0.20 * ramp_col) * np.random.normal(
        0, 1, size=(n, len(freqs))
    ).astype(np.float32)

    freq_mod = (0.00 + 0.10 * ramp) * np.sin(2*np.pi*0.03*t).astype(np.float32)

    for k, (f, a) in enumerate(zip(freqs, amps)):
        ph = phases0[k] + np.cumsum(phase_jitter[:, k]).astype(np.float32) / fs
        inst_freq = f * (1.0 + freq_mod)
        x += a * np.sin(2*np.pi*inst_freq*t + ph).astype(np.float32)

    x += (0.03 + 0.15 * ramp) * (x**2 - np.mean(x**2)).astype(np.float32)
    x = x + noise

    burst_len = int(15 * fs)
    burst_start = max(0, event_time - burst_len // 2)
    burst_end = min(n, burst_start + burst_len)
    burst_t = np.arange(burst_end - burst_start, dtype=np.float32) / fs
    burst = 8.0 * np.exp(-0.5 * ((burst_t - burst_t.mean()) / 2.0)**2).astype(np.float32)
    burst *= np.sin(2*np.pi*7.0*burst_t).astype(np.float32)
    x[burst_start:burst_end] += burst

    meta = {
        "fs": float(fs),
        "n_samples": float(n),
        "pre_event_start_idx": float(pre_event_start),
        "event_time_idx": float(event_time),
        "pre_event_start_sec": float(pre_event_start / fs),
        "event_time_sec": float(event_time / fs),
    }
    return t, x.astype(np.float32), meta


def load_real_waveform_obspy(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if not HAS_OBSPY:
        raise RuntimeError("ObsPy is not available; set USE_REAL_DATA=False or install obspy.")
    if not cfg.REAL_DATA_PATH or not os.path.exists(cfg.REAL_DATA_PATH):
        raise FileNotFoundError(f"REAL_DATA_PATH not found: {cfg.REAL_DATA_PATH}")

    st = obspy_read(cfg.REAL_DATA_PATH)
    tr = st[0]
    x = tr.data.astype(np.float32)
    fs = float(tr.stats.sampling_rate)
    t = np.arange(len(x), dtype=np.float32) / fs

    meta = {
        "fs": float(fs),
        "n_samples": float(len(x)),
        "starttime": str(tr.stats.starttime),
        "station": str(getattr(tr.stats, "station", "")),
        "network": str(getattr(tr.stats, "network", "")),
        "channel": str(getattr(tr.stats, "channel", "")),
    }
    return t, x, meta


# =============================================================================
# 3. PREPROCESSING
# =============================================================================

def detrend_signal(x: np.ndarray) -> np.ndarray:
    n = len(x)
    t = np.arange(n, dtype=np.float32)
    A = np.vstack([t, np.ones(n, dtype=np.float32)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    slope, intercept = coeff.astype(np.float32)
    trend = slope * t + intercept
    return (x - trend).astype(np.float32)


def butter_bandpass_filter_np(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        lowc = low / nyq
        highc = high / nyq
        b, a = butter(order, [lowc, highc], btype="band")
        y = filtfilt(b, a, x).astype(np.float32)
        return y
    except Exception as e:
        print(f"[WARNING] SciPy not available or filtering failed ({e}). Proceeding without bandpass.")
        return x.astype(np.float32)


def robust_scale(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return ((x - med) / mad).astype(np.float32)


def preprocess_signal(x: np.ndarray, cfg: Config, fs: float) -> np.ndarray:
    y = x.astype(np.float32)

    if cfg.DETREND:
        y = detrend_signal(y)

    if cfg.APPLY_BANDPASS:
        y = butter_bandpass_filter_np(y, fs=fs, low=cfg.BP_LOW, high=cfg.BP_HIGH, order=4)

    if cfg.ROBUST_SCALE:
        y = robust_scale(y)
    else:
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)

    return y.astype(np.float32)


# =============================================================================
# 4. WINDOWS + DELAY EMBEDDING
# =============================================================================

def sliding_windows(x: np.ndarray, fs: float, win_sec: float, hop_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    win_len = int(round(win_sec * fs))
    hop_len = int(round(hop_sec * fs))
    n = len(x)

    if win_len < 10 or hop_len < 1:
        raise ValueError("Window/hop too small given FS; increase WINDOW_SEC or HOP_SEC.")

    starts = np.arange(0, n - win_len + 1, hop_len, dtype=int)
    Xw = np.stack([x[s:s+win_len] for s in starts], axis=0).astype(np.float32)

    centers = starts + win_len // 2
    centers_sec = (centers / fs).astype(np.float32)
    return Xw, centers_sec


def delay_embedding_1d(series: np.ndarray, dim: int, delay: int) -> np.ndarray:
    series = np.asarray(series, dtype=np.float32)
    L = len(series)
    T_eff = L - (dim - 1) * delay
    if T_eff <= 5:
        raise ValueError(f"Not enough points for embedding: L={L}, dim={dim}, delay={delay}")
    emb = np.zeros((T_eff, dim), dtype=np.float32)
    for j in range(dim):
        emb[:, j] = series[j * delay : j * delay + T_eff]
    return emb


def subsample_points(X: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    if N <= n_samples:
        return X
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=n_samples, replace=False)
    return X[idx]


# =============================================================================
# 5. AUTOENCODERS
# =============================================================================

def build_dense_autoencoder(input_dim: int, latent_dim: int, hidden: int, lr: float) -> Model:
    inp = keras.Input(shape=(input_dim,), dtype="float32")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.Dense(hidden, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="linear", name=f"latent_{latent_dim}")(x)
    x = layers.Dense(hidden, activation="relu")(z)
    x = layers.Dense(hidden, activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)

    ae = keras.Model(inp, out, name=f"AE_latent_{latent_dim}")
    ae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return ae


def plot_training_history(hist: keras.callbacks.History, outpath: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history.get("loss", []), label="loss")
    if "val_loss" in hist.history:
        plt.plot(hist.history["val_loss"], label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def pointwise_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.mean((a - b) ** 2, axis=1).astype(np.float32)


# =============================================================================
# 6. TDA
# =============================================================================

def compute_diagrams(X: np.ndarray, maxdim: int) -> List[np.ndarray]:
    if not HAS_RIPSER:
        raise RuntimeError("ripser is not available but DO_TDA=True. Install ripser or set DO_TDA=False.")
    X = np.asarray(X, dtype=np.float32)
    res = ripser(X, maxdim=maxdim)
    return res["dgms"]


def wasserstein(diag_a: np.ndarray, diag_b: np.ndarray) -> float:
    if not HAS_PERSIM:
        return float("nan")
    return float(persim.wasserstein(diag_a, diag_b))


def betti_curve_from_diagram(diag: np.ndarray, tgrid: np.ndarray) -> np.ndarray:
    if diag is None or len(diag) == 0:
        return np.zeros_like(tgrid, dtype=int)

    d = np.asarray(diag, dtype=np.float32)
    births = d[:, 0]
    deaths = d[:, 1].copy()

    tmax = float(np.max(tgrid))
    deaths = np.where(np.isfinite(deaths), deaths, tmax + 0.05 * tmax + 1e-6)

    beta = np.zeros_like(tgrid, dtype=int)
    for i, t in enumerate(tgrid):
        beta[i] = int(np.sum((births <= t) & (t < deaths)))
    return beta


def plot_diagrams_side_by_side(dgms_a: List[np.ndarray], dgms_b: List[np.ndarray],
                              outpath: Path, title_a: str, title_b: str) -> None:
    if not HAS_PERSIM:
        return

    max_k = min(len(dgms_a), len(dgms_b))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle("Persistence diagrams (comparison)", fontsize=12)

    plt.sca(axes[0])
    plot_diagrams(dgms_a[:max_k], show=False)
    plt.title(title_a)

    plt.sca(axes[1])
    plot_diagrams(dgms_b[:max_k], show=False)
    plt.title(title_b)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_betti_curves(betti_dict: Dict[str, Dict[int, np.ndarray]],
                      tgrid: np.ndarray, outpath: Path) -> None:
    plt.figure(figsize=(10, 6))
    for system_label, hk_dict in betti_dict.items():
        for k, curve in hk_dict.items():
            plt.plot(tgrid, curve, label=f"{system_label} | H{k}")

    plt.title("Betti curves across filtration scales")
    plt.xlabel("Filtration scale")
    plt.ylabel("Betti number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# =============================================================================
# 7. SCORING / CHANGEPOINTS
# =============================================================================

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / float(w)
    y = np.convolve(xp, kernel, mode="valid").astype(np.float32)
    return y


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return ((x - med) / mad).astype(np.float32)


def simple_changepoints(score: np.ndarray, min_distance: int = 10) -> List[int]:
    z = robust_zscore(score)
    peaks = []
    for i in range(1, len(z) - 1):
        if z[i] > z[i-1] and z[i] > z[i+1]:
            peaks.append(i)

    selected = []
    for p in sorted(peaks, key=lambda i: z[i], reverse=True):
        if all(abs(p - s) >= min_distance for s in selected):
            selected.append(p)
    selected.sort()
    return selected


# =============================================================================
# 8. VISUALIZATION UTILITIES
# =============================================================================

def plot_signal_with_markers(t: np.ndarray, x: np.ndarray, outpath: Path,
                            markers: Dict[str, float], title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(t, x, linewidth=0.8)
    for name, sec in markers.items():
        plt.axvline(sec, linestyle="--", linewidth=1.2, label=name)
    plt.title(title)
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_embedding_3d(emb: np.ndarray, outpath: Path, title: str) -> None:
    if emb.shape[1] < 3:
        return
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], s=3, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t+τ)")
    ax.set_zlabel("x(t+2τ)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_metric_time(centers_sec: np.ndarray, metric: np.ndarray, outpath: Path,
                     title: str, ylabel: str, extra_lines: Optional[Dict[str, float]] = None) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(centers_sec, metric, linewidth=1.2)
    if extra_lines:
        for name, sec in extra_lines.items():
            plt.axvline(sec, linestyle="--", linewidth=1.0, label=name)
        plt.legend()
    plt.title(title)
    plt.xlabel("Time (sec)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# =============================================================================
# 9. MAIN PIPELINE (ACCELERATED)
# =============================================================================

def run_pipeline(cfg: Config) -> None:
    # Output directories
    out_dir = Path(cfg.OUT_DIR)
    fig_dir = out_dir / "figures"
    rep_dir = out_dir / "reports"
    mdl_dir = out_dir / "models"
    safe_mkdir(fig_dir)
    safe_mkdir(rep_dir)
    safe_mkdir(mdl_dir)

    # Reproducibility
    set_global_seed(cfg.SEED)

    # Load/generate data
    if cfg.USE_REAL_DATA:
        t, x_raw, meta = load_real_waveform_obspy(cfg)
        fs = float(meta["fs"])
        markers = {}
        print("[INFO] Loaded real waveform.")
    else:
        t, x_raw, meta = synthetic_seismic_like_signal(cfg)
        fs = float(meta["fs"])
        markers = {
            "pre-event start": float(meta["pre_event_start_sec"]),
            "event time": float(meta["event_time_sec"]),
        }
        print("[INFO] Generated synthetic seismic-like waveform.")

    # Preprocess
    x = preprocess_signal(x_raw, cfg, fs)
    plot_signal_with_markers(
        t, x,
        fig_dir / "signal_preprocessed.png",
        markers=markers,
        title="Preprocessed continuous signal (scaled)"
    )

    # Windowing
    Xw, centers_sec = sliding_windows(x, fs, cfg.WINDOW_SEC, cfg.HOP_SEC)
    n_windows = Xw.shape[0]
    print(f"[INFO] Windows: {n_windows} | win_len={Xw.shape[1]} samples | hop={cfg.HOP_SEC}s")

    # Delay embedding per window
    emb_list: List[np.ndarray] = []
    valid_mask = np.ones(n_windows, dtype=bool)

    for i in range(n_windows):
        try:
            emb = delay_embedding_1d(Xw[i], dim=cfg.EMBED_DIM, delay=cfg.EMBED_DELAY)
            emb_list.append(emb.astype(np.float32))
        except Exception:
            emb_list.append(np.zeros((1, cfg.EMBED_DIM), dtype=np.float32))
            valid_mask[i] = False

    idx_early = int(0.15 * n_windows)
    idx_late = int(0.80 * n_windows)
    plot_embedding_3d(emb_list[idx_early], fig_dir / "embedding_early_3d.png",
                      title="3D delay embedding (early window)")
    plot_embedding_3d(emb_list[idx_late], fig_dir / "embedding_late_3d.png",
                      title="3D delay embedding (late window)")

    # Baseline training set
    baseline_n = max(5, int(cfg.BASELINE_FRACTION_FOR_TRAIN * n_windows))
    baseline_idxs = np.where(valid_mask)[0][:baseline_n]
    X_train_cloud = np.vstack([emb_list[i] for i in baseline_idxs]).astype(np.float32)

    mean_emb = np.mean(X_train_cloud, axis=0)
    std_emb = np.std(X_train_cloud, axis=0) + 1e-8
    X_train_norm = ((X_train_cloud - mean_emb) / std_emb).astype(np.float32)

    print(f"[INFO] Baseline training windows: {len(baseline_idxs)} | "
          f"baseline cloud points: {X_train_norm.shape[0]} | dim={X_train_norm.shape[1]}")

    # Build AEs
    ae1d = build_dense_autoencoder(cfg.EMBED_DIM, cfg.LATENT_1D, cfg.AE_HIDDEN, cfg.LEARNING_RATE)
    ae2d = build_dense_autoencoder(cfg.EMBED_DIM, cfg.LATENT_2D, cfg.AE_HIDDEN, cfg.LEARNING_RATE)

    # Training callbacks (speed + stability), but still produces full artifacts
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, mode="min", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, mode="min", verbose=1
        ),
    ]

    # Train with validation split
    hist1 = ae1d.fit(
        X_train_norm, X_train_norm,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )
    hist2 = ae2d.fit(
        X_train_norm, X_train_norm,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )

    plot_training_history(hist1, fig_dir / "train_curve_AE1D.png", "Training curve (AE-1D)")
    plot_training_history(hist2, fig_dir / "train_curve_AE2D.png", "Training curve (AE-2D)")

    if cfg.SAVE_MODELS:
        ae1d.save(mdl_dir / "AE1D.keras")
        ae2d.save(mdl_dir / "AE2D.keras")

    # -------------------------------------------------------------------------
    # ACCELERATION: Batch inference across ALL windows (instead of per-window predict)
    # -------------------------------------------------------------------------
    mse1_mean = np.full(n_windows, np.nan, dtype=np.float32)
    mse2_mean = np.full(n_windows, np.nan, dtype=np.float32)
    mse1_median = np.full(n_windows, np.nan, dtype=np.float32)
    mse2_median = np.full(n_windows, np.nan, dtype=np.float32)

    # Prepare concatenated arrays for valid windows
    win_ids: List[int] = []
    offsets: List[Tuple[int, int]] = []
    emb_norm_chunks: List[np.ndarray] = []

    cursor = 0
    for i in range(n_windows):
        if not valid_mask[i]:
            offsets.append((cursor, cursor))
            continue
        emb = emb_list[i].astype(np.float32)
        emb_norm = ((emb - mean_emb) / std_emb).astype(np.float32)
        emb_norm_chunks.append(emb_norm)
        start = cursor
        cursor += emb_norm.shape[0]
        end = cursor
        offsets.append((start, end))
        win_ids.append(i)

    if len(emb_norm_chunks) == 0:
        raise RuntimeError("No valid windows after embedding. Check WINDOW_SEC/HOP_SEC/EMBED_DIM/EMBED_DELAY.")

    X_all = np.vstack(emb_norm_chunks).astype(np.float32)
    print(f"[INFO] Batch inference points: {X_all.shape[0]} across {len(win_ids)} valid windows.")

    # Predict once (per AE)
    rec1_all = ae1d.predict(X_all, batch_size=4096, verbose=0).astype(np.float32)
    rec2_all = ae2d.predict(X_all, batch_size=4096, verbose=0).astype(np.float32)

    err1_all = pointwise_mse(X_all, rec1_all)
    err2_all = pointwise_mse(X_all, rec2_all)

    # Aggregate means fast using bincount
    # Build per-point window index array
    point_win = np.empty(X_all.shape[0], dtype=np.int32)
    for i in range(n_windows):
        if not valid_mask[i]:
            continue
        a, b = offsets[i]
        point_win[a:b] = i

    counts = np.bincount(point_win, minlength=n_windows).astype(np.float32)
    sum1 = np.bincount(point_win, weights=err1_all.astype(np.float32), minlength=n_windows).astype(np.float32)
    sum2 = np.bincount(point_win, weights=err2_all.astype(np.float32), minlength=n_windows).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        mse1_mean = (sum1 / np.maximum(counts, 1.0)).astype(np.float32)
        mse2_mean = (sum2 / np.maximum(counts, 1.0)).astype(np.float32)

    # Medians: computed per-window from slices (still cheap compared to predict-per-window)
    for i in range(n_windows):
        if not valid_mask[i]:
            continue
        a, b = offsets[i]
        if b > a:
            mse1_median[i] = float(np.median(err1_all[a:b]))
            mse2_median[i] = float(np.median(err2_all[a:b]))

    # -------------------------------------------------------------------------
    # Topology (still per-window, but strided/subsampled)
    # -------------------------------------------------------------------------
    wdist1 = {k: np.full(n_windows, np.nan, dtype=np.float32) for k in range(cfg.TDA_MAXDIM + 1)}
    wdist2 = {k: np.full(n_windows, np.nan, dtype=np.float32) for k in range(cfg.TDA_MAXDIM + 1)}

    if cfg.DO_TDA and (HAS_RIPSER and HAS_PERSIM):
        for i in range(n_windows):
            if not valid_mask[i]:
                continue
            if i % cfg.TDA_STRIDE != 0:
                continue

            emb = emb_list[i].astype(np.float32)
            emb_norm = ((emb - mean_emb) / std_emb).astype(np.float32)

            # Reconstruct from cached global predictions: slice
            a, b = offsets[i]
            rec1 = rec1_all[a:b]
            rec2 = rec2_all[a:b]

            Xo = subsample_points(emb_norm, cfg.TDA_SUBSAMPLE, seed=cfg.SEED + i)
            X1 = subsample_points(rec1, cfg.TDA_SUBSAMPLE, seed=cfg.SEED + 10_000 + i)
            X2 = subsample_points(rec2, cfg.TDA_SUBSAMPLE, seed=cfg.SEED + 20_000 + i)

            dg_o = compute_diagrams(Xo, maxdim=cfg.TDA_MAXDIM)
            dg_1 = compute_diagrams(X1, maxdim=cfg.TDA_MAXDIM)
            dg_2 = compute_diagrams(X2, maxdim=cfg.TDA_MAXDIM)

            for k in range(cfg.TDA_MAXDIM + 1):
                try:
                    wdist1[k][i] = float(wasserstein(dg_o[k], dg_1[k]))
                    wdist2[k][i] = float(wasserstein(dg_o[k], dg_2[k]))
                except Exception:
                    wdist1[k][i] = np.nan
                    wdist2[k][i] = np.nan

            if i % 25 == 0:
                gc.collect()

    # Interpolate NaNs from stride (only if DO_TDA=True)
    def interpolate_nans(y: np.ndarray) -> np.ndarray:
        y = y.astype(np.float32)
        idx = np.arange(len(y))
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return y
        return np.interp(idx, idx[mask], y[mask]).astype(np.float32)

    if cfg.DO_TDA:
        for k in range(cfg.TDA_MAXDIM + 1):
            wdist1[k] = interpolate_nans(wdist1[k])
            wdist2[k] = interpolate_nans(wdist2[k])

    # -------------------------------------------------------------------------
    # Stability / early-warning scores
    # -------------------------------------------------------------------------
    z_mse2 = robust_zscore(np.nan_to_num(mse2_mean, nan=np.nanmedian(mse2_mean)))

    if cfg.DO_TDA:
        topo2 = np.zeros(n_windows, dtype=np.float32)
        for k in range(cfg.TDA_MAXDIM + 1):
            topo2 += np.nan_to_num(wdist2[k], nan=np.nanmedian(wdist2[k]))
        z_topo2 = robust_zscore(topo2)
    else:
        topo2 = np.zeros(n_windows, dtype=np.float32)
        z_topo2 = np.zeros(n_windows, dtype=np.float32)

    score2_raw = (0.55 * z_mse2 + 0.45 * z_topo2).astype(np.float32)
    score2 = moving_average(score2_raw, cfg.SMOOTH_WIN)

    z_score2 = robust_zscore(score2)
    alert_mask = z_score2 > cfg.ALERT_Z

    cp_idxs = []
    if cfg.USE_CHANGEPOINT:
        cp_idxs = simple_changepoints(score2, min_distance=cfg.CHANGEPOINT_MIN_DISTANCE)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    plot_metric_time(
        centers_sec, mse1_mean,
        fig_dir / "mse_AE1D_mean_over_time.png",
        title="Window mean reconstruction error (AE-1D)",
        ylabel="Mean MSE",
        extra_lines=markers if markers else None
    )
    plot_metric_time(
        centers_sec, mse2_mean,
        fig_dir / "mse_AE2D_mean_over_time.png",
        title="Window mean reconstruction error (AE-2D)",
        ylabel="Mean MSE",
        extra_lines=markers if markers else None
    )
    if cfg.DO_TDA:
        plot_metric_time(
            centers_sec, topo2,
            fig_dir / "topo_distance_sum_AE2D_over_time.png",
            title="Topology deformation proxy (sum Wasserstein distances, AE-2D)",
            ylabel="Sum Wasserstein",
            extra_lines=markers if markers else None
        )

    plt.figure(figsize=(12, 4))
    plt.plot(centers_sec, score2, linewidth=1.4, label="score (smoothed)")
    plt.plot(centers_sec, z_score2, linewidth=1.0, alpha=0.6, label="robust z-score")

    if markers:
        for name, sec in markers.items():
            plt.axvline(sec, linestyle="--", linewidth=1.0, label=name)

    if alert_mask.any():
        plt.scatter(centers_sec[alert_mask], score2[alert_mask], s=18, label="alerts")

    for cp in cp_idxs[:10]:
        plt.axvline(float(centers_sec[cp]), linestyle=":", linewidth=1.0)

    plt.title("Early-warning stability curve (AE-2D + topology)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Score / z-score")
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "early_warning_score.png", dpi=160)
    plt.close()

    # Illustrative topology plots (best score window)
    best_idx = int(np.argmax(z_score2))

    if cfg.DO_TDA and HAS_PERSIM and HAS_RIPSER and valid_mask[best_idx]:
        emb = emb_list[best_idx].astype(np.float32)
        emb_norm = ((emb - mean_emb) / std_emb).astype(np.float32)

        a, b = offsets[best_idx]
        rec1 = rec1_all[a:b]
        rec2 = rec2_all[a:b]

        Xo = subsample_points(emb_norm, cfg.TDA_SUBSAMPLE, seed=cfg.SEED + best_idx)
        X1 = subsample_points(rec1, cfg.TDA_SUBSAMPLE, seed=cfg.SEED + 10_000 + best_idx)
        X2 = subsample_points(rec2, cfg.TDA_SUBSAMPLE, seed=cfg.SEED + 20_000 + best_idx)

        dg_o = compute_diagrams(Xo, maxdim=cfg.TDA_MAXDIM)
        dg_1 = compute_diagrams(X1, maxdim=cfg.TDA_MAXDIM)
        dg_2 = compute_diagrams(X2, maxdim=cfg.TDA_MAXDIM)

        plot_diagrams_side_by_side(
            dg_o, dg_1,
            fig_dir / "persistence_diagrams_original_vs_AE1D.png",
            title_a="Original (best-score window)",
            title_b="AE-1D reconstruction"
        )
        plot_diagrams_side_by_side(
            dg_o, dg_2,
            fig_dir / "persistence_diagrams_original_vs_AE2D.png",
            title_a="Original (best-score window)",
            title_b="AE-2D reconstruction"
        )

        all_vals = []
        for k in range(cfg.TDA_MAXDIM + 1):
            for d in [dg_o[k], dg_1[k], dg_2[k]]:
                if len(d) > 0:
                    finite = d[np.isfinite(d[:, 1])]
                    if len(finite) > 0:
                        all_vals.append(float(np.max(finite[:, 1])))
        tmax = max(all_vals) if len(all_vals) > 0 else 1.0

        tgrid = np.linspace(0.0, tmax, cfg.BETTI_NGRID).astype(np.float32)
        betti_dict = {"Original": {}, "AE1D": {}, "AE2D": {}}
        for k in range(cfg.TDA_MAXDIM + 1):
            betti_dict["Original"][k] = betti_curve_from_diagram(dg_o[k], tgrid)
            betti_dict["AE1D"][k] = betti_curve_from_diagram(dg_1[k], tgrid)
            betti_dict["AE2D"][k] = betti_curve_from_diagram(dg_2[k], tgrid)

        plot_betti_curves(betti_dict, tgrid, fig_dir / "betti_curves_best_score.png")

    # -------------------------------------------------------------------------
    # Export Excel report + extended lightweight reporting (NO extra heavy compute)
    # -------------------------------------------------------------------------
    engine = None
    try:
        import openpyxl  # noqa: F401
        engine = "openpyxl"
    except Exception:
        engine = None

    df = pd.DataFrame({
        "window_index": np.arange(n_windows, dtype=int),
        "center_time_sec": centers_sec.astype(np.float32),
        "valid_window": valid_mask.astype(int),
        "mse_AE1D_mean": mse1_mean.astype(np.float32),
        "mse_AE2D_mean": mse2_mean.astype(np.float32),
        "mse_AE1D_median": mse1_median.astype(np.float32),
        "mse_AE2D_median": mse2_median.astype(np.float32),
        "score_AE2D_topo": score2.astype(np.float32),
        "score_z": z_score2.astype(np.float32),
        "alert": alert_mask.astype(int)
    })

    if cfg.DO_TDA:
        for k in range(cfg.TDA_MAXDIM + 1):
            df[f"wasserstein_AE1D_H{k}"] = wdist1[k].astype(np.float32)
            df[f"wasserstein_AE2D_H{k}"] = wdist2[k].astype(np.float32)
        df["topo_sum_AE2D"] = topo2.astype(np.float32)

    lead_time_sec = np.nan
    earliest_alert_sec = np.nan
    if (not cfg.USE_REAL_DATA) and markers and alert_mask.any():
        earliest_alert_sec = float(np.min(centers_sec[alert_mask]))
        lead_time_sec = float(markers["event time"] - earliest_alert_sec)

    summary = {
        "n_windows": int(n_windows),
        "baseline_windows_used_for_training": int(len(baseline_idxs)),
        "best_score_window_index": int(best_idx),
        "best_score_time_sec": float(centers_sec[best_idx]),
        "best_score_z": float(z_score2[best_idx]),
        "earliest_alert_time_sec": float(earliest_alert_sec) if np.isfinite(earliest_alert_sec) else None,
        "lead_time_sec_event_minus_earliest_alert": float(lead_time_sec) if np.isfinite(lead_time_sec) else None
    }

    cfg_dict = asdict(cfg)
    meta_out = meta.copy()
    meta_out["computed_n_windows"] = float(n_windows)

    # Extended lightweight reporting
    emb_stats_rows = []
    for i in range(n_windows):
        E = np.asarray(emb_list[i], dtype=np.float32)
        if (E.size == 0) or (not np.all(np.isfinite(E))):
            row = {
                "window_index": int(i),
                "center_time_sec": float(centers_sec[i]),
                "n_points": int(E.shape[0]) if E.ndim == 2 else 0,
                "emb_l2_mean": np.nan,
                "emb_l2_std": np.nan,
            }
            for d in range(cfg.EMBED_DIM):
                row[f"emb_mean_dim{d}"] = np.nan
                row[f"emb_std_dim{d}"] = np.nan
            emb_stats_rows.append(row)
            continue

        l2 = np.linalg.norm(E, axis=1)
        row = {
            "window_index": int(i),
            "center_time_sec": float(centers_sec[i]),
            "n_points": int(E.shape[0]),
            "emb_l2_mean": float(np.mean(l2)),
            "emb_l2_std": float(np.std(l2)),
        }
        mu = np.mean(E, axis=0)
        sd = np.std(E, axis=0)
        for d in range(cfg.EMBED_DIM):
            row[f"emb_mean_dim{d}"] = float(mu[d])
            row[f"emb_std_dim{d}"] = float(sd[d])
        emb_stats_rows.append(row)

    df_emb = pd.DataFrame(emb_stats_rows)

    n_valid = int(np.sum(valid_mask))
    n_alert = int(np.sum(alert_mask))
    n_cp = int(len(cp_idxs)) if cfg.USE_CHANGEPOINT else 0
    n_tda_eval = int(df["topo_sum_AE2D"].notna().sum()) if (cfg.DO_TDA and ("topo_sum_AE2D" in df.columns)) else 0

    df_global_plus = pd.DataFrame([{
        "n_windows_total": int(n_windows),
        "n_windows_valid": n_valid,
        "valid_ratio": float(n_valid / max(1, int(n_windows))),
        "n_alert_windows": n_alert,
        "alert_ratio": float(n_alert / max(1, int(n_windows))),
        "n_changepoints": n_cp,
        "tda_enabled": int(bool(cfg.DO_TDA)),
        "n_tda_evaluated_windows": n_tda_eval,
        "tda_eval_ratio": float(n_tda_eval / max(1, int(n_windows))) if cfg.DO_TDA else 0.0,
        "window_sec": float(cfg.WINDOW_SEC),
        "hop_sec": float(cfg.HOP_SEC),
        "embed_dim": int(cfg.EMBED_DIM),
        "embed_delay": int(cfg.EMBED_DELAY),
        "ae_epochs": int(cfg.EPOCHS),
    }])

    percentile_rows = []
    metric_candidates = ["mse_AE1D_mean", "mse_AE2D_mean", "score_AE2D_topo", "score_z"]
    if cfg.DO_TDA and ("topo_sum_AE2D" in df.columns):
        metric_candidates += ["topo_sum_AE2D"]
        for k in range(cfg.TDA_MAXDIM + 1):
            c = f"wasserstein_AE2D_H{k}"
            if c in df.columns:
                metric_candidates.append(c)

    for col in metric_candidates:
        v = df[col].to_numpy(dtype=np.float32)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        percentile_rows.append({
            "metric": col,
            "p01": float(np.percentile(v, 1)),
            "p05": float(np.percentile(v, 5)),
            "p10": float(np.percentile(v, 10)),
            "p25": float(np.percentile(v, 25)),
            "p50": float(np.percentile(v, 50)),
            "p75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "max": float(np.max(v)),
        })
    df_percentiles = pd.DataFrame(percentile_rows)

    smooth_w = max(1, int(cfg.SMOOTH_WIN))
    df_smooth = pd.DataFrame({
        "window_index": df["window_index"].to_numpy(dtype=int),
        "center_time_sec": df["center_time_sec"].to_numpy(dtype=np.float32),
        "mse_AE1D_mean_smooth": moving_average(df["mse_AE1D_mean"].to_numpy(dtype=np.float32), smooth_w),
        "mse_AE2D_mean_smooth": moving_average(df["mse_AE2D_mean"].to_numpy(dtype=np.float32), smooth_w),
        "score_z_smooth": moving_average(df["score_z"].to_numpy(dtype=np.float32), smooth_w),
    })
    if cfg.DO_TDA and ("topo_sum_AE2D" in df.columns):
        df_smooth["topo_sum_AE2D_smooth"] = moving_average(df["topo_sum_AE2D"].to_numpy(dtype=np.float32), smooth_w)

    df_drift = pd.DataFrame({
        "window_index": df["window_index"].to_numpy(dtype=int),
        "center_time_sec": df["center_time_sec"].to_numpy(dtype=np.float32),
        "delta_mse_AE2D_mean": np.r_[np.nan, np.diff(df["mse_AE2D_mean"].to_numpy(dtype=np.float32))],
        "abs_delta_mse_AE2D_mean": np.r_[np.nan, np.abs(np.diff(df["mse_AE2D_mean"].to_numpy(dtype=np.float32)))],
        "delta_score_z": np.r_[np.nan, np.diff(df["score_z"].to_numpy(dtype=np.float32))],
        "abs_delta_score_z": np.r_[np.nan, np.abs(np.diff(df["score_z"].to_numpy(dtype=np.float32)))],
    })
    if cfg.DO_TDA and ("topo_sum_AE2D" in df.columns):
        topo = df["topo_sum_AE2D"].to_numpy(dtype=np.float32)
        df_drift["delta_topo_sum_AE2D"] = np.r_[np.nan, np.diff(topo)]
        df_drift["abs_delta_topo_sum_AE2D"] = np.r_[np.nan, np.abs(np.diff(topo))]

    excel_path = rep_dir / "SEISMIC_TOPO_DEEP_REPORT.xlsx"
    with pd.ExcelWriter(excel_path, engine=engine) as writer:
        df.to_excel(writer, sheet_name="window_metrics", index=False)
        pd.DataFrame([cfg_dict]).to_excel(writer, sheet_name="config", index=False)
        pd.DataFrame([meta_out]).to_excel(writer, sheet_name="meta", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)

        if cfg.USE_CHANGEPOINT:
            df_cp = pd.DataFrame({
                "changepoint_index": cp_idxs,
                "changepoint_time_sec": [float(centers_sec[i]) for i in cp_idxs]
            })
            df_cp.to_excel(writer, sheet_name="changepoints", index=False)

        df_global_plus.to_excel(writer, sheet_name="global_plus", index=False)
        df_emb.to_excel(writer, sheet_name="embedding_stats", index=False)
        df_percentiles.to_excel(writer, sheet_name="percentiles", index=False)
        df_smooth.to_excel(writer, sheet_name="smoothed_metrics", index=False)
        df_drift.to_excel(writer, sheet_name="drift_proxies", index=False)

    with open(rep_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "config": cfg_dict, "meta": meta_out}, f, indent=2)

    # Additional CSV exports
    try:
        df.to_csv(rep_dir / "window_metrics_flat.csv", index=False)
        df_emb.to_csv(rep_dir / "embedding_stats_flat.csv", index=False)
        df_percentiles.to_csv(rep_dir / "metric_percentiles_flat.csv", index=False)
        df_smooth.to_csv(rep_dir / "smoothed_metrics_flat.csv", index=False)
        df_drift.to_csv(rep_dir / "drift_proxies_flat.csv", index=False)
    except Exception as e:
        print(f"[WARN] CSV export failed: {e}")

    # Extra figures derived from existing series (no extra TDA)
    try:
        tsec = df["center_time_sec"].to_numpy(dtype=np.float32)

        if cfg.DO_TDA and ("topo_sum_AE2D" in df.columns):
            fig = plt.figure(figsize=(11, 4))
            ax1 = fig.add_subplot(111)
            ax1.plot(tsec, df["mse_AE2D_mean"].to_numpy(dtype=np.float32), label="MSE AE-2D (mean)")
            ax1.set_xlabel("Time (sec)")
            ax1.set_ylabel("AE-2D MSE")
            ax2 = ax1.twinx()
            ax2.plot(tsec, df["topo_sum_AE2D"].to_numpy(dtype=np.float32), label="Topo sum (AE-2D)")
            ax2.set_ylabel("Topological deformation (sum)")
            ax1.set_title("AE reconstruction error vs topological deformation")
            fig.tight_layout()
            fig.savefig(fig_dir / "ae2d_mse_vs_topo_sum.png", dpi=200)
            plt.close(fig)

        fig = plt.figure(figsize=(11, 4))
        ax = fig.add_subplot(111)
        ax.plot(df_emb["center_time_sec"].to_numpy(dtype=np.float32),
                df_emb["emb_l2_mean"].to_numpy(dtype=np.float32))
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Embedding L2 mean")
        ax.set_title("Delay-embedding magnitude over time (mean L2 norm)")
        fig.tight_layout()
        fig.savefig(fig_dir / "embedding_l2_mean_over_time.png", dpi=200)
        plt.close(fig)

        if cfg.DO_TDA:
            c0 = "wasserstein_AE2D_H0"
            c1 = "wasserstein_AE2D_H1"
            if (c0 in df.columns) and (c1 in df.columns):
                x0 = df[c0].to_numpy(dtype=np.float32)
                x1 = df[c1].to_numpy(dtype=np.float32)
                mask = np.isfinite(x0) & np.isfinite(x1)
                if np.any(mask):
                    fig = plt.figure(figsize=(6, 5))
                    ax = fig.add_subplot(111)
                    sc = ax.scatter(x0[mask], x1[mask], c=tsec[mask], s=12)
                    ax.set_xlabel("Wasserstein (AE-2D) H0")
                    ax.set_ylabel("Wasserstein (AE-2D) H1")
                    ax.set_title("Topological phase portrait (colored by time)")
                    fig.colorbar(sc, ax=ax, label="Time (sec)")
                    fig.tight_layout()
                    fig.savefig(fig_dir / "topo_phase_wass_H0_vs_H1.png", dpi=220)
                    plt.close(fig)

        fig = plt.figure(figsize=(11, 4))
        ax = fig.add_subplot(111)
        ax.plot(tsec, df["score_z"].to_numpy(dtype=np.float32), label="Score z")
        if np.any(alert_mask):
            ax.scatter(tsec[alert_mask], df["score_z"].to_numpy(dtype=np.float32)[alert_mask],
                       s=20, marker="o", label="Alerts")
        if cfg.USE_CHANGEPOINT and (len(cp_idxs) > 0):
            for i_cp in cp_idxs:
                ax.axvline(float(tsec[i_cp]), linestyle="--", linewidth=1.0)
        if (not cfg.USE_REAL_DATA) and markers and ("event time" in markers):
            ax.axvline(float(markers["event time"]), linestyle="-", linewidth=2.0)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Score z")
        ax.set_title("Unified timeline: score_z, alerts, changepoints, event")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / "timeline_score_alerts_changepoints.png", dpi=220)
        plt.close(fig)

    except Exception as e:
        print(f"[WARN] Extra plotting failed: {e}")

    # README
    try:
        readme = []
        readme.append("# Results Artifacts\n")
        readme.append("This folder contains lightweight, presentation-oriented artifacts derived from already-computed arrays.\n")
        readme.append("No additional persistent homology computations are performed by these exports.\n\n")

        readme.append("## Excel report\n")
        readme.append("- `SEISMIC_TOPO_DEEP_REPORT.xlsx` includes:\n")
        readme.append("  - `window_metrics`: core per-window metrics.\n")
        readme.append("  - `config`, `meta`, `summary`, `changepoints` (if enabled).\n")
        readme.append("  - `global_plus`: extended global counts/coverage.\n")
        readme.append("  - `embedding_stats`: per-window delay-embedding descriptive statistics.\n")
        readme.append("  - `percentiles`: distributional summaries.\n")
        readme.append("  - `smoothed_metrics`: moving-average versions.\n")
        readme.append("  - `drift_proxies`: deltas between consecutive windows.\n\n")

        readme.append("## CSV exports\n")
        readme.append("- `window_metrics_flat.csv`\n")
        readme.append("- `embedding_stats_flat.csv`\n")
        readme.append("- `metric_percentiles_flat.csv`\n")
        readme.append("- `smoothed_metrics_flat.csv`\n")
        readme.append("- `drift_proxies_flat.csv`\n\n")

        readme.append("## Extra figures\n")
        readme.append("- `ae2d_mse_vs_topo_sum.png` (if topology enabled)\n")
        readme.append("- `embedding_l2_mean_over_time.png`\n")
        readme.append("- `topo_phase_wass_H0_vs_H1.png` (if Wasserstein H0/H1 available)\n")
        readme.append("- `timeline_score_alerts_changepoints.png`\n")

        (rep_dir / "README_results.md").write_text("".join(readme), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] README export failed: {e}")

    print("=" * 80)
    print("[DONE] Pipeline finished successfully.")
    print(f"[OUTPUT] Figures: {fig_dir}")
    print(f"[OUTPUT] Report:  {excel_path}")
    print(f"[OUTPUT] Models:  {mdl_dir if cfg.SAVE_MODELS else '(not saved)'}")
    print("=" * 80)


# =============================================================================
# 10. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    cfg = Config()

    # -------------------------------------------------------------------------
    # FAST + FULL-ARTIFACTS PRESET (enabled by default in this modified version)
    # -------------------------------------------------------------------------
    # If a full 3-hour run is desired, set:
    #   cfg.FAST_PRESET = False
    #
    # If stronger topology (H2) is desired and compute budget allows:
    #   cfg.FAST_PRESET = False
    #   cfg.EMBED_DIM = 5
    #   cfg.TDA_MAXDIM = 2
    #   cfg.TDA_SUBSAMPLE = 600
    # -------------------------------------------------------------------------

    cfg = apply_fast_preset(cfg)
    run_pipeline(cfg)


