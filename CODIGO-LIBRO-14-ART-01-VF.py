#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Topological Intelligence for Anomaly Detection in Smart Electrical Grids
============================================================================

This script implements an executable research pipeline inspired by Sections 1, 2,
3 and 4 of the attached manuscript:

- Section 1: The problem is framed as anomaly detection in a cyber-physical smart
  grid where physical, informational and malicious anomalies reorganize the joint
  geometry of the operating state.
- Section 2: The implementation follows the manuscript's mathematical logic:
  graph-temporal state construction, residual and power-balance consistency,
  deep graph-temporal representation learning, persistent homology in latent
  space, and regime-aware dual-evidence scoring.
- Section 3: The methodology is operationalized as a reproducible workflow:
  robust normalization, smoothing, windowing, regime conditioning, deep learning,
  persistent homology, dual-evidence scoring, persistence-based alerting,
  localization and evaluation.
- Section 4: The code is organized as a modular computational pipeline that
  produces audit-ready artifacts and separates results into folders aligned with
  manuscript sections 5.1, 5.2 and 5.3.

Important note
--------------
The script is designed to run end-to-end even when no real utility dataset is
available. It includes a synthetic smart-grid cyber-physical simulator that
generates:
    * normal operating regimes,
    * physical faults and topology inconsistencies,
    * cyber attacks and data-quality anomalies,
    * graph-aware telemetry,
    * labels for evaluation and explainability studies.

The generated result folders are intentionally organized so that figures and
tables can be inserted directly into:
    * 5.1 Detection of Physical Faults and Topology Inconsistencies
    * 5.2 Identification of Cyber Events and Data-Quality Anomalies
    * 5.3 Localization, Explainability, and Operational Decision Support

Requested libraries
-------------------
This implementation explicitly uses:
    * torch         -> deep graph-temporal autoencoding/prediction
    * ripser        -> persistent homology
    * persim        -> persistence diagrams and diagram distances
    * kmapper       -> Mapper graph generation and visualization
    * toponetx      -> simplicial-complex construction for higher-order
                       explainability analysis

The library GUDHI is intentionally NOT used.

Typical execution
-----------------
    python smart_grid_topological_pipeline.py --output_root results_smart_grid

The script saves:
    * figures (.png),
    * Mapper HTML visualization (.html),
    * metrics (.csv),
    * incident summaries (.json),
    * model weights (.pt),
    * a manifest linking every artifact to section 5.1 / 5.2 / 5.3.


Manuscript author metadata
--------------------------
Jaime Aguilar-Ortiz1, Francisco R. Trejo-Macotela1*, Ocotlan Diaz-Parra1,
Jorge A. Ruiz-Vanoye1, Marco A. Vera-Jimenez1, Carlos R. Dominguez-Mayorga1

1 Direccion de Investigacion, Innovacion y Posgrado, Universidad Politecnica de
Pachuca, Carretera Pachuca-Cd. Sahagun Km 20, Ex-Hacienda de Santa Barbara,
Zempoala, Hidalgo 43830, Mexico.

Corresponding author e-mail: trejo_macotela@upp.edu.mx
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Optional imports for the topological stack.  The script checks them explicitly
# in `require_topological_stack()` and provides a clear installation message if
# any package is missing.  The import pattern below allows static syntax checks
# to succeed even when the current environment does not ship these libraries.
# -----------------------------------------------------------------------------

try:
    import kmapper as km
    from kmapper import adapter as km_adapter
    from kmapper import draw_matplotlib as km_draw_matplotlib
except Exception:  # pragma: no cover - runtime dependency check handles this
    km = None
    km_adapter = None
    km_draw_matplotlib = None

try:
    import toponetx as tnx
except Exception:  # pragma: no cover
    tnx = None

try:
    from ripser import ripser
except Exception:  # pragma: no cover
    ripser = None

try:
    from persim import bottleneck, plot_diagrams, wasserstein
except Exception:  # pragma: no cover
    bottleneck = None
    plot_diagrams = None
    wasserstein = None


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Configuration for the synthetic smart-grid simulator."""

    n_rows: int = 3
    n_cols: int = 6
    total_steps: int = 1320              # about 4.5 days at 5-minute resolution
    sampling_minutes: int = 5
    seed: int = 42

    # Windowing and regime conditioning
    window_length: int = 24              # 2 hours if 5-minute measurements
    prediction_horizon: int = 1
    smoothing_alpha: float = 0.30
    n_regimes: int = 4

    # Graph and electrical simulation
    nominal_voltage: float = 1.0
    nominal_frequency: float = 60.0
    edge_weight_low: float = 1.0
    edge_weight_high: float = 2.5

    # Feature engineering
    clip_value: float = 6.0
    epsilon: float = 1e-6

    # Dataset split (chronological)
    train_fraction: float = 0.55
    val_fraction: float = 0.20

    # Topological horizon over latent windows
    topological_horizon: int = 18
    betti_grid_points: int = 25
    landscape_grid_points: int = 48
    landscape_layers: int = 3

    # Fast topological approximation
    fast_topology_mode: bool = True
    sparse_rips_epsilon: float = 0.20
    incremental_topology: bool = True
    sparse_filtration_min_points: int = 10

    # Simulator behavior
    observation_noise_std: float = 0.010
    angle_noise_std: float = 0.004
    power_noise_std: float = 0.030
    frequency_noise_std: float = 0.008
    derivative_scale: float = 1.0

    # Visualization and publication outputs
    top_k_localization_nodes: int = 8
    max_mapper_points: int = 320
    mapper_label_top_k: int = 10
    publication_table_precision: int = 4
    figure_dpi: int = 240


@dataclass
class ModelConfig:
    """Configuration for the deep graph-temporal encoder."""

    hidden_dim: int = 32
    temporal_dim: int = 48
    latent_dim: int = 16
    dropout: float = 0.15
    batch_size: int = 48
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 6

    # Loss coefficients aligned with the manuscript
    lambda_recon: float = 1.0
    lambda_pred: float = 1.0
    lambda_contrast: float = 0.12
    lambda_topology_proxy: float = 0.08

    # Contrastive settings
    contrastive_margin: float = 1.25
    augmentation_noise_std: float = 0.03


@dataclass
class DetectionConfig:
    """Configuration for anomaly scoring, calibration and persistence."""

    # Statistical score weights
    a_recon: float = 0.35
    a_pred: float = 0.30
    a_residual: float = 0.20
    a_uncertainty: float = 0.15

    # Topological score weights
    b_fragmentation: float = 0.25
    b_loop: float = 0.20
    b_summary: float = 0.25
    b_reference_distance: float = 0.15
    b_transition: float = 0.15

    # Composite score
    lambda_statistical: float = 0.55

    # Alert calibration
    alert_tail_probability: float = 0.025
    persistence_k: int = 2
    persistence_h: int = 3

    # Topology parameters
    max_homology_dim: int = 1
    loop_distance_balance: float = 0.50
    reference_sample_limit: int = 35

    # Explainability fusion
    gradient_weight: float = 0.70
    physics_weight: float = 0.30


@dataclass
class OutputPaths:
    """Container for output folders."""

    root: Path
    global_dir: Path
    sec_5_1: Path
    sec_5_2: Path
    sec_5_3: Path


@dataclass
class ExperimentConfig:
    """Master configuration that groups all sub-configurations."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    output_root: str = "results_smart_grid_topology"


# -----------------------------------------------------------------------------
# Small utility helpers
# -----------------------------------------------------------------------------

FEATURE_NAMES = [
    "voltage",
    "angle",
    "active_power",
    "reactive_power",
    "frequency",
    "state_residual",
    "balance_inconsistency",
    "breaker_ratio",
    "renewable_share",
    "missing_indicator",
    "d_voltage",
    "d_frequency",
]

PHYSICAL_SUBTYPES = {
    "line_trip",
    "transformer_stress",
    "oscillation",
    "topology_inconsistency",
}

CYBER_DATA_SUBTYPES = {
    "false_data_injection",
    "replay_attack",
    "missingness_burst",
    "timestamp_disorder",
    "sensor_drift",
}



def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def select_torch_device() -> torch.device:
    """
    Select a safe torch device.

    Some Windows/Anaconda installations expose CUDA as available even though the
    runtime later fails during tensor allocation.  This helper validates the
    device before the pipeline commits to it.
    """
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([0.0], device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            warnings.warn(
                f"CUDA was reported as available but could not be initialized ({exc}). Falling back to CPU."
            )
    return torch.device("cpu")


def require_topological_stack() -> None:
    """
    Stop execution early if the requested topological libraries are missing.

    The user explicitly asked for kmapper / KeplerMapper, TopoNetX, torch,
    Ripser and Persim.  This function provides a single, clear installation
    message rather than allowing failures to appear deep inside the pipeline.
    """
    missing = []
    if km is None or km_adapter is None or km_draw_matplotlib is None:
        missing.append("kmapper>=2.1.0")
    if tnx is None:
        missing.append("toponetx>=0.4.0")
    if ripser is None:
        missing.append("ripser>=0.6.14")
    if bottleneck is None or plot_diagrams is None or wasserstein is None:
        missing.append("persim>=0.3.8")

    if missing:
        install_cmd = "pip install " + " ".join(missing)
        message = (
            "The required topological stack is incomplete.\n"
            f"Missing packages: {', '.join(missing)}\n"
            f"Install them with:\n    {install_cmd}\n"
            "The script intentionally does not use GUDHI."
        )
        raise ImportError(message)


def build_output_tree(root: str) -> OutputPaths:
    """Create the output folder structure aligned with manuscript sections."""
    root_path = Path(root)
    global_dir = root_path / "global_artifacts"
    sec_5_1 = root_path / "section_5_1_physical_faults_and_topology"
    sec_5_2 = root_path / "section_5_2_cyber_and_data_quality"
    sec_5_3 = root_path / "section_5_3_localization_and_decision_support"

    for path in [root_path, global_dir, sec_5_1, sec_5_2, sec_5_3]:
        path.mkdir(parents=True, exist_ok=True)

    return OutputPaths(
        root=root_path,
        global_dir=global_dir,
        sec_5_1=sec_5_1,
        sec_5_2=sec_5_2,
        sec_5_3=sec_5_3,
    )


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file with consistent formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def register_artifact(
    manifest: List[Dict[str, str]],
    section: str,
    path: Path,
    description: str,
) -> None:
    """Append one artifact entry to the section manifest."""
    manifest.append(
        {
            "section": section,
            "file": str(path),
            "description": description,
        }
    )



def flatten_columns_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns so that CSV and LaTeX exports remain stable."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join([str(part) for part in col if str(part) != ""]).strip()
            for col in out.columns.to_flat_index()
        ]
    else:
        out.columns = [str(col) for col in out.columns]
    return out


def save_table_bundle(
    df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    caption: str,
    label: str,
    precision: int = 4,
) -> pd.DataFrame:
    """
    Save a table as CSV and LaTeX, ready for manuscript insertion.

    The LaTeX export uses a flattened column structure and a controlled
    float format so the table can be copied directly into the chapter.
    """
    flat = flatten_columns_for_export(df)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(csv_path, index=False)

    float_format = lambda x: f"{x:.{precision}f}" if pd.notnull(x) and np.isfinite(x) else ""
    latex = flat.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        na_rep="",
        float_format=float_format,
    )
    tex_path.write_text(latex, encoding="utf-8")
    return flat


def robust_center_scale(values: np.ndarray, epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute robust centering and scaling statistics.

    Median / IQR scaling is used instead of mean / standard deviation because the
    manuscript explicitly highlights the need for outlier-resistant geometry.
    """
    median = np.nanmedian(values, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr < epsilon, 1.0, iqr)
    return median, iqr


def robust_positive_zscore(values: np.ndarray, reference: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Convert a positive-valued score into a robust z-like deviation above normal.

    Negative values are clipped to zero because the anomaly logic only needs to
    accumulate excess deviation above the normal reference.
    """
    z = (values - reference) / (scale + 1e-8)
    return np.clip(z, 0.0, None)


def rolling_persistent_alert(raw_alerts: np.ndarray, k: int, h: int) -> np.ndarray:
    """Implement the persistence rule described in the manuscript."""
    persistent = np.zeros_like(raw_alerts, dtype=int)
    for i in range(len(raw_alerts)):
        start = max(0, i - h + 1)
        if raw_alerts[start : i + 1].sum() >= k:
            persistent[i] = 1
    return persistent


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    This metric is included because the manuscript emphasizes that operational
    alerting systems must be calibrated, not merely rank-sensitive.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / max(n, 1)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score."""
    return float(np.mean((y_prob - y_true) ** 2))


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid used in fallback calibration."""
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------------------------------------------------------
# Synthetic smart-grid construction
# -----------------------------------------------------------------------------

def build_grid_graph(cfg: SimulationConfig) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]], Dict[int, str], Dict[int, str]]:
    """
    Create a moderate-size smart-grid graph.

    The graph is based on a 2D lattice because it produces a visually readable
    substation/feeder layout.  A few edges are removed to avoid an unrealistically
    dense mesh, while a few cross-links are kept so that topology changes and
    loops remain meaningful for the analysis.
    """
    G = nx.Graph()
    positions: Dict[int, Tuple[float, float]] = {}
    node_roles: Dict[int, str] = {}
    regions: Dict[int, str] = {}

    node_id = 0
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            G.add_node(node_id)
            positions[node_id] = (float(c), float(cfg.n_rows - 1 - r))
            if r == 0 and c in {1, 3, 5}:
                node_roles[node_id] = "renewable"
            elif r == 1 and c in {0, 4}:
                node_roles[node_id] = "storage"
            else:
                node_roles[node_id] = "load"

            if c <= 1:
                regions[node_id] = "west"
            elif c <= 3:
                regions[node_id] = "central"
            else:
                regions[node_id] = "east"

            node_id += 1

    def node_at(row: int, col: int) -> int:
        return row * cfg.n_cols + col

    # Base lattice connections.
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            current = node_at(r, c)
            if c + 1 < cfg.n_cols:
                G.add_edge(current, node_at(r, c + 1))
            if r + 1 < cfg.n_rows:
                G.add_edge(current, node_at(r + 1, c))

    # Remove a few links to avoid an over-meshed graph.
    edges_to_remove = [
        (node_at(0, 2), node_at(1, 2)),
        (node_at(1, 1), node_at(1, 2)),
        (node_at(1, 4), node_at(2, 4)),
    ]
    for edge in edges_to_remove:
        if G.has_edge(*edge):
            G.remove_edge(*edge)

    # Add some cross-links to preserve meaningful loop structure.
    extra_edges = [
        (node_at(0, 0), node_at(1, 1)),
        (node_at(0, 3), node_at(1, 4)),
        (node_at(1, 2), node_at(2, 3)),
        (node_at(0, 4), node_at(2, 5)),
    ]
    for edge in extra_edges:
        G.add_edge(*edge)

    # Electrical weights.
    rng = np.random.default_rng(cfg.seed)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(cfg.edge_weight_low, cfg.edge_weight_high))
        G[u][v]["capacity"] = float(rng.uniform(0.9, 1.8))

    return G, positions, node_roles, regions


def edge_index_map(G: nx.Graph) -> List[Tuple[int, int]]:
    """Return a stable ordered edge list."""
    return [tuple(sorted(edge)) for edge in sorted(G.edges())]


def graph_laplacian_from_status(
    G: nx.Graph,
    edge_list: List[Tuple[int, int]],
    line_status: np.ndarray,
) -> np.ndarray:
    """
    Build a weighted Laplacian with per-edge line status.

    line_status[e] == 0 means the line is unavailable.
    """
    n = G.number_of_nodes()
    A = np.zeros((n, n), dtype=float)
    for idx, (u, v) in enumerate(edge_list):
        status = line_status[idx]
        if status <= 0:
            continue
        w = G[u][v]["weight"] * status
        A[u, v] = w
        A[v, u] = w
    D = np.diag(A.sum(axis=1))
    return D - A


def solve_dc_angles(
    laplacian: np.ndarray,
    injections: np.ndarray,
    slack: int = 0,
) -> np.ndarray:
    """
    Solve a DC-style angle system using a pseudo-inverse.

    The pseudo-inverse is intentionally used because:
    1) Laplacians are singular by construction,
    2) topology inconsistency scenarios can temporarily disconnect the graph,
    3) the synthetic study needs a robust solver rather than a fragile exact one.
    """
    n = laplacian.shape[0]
    injections = injections - injections.mean()
    regularized = laplacian + np.ones((n, n), dtype=float) / n
    theta = np.linalg.pinv(regularized) @ injections
    theta = theta - theta[slack]
    return theta


def compute_branch_flows(
    G: nx.Graph,
    edge_list: List[Tuple[int, int]],
    theta: np.ndarray,
    line_status: np.ndarray,
) -> np.ndarray:
    """
    Compute branch flows from phase-angle differences.

    This is a lightweight DC-inspired approximation suitable for a reproducible
    synthetic study.  The goal is not high-fidelity power-flow simulation, but a
    coherent graph-physical signal that supports anomaly analytics.
    """
    flows = np.zeros(len(edge_list), dtype=float)
    for idx, (u, v) in enumerate(edge_list):
        status = line_status[idx]
        if status <= 0:
            flows[idx] = 0.0
            continue
        w = G[u][v]["weight"]
        flows[idx] = status * w * (theta[u] - theta[v])
    return flows


def local_flow_aggregate(
    n_nodes: int,
    edge_list: List[Tuple[int, int]],
    flows: np.ndarray,
) -> np.ndarray:
    """
    Aggregate branch-flow magnitude back to nodes.

    The result is used both for voltage synthesis and for localization plots.
    """
    agg = np.zeros(n_nodes, dtype=float)
    for idx, (u, v) in enumerate(edge_list):
        agg[u] += abs(flows[idx])
        agg[v] += abs(flows[idx])
    return agg


def build_anomaly_schedule(
    cfg: SimulationConfig,
    G: nx.Graph,
    edge_list: List[Tuple[int, int]],
    positions: Dict[int, Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """
    Construct a deterministic event schedule.

    The schedule intentionally places some anomalies in the validation region and
    some in the test region so that:
        * the detector can be calibrated,
        * the manuscript's evaluation metrics remain meaningful,
        * sections 5.1 and 5.2 receive multiple representative cases.
    """
    # Helper selections
    def edge_by_nodes(u: int, v: int) -> Tuple[int, int]:
        return tuple(sorted((u, v)))

    all_nodes = sorted(G.nodes())
    west_nodes = [n for n, x in positions.items() if x[0] <= 1.5]
    center_nodes = [n for n, x in positions.items() if 1.5 < x[0] <= 3.5]
    east_nodes = [n for n, x in positions.items() if x[0] > 3.5]

    schedule = [
        {
            "episode_id": 1,
            "family_group": "physical_topology",
            "subtype": "line_trip",
            "start": 210,
            "end": 255,
            "target_edge": edge_by_nodes(3, 9),
            "target_nodes": [3, 9, 4, 10],
            "description": "Unreported feeder trip producing state-estimation inconsistency.",
        },
        {
            "episode_id": 2,
            "family_group": "physical_topology",
            "subtype": "oscillation",
            "start": 440,
            "end": 495,
            "target_nodes": center_nodes,
            "description": "Electromechanical oscillation concentrated in the central region.",
        },
        {
            "episode_id": 3,
            "family_group": "physical_topology",
            "subtype": "transformer_stress",
            "start": 740,
            "end": 790,
            "target_nodes": west_nodes,
            "description": "Regional overload with voltage sag and reactive stress.",
        },
        {
            "episode_id": 4,
            "family_group": "physical_topology",
            "subtype": "topology_inconsistency",
            "start": 900,
            "end": 950,
            "target_edge": edge_by_nodes(4, 10),
            "target_nodes": [4, 10, 5, 11],
            "description": "Reported switching status inconsistent with the physical configuration.",
        },
        {
            "episode_id": 5,
            "family_group": "cyber_data_quality",
            "subtype": "false_data_injection",
            "start": 1030,
            "end": 1075,
            "target_nodes": east_nodes[:4],
            "description": "Structured stealth bias on active power and voltage channels.",
        },
        {
            "episode_id": 6,
            "family_group": "cyber_data_quality",
            "subtype": "replay_attack",
            "start": 1125,
            "end": 1170,
            "target_nodes": center_nodes[:4],
            "description": "Replay of previously normal telemetry in a subset of sensors.",
        },
        {
            "episode_id": 7,
            "family_group": "cyber_data_quality",
            "subtype": "missingness_burst",
            "start": 1210,
            "end": 1245,
            "target_nodes": west_nodes[:4] + east_nodes[:2],
            "description": "Communication burst with structured missingness.",
        },
        {
            "episode_id": 8,
            "family_group": "cyber_data_quality",
            "subtype": "timestamp_disorder",
            "start": 1260,
            "end": 1295,
            "target_nodes": all_nodes[::3],
            "description": "Temporal desynchronization affecting a subset of channels.",
        },
        {
            "episode_id": 9,
            "family_group": "cyber_data_quality",
            "subtype": "sensor_drift",
            "start": 1298,
            "end": 1318,
            "target_nodes": east_nodes,
            "description": "Gradual bias drift in voltage and frequency channels.",
        },
    ]
    return schedule


def regime_name_from_context(load_factor: float, solar_factor: float, hour: float) -> str:
    """Assign a human-readable operating regime name used in the simulator."""
    if 10.0 <= hour <= 15.0 and solar_factor > 0.45:
        return "solar_midday"
    if 17.0 <= hour <= 22.0 and load_factor > 1.00:
        return "evening_peak"
    if hour < 6.0 or hour >= 23.0:
        return "night_low_load"
    return "transitional_day"


def simulate_smart_grid(cfg: SimulationConfig) -> Dict[str, Any]:
    """
    Create a multimodal synthetic dataset aligned with the chapter objective.

    The simulator encodes:
        * graph-aware physical relationships,
        * multiple normal regimes,
        * physically grounded residual signals,
        * cyber and data-quality corruption,
        * localization labels by node and episode,
        * timestamp metadata for chronological evaluation.
    """
    rng = np.random.default_rng(cfg.seed)
    G, positions, node_roles, regions = build_grid_graph(cfg)
    edge_list = edge_index_map(G)
    n_nodes = G.number_of_nodes()
    n_edges = len(edge_list)
    schedule = build_anomaly_schedule(cfg, G, edge_list, positions)

    # Stable node metadata
    load_base = rng.uniform(0.55, 1.25, size=n_nodes)
    solar_cap = np.array([rng.uniform(0.55, 1.05) if node_roles[i] == "renewable" else 0.0 for i in range(n_nodes)])
    storage_cap = np.array([rng.uniform(0.20, 0.50) if node_roles[i] == "storage" else 0.0 for i in range(n_nodes)])

    # Time-varying line status for the physical grid and for the estimator.
    true_status = np.ones((cfg.total_steps, n_edges), dtype=float)
    est_status = np.ones((cfg.total_steps, n_edges), dtype=float)

    # Target edge indexing helper
    edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}

    # Apply line-status modifications for physical/topology events.
    for event in schedule:
        if "target_edge" not in event:
            continue
        edge = tuple(sorted(event["target_edge"]))
        edge_idx = edge_to_idx[edge]
        if event["subtype"] == "line_trip":
            true_status[event["start"] : event["end"] + 1, edge_idx] = 0.0
            est_status[event["start"] : event["end"] + 1, edge_idx] = 1.0
        elif event["subtype"] == "topology_inconsistency":
            true_status[event["start"] : event["end"] + 1, edge_idx] = 1.0
            est_status[event["start"] : event["end"] + 1, edge_idx] = 0.0

    # Allocate arrays
    voltage = np.zeros((cfg.total_steps, n_nodes), dtype=float)
    angle = np.zeros((cfg.total_steps, n_nodes), dtype=float)
    active_power = np.zeros((cfg.total_steps, n_nodes), dtype=float)
    reactive_power = np.zeros((cfg.total_steps, n_nodes), dtype=float)
    frequency = np.zeros((cfg.total_steps, n_nodes), dtype=float)

    state_residual = np.zeros((cfg.total_steps, n_nodes), dtype=float)
    balance_inconsistency = np.zeros((cfg.total_steps, n_nodes), dtype=float)
    breaker_ratio = np.zeros((cfg.total_steps, n_nodes), dtype=float)
    renewable_share = np.zeros((cfg.total_steps, n_nodes), dtype=float)
    missing_indicator = np.zeros((cfg.total_steps, n_nodes), dtype=float)

    anomaly_binary = np.zeros(cfg.total_steps, dtype=int)
    family_group = np.array(["normal"] * cfg.total_steps, dtype=object)
    subtype = np.array(["normal"] * cfg.total_steps, dtype=object)
    episode_id = np.zeros(cfg.total_steps, dtype=int)
    node_mask = np.zeros((cfg.total_steps, n_nodes), dtype=int)
    regime_names = np.array(["normal"] * cfg.total_steps, dtype=object)
    timestamps = np.arange(cfg.total_steps) * cfg.sampling_minutes
    context_features = np.zeros((cfg.total_steps, 5), dtype=float)

    clean_voltage_history = []
    clean_angle_history = []
    clean_active_history = []
    clean_reactive_history = []
    clean_frequency_history = []

    low_pass_noise = np.zeros(n_nodes, dtype=float)

    # Main simulation loop
    for t in range(cfg.total_steps):
        hour = ((t * cfg.sampling_minutes) / 60.0) % 24.0
        day_phase = 2.0 * np.pi * ((t * cfg.sampling_minutes) % (24 * 60)) / (24 * 60)

        # Normal operating context
        load_factor = 0.88 + 0.18 * np.sin(day_phase - np.pi / 2) + 0.08 * np.sin(2 * day_phase + 0.5)
        load_factor = float(np.clip(load_factor, 0.55, 1.25))
        solar_factor = float(np.clip(np.sin(day_phase - np.pi / 2), 0.0, None))
        wind_factor = 0.55 + 0.15 * np.sin(day_phase + 1.2) + 0.05 * np.sin(4 * day_phase)
        wind_factor = float(np.clip(wind_factor, 0.25, 0.95))
        regime_names[t] = regime_name_from_context(load_factor, solar_factor, hour)

        low_pass_noise = 0.92 * low_pass_noise + 0.08 * rng.standard_normal(n_nodes)

        demand = load_base * (
            1.0
            + 0.35 * load_factor
            + 0.08 * low_pass_noise
            + 0.05 * rng.standard_normal(n_nodes)
        )

        renewable = np.zeros(n_nodes, dtype=float)
        for i in range(n_nodes):
            if node_roles[i] == "renewable":
                renewable[i] = solar_cap[i] * (0.12 + 0.88 * solar_factor) + 0.02 * rng.standard_normal()
            elif node_roles[i] == "storage":
                renewable[i] = storage_cap[i] * (0.25 + 0.10 * np.sin(day_phase + i / 10.0))
            else:
                renewable[i] = 0.02 * rng.random()

        injections = renewable - demand
        injections = injections - injections.mean()

        # Default active event if any.
        active_event: Optional[Dict[str, Any]] = None
        for event in schedule:
            if event["start"] <= t <= event["end"]:
                active_event = event
                break

        # Event-dependent physical perturbations before solving the "true" state.
        if active_event is not None:
            anomaly_binary[t] = 1
            family_group[t] = active_event["family_group"]
            subtype[t] = active_event["subtype"]
            episode_id[t] = active_event["episode_id"]
            node_mask[t, active_event["target_nodes"]] = 1

            if active_event["subtype"] == "transformer_stress":
                injections[active_event["target_nodes"]] -= 0.25
            elif active_event["subtype"] == "oscillation":
                oscillation = 0.18 * np.sin(2.0 * np.pi * (t - active_event["start"]) / 6.0)
                injections[active_event["target_nodes"]] += oscillation
            # line_trip and topology_inconsistency are mainly reflected through line status
            # and the mismatch between true and estimated topologies.

        true_laplacian = graph_laplacian_from_status(G, edge_list, true_status[t])
        theta_true = solve_dc_angles(true_laplacian, injections, slack=0)
        true_flows = compute_branch_flows(G, edge_list, theta_true, true_status[t])
        flow_agg = local_flow_aggregate(n_nodes, edge_list, true_flows)

        # Construct clean physical states.
        voltage_clean = (
            cfg.nominal_voltage
            - 0.05 * demand
            + 0.04 * renewable
            - 0.008 * flow_agg
            + 0.01 * rng.standard_normal(n_nodes)
        )
        reactive_clean = 0.28 * demand - 0.10 * renewable + 0.03 * rng.standard_normal(n_nodes)
        frequency_clean = (
            cfg.nominal_frequency
            + 0.02 * np.sin(day_phase + 0.3)
            + 0.01 * np.tanh(injections)
            + 0.005 * rng.standard_normal(n_nodes)
        )

        # Stronger physical signatures for some fault types.
        if active_event is not None:
            targets = active_event["target_nodes"]
            if active_event["subtype"] == "line_trip":
                voltage_clean[targets] -= 0.05
                frequency_clean[targets] -= 0.01
            elif active_event["subtype"] == "transformer_stress":
                voltage_clean[targets] -= 0.07
                reactive_clean[targets] += 0.10
            elif active_event["subtype"] == "oscillation":
                phase = 2.0 * np.pi * (t - active_event["start"]) / 4.0
                theta_true[targets] += 0.05 * np.sin(phase)
                frequency_clean[targets] += 0.06 * np.sin(phase)
            elif active_event["subtype"] == "topology_inconsistency":
                voltage_clean[targets] -= 0.03

        # Add measurement noise to obtain the observed channels.
        voltage_obs = voltage_clean + cfg.observation_noise_std * rng.standard_normal(n_nodes)
        angle_obs = theta_true + cfg.angle_noise_std * rng.standard_normal(n_nodes)
        active_obs = injections + cfg.power_noise_std * rng.standard_normal(n_nodes)
        reactive_obs = reactive_clean + 0.02 * rng.standard_normal(n_nodes)
        frequency_obs = frequency_clean + cfg.frequency_noise_std * rng.standard_normal(n_nodes)

        # Cyber/data-quality perturbations applied to observations.
        if active_event is not None:
            targets = active_event["target_nodes"]

            if active_event["subtype"] == "false_data_injection":
                attack_shape = np.linspace(0.7, 1.3, len(targets))
                active_obs[targets] += 0.22 * attack_shape
                voltage_obs[targets] += 0.025 * attack_shape
                # Preserve a portion of global aggregate behavior to emulate a
                # stealthier multivariate corruption.
                active_obs -= active_obs.mean() - injections.mean()

            elif active_event["subtype"] == "replay_attack":
                lag = 36
                if len(clean_active_history) > lag:
                    voltage_obs[targets] = clean_voltage_history[-lag][targets] + 0.002 * rng.standard_normal(len(targets))
                    angle_obs[targets] = clean_angle_history[-lag][targets] + 0.001 * rng.standard_normal(len(targets))
                    active_obs[targets] = clean_active_history[-lag][targets] + 0.003 * rng.standard_normal(len(targets))
                    reactive_obs[targets] = clean_reactive_history[-lag][targets] + 0.003 * rng.standard_normal(len(targets))
                    frequency_obs[targets] = clean_frequency_history[-lag][targets] + 0.001 * rng.standard_normal(len(targets))

            elif active_event["subtype"] == "missingness_burst":
                voltage_obs[targets] = np.nan
                active_obs[targets] = np.nan
                reactive_obs[targets] = np.nan
                frequency_obs[targets] = np.nan
                missing_indicator[t, targets] = 1.0

            elif active_event["subtype"] == "timestamp_disorder":
                if t > 1:
                    voltage_obs[targets] = voltage[max(0, t - 1), targets]
                    active_obs[targets] = active_power[max(0, t - 2), targets]
                    frequency_obs[targets] = frequency[max(0, t - 1), targets]
                    missing_indicator[t, targets] = 1.0

            elif active_event["subtype"] == "sensor_drift":
                drift_strength = (t - active_event["start"] + 1) / max(1, (active_event["end"] - active_event["start"] + 1))
                voltage_obs[targets] += 0.035 * drift_strength
                frequency_obs[targets] += 0.018 * drift_strength

        # Replace NaN values only for the state-estimation stage.  Missingness is
        # still stored explicitly and later becomes part of the model input.
        if t == 0:
            prev_voltage = np.full(n_nodes, cfg.nominal_voltage, dtype=float)
            prev_active = np.zeros(n_nodes, dtype=float)
            prev_reactive = np.zeros(n_nodes, dtype=float)
            prev_frequency = np.full(n_nodes, cfg.nominal_frequency, dtype=float)
        else:
            prev_voltage = voltage[t - 1]
            prev_active = active_power[t - 1]
            prev_reactive = reactive_power[t - 1]
            prev_frequency = frequency[t - 1]

        voltage_est = np.where(np.isnan(voltage_obs), prev_voltage, voltage_obs)
        active_est = np.where(np.isnan(active_obs), prev_active, active_obs)
        reactive_est = np.where(np.isnan(reactive_obs), prev_reactive, reactive_obs)
        frequency_est = np.where(np.isnan(frequency_obs), prev_frequency, frequency_obs)

        # State estimation and physically informed residuals.
        est_laplacian = graph_laplacian_from_status(G, edge_list, est_status[t])
        theta_hat = solve_dc_angles(est_laplacian, active_est, slack=0)
        est_flows = compute_branch_flows(G, edge_list, theta_hat, est_status[t])

        injection_hat = est_laplacian @ theta_hat
        voltage_hat = 0.60 * voltage_est + 0.40 * np.clip(
            1.0 - 0.03 * np.abs(injection_hat),
            0.85,
            1.10,
        )

        # Nodal residual mixes active-power mismatch and voltage consistency.
        residual = np.abs(active_est - injection_hat) + 0.50 * np.abs(voltage_est - voltage_hat)

        # Nodal balance inconsistency combines active and reactive proxies.
        q_proxy = np.zeros(n_nodes, dtype=float)
        for idx, (u, v) in enumerate(edge_list):
            status = est_status[t, idx]
            if status <= 0:
                continue
            w = G[u][v]["weight"]
            q = 0.25 * status * w * (voltage_est[u] - voltage_est[v])
            q_proxy[u] += q
            q_proxy[v] -= q

        p_balance = np.abs(active_est - injection_hat)
        q_balance = np.abs(reactive_est - q_proxy)
        balance = p_balance + q_balance

        # Breaker ratio uses estimator-visible topology, not true topology.  This
        # reflects what the control room believes is currently connected.
        degree_nominal = dict(G.degree())
        degree_estimated = {node: 0.0 for node in G.nodes()}
        for idx, (u, v) in enumerate(edge_list):
            if est_status[t, idx] > 0:
                degree_estimated[u] += 1.0
                degree_estimated[v] += 1.0
        breaker = np.array(
            [
                degree_estimated[node] / max(degree_nominal[node], 1)
                for node in sorted(G.nodes())
            ],
            dtype=float,
        )

        renewable_ratio = renewable / (np.abs(renewable) + np.abs(demand) + cfg.epsilon)

        # Persist all outputs.
        voltage[t] = np.where(np.isnan(voltage_obs), prev_voltage, voltage_obs)
        angle[t] = np.where(np.isnan(angle_obs), 0.0, angle_obs)
        active_power[t] = np.where(np.isnan(active_obs), prev_active, active_obs)
        reactive_power[t] = np.where(np.isnan(reactive_obs), prev_reactive, reactive_obs)
        frequency[t] = np.where(np.isnan(frequency_obs), prev_frequency, frequency_obs)

        state_residual[t] = residual
        balance_inconsistency[t] = balance
        breaker_ratio[t] = breaker
        renewable_share[t] = renewable_ratio

        context_features[t] = np.array(
            [
                load_factor,
                solar_factor,
                math.sin(2.0 * np.pi * hour / 24.0),
                math.cos(2.0 * np.pi * hour / 24.0),
                breaker.mean(),
            ],
            dtype=float,
        )

        # Save clean history to support replay attacks.
        clean_voltage_history.append(voltage_clean.copy())
        clean_angle_history.append(theta_true.copy())
        clean_active_history.append(injections.copy())
        clean_reactive_history.append(reactive_clean.copy())
        clean_frequency_history.append(frequency_clean.copy())

    # Derivative features
    d_voltage = np.zeros_like(voltage)
    d_frequency = np.zeros_like(frequency)
    d_voltage[1:] = cfg.derivative_scale * (voltage[1:] - voltage[:-1])
    d_frequency[1:] = cfg.derivative_scale * (frequency[1:] - frequency[:-1])

    raw_features = np.stack(
        [
            voltage,
            angle,
            active_power,
            reactive_power,
            frequency,
            state_residual,
            balance_inconsistency,
            breaker_ratio,
            renewable_share,
            missing_indicator,
            d_voltage,
            d_frequency,
        ],
        axis=-1,
    )

    # Store schedule for reproducibility and later incident reporting.
    return {
        "graph": G,
        "positions": positions,
        "node_roles": node_roles,
        "regions": regions,
        "edge_list": edge_list,
        "schedule": schedule,
        "raw_features": raw_features,
        "context_features": context_features,
        "timestamps": timestamps,
        "anomaly_binary": anomaly_binary,
        "family_group": family_group,
        "subtype": subtype,
        "episode_id": episode_id,
        "node_mask": node_mask,
        "regime_names": regime_names,
        "true_status": true_status,
        "est_status": est_status,
    }


# -----------------------------------------------------------------------------
# Preprocessing and window construction
# -----------------------------------------------------------------------------

def normalize_and_smooth(
    raw_features: np.ndarray,
    anomaly_binary: np.ndarray,
    cfg: SimulationConfig,
) -> Dict[str, np.ndarray]:
    """
    Perform robust normalization, clipping, imputation and temporal smoothing.

    The training split uses only early, normal observations to estimate robust
    location/scale statistics.  This design prevents future information leakage
    and remains consistent with the manuscript's chronological evaluation logic.
    """
    n_steps = raw_features.shape[0]
    train_end_step = int(cfg.train_fraction * n_steps)
    train_normal_mask = (np.arange(n_steps) < train_end_step) & (anomaly_binary == 0)
    if train_normal_mask.sum() < 10:
        raise RuntimeError("Not enough normal observations in the training split.")

    ref = raw_features[train_normal_mask]
    median, iqr = robust_center_scale(ref, epsilon=cfg.epsilon)

    normalized = (raw_features - median) / (iqr + cfg.epsilon)
    normalized = np.clip(normalized, -cfg.clip_value, cfg.clip_value)

    # Missing values are imputed after robust normalization so that all later
    # geometry operates in the same scale domain.
    median_impute = np.nanmedian(normalized[train_normal_mask], axis=0)
    filled = np.where(np.isnan(normalized), median_impute, normalized)

    # Preserve the binary missingness indicator without smoothing.
    missing_idx = FEATURE_NAMES.index("missing_indicator")
    smoothed = filled.copy()
    continuous_indices = [i for i in range(len(FEATURE_NAMES)) if i != missing_idx]
    for t in range(1, n_steps):
        smoothed[t, :, continuous_indices] = (
            cfg.smoothing_alpha * filled[t, :, continuous_indices]
            + (1.0 - cfg.smoothing_alpha) * smoothed[t - 1, :, continuous_indices]
        )
        smoothed[t, :, missing_idx] = filled[t, :, missing_idx]

    return {
        "median": median,
        "iqr": iqr,
        "normalized": filled,
        "smoothed": smoothed,
    }


def make_windows(
    smoothed_features: np.ndarray,
    context_features: np.ndarray,
    timestamps: np.ndarray,
    anomaly_binary: np.ndarray,
    family_group: np.ndarray,
    subtype: np.ndarray,
    episode_id: np.ndarray,
    node_mask: np.ndarray,
    regime_names: np.ndarray,
    cfg: SimulationConfig,
) -> Dict[str, np.ndarray]:
    """
    Convert time steps into graph-temporal windows.

    Each window is represented as:
        X_t  = [window_length, n_nodes, n_features]
        Y_t  = next-step target for predictive learning
    """
    L = cfg.window_length
    H = cfg.prediction_horizon
    n_steps = smoothed_features.shape[0]

    windows = []
    next_steps = []
    contexts = []
    end_times = []
    y_binary = []
    y_family = []
    y_subtype = []
    y_episode = []
    y_node_mask = []
    y_regime_name = []

    for end_idx in range(L - 1, n_steps - H):
        start_idx = end_idx - L + 1
        future_idx = end_idx + H

        windows.append(smoothed_features[start_idx : end_idx + 1])
        next_steps.append(smoothed_features[future_idx])
        contexts.append(context_features[start_idx : end_idx + 1].mean(axis=0))
        end_times.append(timestamps[end_idx])

        # The monitoring window ends at end_idx.  We label the window with the
        # state of the last time step because the detector is expected to decide
        # at that moment.
        y_binary.append(anomaly_binary[end_idx])
        y_family.append(family_group[end_idx])
        y_subtype.append(subtype[end_idx])
        y_episode.append(episode_id[end_idx])
        y_node_mask.append(node_mask[end_idx])
        y_regime_name.append(regime_names[end_idx])

    return {
        "X": np.asarray(windows, dtype=np.float32),
        "Y": np.asarray(next_steps, dtype=np.float32),
        "context": np.asarray(contexts, dtype=np.float32),
        "end_time": np.asarray(end_times, dtype=float),
        "label_binary": np.asarray(y_binary, dtype=int),
        "family_group": np.asarray(y_family, dtype=object),
        "subtype": np.asarray(y_subtype, dtype=object),
        "episode_id": np.asarray(y_episode, dtype=int),
        "node_mask": np.asarray(y_node_mask, dtype=int),
        "regime_name": np.asarray(y_regime_name, dtype=object),
    }


def chronological_split(n_windows: int, cfg: SimulationConfig) -> Dict[str, np.ndarray]:
    """Create train/validation/test index arrays."""
    train_end = int(cfg.train_fraction * n_windows)
    val_end = int((cfg.train_fraction + cfg.val_fraction) * n_windows)
    idx = np.arange(n_windows)
    return {
        "train": idx[:train_end],
        "val": idx[train_end:val_end],
        "test": idx[val_end:],
    }


def fit_regime_model(
    window_context: np.ndarray,
    label_binary: np.ndarray,
    split_idx: Dict[str, np.ndarray],
    cfg: SimulationConfig,
) -> Dict[str, Any]:
    """
    Fit a regime model on normal training windows.

    The implementation uses KMeans because the manuscript's regime assignment is
    conceptually a contextual partition of normal operation.  The human-readable
    simulated regime names remain stored separately for interpretation.
    """
    train_idx = split_idx["train"]
    normal_train = train_idx[label_binary[train_idx] == 0]
    if len(normal_train) < cfg.n_regimes:
        raise RuntimeError("Not enough normal training windows to fit regime conditioning.")

    kmeans = KMeans(n_clusters=cfg.n_regimes, random_state=cfg.seed, n_init=20)
    kmeans.fit(window_context[normal_train])

    regime_id = kmeans.predict(window_context)
    return {
        "kmeans": kmeans,
        "regime_id": regime_id,
    }


def augment_normal_windows(
    X: np.ndarray,
    Y: np.ndarray,
    regime_id: np.ndarray,
    model_cfg: ModelConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment normal windows with small, semantics-preserving perturbations.

    The purpose is not to fabricate anomalies.  The goal is to widen the manifold
    of normal operation so that the learned representation remains robust to mild
    shifts, noise and timestamp micro-jitter.
    """
    rng = np.random.default_rng(seed)

    noise_aug = X + model_cfg.augmentation_noise_std * rng.standard_normal(X.shape).astype(np.float32)
    y_noise_aug = Y + model_cfg.augmentation_noise_std * rng.standard_normal(Y.shape).astype(np.float32)

    jitter_aug = X.copy()
    if X.shape[1] > 2:
        jitter_aug[:, 1:] = 0.85 * jitter_aug[:, 1:] + 0.15 * jitter_aug[:, :-1]

    X_all = np.concatenate([X, noise_aug.astype(np.float32), jitter_aug.astype(np.float32)], axis=0)
    Y_all = np.concatenate([Y, y_noise_aug.astype(np.float32), Y.astype(np.float32)], axis=0)
    regime_all = np.concatenate([regime_id, regime_id, regime_id], axis=0)
    return X_all, Y_all, regime_all


class WindowDataset(Dataset):
    """PyTorch dataset for graph-temporal windows."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, regime_id: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.regime_id = torch.tensor(regime_id, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx], self.regime_id[idx]


# -----------------------------------------------------------------------------
# Deep graph-temporal model
# -----------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """
    Simple graph convolution using a fixed normalized adjacency matrix.

    The goal is to preserve explicit graph inductive bias without introducing a
    heavy dependency on a specialized graph-neural framework.  This keeps the
    script reproducible and focused on the chapter methodology.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x        : [batch, nodes, features]
        adj_norm : [nodes, nodes]
        """
        msg = torch.einsum("ij,bjf->bif", adj_norm, x)
        out = self.linear(msg)
        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class GraphTemporalAutoencoder(nn.Module):
    """
    Graph-temporal autoencoder with forecast and uncertainty heads.

    Architecture:
        per time step:
            graph convolution -> graph convolution -> graph pooling
        sequence:
            pooled graph tokens -> GRU
        latent:
            final GRU state -> latent vector
        heads:
            reconstruction of the full window,
            next-step prediction,
            predictive log-variance.

    The model supports:
        * reconstruction evidence,
        * predictive evidence,
        * uncertainty-aware scoring,
        * gradient-based attribution.

    Important implementation note
    -----------------------------
    The decoder heads are instantiated *eagerly* in ``__init__`` using the known
    monitoring-window length from the experiment configuration.  This is crucial
    for CUDA correctness and for training correctness:

        1) If these heads are created lazily inside ``forward`` after
           ``model.to(device)``, the new modules stay on CPU unless they are
           moved manually.
        2) If the optimizer is created before the lazy heads exist, their
           parameters are not registered inside the optimizer and therefore are
           not updated during training.

    By building the heads up front, all parameters are present before
    ``model.to(device)`` and before ``torch.optim.Adam(model.parameters(), ...)``.
    """

    def __init__(
        self,
        n_features: int,
        n_nodes: int,
        window_length: int,
        model_cfg: ModelConfig,
    ) -> None:
        super().__init__()
        if window_length <= 0:
            raise ValueError("window_length must be a strictly positive integer.")

        self.n_features = n_features
        self.n_nodes = n_nodes
        self.window_length = int(window_length)
        self.model_cfg = model_cfg

        self.gc1 = GraphConvolution(n_features, model_cfg.hidden_dim, model_cfg.dropout)
        self.gc2 = GraphConvolution(model_cfg.hidden_dim, model_cfg.hidden_dim, model_cfg.dropout)

        pooled_dim = model_cfg.hidden_dim * 2
        self.gru = nn.GRU(
            input_size=pooled_dim,
            hidden_size=model_cfg.temporal_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.to_latent = nn.Sequential(
            nn.Linear(model_cfg.temporal_dim, model_cfg.temporal_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.temporal_dim, model_cfg.latent_dim),
        )

        # Decoder heads are created during initialization so that:
        #   * they are moved correctly by model.to(device),
        #   * they are visible to the optimizer from the start,
        #   * the code cannot silently mix CPU and CUDA tensors.
        self._build_output_heads()

    def _build_output_heads(self) -> None:
        """
        Build decoder heads using the fixed monitoring-window length.

        The chapter uses a fixed-length temporal window, therefore there is no
        benefit in creating these layers lazily.  Eager construction is safer
        and guarantees that all parameters are registered immediately.
        """
        recon_dim = self.window_length * self.n_nodes * self.n_features
        pred_dim = self.n_nodes * self.n_features

        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.model_cfg.latent_dim, self.model_cfg.temporal_dim),
            nn.ReLU(),
            nn.Linear(self.model_cfg.temporal_dim, recon_dim),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.model_cfg.latent_dim, self.model_cfg.temporal_dim),
            nn.ReLU(),
            nn.Linear(self.model_cfg.temporal_dim, pred_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(self.model_cfg.latent_dim, self.model_cfg.temporal_dim),
            nn.ReLU(),
            nn.Linear(self.model_cfg.temporal_dim, pred_dim),
        )

    def _validate_window_shape(self, x: torch.Tensor) -> Tuple[int, int, int, int]:
        """
        Validate the incoming tensor shape.

        The decoder dimensions are tied to the configured window length.
        Failing early with a clear error message is much better than producing a
        silent shape mismatch deeper in the computational graph.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected an input tensor of shape [batch, time, nodes, features], got ndim={x.ndim}."
            )

        B, T, N, F_ = x.shape
        if T != self.window_length:
            raise ValueError(
                "GraphTemporalAutoencoder received a window with time dimension "
                f"{T}, but the model was initialized for window_length={self.window_length}."
            )
        if N != self.n_nodes:
            raise ValueError(
                f"Expected {self.n_nodes} nodes, but received {N}."
            )
        if F_ != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, but received {F_}."
            )
        return B, T, N, F_

    def spatial_encode(self, x_t: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Encode one graph snapshot and pool over nodes."""
        h = self.gc1(x_t, adj_norm)
        h = self.gc2(h, adj_norm)
        pooled_mean = h.mean(dim=1)
        pooled_max = h.max(dim=1).values
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return pooled

    def encode(self, x: torch.Tensor, adj_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode a window into a latent representation.

        x : [batch, time, nodes, features]
        """
        B, T, N, F_ = self._validate_window_shape(x)

        seq_tokens = []
        for t in range(T):
            pooled = self.spatial_encode(x[:, t], adj_norm)
            seq_tokens.append(pooled)

        seq = torch.stack(seq_tokens, dim=1)  # [B, T, pooled_dim]
        gru_out, h_n = self.gru(seq)
        latent = self.to_latent(h_n[-1])

        return {
            "latent": latent,
            "seq_repr": seq,
            "gru_out": gru_out,
        }

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass with reconstruction, prediction and uncertainty."""
        enc = self.encode(x, adj_norm)
        z = enc["latent"]

        B, T, N, F_ = self._validate_window_shape(x)
        recon_flat = self.reconstruction_head(z)
        pred_flat = self.prediction_head(z)
        logvar_flat = self.logvar_head(z)

        recon = recon_flat.view(B, T, N, F_)
        pred = pred_flat.view(B, N, F_)
        logvar = logvar_flat.view(B, N, F_)

        return {
            "latent": z,
            "seq_repr": enc["seq_repr"],
            "gru_out": enc["gru_out"],
            "recon": recon,
            "pred": pred,
            "logvar": logvar,
        }


def normalized_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """
    Compute a symmetrically normalized adjacency with self-loops.

    This matrix is the fixed message-passing scaffold for the graph encoder.
    """
    A = nx.to_numpy_array(G, weight="weight", dtype=float)
    A = A + np.eye(A.shape[0], dtype=float)
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(deg, 1e-8, None)))
    return deg_inv_sqrt @ A @ deg_inv_sqrt


def gaussian_nll_loss(pred: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for a diagonal Gaussian predictive head.

    This operationalizes the manuscript's idea that uncertainty should enter the
    anomaly signal explicitly rather than being ignored.
    """
    precision = torch.exp(-torch.clamp(logvar, -6.0, 6.0))
    loss = 0.5 * (precision * (pred - target) ** 2 + logvar)
    return loss.mean()


def within_regime_compactness(latent: torch.Tensor, regime_id: torch.Tensor) -> torch.Tensor:
    """
    Penalize excessive splitting of normal windows from the same regime.

    This is a differentiable proxy for topology-aware regularization:
    normal points that belong to the same operating context should not fragment
    unnecessarily in latent space.
    """
    loss = torch.tensor(0.0, device=latent.device)
    unique = torch.unique(regime_id)
    count = 0
    for rid in unique:
        mask = regime_id == rid
        if mask.sum() < 2:
            continue
        z = latent[mask]
        center = z.mean(dim=0, keepdim=True)
        loss = loss + ((z - center) ** 2).mean()
        count += 1
    if count == 0:
        return loss
    return loss / count


def temporal_smoothness(seq_repr: torch.Tensor) -> torch.Tensor:
    """
    Penalize unnecessary temporal churn inside each window representation.
    """
    if seq_repr.shape[1] < 2:
        return torch.tensor(0.0, device=seq_repr.device)
    diff = seq_repr[:, 1:] - seq_repr[:, :-1]
    return (diff ** 2).mean()


def contrastive_consistency_loss(
    model: GraphTemporalAutoencoder,
    x: torch.Tensor,
    adj_norm: torch.Tensor,
    model_cfg: ModelConfig,
) -> torch.Tensor:
    """
    Lightweight contrastive objective based on consistency under mild augmentations.

    This is intentionally simple and stable:
        * positive pair: original window vs. mild noisy version
        * negative pair: original latent vs. shuffled latent
    """
    noise = model_cfg.augmentation_noise_std * torch.randn_like(x)
    x_aug = x + noise

    z = model.encode(x, adj_norm)["latent"]
    z_aug = model.encode(x_aug, adj_norm)["latent"]

    positive = ((z - z_aug) ** 2).mean()

    perm = torch.randperm(z.shape[0], device=z.device)
    z_neg = z[perm]
    neg_dist = torch.norm(z - z_neg, dim=1)
    negative = torch.relu(model_cfg.contrastive_margin - neg_dist).mean()

    return positive + negative


def train_model(
    model: GraphTemporalAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj_norm: torch.Tensor,
    model_cfg: ModelConfig,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, List[float]]:
    """
    Train the graph-temporal autoencoder with early stopping.

    Training uses only normal windows (plus their semantics-preserving
    augmentations).  Validation also monitors normal windows to preserve the
    unsupervised character of the learned representation.
    """
    # Defensive move: even though the caller already sends `adj_norm` on the
    # target device, explicitly re-anchoring it here prevents accidental CPU/GPU
    # mismatches if this function is reused in a different context.
    adj_norm = adj_norm.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg.learning_rate,
        weight_decay=model_cfg.weight_decay,
    )

    history = {
        "train_total": [],
        "train_recon": [],
        "train_pred": [],
        "val_total": [],
        "val_recon": [],
        "val_pred": [],
    }

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(model_cfg.epochs):
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_pred = 0.0
        train_count = 0

        for x, y, regime in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            regime = regime.to(device, non_blocking=True)

            out = model(x, adj_norm)
            recon_loss = F.mse_loss(out["recon"], x)
            pred_loss = gaussian_nll_loss(out["pred"], y, out["logvar"])
            contrast_loss = contrastive_consistency_loss(model, x, adj_norm, model_cfg)
            topo_proxy = within_regime_compactness(out["latent"], regime) + temporal_smoothness(out["seq_repr"])

            total_loss = (
                model_cfg.lambda_recon * recon_loss
                + model_cfg.lambda_pred * pred_loss
                + model_cfg.lambda_contrast * contrast_loss
                + model_cfg.lambda_topology_proxy * topo_proxy
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            batch_size = x.shape[0]
            train_total += total_loss.item() * batch_size
            train_recon += recon_loss.item() * batch_size
            train_pred += pred_loss.item() * batch_size
            train_count += batch_size

        history["train_total"].append(train_total / max(train_count, 1))
        history["train_recon"].append(train_recon / max(train_count, 1))
        history["train_pred"].append(train_pred / max(train_count, 1))

        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_pred = 0.0
        val_count = 0

        with torch.no_grad():
            for x, y, regime in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                regime = regime.to(device, non_blocking=True)

                out = model(x, adj_norm)
                recon_loss = F.mse_loss(out["recon"], x)
                pred_loss = gaussian_nll_loss(out["pred"], y, out["logvar"])
                contrast_loss = contrastive_consistency_loss(model, x, adj_norm, model_cfg)
                topo_proxy = within_regime_compactness(out["latent"], regime) + temporal_smoothness(out["seq_repr"])

                total_loss = (
                    model_cfg.lambda_recon * recon_loss
                    + model_cfg.lambda_pred * pred_loss
                    + model_cfg.lambda_contrast * contrast_loss
                    + model_cfg.lambda_topology_proxy * topo_proxy
                )

                batch_size = x.shape[0]
                val_total += total_loss.item() * batch_size
                val_recon += recon_loss.item() * batch_size
                val_pred += pred_loss.item() * batch_size
                val_count += batch_size

        val_total_mean = val_total / max(val_count, 1)
        history["val_total"].append(val_total_mean)
        history["val_recon"].append(val_recon / max(val_count, 1))
        history["val_pred"].append(val_pred / max(val_count, 1))

        if val_total_mean < best_val:
            best_val = val_total_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= model_cfg.early_stopping_patience:
            break

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    return history


def plot_training_history(history: Dict[str, List[float]], path: Path) -> None:
    """Save training curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_total"], label="train_total")
    ax.plot(history["val_total"], label="val_total")
    ax.plot(history["train_recon"], label="train_recon", linestyle="--")
    ax.plot(history["val_recon"], label="val_recon", linestyle="--")
    ax.set_title("Training and validation losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def infer_model_outputs(
    model: GraphTemporalAutoencoder,
    X: np.ndarray,
    Y: np.ndarray,
    adj_norm: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run inference over all windows and return numpy arrays.

    This function centralizes all model-driven evidence required later:
        * latent vectors,
        * reconstruction error,
        * prediction error,
        * uncertainty,
        * reconstructed windows,
        * next-step forecasts.
    """
    model.eval()
    adj_norm = adj_norm.to(device)
    batch_size = 128
    latents = []
    recon_errors = []
    pred_errors = []
    uncertainty = []
    reconstructions = []
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(len(X), start + batch_size)
            x_batch = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            y_batch = torch.tensor(Y[start:end], dtype=torch.float32, device=device)

            out = model(x_batch, adj_norm)
            latents.append(out["latent"].cpu().numpy())
            reconstructions.append(out["recon"].cpu().numpy())
            predictions.append(out["pred"].cpu().numpy())

            recon_err = ((out["recon"] - x_batch) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            pred_err = ((out["pred"] - y_batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            unc = torch.exp(torch.clamp(out["logvar"], -6.0, 6.0)).mean(dim=(1, 2)).cpu().numpy()

            recon_errors.append(recon_err)
            pred_errors.append(pred_err)
            uncertainty.append(unc)

    return {
        "latent": np.concatenate(latents, axis=0),
        "recon_error": np.concatenate(recon_errors, axis=0),
        "pred_error": np.concatenate(pred_errors, axis=0),
        "uncertainty": np.concatenate(uncertainty, axis=0),
        "reconstruction": np.concatenate(reconstructions, axis=0),
        "prediction": np.concatenate(predictions, axis=0),
    }


# -----------------------------------------------------------------------------
# Latent whitening and persistent homology
# -----------------------------------------------------------------------------

def fit_whitening_transform(z_train_normal: np.ndarray, epsilon: float = 1e-6) -> Dict[str, np.ndarray]:
    """
    Fit a whitening transform so that Euclidean latent distances approximate a
    covariance-aware metric, in the spirit of the manuscript's Mahalanobis logic.
    """
    mean = z_train_normal.mean(axis=0)
    centered = z_train_normal - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov + epsilon * np.eye(cov.shape[0]))
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.clip(eigvals, epsilon, None))) @ eigvecs.T
    return {"mean": mean, "inv_sqrt": inv_sqrt}


def apply_whitening(z: np.ndarray, whitening: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply the latent whitening transform."""
    return (z - whitening["mean"]) @ whitening["inv_sqrt"].T


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    """
    Remove infinite points and keep only valid birth/death pairs.
    """
    if diagram is None or len(diagram) == 0:
        return np.empty((0, 2), dtype=float)
    diagram = np.asarray(diagram, dtype=float)
    mask = np.isfinite(diagram).all(axis=1) & (diagram[:, 1] > diagram[:, 0])
    return diagram[mask]


def persistent_entropy_manual(diagram: np.ndarray) -> float:
    """
    Compute persistent entropy from finite lifetimes.
    """
    d = finite_diagram(diagram)
    if len(d) == 0:
        return 0.0
    lifetimes = d[:, 1] - d[:, 0]
    total = lifetimes.sum()
    if total <= 1e-12:
        return 0.0
    p = lifetimes / total
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum())


def total_persistence(diagram: np.ndarray, power: float = 1.0) -> float:
    """Compute total persistence of a finite diagram."""
    d = finite_diagram(diagram)
    if len(d) == 0:
        return 0.0
    lifetimes = d[:, 1] - d[:, 0]
    return float(np.sum(lifetimes ** power))


def betti_curve(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Compute a Betti curve on a fixed filtration grid.
    """
    d = finite_diagram(diagram)
    if len(d) == 0:
        return np.zeros_like(grid)
    curve = np.zeros_like(grid, dtype=float)
    for i, value in enumerate(grid):
        curve[i] = np.sum((d[:, 0] <= value) & (value < d[:, 1]))
    return curve


def safe_bottleneck(d1: np.ndarray, d2: np.ndarray) -> float:
    """Distance wrapper that handles empty diagrams safely."""
    f1 = finite_diagram(d1)
    f2 = finite_diagram(d2)
    if len(f1) == 0 and len(f2) == 0:
        return 0.0
    return float(bottleneck(f1, f2))


def safe_wasserstein(d1: np.ndarray, d2: np.ndarray) -> float:
    """Distance wrapper that handles empty diagrams safely."""
    f1 = finite_diagram(d1)
    f2 = finite_diagram(d2)
    if len(f1) == 0 and len(f2) == 0:
        return 0.0
    return float(wasserstein(f1, f2))




def empty_topology_record(
    cfg: SimulationConfig,
    max_scale: float = 1.0,
    method: str = "empty",
    num_edges: int = 0,
    approximation_used: bool = False,
) -> Dict[str, Any]:
    """Create a shape-stable empty topological record."""
    betti_grid = np.linspace(0.0, max_scale, cfg.betti_grid_points)
    landscape_grid = np.linspace(0.0, max_scale, cfg.landscape_grid_points)
    landscape_shape = (cfg.landscape_layers, cfg.landscape_grid_points)
    summary_vector = np.concatenate(
        [
            np.zeros(6, dtype=float),
            np.zeros(cfg.betti_grid_points, dtype=float),
            np.zeros(cfg.betti_grid_points, dtype=float),
            np.zeros(cfg.landscape_layers * cfg.landscape_grid_points, dtype=float),
            np.zeros(cfg.landscape_layers * cfg.landscape_grid_points, dtype=float),
        ]
    )
    empty = np.empty((0, 2), dtype=float)
    return {
        "dgms": [empty, empty],
        "betti_grid": betti_grid,
        "betti0": np.zeros_like(betti_grid),
        "betti1": np.zeros_like(betti_grid),
        "landscape_grid": landscape_grid,
        "landscape_h0": np.zeros(landscape_shape, dtype=float),
        "landscape_h1": np.zeros(landscape_shape, dtype=float),
        "fragmentation": 0.0,
        "loop_persistence": 0.0,
        "entropy0": 0.0,
        "entropy1": 0.0,
        "landscape_energy0": 0.0,
        "landscape_energy1": 0.0,
        "summary_vector": summary_vector,
        "method": method,
        "num_edges": int(num_edges),
        "approximation_used": bool(approximation_used),
        "incremental_update": False,
    }


def stabilize_cloud_with_indices(cloud: np.ndarray, global_indices: np.ndarray) -> np.ndarray:
    """
    Add deterministic micro-jitter tied to global indices.

    This avoids exact duplicate points without breaking the overlap structure of
    consecutive sliding clouds, which is necessary for incremental updates.
    """
    if cloud.size == 0:
        return cloud
    basis = np.linspace(1.0, 2.0, cloud.shape[1], dtype=float)
    jitter = 1e-10 * np.outer(global_indices.astype(float) + 1.0, basis)
    return cloud + jitter


def dense_distance_matrix(cloud: np.ndarray) -> np.ndarray:
    """Compute a dense Euclidean distance matrix."""
    n = cloud.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    return squareform(pdist(cloud, metric="euclidean"))


def incremental_distance_matrix(
    previous_distance: Optional[np.ndarray],
    previous_indices: Optional[np.ndarray],
    new_cloud: np.ndarray,
    new_indices: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Update the pairwise distance matrix incrementally when the cloud slides by one point.
    """
    n = new_cloud.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float), False

    if (
        previous_distance is None
        or previous_indices is None
        or previous_distance.shape[0] < 2
        or len(previous_indices) != n
        or not np.array_equal(previous_indices[1:], new_indices[:-1])
    ):
        return dense_distance_matrix(new_cloud), False

    kept = previous_distance[1:, 1:].copy()
    new_row = np.linalg.norm(new_cloud[:-1] - new_cloud[-1], axis=1)

    updated = np.zeros((n, n), dtype=float)
    updated[:-1, :-1] = kept
    updated[-1, :-1] = new_row
    updated[:-1, -1] = new_row
    return updated, True


def greedy_insertion_radii(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Furthest-point sampling insertion radii aligned with the sparse-filtration notebook of Ripser.py.
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    perm = np.zeros(n, dtype=np.int64)
    lambdas = np.zeros(n, dtype=float)
    ds = distance_matrix[0, :].copy()
    for i in range(1, n):
        idx = int(np.argmax(ds))
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, distance_matrix[idx, :])
    return lambdas[perm]


def approximate_sparse_distance_matrix(distance_matrix: np.ndarray, eps: float) -> sp.csr_matrix:
    """
    Build a sparse approximate Vietoris-Rips distance matrix.

    The implementation follows the construction illustrated in the official
    Ripser.py sparse-filtration notebook.
    """
    if eps <= 0:
        raise ValueError("sparse_rips_epsilon must be strictly positive.")

    D = np.asarray(distance_matrix, dtype=float).copy()
    n = D.shape[0]
    if n == 0:
        return sp.csr_matrix((0, 0), dtype=float)

    lambdas = greedy_insertion_radii(D)
    E0 = (1.0 + eps) / eps
    E1 = (1.0 + eps) ** 2 / eps

    n_bounds = ((eps ** 2 + 3.0 * eps + 2.0) / eps) * lambdas
    D[D > n_bounds[:, None]] = np.inf

    II, JJ = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = (II < JJ) & np.isfinite(D)
    II = II[mask]
    JJ = JJ[mask]
    vals = D[mask]

    if len(vals) == 0:
        out = sp.csr_matrix((n, n), dtype=float)
        out.setdiag(0.0)
        return out

    minlam = np.minimum(lambdas[II], lambdas[JJ])
    maxlam = np.maximum(lambdas[II], lambdas[JJ])
    M = np.minimum((E0 + E1) * minlam, E0 * (minlam + maxlam))

    keep = vals <= M
    II = II[keep]
    JJ = JJ[keep]
    vals = vals[keep]
    minlam = minlam[keep]

    if len(vals) == 0:
        out = sp.csr_matrix((n, n), dtype=float)
        out.setdiag(0.0)
        return out

    warp_mask = vals > 2.0 * minlam * E0
    vals = vals.copy()
    vals[warp_mask] = 2.0 * (vals[warp_mask] - minlam[warp_mask] * E0)

    rows = np.concatenate([II, JJ])
    cols = np.concatenate([JJ, II])
    data = np.concatenate([vals, vals])

    sparse_dm = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    sparse_dm.setdiag(0.0)
    sparse_dm.eliminate_zeros()
    return sparse_dm


def diagram_stat_fallback_distance(d1: np.ndarray, d2: np.ndarray) -> float:
    """Fallback distance when a diagram metric fails numerically."""
    f1 = finite_diagram(d1)
    f2 = finite_diagram(d2)
    stats1 = np.array(
        [
            len(f1),
            total_persistence(f1, power=1.0),
            persistent_entropy_manual(f1),
        ],
        dtype=float,
    )
    stats2 = np.array(
        [
            len(f2),
            total_persistence(f2, power=1.0),
            persistent_entropy_manual(f2),
        ],
        dtype=float,
    )
    return float(np.linalg.norm(stats1 - stats2))


def truncated_persistence_landscape(
    diagram: np.ndarray,
    grid: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """
    Compute a truncated persistence landscape on a fixed grid.

    This lightweight implementation is sufficient for publication plots and for
    stable vectorization without the overhead of exact critical-point algebra.
    """
    d = finite_diagram(diagram)
    if len(d) == 0 or len(grid) == 0:
        return np.zeros((n_layers, len(grid)), dtype=float)

    tents = []
    for birth, death in d:
        tents.append(np.maximum(0.0, np.minimum(grid - birth, death - grid)))
    values = np.stack(tents, axis=0)
    values = np.sort(values, axis=0)[::-1]

    out = np.zeros((n_layers, len(grid)), dtype=float)
    depth = min(n_layers, values.shape[0])
    out[:depth] = values[:depth]
    return out


def compute_persistence_record_from_distance_matrix(
    distance_matrix: np.ndarray,
    cfg: SimulationConfig,
) -> Dict[str, Any]:
    """
    Compute persistent homology from a precomputed distance matrix.

    The function chooses a sparse approximate filtration in fast mode and falls
    back to dense exact Ripser when needed.
    """
    n = distance_matrix.shape[0]
    if n < 4:
        return empty_topology_record(cfg, method="too_small")

    tri = distance_matrix[np.triu_indices(n, k=1)]
    tri = tri[np.isfinite(tri)]
    if len(tri) == 0 or np.nanmax(np.abs(tri)) < 1e-12:
        return empty_topology_record(cfg, method="degenerate")

    max_scale = max(float(np.percentile(tri, 90)), 1e-3)
    approximation_used = False
    method = "dense_exact"

    try:
        if cfg.fast_topology_mode and n >= cfg.sparse_filtration_min_points:
            sparse_dm = approximate_sparse_distance_matrix(distance_matrix, cfg.sparse_rips_epsilon)
            if sparse_dm.nnz > 0:
                result = ripser(sparse_dm, distance_matrix=True, maxdim=1)
                approximation_used = True
                method = "sparse_rips"
            else:
                result = ripser(distance_matrix, distance_matrix=True, maxdim=1, thresh=max_scale)
                method = "dense_fallback_no_edges"
        else:
            result = ripser(distance_matrix, distance_matrix=True, maxdim=1, thresh=max_scale)
    except Exception:
        try:
            result = ripser(distance_matrix, distance_matrix=True, maxdim=1, thresh=max_scale)
            method = "dense_fallback_exception"
        except Exception:
            return empty_topology_record(
                cfg,
                max_scale=max_scale,
                method="failed",
                approximation_used=approximation_used,
            )

    dgms = result.get("dgms", [np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)])
    if len(dgms) == 1:
        dgms = [dgms[0], np.empty((0, 2), dtype=float)]

    betti_grid = np.linspace(0.0, max_scale, cfg.betti_grid_points)
    landscape_grid = np.linspace(0.0, max_scale, cfg.landscape_grid_points)

    b0 = betti_curve(dgms[0], betti_grid)
    b1 = betti_curve(dgms[1], betti_grid)

    landscape_h0 = truncated_persistence_landscape(dgms[0], landscape_grid, cfg.landscape_layers)
    landscape_h1 = truncated_persistence_landscape(dgms[1], landscape_grid, cfg.landscape_layers)

    fragmentation = float(np.trapz(b0, betti_grid))
    loop_persistence = total_persistence(dgms[1], power=1.0)
    entropy0 = persistent_entropy_manual(dgms[0])
    entropy1 = persistent_entropy_manual(dgms[1])
    landscape_energy0 = float(np.trapz(landscape_h0.sum(axis=0), landscape_grid))
    landscape_energy1 = float(np.trapz(landscape_h1.sum(axis=0), landscape_grid))

    summary_vector = np.concatenate(
        [
            np.array(
                [
                    fragmentation,
                    loop_persistence,
                    entropy0,
                    entropy1,
                    landscape_energy0,
                    landscape_energy1,
                ],
                dtype=float,
            ),
            b0,
            b1,
            landscape_h0.reshape(-1),
            landscape_h1.reshape(-1),
        ]
    )

    return {
        "dgms": dgms,
        "betti_grid": betti_grid,
        "betti0": b0,
        "betti1": b1,
        "landscape_grid": landscape_grid,
        "landscape_h0": landscape_h0,
        "landscape_h1": landscape_h1,
        "fragmentation": fragmentation,
        "loop_persistence": loop_persistence,
        "entropy0": entropy0,
        "entropy1": entropy1,
        "landscape_energy0": landscape_energy0,
        "landscape_energy1": landscape_energy1,
        "summary_vector": summary_vector,
        "method": method,
        "num_edges": int(result.get("num_edges", 0)),
        "approximation_used": approximation_used,
        "incremental_update": False,
    }


def compute_persistence_for_cloud(
    cloud: np.ndarray,
    cfg: SimulationConfig,
    global_indices: Optional[np.ndarray] = None,
    previous_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute persistent homology and vectorized summaries for one latent cloud.
    """
    if global_indices is None:
        global_indices = np.arange(cloud.shape[0], dtype=int)

    stable_cloud = stabilize_cloud_with_indices(cloud, global_indices)

    previous_distance = None if previous_cache is None else previous_cache.get("distance_matrix")
    previous_indices = None if previous_cache is None else previous_cache.get("global_indices")

    if cfg.incremental_topology:
        distance_matrix, incremental_used = incremental_distance_matrix(
            previous_distance,
            previous_indices,
            stable_cloud,
            global_indices,
        )
    else:
        distance_matrix = dense_distance_matrix(stable_cloud)
        incremental_used = False

    record = compute_persistence_record_from_distance_matrix(distance_matrix, cfg)
    record["incremental_update"] = bool(incremental_used)
    record["_cache"] = {
        "distance_matrix": distance_matrix,
        "global_indices": global_indices.copy(),
    }
    return record


def compute_latent_topology_sequence(
    latent_whitened: np.ndarray,
    regime_id: np.ndarray,
    split_idx: Dict[str, np.ndarray],
    cfg: SimulationConfig,
    det_cfg: DetectionConfig,
) -> Dict[str, Any]:
    """
    Compute PH summaries for every monitoring window and build regime references.
    """
    n_windows = len(latent_whitened)

    topo_records: List[Dict[str, Any]] = []
    acceleration_rows: List[Dict[str, Any]] = []
    previous_cache: Optional[Dict[str, Any]] = None

    for idx in range(n_windows):
        start = max(0, idx - cfg.topological_horizon + 1)
        cloud = latent_whitened[start : idx + 1]
        global_indices = np.arange(start, idx + 1, dtype=int)
        info = compute_persistence_for_cloud(
            cloud,
            cfg,
            global_indices=global_indices,
            previous_cache=previous_cache,
        )
        previous_cache = info.pop("_cache")
        info["window_index"] = idx
        topo_records.append(info)
        acceleration_rows.append(
            {
                "window_index": idx,
                "cloud_size": int(cloud.shape[0]),
                "method": str(info["method"]),
                "approximation_used": int(info["approximation_used"]),
                "incremental_update": int(info["incremental_update"]),
                "num_edges": int(info["num_edges"]),
            }
        )

    # Build references using normal validation windows grouped by regime.
    val_idx = split_idx["val"]
    references: Dict[int, Dict[str, Any]] = {}
    all_summary = np.stack([r["summary_vector"] for r in topo_records], axis=0)
    val_summary = all_summary[val_idx]

    for rid in np.unique(regime_id):
        regime_val = [i for i in val_idx if regime_id[i] == rid]
        if len(regime_val) == 0:
            continue

        summary_vectors = np.stack([topo_records[i]["summary_vector"] for i in regime_val], axis=0)
        fragmentation_values = np.array([topo_records[i]["fragmentation"] for i in regime_val], dtype=float)
        loop_values = np.array([topo_records[i]["loop_persistence"] for i in regime_val], dtype=float)

        ref_vector = np.median(summary_vectors, axis=0)
        ref_scale = np.nanpercentile(summary_vectors, 75, axis=0) - np.nanpercentile(summary_vectors, 25, axis=0)
        ref_scale = np.where(ref_scale < 1e-6, 1.0, ref_scale)

        # Medoid persistence diagrams from a limited sample to control cost.
        sampled = regime_val[: det_cfg.reference_sample_limit]
        sampled_h0 = [topo_records[i]["dgms"][0] for i in sampled]
        sampled_h1 = [topo_records[i]["dgms"][1] for i in sampled]

        def medoid(diagrams: List[np.ndarray], distance_fn) -> np.ndarray:
            clean = [finite_diagram(d) for d in diagrams if d is not None]
            if len(clean) == 0:
                return np.empty((0, 2), dtype=float)
            if len(clean) == 1:
                return clean[0]

            dist_mat = np.zeros((len(clean), len(clean)), dtype=float)
            for i in range(len(clean)):
                for j in range(i + 1, len(clean)):
                    try:
                        d = float(distance_fn(clean[i], clean[j]))
                        if not np.isfinite(d):
                            raise ValueError("non-finite diagram distance")
                    except Exception:
                        d = diagram_stat_fallback_distance(clean[i], clean[j])
                    dist_mat[i, j] = d
                    dist_mat[j, i] = d

            idx_best = int(np.argmin(dist_mat.sum(axis=1)))
            return clean[idx_best]

        references[int(rid)] = {
            "summary_vector": ref_vector,
            "summary_scale": ref_scale,
            "fragmentation": float(np.median(fragmentation_values)),
            "loop_persistence": float(np.median(loop_values)),
            "diagram_h0": medoid(sampled_h0, safe_bottleneck),
            "diagram_h1": medoid(sampled_h1, safe_wasserstein),
        }

    # Compute per-window topological scores.
    fragmentation = np.array([r["fragmentation"] for r in topo_records], dtype=float)
    loop_persistence_vals = np.array([r["loop_persistence"] for r in topo_records], dtype=float)
    entropy0 = np.array([r["entropy0"] for r in topo_records], dtype=float)
    entropy1 = np.array([r["entropy1"] for r in topo_records], dtype=float)
    landscape_energy0 = np.array([r["landscape_energy0"] for r in topo_records], dtype=float)
    landscape_energy1 = np.array([r["landscape_energy1"] for r in topo_records], dtype=float)

    transition_score = np.zeros(n_windows, dtype=float)
    reference_distance = np.zeros(n_windows, dtype=float)
    summary_deviation = np.zeros(n_windows, dtype=float)
    frag_dev = np.zeros(n_windows, dtype=float)
    loop_dev = np.zeros(n_windows, dtype=float)

    for i in range(n_windows):
        rid = int(regime_id[i])
        ref = references.get(rid, None)

        if i > 0:
            prev = topo_records[i - 1]["dgms"]
            cur = topo_records[i]["dgms"]
            transition_score[i] = (
                safe_bottleneck(cur[0], prev[0])
                + det_cfg.loop_distance_balance * safe_wasserstein(cur[1], prev[1])
            )

        if ref is None:
            continue

        cur = topo_records[i]["dgms"]
        reference_distance[i] = (
            safe_bottleneck(cur[0], ref["diagram_h0"])
            + det_cfg.loop_distance_balance * safe_wasserstein(cur[1], ref["diagram_h1"])
        )

        summary_vector = topo_records[i]["summary_vector"]
        summary_deviation[i] = float(
            np.linalg.norm((summary_vector - ref["summary_vector"]) / ref["summary_scale"])
            / np.sqrt(len(summary_vector))
        )

        frag_dev[i] = abs(topo_records[i]["fragmentation"] - ref["fragmentation"]) / (abs(ref["fragmentation"]) + 1e-6)
        loop_dev[i] = abs(topo_records[i]["loop_persistence"] - ref["loop_persistence"]) / (abs(ref["loop_persistence"]) + 1e-6)

    topo_df = pd.DataFrame(
        {
            "fragmentation": fragmentation,
            "loop_persistence": loop_persistence_vals,
            "persistent_entropy_h0": entropy0,
            "persistent_entropy_h1": entropy1,
            "landscape_energy_h0": landscape_energy0,
            "landscape_energy_h1": landscape_energy1,
            "transition_score": transition_score,
            "reference_distance": reference_distance,
            "summary_deviation": summary_deviation,
            "frag_dev": frag_dev,
            "loop_dev": loop_dev,
        }
    )

    return {
        "topo_records": topo_records,
        "references": references,
        "topo_df": topo_df,
        "all_summary_vectors": all_summary,
        "val_summary_vectors": val_summary,
        "acceleration_df": pd.DataFrame(acceleration_rows),
    }


# -----------------------------------------------------------------------------
# Scoring, calibration and evaluation
# -----------------------------------------------------------------------------

def derive_statistical_signals(
    X_windows: np.ndarray,
    model_outputs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Derive the statistical evidence channels required by the dual-evidence score.
    """
    residual_idx = FEATURE_NAMES.index("state_residual")
    balance_idx = FEATURE_NAMES.index("balance_inconsistency")

    last_residual = X_windows[:, -1, :, residual_idx].mean(axis=1)
    last_balance = X_windows[:, -1, :, balance_idx].mean(axis=1)
    physical_signal = last_residual + last_balance

    return pd.DataFrame(
        {
            "recon_error": model_outputs["recon_error"],
            "pred_error": model_outputs["pred_error"],
            "uncertainty": model_outputs["uncertainty"],
            "physical_signal": physical_signal,
        }
    )


def fit_regime_thresholds(
    scores: np.ndarray,
    regime_id: np.ndarray,
    label_binary: np.ndarray,
    val_idx: np.ndarray,
    tail_probability: float,
) -> Dict[int, float]:
    """
    Fit regime-specific thresholds using normal validation windows.
    """
    thresholds: Dict[int, float] = {}
    for rid in np.unique(regime_id):
        mask = (regime_id[val_idx] == rid) & (label_binary[val_idx] == 0)
        values = scores[val_idx][mask]
        if len(values) == 0:
            thresholds[int(rid)] = float(np.quantile(scores[val_idx], 1.0 - tail_probability))
        else:
            thresholds[int(rid)] = float(np.quantile(values, 1.0 - tail_probability))
    return thresholds


def apply_regime_thresholds(
    scores: np.ndarray,
    regime_id: np.ndarray,
    thresholds: Dict[int, float],
) -> np.ndarray:
    """Return one binary raw alert per window."""
    out = np.zeros_like(scores, dtype=int)
    for i in range(len(scores)):
        out[i] = int(scores[i] > thresholds[int(regime_id[i])])
    return out


def calibrate_probabilities(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    test_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate anomaly probabilities with isotonic regression when possible.

    If the validation region is class-degenerate, the function falls back to a
    smooth sigmoid transform centered on the validation median.
    """
    if len(np.unique(val_labels)) >= 2 and np.sum(val_labels) >= 2:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_scores, val_labels)
        return iso.predict(val_scores), iso.predict(test_scores)

    center = np.median(val_scores)
    scale = np.percentile(val_scores, 75) - np.percentile(val_scores, 25)
    scale = max(scale, 1e-6)
    return safe_sigmoid((val_scores - center) / scale), safe_sigmoid((test_scores - center) / scale)


def detection_delays(
    episode_id: np.ndarray,
    persistent_alert: np.ndarray,
    subset_idx: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-episode detection delays on a given subset (typically the test set).
    """
    subset_positions = {global_idx: local_pos for local_pos, global_idx in enumerate(subset_idx)}
    rows = []
    for eid in sorted(set(episode_id[subset_idx].tolist())):
        if eid == 0:
            continue
        episode_global = [idx for idx in subset_idx if episode_id[idx] == eid]
        onset_global = min(episode_global)
        end_global = max(episode_global)
        onset_local = subset_positions[onset_global]
        end_local = subset_positions[end_global]

        alert_candidates = np.where(persistent_alert[subset_idx][onset_local : end_local + 1] == 1)[0]
        if len(alert_candidates) == 0:
            delay = np.nan
            detected = 0
        else:
            delay = float(alert_candidates[0])
            detected = 1

        rows.append(
            {
                "episode_id": int(eid),
                "onset_window_global": int(onset_global),
                "end_window_global": int(end_global),
                "detected": int(detected),
                "detection_delay_windows": delay,
            }
        )

    return pd.DataFrame(rows)


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute a compact classification summary used in tables.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    summary = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": compute_brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }

    if len(np.unique(y_true)) >= 2:
        summary["roc_auc"] = float(roc_auc_score(y_true, y_score))
        summary["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        summary["roc_auc"] = np.nan
        summary["average_precision"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    summary["tn"] = int(tn)
    summary["fp"] = int(fp)
    summary["fn"] = int(fn)
    summary["tp"] = int(tp)
    return summary


def prepare_result_table(
    metadata: pd.DataFrame,
    split_idx: Dict[str, np.ndarray],
    regime_id: np.ndarray,
    statistical_df: pd.DataFrame,
    topo_df: pd.DataFrame,
    composite_score: np.ndarray,
    raw_alert: np.ndarray,
    persistent_alert: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    """
    Merge metadata and all score channels into one per-window table.
    """
    result = pd.concat([metadata.reset_index(drop=True), statistical_df, topo_df], axis=1)
    result["regime_id"] = regime_id
    result["composite_score"] = composite_score
    result["raw_alert"] = raw_alert
    result["persistent_alert"] = persistent_alert
    result["probability"] = probability

    result["split"] = "test"
    result.loc[split_idx["train"], "split"] = "train"
    result.loc[split_idx["val"], "split"] = "val"
    return result


# -----------------------------------------------------------------------------
# Explainability helpers
# -----------------------------------------------------------------------------

def gradient_based_attribution(
    model: GraphTemporalAutoencoder,
    x_window: np.ndarray,
    y_next: np.ndarray,
    adj_norm: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Compute gradient-based input attribution for one window.

    The score uses the differentiable statistical evidence component
    (reconstruction + prediction loss).  This aligns with the manuscript's
    gradient-based localization equation.

    Important implementation note
    -----------------------------
    When the main model runs on CUDA, PyTorch may route GRU/LSTM operations
    through cuDNN.  Backpropagation through a cuDNN RNN that was executed in
    evaluation mode raises:

        RuntimeError: cudnn RNN backward can only be called in training mode

    The attribution stage intentionally uses evaluation mode to keep dropout
    disabled and the explanations deterministic.  To make the routine backend-
    agnostic and robust, attribution is executed on a CPU clone of the trained
    model whenever the main model lives on CUDA.  Because only one
    representative window is explained, the overhead is negligible relative to
    the end-to-end pipeline time.
    """
    original_mode = model.training

    if device.type == "cuda":
        attr_device = torch.device("cpu")
        model_for_attr = GraphTemporalAutoencoder(
            n_features=model.n_features,
            n_nodes=model.n_nodes,
            window_length=model.window_length,
            model_cfg=model.model_cfg,
        ).to(attr_device)
        state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model_for_attr.load_state_dict(state_cpu)
    else:
        attr_device = device
        model_for_attr = model

    try:
        model_for_attr.eval()
        model_for_attr.zero_grad(set_to_none=True)
        adj_norm_attr = adj_norm.detach().to(attr_device)

        x = torch.tensor(x_window[None], dtype=torch.float32, device=attr_device)
        x = x.clone().detach().requires_grad_(True)
        y = torch.tensor(y_next[None], dtype=torch.float32, device=attr_device)

        out = model_for_attr(x, adj_norm_attr)
        score = F.mse_loss(out["recon"], x) + F.mse_loss(out["pred"], y)
        score.backward()

        if x.grad is None:
            raise RuntimeError(
                "Gradient attribution failed because no gradients were produced for the input window."
            )

        grad = x.grad.detach().abs().cpu().numpy()[0]  # [time, nodes, features]
    finally:
        model.zero_grad(set_to_none=True)
        model.train(original_mode)
        if model_for_attr is not model:
            model_for_attr.zero_grad(set_to_none=True)

    node_attr = grad.mean(axis=(0, 2))
    feature_attr = grad.mean(axis=(0, 1))
    time_attr = grad.mean(axis=(1, 2))

    return {
        "full_grad": grad,
        "node_attr": node_attr,
        "feature_attr": feature_attr,
        "time_attr": time_attr,
    }


def combine_gradient_and_physics_attribution(
    grad_node_attr: np.ndarray,
    x_window: np.ndarray,
    det_cfg: DetectionConfig,
) -> np.ndarray:
    """
    Fuse gradient attribution with physics-aware local evidence.

    The physical component uses the manuscript's core local signals:
        * state-estimation residual,
        * nodal balance inconsistency.
    """
    residual_idx = FEATURE_NAMES.index("state_residual")
    balance_idx = FEATURE_NAMES.index("balance_inconsistency")

    physics = (
        x_window[-1, :, residual_idx]
        + x_window[-1, :, balance_idx]
    )
    physics = np.clip(physics, 0.0, None)

    if grad_node_attr.max() > 0:
        grad_norm = grad_node_attr / (grad_node_attr.max() + 1e-8)
    else:
        grad_norm = grad_node_attr

    if physics.max() > 0:
        physics_norm = physics / (physics.max() + 1e-8)
    else:
        physics_norm = physics

    combined = det_cfg.gradient_weight * grad_norm + det_cfg.physics_weight * physics_norm
    return combined


def build_simplicial_complex_for_region(
    G: nx.Graph,
    selected_nodes: List[int],
) -> Dict[str, Any]:
    """
    Build a TopoNetX simplicial complex from the anomalous subgraph.

    The complex includes:
        * 1-simplices from graph edges,
        * 2-simplices from 3-cliques,
    and derives a 1-Hodge Laplacian using TopoNetX incidence matrices.
    """
    subgraph = G.subgraph(selected_nodes).copy()
    simplices: List[List[int]] = []

    for edge in subgraph.edges():
        simplices.append([int(edge[0]), int(edge[1])])

    triangles = [list(clique) for clique in nx.enumerate_all_cliques(subgraph) if len(clique) == 3]
    simplices.extend(triangles)

    if len(simplices) == 0:
        simplices = [[int(n)] for n in subgraph.nodes()]

    sc = tnx.SimplicialComplex(simplices)
    B1 = sc.incidence_matrix(1)
    try:
        B2 = sc.incidence_matrix(2)
    except Exception:
        B2 = sp.csr_matrix((B1.shape[1], 0), dtype=float)

    if not sp.issparse(B1):
        B1 = sp.csr_matrix(B1)
    if not sp.issparse(B2):
        B2 = sp.csr_matrix(B2)

    L1 = (B1.T @ B1) + (B2 @ B2.T)
    L1_dense = L1.toarray() if sp.issparse(L1) else np.asarray(L1, dtype=float)
    eigenvalues = np.linalg.eigvalsh(L1_dense) if L1_dense.size > 0 else np.array([], dtype=float)

    return {
        "subgraph": subgraph,
        "simplicial_complex": sc,
        "B1_shape": B1.shape,
        "B2_shape": B2.shape,
        "L1_eigenvalues": eigenvalues,
        "triangles": triangles,
    }


def topological_narrative(
    row: pd.Series,
) -> List[str]:
    """
    Convert topological quantities into operator-friendly interpretation hints.
    """
    notes = []
    if row["frag_dev"] > 1.0:
        notes.append("Latent manifold splitting is elevated relative to the regime baseline, suggesting structural fragmentation or topology inconsistency.")
    if row["loop_dev"] > 1.0:
        notes.append("Persistent loop activity is elevated, which is consistent with oscillatory or recurrent anomalous behavior.")
    if row["transition_score"] > np.percentile([row["transition_score"]], 50):
        notes.append("Topological transition between adjacent monitoring horizons is abrupt.")
    if len(notes) == 0:
        notes.append("Topological deviation is moderate and should be interpreted jointly with residual and localization evidence.")
    return notes


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_score_timeline(
    df: pd.DataFrame,
    thresholds: Dict[int, float],
    section_mask: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """
    Plot composite score, regime-specific threshold and persistent alerts.

    The same plotting function is reused for sections 5.1 and 5.2 with different
    masks and titles.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(df))

    ax.plot(x, df["composite_score"], label="Composite anomaly score")
    threshold_series = np.array([thresholds[int(r)] for r in df["regime_id"]], dtype=float)
    ax.plot(x, threshold_series, label="Regime threshold", linestyle="--")

    if section_mask.any():
        idx = np.where(section_mask)[0]
        for i in idx:
            ax.axvspan(max(0, i - 0.5), i + 0.5, alpha=0.10, color="red")

    alert_idx = np.where(df["persistent_alert"].values == 1)[0]
    if len(alert_idx) > 0:
        ax.scatter(alert_idx, df["composite_score"].values[alert_idx], marker="x", label="Persistent alerts")

    ax.set_title(title)
    ax.set_xlabel("Monitoring window")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_persistence_comparison(
    before_record: Dict[str, Any],
    during_record: Dict[str, Any],
    title: str,
    path: Path,
) -> None:
    """
    Save side-by-side persistence diagrams for a representative event comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_diagrams(before_record["dgms"], ax=axes[0], title="Before event")
    plot_diagrams(during_record["dgms"], ax=axes[1], title="During event")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_betti_comparison(
    before_record: Dict[str, Any],
    during_record: Dict[str, Any],
    title: str,
    path: Path,
) -> None:
    """
    Save Betti-curve comparison before vs during a representative event.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(before_record["betti_grid"], before_record["betti0"], label="Before event Betti-0")
    ax.plot(before_record["betti_grid"], before_record["betti1"], label="Before event Betti-1")
    ax.plot(during_record["betti_grid"], during_record["betti0"], label="During event Betti-0", linestyle="--")
    ax.plot(during_record["betti_grid"], during_record["betti1"], label="During event Betti-1", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Filtration scale")
    ax.set_ylabel("Betti count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_feature_boxplot(
    df: pd.DataFrame,
    column: str,
    group_col: str,
    title: str,
    path: Path,
) -> None:
    """Create a classical boxplot with matplotlib only."""
    groups = [g for g in sorted(df[group_col].unique()) if g != "normal"]
    data = [df.loc[df[group_col] == g, column].values for g in groups]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=groups, vert=True)
    ax.set_title(title)
    ax.set_ylabel(column)
    ax.set_xlabel(group_col)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_attribution_heatmap(
    grad: np.ndarray,
    selected_nodes: List[int],
    title: str,
    path: Path,
) -> None:
    """
    Plot a node-by-time attribution heatmap for the selected nodes.
    """
    # Aggregate gradients over features and restrict to selected nodes.
    heat = grad[:, selected_nodes, :].mean(axis=2).T

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heat, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Time step inside monitoring window")
    ax.set_ylabel("Selected nodes")
    ax.set_yticks(np.arange(len(selected_nodes)))
    ax.set_yticklabels([str(n) for n in selected_nodes])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_grid_localization(
    G: nx.Graph,
    positions: Dict[int, Tuple[float, float]],
    node_scores: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """
    Plot the full grid with node-wise anomaly responsibility.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    node_list = sorted(G.nodes())
    values = np.array([node_scores[n] for n in node_list], dtype=float)
    nx.draw_networkx_edges(G, positions, ax=ax, alpha=0.5)
    nodes = nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=node_list,
        node_color=values,
        node_size=550,
        cmap="viridis",
        ax=ax,
    )
    nx.draw_networkx_labels(G, positions, ax=ax, font_size=8)
    fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04, label="Localization score")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_region_influence(
    region_scores: Dict[str, float],
    title: str,
    path: Path,
) -> None:
    """Bar plot of aggregated regional influence."""
    labels = list(region_scores.keys())
    values = [region_scores[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Aggregated influence")
    ax.set_xlabel("Region")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_simplicial_region(
    simplicial_info: Dict[str, Any],
    positions: Dict[int, Tuple[float, float]],
    node_scores: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """
    Visualize the anomalous subgraph and its 2-simplices.
    """
    subgraph = simplicial_info["subgraph"]
    triangles = simplicial_info["triangles"]

    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw_networkx_edges(subgraph, positions, ax=ax, width=2.0, alpha=0.65)

    # Fill 2-simplices to show the higher-order lift.
    for tri in triangles:
        coords = np.array([positions[n] for n in tri], dtype=float)
        polygon = plt.Polygon(coords, closed=True, alpha=0.18)
        ax.add_patch(polygon)

    node_list = sorted(subgraph.nodes())
    values = np.array([node_scores[n] for n in node_list], dtype=float)
    nodes = nx.draw_networkx_nodes(
        subgraph,
        positions,
        nodelist=node_list,
        node_color=values,
        node_size=700,
        cmap="plasma",
        ax=ax,
    )
    nx.draw_networkx_labels(subgraph, positions, ax=ax, font_size=8)
    fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04, label="Node influence")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_hodge_spectrum(
    eigenvalues: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Plot eigenvalues of the local 1-Hodge Laplacian."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(eigenvalues) > 0:
        ax.plot(np.arange(len(eigenvalues)), np.sort(eigenvalues), marker="o")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)




def plot_landscape_comparison(
    before_record: Dict[str, Any],
    during_record: Dict[str, Any],
    title: str,
    path: Path,
) -> None:
    """
    Save a truncated persistence-landscape comparison focused on H1.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    max_layers = min(
        before_record["landscape_h1"].shape[0],
        during_record["landscape_h1"].shape[0],
        3,
    )
    for layer in range(max_layers):
        ax.plot(
            before_record["landscape_grid"],
            before_record["landscape_h1"][layer],
            label=f"Before λ{layer + 1}",
        )
        ax.plot(
            during_record["landscape_grid"],
            during_record["landscape_h1"][layer],
            linestyle="--",
            label=f"During λ{layer + 1}",
        )
    ax.set_title(title)
    ax.set_xlabel("Filtration scale")
    ax.set_ylabel("Landscape amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize_mapper_nodes(
    mapper_graph: Dict[str, Any],
    sample_scores: np.ndarray,
    sample_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Build a publication-friendly Mapper node summary table.
    """
    rows = []
    nodes = mapper_graph.get("nodes", {})
    for node_id, members in nodes.items():
        member_list = list(members)
        if len(member_list) == 0:
            dominant_label = "empty"
            purity = 0.0
            mean_score = 0.0
            max_score = 0.0
        else:
            label_counts = Counter([str(sample_labels[i]) for i in member_list])
            dominant_label, dominant_count = label_counts.most_common(1)[0]
            purity = dominant_count / len(member_list)
            mean_score = float(np.mean(sample_scores[member_list]))
            max_score = float(np.max(sample_scores[member_list]))

        rows.append(
            {
                "mapper_node": str(node_id),
                "n_members": int(len(member_list)),
                "mean_score": mean_score,
                "max_score": max_score,
                "dominant_subtype": dominant_label,
                "dominant_purity": float(purity),
            }
        )

    if len(rows) == 0:
        return pd.DataFrame(
            columns=["mapper_node", "n_members", "mean_score", "max_score", "dominant_subtype", "dominant_purity"]
        )

    return pd.DataFrame(rows).sort_values(["mean_score", "n_members"], ascending=[False, False]).reset_index(drop=True)


def plot_mapper_static(
    mapper_graph: Any,
    title: str,
    path: Path,
    sample_scores: Optional[np.ndarray] = None,
    sample_labels: Optional[Sequence[str]] = None,
    label_top_k: int = 8,
) -> None:
    """
    Save a publication-oriented static Mapper figure.

    Node size encodes membership cardinality, node color encodes mean anomaly
    score, and the highest-scoring nodes are annotated with their dominant class.
    """
    if sample_scores is None or sample_labels is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        km_draw_matplotlib(mapper_graph, ax=ax, fig=fig, layout="kk")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return

    nodes = mapper_graph.get("nodes", {})
    links = mapper_graph.get("links", {})
    Gm = nx.Graph()
    for node_id, members in nodes.items():
        member_list = list(members)
        scores = np.asarray(sample_scores[member_list], dtype=float) if len(member_list) > 0 else np.array([0.0])
        labels = [str(sample_labels[i]) for i in member_list] if len(member_list) > 0 else ["empty"]
        dominant_label, dominant_count = Counter(labels).most_common(1)[0]
        Gm.add_node(
            str(node_id),
            n_members=int(len(member_list)),
            mean_score=float(np.mean(scores)),
            dominant_label=dominant_label,
            purity=float(dominant_count / max(len(member_list), 1)),
            members=member_list,
        )

    for src, targets in links.items():
        for tgt in targets:
            src_key = str(src)
            tgt_key = str(tgt)
            src_members = set(nodes.get(src, []))
            tgt_members = set(nodes.get(tgt, []))
            overlap = len(src_members.intersection(tgt_members))
            Gm.add_edge(src_key, tgt_key, overlap=max(overlap, 1))

    if Gm.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title)
        ax.text(0.5, 0.5, "Mapper graph is empty", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return

    try:
        pos = nx.kamada_kawai_layout(Gm)
    except Exception:
        pos = nx.spring_layout(Gm, seed=42)

    node_list = list(Gm.nodes())
    node_sizes = [250.0 + 65.0 * math.sqrt(max(Gm.nodes[n]["n_members"], 1)) for n in node_list]
    node_colors = [Gm.nodes[n]["mean_score"] for n in node_list]
    edge_widths = [0.8 + 0.35 * math.log1p(Gm.edges[e]["overlap"]) for e in Gm.edges()]

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_edges(Gm, pos, width=edge_widths, alpha=0.35, ax=ax)
    nodes_artist = nx.draw_networkx_nodes(
        Gm,
        pos,
        nodelist=node_list,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
        ax=ax,
        linewidths=0.5,
        edgecolors="black",
    )

    ranked_nodes = sorted(node_list, key=lambda n: Gm.nodes[n]["mean_score"], reverse=True)[:label_top_k]
    labels = {
        n: f"{Gm.nodes[n]['dominant_label']}\nN={Gm.nodes[n]['n_members']}"
        for n in ranked_nodes
    }
    nx.draw_networkx_labels(Gm, pos, labels=labels, font_size=8, ax=ax)

    cbar = fig.colorbar(nodes_artist, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean composite score")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main analysis orchestration
# -----------------------------------------------------------------------------

def run_pipeline(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the full smart-grid topological-anomaly pipeline."""
    require_topological_stack()
    set_global_seed(config.simulation.seed)
    paths = build_output_tree(config.output_root)

    # Persist configuration for auditability.
    save_json(paths.global_dir / "experiment_config.json", asdict(config))

    # ---------------------------------------------------------------------
    # 1) Synthetic cyber-physical data generation
    # ---------------------------------------------------------------------
    synthetic = simulate_smart_grid(config.simulation)
    save_json(paths.global_dir / "anomaly_schedule.json", {"events": synthetic["schedule"]})

    preprocessing = normalize_and_smooth(
        synthetic["raw_features"],
        synthetic["anomaly_binary"],
        config.simulation,
    )

    windows = make_windows(
        preprocessing["smoothed"],
        synthetic["context_features"],
        synthetic["timestamps"],
        synthetic["anomaly_binary"],
        synthetic["family_group"],
        synthetic["subtype"],
        synthetic["episode_id"],
        synthetic["node_mask"],
        synthetic["regime_names"],
        config.simulation,
    )

    split_idx = chronological_split(len(windows["X"]), config.simulation)
    regime_model = fit_regime_model(
        windows["context"],
        windows["label_binary"],
        split_idx,
        config.simulation,
    )
    regime_id = regime_model["regime_id"]

    # Metadata table
    metadata = pd.DataFrame(
        {
            "end_time_minutes": windows["end_time"],
            "label_binary": windows["label_binary"],
            "family_group": windows["family_group"],
            "subtype": windows["subtype"],
            "episode_id": windows["episode_id"],
            "regime_name": windows["regime_name"],
        }
    )

    # ---------------------------------------------------------------------
    # 2) Build deep-learning train/validation sets
    # ---------------------------------------------------------------------
    train_idx = split_idx["train"]
    val_idx = split_idx["val"]

    train_normal_idx = train_idx[windows["label_binary"][train_idx] == 0]
    val_normal_idx = val_idx[windows["label_binary"][val_idx] == 0]

    X_train = windows["X"][train_normal_idx]
    Y_train = windows["Y"][train_normal_idx]
    regime_train = regime_id[train_normal_idx]

    X_train_aug, Y_train_aug, regime_train_aug = augment_normal_windows(
        X_train,
        Y_train,
        regime_train,
        config.model,
        config.simulation.seed,
    )

    X_val = windows["X"][val_normal_idx]
    Y_val = windows["Y"][val_normal_idx]
    regime_val = regime_id[val_normal_idx]

    train_dataset = WindowDataset(X_train_aug, Y_train_aug, regime_train_aug)
    val_dataset = WindowDataset(X_val, Y_val, regime_val)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.model.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.model.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    # Model and adjacency
    G = synthetic["graph"]
    adj_norm_np = normalized_adjacency_matrix(G).astype(np.float32)
    device = select_torch_device()
    adj_norm = torch.tensor(adj_norm_np, dtype=torch.float32, device=device)

    try:
        model = GraphTemporalAutoencoder(
            n_features=len(FEATURE_NAMES),
            n_nodes=G.number_of_nodes(),
            window_length=config.simulation.window_length,
            model_cfg=config.model,
        ).to(device)
    except Exception:
        device = torch.device("cpu")
        adj_norm = torch.tensor(adj_norm_np, dtype=torch.float32, device=device)
        model = GraphTemporalAutoencoder(
            n_features=len(FEATURE_NAMES),
            n_nodes=G.number_of_nodes(),
            window_length=config.simulation.window_length,
            model_cfg=config.model,
        ).to(device)

    history = train_model(
        model,
        train_loader,
        val_loader,
        adj_norm,
        config.model,
        device,
        paths.global_dir,
    )
    plot_training_history(history, paths.global_dir / "training_history.png")

    # ---------------------------------------------------------------------
    # 3) Deep inference
    # ---------------------------------------------------------------------
    model_outputs = infer_model_outputs(
        model,
        windows["X"],
        windows["Y"],
        adj_norm,
        device,
    )

    statistical_df = derive_statistical_signals(windows["X"], model_outputs)

    # ---------------------------------------------------------------------
    # 4) Latent whitening and persistent homology
    # ---------------------------------------------------------------------
    whiten_ref = fit_whitening_transform(model_outputs["latent"][train_normal_idx])
    latent_whitened = apply_whitening(model_outputs["latent"], whiten_ref)
    topology = compute_latent_topology_sequence(
        latent_whitened,
        regime_id,
        split_idx,
        config.simulation,
        config.detection,
    )
    topo_df = topology["topo_df"]

    # ---------------------------------------------------------------------
    # 5) Regime-aware dual-evidence scoring
    # ---------------------------------------------------------------------
    val_normal_mask = np.zeros(len(windows["X"]), dtype=bool)
    val_normal_mask[val_idx] = True
    val_normal_mask &= (windows["label_binary"] == 0)

    # Robust score normalization using normal validation windows.
    stat_ref = {}
    stat_z = {}
    for col in ["recon_error", "pred_error", "uncertainty", "physical_signal"]:
        ref, scale = robust_center_scale(statistical_df.loc[val_normal_mask, col].values.reshape(-1, 1))
        stat_ref[col] = (ref[0], scale[0])
        stat_z[col] = robust_positive_zscore(statistical_df[col].values, ref[0], scale[0])

    topo_ref = {}
    topo_z = {}
    for col in ["frag_dev", "loop_dev", "summary_deviation", "reference_distance", "transition_score"]:
        ref, scale = robust_center_scale(topo_df.loc[val_normal_mask, col].values.reshape(-1, 1))
        topo_ref[col] = (ref[0], scale[0])
        topo_z[col] = robust_positive_zscore(topo_df[col].values, ref[0], scale[0])

    statistical_score = (
        config.detection.a_recon * stat_z["recon_error"]
        + config.detection.a_pred * stat_z["pred_error"]
        + config.detection.a_residual * stat_z["physical_signal"]
        + config.detection.a_uncertainty * stat_z["uncertainty"]
    )

    topological_score = (
        config.detection.b_fragmentation * topo_z["frag_dev"]
        + config.detection.b_loop * topo_z["loop_dev"]
        + config.detection.b_summary * topo_z["summary_deviation"]
        + config.detection.b_reference_distance * topo_z["reference_distance"]
        + config.detection.b_transition * topo_z["transition_score"]
    )

    composite_score = (
        config.detection.lambda_statistical * statistical_score
        + (1.0 - config.detection.lambda_statistical) * topological_score
    )

    thresholds = fit_regime_thresholds(
        composite_score,
        regime_id,
        windows["label_binary"],
        val_idx,
        config.detection.alert_tail_probability,
    )
    raw_alert = apply_regime_thresholds(composite_score, regime_id, thresholds)
    persistent_alert = rolling_persistent_alert(
        raw_alert,
        config.detection.persistence_k,
        config.detection.persistence_h,
    )

    val_prob, test_prob = calibrate_probabilities(
        composite_score[val_idx],
        windows["label_binary"][val_idx],
        composite_score[split_idx["test"]],
    )
    probability = np.zeros_like(composite_score, dtype=float)
    probability[val_idx] = val_prob
    probability[split_idx["test"]] = test_prob
    probability[split_idx["train"]] = safe_sigmoid(
        (composite_score[split_idx["train"]] - np.median(composite_score[val_idx]))
        / max(np.percentile(composite_score[val_idx], 75) - np.percentile(composite_score[val_idx], 25), 1e-6)
    )

    # ---------------------------------------------------------------------
    # 6) Aggregate final result table
    # ---------------------------------------------------------------------
    result_df = prepare_result_table(
        metadata=metadata,
        split_idx=split_idx,
        regime_id=regime_id,
        statistical_df=statistical_df.assign(
            statistical_score=statistical_score,
        ),
        topo_df=topo_df.assign(
            topological_score=topological_score,
        ),
        composite_score=composite_score,
        raw_alert=raw_alert,
        persistent_alert=persistent_alert,
        probability=probability,
    )
    result_df["window_index"] = np.arange(len(result_df))
    result_df.to_csv(paths.global_dir / "per_window_results.csv", index=False)

    # Global evaluation summaries.
    test_mask = result_df["split"] == "test"
    test_df = result_df.loc[test_mask].copy()
    test_idx = split_idx["test"]

    physical_binary_test = (test_df["family_group"].values == "physical_topology").astype(int)
    cyber_binary_test = (test_df["family_group"].values == "cyber_data_quality").astype(int)

    physical_prob = np.clip(test_df["probability"].values, 0.0, 1.0)
    cyber_prob = np.clip(test_df["probability"].values, 0.0, 1.0)

    summary_rows = []

    physical_summary = classification_summary(
        y_true=physical_binary_test,
        y_pred=test_df["persistent_alert"].values.astype(int),
        y_score=test_df["composite_score"].values,
        y_prob=physical_prob,
    )
    physical_summary["target_group"] = "physical_topology"
    summary_rows.append(physical_summary)

    cyber_summary = classification_summary(
        y_true=cyber_binary_test,
        y_pred=test_df["persistent_alert"].values.astype(int),
        y_score=test_df["composite_score"].values,
        y_prob=cyber_prob,
    )
    cyber_summary["target_group"] = "cyber_data_quality"
    summary_rows.append(cyber_summary)

    evaluation_summary_df = pd.DataFrame(summary_rows)
    save_table_bundle(
        evaluation_summary_df,
        paths.global_dir / "evaluation_summary.csv",
        paths.global_dir / "evaluation_summary.tex",
        caption="Global evaluation summary on the test split.",
        label="tab:global_evaluation_summary",
        precision=config.simulation.publication_table_precision,
    )

    delay_df = detection_delays(
        windows["episode_id"],
        persistent_alert,
        test_idx,
    )
    save_table_bundle(
        delay_df,
        paths.global_dir / "episode_detection_delays.csv",
        paths.global_dir / "episode_detection_delays.tex",
        caption="Per-episode persistent-alert detection delays on the test split.",
        label="tab:episode_detection_delays",
        precision=config.simulation.publication_table_precision,
    )

    subtype_summary = (
        test_df.groupby("subtype")[["statistical_score", "topological_score", "composite_score", "persistent_alert"]]
        .agg(["mean", "max", "count"])
        .reset_index()
    )
    save_table_bundle(
        subtype_summary,
        paths.global_dir / "subtype_score_summary.csv",
        paths.global_dir / "subtype_score_summary.tex",
        caption="Score summary by anomaly subtype on the test split.",
        label="tab:subtype_score_summary",
        precision=config.simulation.publication_table_precision,
    )

    # ---------------------------------------------------------------------
    # 7) Section 5.1 artifacts
    # ---------------------------------------------------------------------
    manifest: List[Dict[str, str]] = []

    physical_mask_all = result_df["family_group"].values == "physical_topology"
    plot_score_timeline(
        result_df,
        thresholds,
        physical_mask_all,
        "Section 5.1 - Composite score under physical faults and topology inconsistencies",
        paths.sec_5_1 / "5_1_composite_score_timeline.png",
    )
    register_artifact(
        manifest,
        "5.1",
        paths.sec_5_1 / "5_1_composite_score_timeline.png",
        "Timeline of composite anomaly score, regime threshold and persistent alerts for physical/topology events.",
    )

    phys_candidates = result_df.index[result_df["family_group"] == "physical_topology"].tolist()
    if len(phys_candidates) > 0:
        representative_phys = int(result_df.loc[phys_candidates, "composite_score"].idxmax())
        same_regime_normal_before = result_df.index[
            (result_df.index < representative_phys)
            & (result_df["family_group"] == "normal")
            & (result_df["regime_id"] == result_df.loc[representative_phys, "regime_id"])
        ].tolist()
        before_phys = same_regime_normal_before[-1] if len(same_regime_normal_before) > 0 else max(0, representative_phys - 5)

        plot_persistence_comparison(
            topology["topo_records"][before_phys],
            topology["topo_records"][representative_phys],
            "Section 5.1 - Persistence diagrams before vs during representative physical/topology event",
            paths.sec_5_1 / "5_1_persistence_diagrams_physical.png",
        )
        register_artifact(
            manifest,
            "5.1",
            paths.sec_5_1 / "5_1_persistence_diagrams_physical.png",
            "Persistence diagrams before and during a representative physical/topology event.",
        )

        plot_betti_comparison(
            topology["topo_records"][before_phys],
            topology["topo_records"][representative_phys],
            "Section 5.1 - Betti curves before vs during representative physical/topology event",
            paths.sec_5_1 / "5_1_betti_curves_physical.png",
        )
        register_artifact(
            manifest,
            "5.1",
            paths.sec_5_1 / "5_1_betti_curves_physical.png",
            "Betti-0 and Betti-1 curves before and during a representative physical/topology event.",
        )

        plot_landscape_comparison(
            topology["topo_records"][before_phys],
            topology["topo_records"][representative_phys],
            "Section 5.1 - Truncated persistence landscapes before vs during representative physical/topology event",
            paths.sec_5_1 / "5_1_persistence_landscapes_physical.png",
        )
        register_artifact(
            manifest,
            "5.1",
            paths.sec_5_1 / "5_1_persistence_landscapes_physical.png",
            "Truncated H1 persistence landscapes before and during a representative physical/topology event.",
        )

    physical_test = result_df.loc[test_mask & (result_df["family_group"].isin(["physical_topology", "normal"]))].copy()
    physical_test_summary = (
        physical_test.groupby("subtype")[["statistical_score", "topological_score", "composite_score", "persistent_alert"]]
        .agg(["mean", "max", "count"])
        .reset_index()
    )
    save_table_bundle(
        physical_test_summary,
        paths.sec_5_1 / "5_1_physical_summary_table.csv",
        paths.sec_5_1 / "5_1_physical_summary_table.tex",
        caption="Section 5.1 summary of physical/topology score behavior.",
        label="tab:section_5_1_physical_summary",
        precision=config.simulation.publication_table_precision,
    )
    register_artifact(
        manifest,
        "5.1",
        paths.sec_5_1 / "5_1_physical_summary_table.csv",
        "Summary table of score behavior across physical and topology-inconsistency subtypes.",
    )
    register_artifact(
        manifest,
        "5.1",
        paths.sec_5_1 / "5_1_physical_summary_table.tex",
        "LaTeX-ready table of physical and topology-inconsistency score behavior.",
    )

    # ---------------------------------------------------------------------
    # 8) Section 5.2 artifacts
    # ---------------------------------------------------------------------
    cyber_mask_all = result_df["family_group"].values == "cyber_data_quality"
    plot_score_timeline(
        result_df,
        thresholds,
        cyber_mask_all,
        "Section 5.2 - Composite score under cyber and data-quality anomalies",
        paths.sec_5_2 / "5_2_composite_score_timeline.png",
    )
    register_artifact(
        manifest,
        "5.2",
        paths.sec_5_2 / "5_2_composite_score_timeline.png",
        "Timeline of composite anomaly score, regime threshold and persistent alerts for cyber/data-quality events.",
    )

    plot_feature_boxplot(
        result_df.loc[test_mask].copy(),
        column="reference_distance",
        group_col="subtype",
        title="Section 5.2 - Diagram reference distance by anomaly subtype",
        path=paths.sec_5_2 / "5_2_reference_distance_boxplot.png",
    )
    register_artifact(
        manifest,
        "5.2",
        paths.sec_5_2 / "5_2_reference_distance_boxplot.png",
        "Boxplot of persistence-diagram reference distances by anomaly subtype.",
    )

    plot_feature_boxplot(
        result_df.loc[test_mask].copy(),
        column="persistent_entropy_h1",
        group_col="subtype",
        title="Section 5.2 - Persistent entropy (H1) by anomaly subtype",
        path=paths.sec_5_2 / "5_2_persistent_entropy_h1_boxplot.png",
    )
    register_artifact(
        manifest,
        "5.2",
        paths.sec_5_2 / "5_2_persistent_entropy_h1_boxplot.png",
        "Boxplot of persistent entropy in H1 by anomaly subtype.",
    )

    cyber_candidates = result_df.index[result_df["family_group"] == "cyber_data_quality"].tolist()
    if len(cyber_candidates) > 0:
        representative_cyber = int(result_df.loc[cyber_candidates, "composite_score"].idxmax())
        same_regime_normal_before = result_df.index[
            (result_df.index < representative_cyber)
            & (result_df["family_group"] == "normal")
            & (result_df["regime_id"] == result_df.loc[representative_cyber, "regime_id"])
        ].tolist()
        before_cyber = same_regime_normal_before[-1] if len(same_regime_normal_before) > 0 else max(0, representative_cyber - 5)

        plot_persistence_comparison(
            topology["topo_records"][before_cyber],
            topology["topo_records"][representative_cyber],
            "Section 5.2 - Persistence diagrams before vs during representative cyber/data anomaly",
            paths.sec_5_2 / "5_2_persistence_diagrams_cyber.png",
        )
        register_artifact(
            manifest,
            "5.2",
            paths.sec_5_2 / "5_2_persistence_diagrams_cyber.png",
            "Persistence diagrams before and during a representative cyber/data-quality event.",
        )

        plot_landscape_comparison(
            topology["topo_records"][before_cyber],
            topology["topo_records"][representative_cyber],
            "Section 5.2 - Truncated persistence landscapes before vs during representative cyber/data anomaly",
            paths.sec_5_2 / "5_2_persistence_landscapes_cyber.png",
        )
        register_artifact(
            manifest,
            "5.2",
            paths.sec_5_2 / "5_2_persistence_landscapes_cyber.png",
            "Truncated H1 persistence landscapes before and during a representative cyber/data-quality event.",
        )


    # Mapper graph for the test latent manifold
    test_latent = np.nan_to_num(latent_whitened[test_idx], nan=0.0, posinf=0.0, neginf=0.0)
    test_df_mapper = result_df.loc[test_idx].copy()

    if len(test_latent) > config.simulation.max_mapper_points:
        sampled_idx_local = np.linspace(
            0,
            len(test_latent) - 1,
            config.simulation.max_mapper_points,
            dtype=int,
        )
        mapper_latent = test_latent[sampled_idx_local]
        mapper_scores = np.nan_to_num(
            test_df_mapper.iloc[sampled_idx_local]["composite_score"].values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mapper_subtypes = test_df_mapper.iloc[sampled_idx_local]["subtype"].astype(str).values
        mapper_tooltips = np.array(
            [
                f"window={int(test_df_mapper.iloc[i]['window_index'])} | subtype={test_df_mapper.iloc[i]['subtype']} | score={test_df_mapper.iloc[i]['composite_score']:.3f}"
                for i in sampled_idx_local
            ]
        )
    else:
        mapper_latent = test_latent
        mapper_scores = np.nan_to_num(
            test_df_mapper["composite_score"].values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mapper_subtypes = test_df_mapper["subtype"].astype(str).values
        mapper_tooltips = np.array(
            [
                f"window={int(row.window_index)} | subtype={row.subtype} | score={row.composite_score:.3f}"
                for row in test_df_mapper.itertuples()
            ]
        )

    pca = PCA(n_components=2, random_state=config.simulation.seed)
    mapper_lens = pca.fit_transform(mapper_latent)

    mapper = km.KeplerMapper(verbose=0)
    lens = mapper.fit_transform(mapper_lens, projection=[0, 1])
    mapper_graph = mapper.map(
        lens,
        mapper_latent,
        cover=km.Cover(n_cubes=10, perc_overlap=0.45),
        clusterer=DBSCAN(eps=0.70, min_samples=3),
    )

    mapper_html_path = paths.sec_5_2 / "5_2_mapper_graph.html"
    mapper.visualize(
        mapper_graph,
        path_html=str(mapper_html_path),
        title="Section 5.2 - KeplerMapper graph of test latent trajectories",
        color_values=mapper_scores,
        color_function_name="Composite anomaly score",
        custom_tooltips=mapper_tooltips,
    )
    register_artifact(
        manifest,
        "5.2",
        mapper_html_path,
        "Interactive KeplerMapper HTML of the latent test manifold colored by composite score.",
    )

    plot_mapper_static(
        mapper_graph,
        title="Section 5.2 - Publication-oriented Mapper graph of latent test windows",
        path=paths.sec_5_2 / "5_2_mapper_graph_static.png",
        sample_scores=mapper_scores,
        sample_labels=mapper_subtypes,
        label_top_k=config.simulation.mapper_label_top_k,
    )
    register_artifact(
        manifest,
        "5.2",
        paths.sec_5_2 / "5_2_mapper_graph_static.png",
        "Publication-oriented static Mapper graph for direct insertion into the manuscript.",
    )

    mapper_node_summary = summarize_mapper_nodes(
        mapper_graph,
        mapper_scores,
        mapper_subtypes,
    )
    save_table_bundle(
        mapper_node_summary,
        paths.sec_5_2 / "5_2_mapper_node_summary.csv",
        paths.sec_5_2 / "5_2_mapper_node_summary.tex",
        caption="Section 5.2 Mapper node summary with dominant subtype and mean score.",
        label="tab:section_5_2_mapper_node_summary",
        precision=config.simulation.publication_table_precision,
    )
    register_artifact(
        manifest,
        "5.2",
        paths.sec_5_2 / "5_2_mapper_node_summary.csv",
        "Tabular summary of Mapper nodes with dominant subtype and score statistics.",
    )
    register_artifact(
        manifest,
        "5.2",
        paths.sec_5_2 / "5_2_mapper_node_summary.tex",
        "LaTeX-ready summary of Mapper nodes with dominant subtype and score statistics.",
    )

    cyber_test_summary = (
        result_df.loc[test_mask & (result_df["family_group"].isin(["cyber_data_quality", "normal"]))]
        .groupby("subtype")[["statistical_score", "topological_score", "composite_score", "persistent_alert"]]
        .agg(["mean", "max", "count"])
        .reset_index()
    )
    save_table_bundle(
        cyber_test_summary,
        paths.sec_5_2 / "5_2_cyber_summary_table.csv",
        paths.sec_5_2 / "5_2_cyber_summary_table.tex",
        caption="Section 5.2 summary of cyber/data-quality score behavior.",
        label="tab:section_5_2_cyber_summary",
        precision=config.simulation.publication_table_precision,
    )
    register_artifact(
        manifest,
        "5.2",
        paths.sec_5_2 / "5_2_cyber_summary_table.csv",
        "Summary table of score behavior across cyber and data-quality subtypes.",
    )
    register_artifact(
        manifest,
        "5.2",
        paths.sec_5_2 / "5_2_cyber_summary_table.tex",
        "LaTeX-ready table of score behavior across cyber and data-quality subtypes.",
    )
    # ---------------------------------------------------------------------
    # 9) Section 5.3 artifacts
    # ---------------------------------------------------------------------
    representative_alerts = result_df.index[result_df["persistent_alert"] == 1].tolist()
    if len(representative_alerts) > 0:
        representative_idx = int(result_df.loc[representative_alerts, "composite_score"].idxmax())
    else:
        representative_idx = int(result_df["composite_score"].idxmax())

    attribution = gradient_based_attribution(
        model=model,
        x_window=windows["X"][representative_idx],
        y_next=windows["Y"][representative_idx],
        adj_norm=adj_norm,
        device=device,
    )
    combined_node_attr = combine_gradient_and_physics_attribution(
        attribution["node_attr"],
        windows["X"][representative_idx],
        config.detection,
    )

    top_nodes = np.argsort(combined_node_attr)[::-1][: config.simulation.top_k_localization_nodes].tolist()

    plot_attribution_heatmap(
        attribution["full_grad"],
        top_nodes,
        "Section 5.3 - Node/time attribution heatmap for a representative alert",
        paths.sec_5_3 / "5_3_attribution_heatmap.png",
    )
    register_artifact(
        manifest,
        "5.3",
        paths.sec_5_3 / "5_3_attribution_heatmap.png",
        "Node-by-time gradient attribution heatmap for a representative alert window.",
    )

    plot_grid_localization(
        G,
        synthetic["positions"],
        combined_node_attr,
        "Section 5.3 - Grid-level localization map",
        paths.sec_5_3 / "5_3_grid_localization_map.png",
    )
    register_artifact(
        manifest,
        "5.3",
        paths.sec_5_3 / "5_3_grid_localization_map.png",
        "Full smart-grid localization map showing node responsibility for the representative alert.",
    )

    region_scores = {}
    for region_name in sorted(set(synthetic["regions"].values())):
        nodes_in_region = [n for n, r in synthetic["regions"].items() if r == region_name]
        region_scores[region_name] = float(np.mean(combined_node_attr[nodes_in_region]))
    plot_region_influence(
        region_scores,
        "Section 5.3 - Regional influence aggregation",
        paths.sec_5_3 / "5_3_regional_influence.png",
    )
    register_artifact(
        manifest,
        "5.3",
        paths.sec_5_3 / "5_3_regional_influence.png",
        "Regional aggregation of localization evidence.",
    )

    simplicial_info = build_simplicial_complex_for_region(G, top_nodes)
    plot_simplicial_region(
        simplicial_info,
        synthetic["positions"],
        combined_node_attr,
        "Section 5.3 - TopoNetX simplicial lifting of the anomalous region",
        paths.sec_5_3 / "5_3_simplicial_region.png",
    )
    register_artifact(
        manifest,
        "5.3",
        paths.sec_5_3 / "5_3_simplicial_region.png",
        "TopoNetX simplicial complex built from the most influential anomalous region.",
    )

    plot_hodge_spectrum(
        simplicial_info["L1_eigenvalues"],
        "Section 5.3 - Local 1-Hodge Laplacian spectrum",
        paths.sec_5_3 / "5_3_hodge_spectrum.png",
    )
    register_artifact(
        manifest,
        "5.3",
        paths.sec_5_3 / "5_3_hodge_spectrum.png",
        "Spectrum of the local 1-Hodge Laplacian derived from the anomalous simplicial region.",
    )

    feature_order = np.argsort(attribution["feature_attr"])[::-1]
    top_features = [
        {"feature": FEATURE_NAMES[int(i)], "importance": float(attribution["feature_attr"][i])}
        for i in feature_order[:5]
    ]
    top_nodes_payload = [
        {"node": int(n), "importance": float(combined_node_attr[n]), "region": synthetic["regions"][n], "role": synthetic["node_roles"][n]}
        for n in top_nodes
    ]

    representative_row = result_df.loc[representative_idx]
    incident_report = {
        "representative_window_index": int(representative_idx),
        "split": str(representative_row["split"]),
        "family_group": str(representative_row["family_group"]),
        "subtype": str(representative_row["subtype"]),
        "episode_id": int(representative_row["episode_id"]),
        "regime_name": str(representative_row["regime_name"]),
        "regime_id": int(representative_row["regime_id"]),
        "scores": {
            "statistical_score": float(representative_row["statistical_score"]),
            "topological_score": float(representative_row["topological_score"]),
            "composite_score": float(representative_row["composite_score"]),
            "transition_score": float(representative_row["transition_score"]),
            "fragmentation": float(representative_row["fragmentation"]),
            "loop_persistence": float(representative_row["loop_persistence"]),
        },
        "top_nodes": top_nodes_payload,
        "top_features": top_features,
        "topological_narrative": topological_narrative(representative_row),
        "simplicial_complex": {
            "nodes": [int(n) for n in simplicial_info["subgraph"].nodes()],
            "edges": [[int(u), int(v)] for u, v in simplicial_info["subgraph"].edges()],
            "triangles": [[int(n) for n in tri] for tri in simplicial_info["triangles"]],
            "B1_shape": list(simplicial_info["B1_shape"]),
            "B2_shape": list(simplicial_info["B2_shape"]),
            "L1_eigenvalues": [float(v) for v in simplicial_info["L1_eigenvalues"]],
        },
    }
    save_json(paths.sec_5_3 / "5_3_incident_report.json", incident_report)
    register_artifact(
        manifest,
        "5.3",
        paths.sec_5_3 / "5_3_incident_report.json",
        "Machine-readable operator report for the representative alert.",
    )

    # ---------------------------------------------------------------------
    # 10) Final manifest and auxiliary tables
    # ---------------------------------------------------------------------
    manifest_df = pd.DataFrame(manifest)
    save_table_bundle(
        manifest_df,
        paths.root / "section_5_artifact_manifest.csv",
        paths.root / "section_5_artifact_manifest.tex",
        caption="Artifact manifest aligned to manuscript sections 5.1--5.3.",
        label="tab:artifact_manifest",
        precision=config.simulation.publication_table_precision,
    )

    # Save a compact topological/reference summary for auditability.
    ref_rows = []
    for rid, ref in topology["references"].items():
        ref_rows.append(
            {
                "regime_id": int(rid),
                "reference_fragmentation": float(ref["fragmentation"]),
                "reference_loop_persistence": float(ref["loop_persistence"]),
                "reference_summary_dim": int(len(ref["summary_vector"])),
                "reference_h0_points": int(len(finite_diagram(ref["diagram_h0"]))),
                "reference_h1_points": int(len(finite_diagram(ref["diagram_h1"]))),
            }
        )
    ref_df = pd.DataFrame(ref_rows)
    save_table_bundle(
        ref_df,
        paths.global_dir / "topological_references.csv",
        paths.global_dir / "topological_references.tex",
        caption="Topological regime references used by the dual-evidence detector.",
        label="tab:topological_references",
        precision=config.simulation.publication_table_precision,
    )

    save_table_bundle(
        topology["acceleration_df"],
        paths.global_dir / "topology_acceleration_audit.csv",
        paths.global_dir / "topology_acceleration_audit.tex",
        caption="Audit table for sparse/incremental topological computation across monitoring windows.",
        label="tab:topology_acceleration_audit",
        precision=config.simulation.publication_table_precision,
    )

    incident_table = pd.DataFrame(incident_report["top_nodes"])
    save_table_bundle(
        incident_table,
        paths.sec_5_3 / "5_3_top_nodes_table.csv",
        paths.sec_5_3 / "5_3_top_nodes_table.tex",
        caption="Section 5.3 top localized nodes for the representative alert.",
        label="tab:section_5_3_top_nodes",
        precision=config.simulation.publication_table_precision,
    )
    feature_table = pd.DataFrame(incident_report["top_features"])
    save_table_bundle(
        feature_table,
        paths.sec_5_3 / "5_3_top_features_table.csv",
        paths.sec_5_3 / "5_3_top_features_table.tex",
        caption="Section 5.3 top explanatory features for the representative alert.",
        label="tab:section_5_3_top_features",
        precision=config.simulation.publication_table_precision,
    )

    # Return selected outputs to the caller.
    return {
        "paths": paths,
        "result_df": result_df,
        "evaluation_summary_df": evaluation_summary_df,
        "delay_df": delay_df,
        "manifest_df": manifest_df,
    }


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser(
        description="Deep topological intelligence pipeline for smart-grid anomaly detection."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results_smart_grid_topology",
        help="Root folder for all generated artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs for the deep graph-temporal model.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=1320,
        help="Number of synthetic time steps to generate.",
    )
    parser.add_argument(
        "--topological_horizon",
        type=int,
        default=18,
        help="Number of recent latent windows used to build each PH cloud.",
    )
    parser.add_argument(
        "--exact_topology",
        action="store_true",
        help="Disable sparse/incremental topological acceleration and use dense exact Ripser for every cloud.",
    )
    parser.add_argument(
        "--max_mapper_points",
        type=int,
        default=320,
        help="Maximum number of test windows sampled for Mapper visualization.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    config = ExperimentConfig()
    config.output_root = args.output_root
    config.simulation.seed = args.seed
    config.simulation.total_steps = args.total_steps
    config.simulation.topological_horizon = args.topological_horizon
    config.simulation.fast_topology_mode = not args.exact_topology
    config.simulation.max_mapper_points = args.max_mapper_points
    config.model.epochs = args.epochs

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    outputs = run_pipeline(config)

    print("=" * 79)
    print("Deep topological smart-grid pipeline finished successfully.")
    print(f"Results root: {outputs['paths'].root}")
    print(f"Manifest:      {outputs['paths'].root / 'section_5_artifact_manifest.csv'}")
    print(f"Global table:  {outputs['paths'].global_dir / 'per_window_results.csv'}")
    print("=" * 79)


if __name__ == "__main__":
    main()
