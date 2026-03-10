# -*- coding: utf-8 -*-
"""
TOPOLOGICAL EFFICIENCY IN DIGITAL TWINS OF AUTONOMOUS VEHICLES
Results Generator for Section 5 (5.1, 5.2, 5.3) + Required Enhancements:

Enhancement (i): DISJOINT SPLITS
  - The real dataset is split into three disjoint partitions:
      * CALIBRATION: used ONLY to calibrate tau (topology tolerance) and compute per-condition references.
      * DEVELOPMENT: used to generate synthetic candidates and (optionally) tune a tau multiplier.
      * HOLD-OUT: used ONLY for final reporting (Section 5 tables/figures), including robustness evaluation.

Enhancement (ii): ONLINE AUDITING (SIMULATED STREAM)
  - A simulated incremental stream of candidate updates is audited by the topology gate.
  - A per-condition reference diagram is maintained and updated (medoid of recent accepted diagrams).
  - Produces:
      * 5_3_efficiency/table_5_3_online_audit_log.csv
      * 5_3_efficiency/fig_5_3_online_audit_latency_divergence.png

Critical fix (this version):
  - Robust stratified splitting:
      * Try stratify by condition|hazard
      * If infeasible (e.g., a class has only 1 member), fallback to condition-only
      * If still infeasible, fallback to non-stratified random split
  - Guarantees disjoint calibration/development/hold-out splits with valid sizes.

Core pipeline remains unchanged:
  * VR persistent homology (H0, H1) up to triangles via Z2 reduction.
  * Bottleneck + 1-Wasserstein distances with diagonal matching.
  * Weighted divergence and acceptance gate.
  * Downstream AI evaluation (3 regimes) + robustness perturbations.
  * Efficiency/scalability analysis and Section 5 outputs.

Data input (optional):
  data_root/
    real/
      <condition_name>/
        *.npy  (Nx2 point cloud) OR *.csv (columns: x,y)

If no real data are found, the script generates structured "real" scenarios and synthetic variants.

Outputs:
  results_topological_efficiency_run_<timestamp>/
    5_1_topological_fidelity/  (tables+figures)
    5_2_ai_tasks/              (tables+figures)
    5_3_efficiency/            (tables+figures + online auditing log/figure)
    run_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Reproducibility + I/O helpers
# -----------------------------

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_point_cloud(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected Nx2 array in {path}, got shape {arr.shape}")
        return arr.astype(float)
    if ext == ".csv":
        df = pd.read_csv(path)
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError(f"CSV must contain columns x,y: {path}")
        return df[["x", "y"]].values.astype(float)
    raise ValueError(f"Unsupported file type: {path}")


# -----------------------------------------
# Synthetic AV-like point cloud generation
# -----------------------------------------

@dataclass
class ScenarioMeta:
    scenario_id: str
    condition: str
    is_real: bool
    is_hazard: bool
    generator: str
    noise_sigma: float
    dropout_rate: float
    occlusion_strength: float


def _make_roundabout(center=(0.0, 0.0), radius=1.0, n=180, jitter=0.0) -> np.ndarray:
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([center[0] + radius*np.cos(angles), center[1] + radius*np.sin(angles)], axis=1)
    if jitter > 0:
        pts = pts + np.random.normal(0, jitter, size=pts.shape)
    return pts


def _make_lane_lines(length=4.0, width=1.5) -> np.ndarray:
    x = np.linspace(-length/2, length/2, 50)
    y1 = np.full_like(x, -width/2)
    y2 = np.full_like(x, +width/2)
    l1 = np.stack([x, y1], axis=1)
    l2 = np.stack([x, y2], axis=1)
    return np.vstack([l1, l2])


def _make_intersection(size=3.5) -> np.ndarray:
    x = np.linspace(-size/2, size/2, 60)
    y = np.linspace(-size/2, size/2, 60)
    horiz = np.stack([x, np.zeros_like(x)], axis=1)
    vert = np.stack([np.zeros_like(y), y], axis=1)
    return np.vstack([horiz, vert])


def _make_obstacle_cluster(center=(0.5, 0.5), spread=0.12, n=80) -> np.ndarray:
    return np.random.normal(loc=np.array(center), scale=spread, size=(n, 2))


def generate_real_scenario(condition: str, base_id: int) -> Tuple[np.ndarray, ScenarioMeta]:
    """
    Produces a structured "real" point cloud with condition-dependent noise/occlusion
    and a stochastic hazard (obstacle cluster) label.
    """
    if condition == "clear_day":
        base = np.vstack([_make_lane_lines(), _make_roundabout(radius=0.9, n=140)])
        jitter = 0.015
        dropout_rate = 0.05
        hazard_prob = 0.20
    elif condition == "night":
        base = np.vstack([_make_lane_lines(), _make_intersection(size=3.2)])
        jitter = 0.03
        dropout_rate = 0.12
        hazard_prob = 0.28
    elif condition == "rain":
        base = np.vstack([_make_lane_lines(), _make_roundabout(radius=0.8, n=120)])
        jitter = 0.04
        dropout_rate = 0.18
        hazard_prob = 0.35
    elif condition == "fog":
        base = np.vstack([_make_intersection(size=3.6), _make_roundabout(radius=0.85, n=110)])
        jitter = 0.06
        dropout_rate = 0.25
        hazard_prob = 0.40
    else:
        base = np.vstack([_make_lane_lines(), _make_roundabout(radius=0.85, n=120)])
        jitter = 0.03
        dropout_rate = 0.10
        hazard_prob = 0.25

    pts = base + np.random.normal(0, jitter, size=base.shape)

    keep = np.random.rand(len(pts)) > dropout_rate
    pts = pts[keep]

    is_hazard = (np.random.rand() < hazard_prob)
    if is_hazard:
        obs_center = (np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4))
        obs = _make_obstacle_cluster(center=obs_center, spread=0.10 + 0.03*np.random.rand(), n=90)
        pts = np.vstack([pts, obs])

    # standardize
    pts = pts.astype(float)
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(pts, axis=1), 90) + 1e-9
    pts = pts / scale

    meta = ScenarioMeta(
        scenario_id=f"real_{condition}_{base_id:04d}",
        condition=condition,
        is_real=True,
        is_hazard=is_hazard,
        generator="real_baseline_generator",
        noise_sigma=jitter,
        dropout_rate=dropout_rate,
        occlusion_strength=dropout_rate,
    )
    return pts, meta


def generate_synthetic_candidate(
    real_points: np.ndarray,
    condition: str,
    base_id: int,
    true_hazard: bool,
    label_flip_prob: float = 0.18
) -> Tuple[np.ndarray, ScenarioMeta]:
    """
    Produces a synthetic candidate by applying transformations that emulate generative variability.
    Some candidates intentionally drift structurally to test the topology gate.
    Synthetic labels are derived from the true hazard label with controlled flip probability.
    """
    pts = real_points.copy()

    theta = np.random.uniform(-0.25, 0.25)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]], dtype=float)
    pts = pts @ R.T
    pts = pts + np.random.normal(0, 0.02, size=pts.shape)

    drift_mode = np.random.choice(
        ["benign", "benign", "drift_break_loop", "drift_merge_clusters"],
        p=[0.45, 0.25, 0.20, 0.10]
    )

    if drift_mode == "drift_break_loop":
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        wedge_center = np.random.uniform(-np.pi, np.pi)
        wedge_width = np.random.uniform(0.6, 1.1)
        keep = np.abs(((angles - wedge_center + np.pi) % (2*np.pi)) - np.pi) > (wedge_width/2)
        pts = pts[keep]
        dropout_rate = 0.25
    elif drift_mode == "drift_merge_clusters":
        bridge = _make_obstacle_cluster(center=(0.0, 0.0), spread=0.20, n=80)
        pts = np.vstack([pts, bridge])
        dropout_rate = 0.08
    else:
        dropout_rate = 0.10

    keep = np.random.rand(len(pts)) > dropout_rate
    pts = pts[keep]

    if np.random.rand() < label_flip_prob:
        syn_hazard = (not true_hazard)
    else:
        syn_hazard = true_hazard

    meta = ScenarioMeta(
        scenario_id=f"syn_{condition}_{base_id:04d}_{drift_mode}",
        condition=condition,
        is_real=False,
        is_hazard=bool(syn_hazard),
        generator=f"synthetic_transform_{drift_mode}",
        noise_sigma=0.02,
        dropout_rate=dropout_rate,
        occlusion_strength=dropout_rate,
    )
    return pts.astype(float), meta


# -----------------------------------------
# Persistent homology (VR up to triangles)
# -----------------------------------------

@dataclass
class PersistenceDiagram:
    dim: int
    points: np.ndarray  # shape (k,2) birth, death
    essential: int


def _build_vr_simplices(points: np.ndarray, max_dim: int = 2) -> Tuple[List[Tuple[int, ...]], List[int], List[float]]:
    n = points.shape[0]
    if n < 2:
        simplices = [(0,)]
        return simplices, [0], [0.0]

    D = squareform(pdist(points))
    simplices: List[Tuple[int, ...]] = []
    dims: List[int] = []
    filts: List[float] = []

    for i in range(n):
        simplices.append((i,))
        dims.append(0)
        filts.append(0.0)

    if max_dim >= 1:
        for i in range(n):
            for j in range(i + 1, n):
                simplices.append((i, j))
                dims.append(1)
                filts.append(float(D[i, j]))

    if max_dim >= 2:
        for i, j, k in combinations(range(n), 3):
            f = max(D[i, j], D[i, k], D[j, k])
            simplices.append((i, j, k))
            dims.append(2)
            filts.append(float(f))

    order = sorted(range(len(simplices)), key=lambda idx: (filts[idx], dims[idx], simplices[idx]))
    simplices = [simplices[i] for i in order]
    dims = [dims[i] for i in order]
    filts = [filts[i] for i in order]
    return simplices, dims, filts


def _boundary_of_simplex(simplex: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    if len(simplex) <= 1:
        return []
    faces = []
    for r in range(len(simplex)):
        face = simplex[:r] + simplex[r+1:]
        faces.append(face)
    return faces


def compute_persistence_diagrams(points: np.ndarray, max_dim: int = 2) -> Dict[int, PersistenceDiagram]:
    simplices, dims, filts = _build_vr_simplices(points, max_dim=max_dim)
    index_of: Dict[Tuple[int, ...], int] = {s: i for i, s in enumerate(simplices)}

    boundary_cols: List[set] = []
    for s, d in zip(simplices, dims):
        if d == 0:
            boundary_cols.append(set())
        else:
            faces = _boundary_of_simplex(s)
            col = set(index_of[f] for f in faces)
            boundary_cols.append(col)

    low: Dict[int, int] = {}
    reduced_cols: Dict[int, set] = {}
    births: List[int] = []
    pairs: Dict[int, int] = {}

    for j in range(len(simplices)):
        col = set(boundary_cols[j])
        while len(col) > 0:
            pivot = max(col)
            if pivot in low:
                col ^= reduced_cols[low[pivot]]
            else:
                break

        if len(col) == 0:
            births.append(j)
        else:
            pivot = max(col)
            low[pivot] = j
            reduced_cols[j] = col
            pairs[pivot] = j

    diag_points: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
    essential_count: Dict[int, int] = {0: 0, 1: 0}

    for b in births:
        d_birth = dims[b]
        if b in pairs:
            d = pairs[b]
            birth_time = filts[b]
            death_time = filts[d]
            if d_birth in (0, 1):
                diag_points[d_birth].append((birth_time, death_time))
        else:
            if d_birth in (0, 1):
                essential_count[d_birth] += 1

    out: Dict[int, PersistenceDiagram] = {}
    for dim in (0, 1):
        pts = np.array(diag_points[dim], dtype=float) if len(diag_points[dim]) > 0 else np.zeros((0, 2), dtype=float)
        if pts.shape[0] > 0:
            pts = pts[pts[:, 1] > pts[:, 0] + 1e-12]
        out[dim] = PersistenceDiagram(dim=dim, points=pts, essential=essential_count[dim])
    return out


# -----------------------------
# Diagram distances
# -----------------------------

def linf_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(max(abs(p[0] - q[0]), abs(p[1] - q[1])))


def dist_to_diagonal(p: np.ndarray) -> float:
    return float((p[1] - p[0]) / 2.0)


def wasserstein_1_distance(D1: np.ndarray, D2: np.ndarray) -> float:
    n = D1.shape[0]
    m = D2.shape[0]
    N = n + m
    if N == 0:
        return 0.0

    C = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i < n and j < m:
                C[i, j] = linf_dist(D1[i], D2[j])
            elif i < n and j >= m:
                C[i, j] = dist_to_diagonal(D1[i])
            elif i >= n and j < m:
                C[i, j] = dist_to_diagonal(D2[j])
            else:
                C[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(C)
    return float(C[row_ind, col_ind].sum())


def _hopcroft_karp_bipartite(adj: List[List[int]], n_left: int, n_right: int) -> int:
    from collections import deque
    INF = 10**9

    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs() -> bool:
        q = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu != -1 and dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
                if pu == -1:
                    found = True
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching


def bottleneck_distance(D1: np.ndarray, D2: np.ndarray) -> float:
    n = D1.shape[0]
    m = D2.shape[0]
    if n == 0 and m == 0:
        return 0.0

    costs = []
    for i in range(n):
        costs.append(dist_to_diagonal(D1[i]))
        for j in range(m):
            costs.append(linf_dist(D1[i], D2[j]))
    for j in range(m):
        costs.append(dist_to_diagonal(D2[j]))

    costs = sorted(set(float(c) for c in costs))
    if len(costs) == 0:
        return 0.0

    def feasible(eps: float) -> bool:
        n_left = n + m
        n_right = m + n
        adj: List[List[int]] = [[] for _ in range(n_left)]

        d1_diag_ok = [dist_to_diagonal(D1[i]) <= eps for i in range(n)]
        d2_diag_ok = [dist_to_diagonal(D2[j]) <= eps for j in range(m)]

        for i in range(n):
            for j in range(m):
                if linf_dist(D1[i], D2[j]) <= eps:
                    adj[i].append(j)
            if d1_diag_ok[i]:
                adj[i].extend(list(range(m, m + n)))

        for li in range(n, n + m):
            for j in range(m):
                if d2_diag_ok[j]:
                    adj[li].append(j)
            adj[li].extend(list(range(m, m + n)))

        match_size = _hopcroft_karp_bipartite(adj, n_left=n_left, n_right=n_right)
        return match_size == n_left

    lo, hi = 0, len(costs) - 1
    best = costs[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        eps = costs[mid]
        if feasible(eps):
            best = eps
            hi = mid - 1
        else:
            lo = mid + 1
    return float(best)


def persistent_entropy(D: np.ndarray) -> float:
    if D.shape[0] == 0:
        return 0.0
    L = D[:, 1] - D[:, 0]
    L = L[L > 1e-12]
    if L.size == 0:
        return 0.0
    p = L / (L.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def betti_curve_from_diagram(D: np.ndarray, eps_grid: np.ndarray, max_death: float) -> np.ndarray:
    if D.shape[0] == 0:
        return np.zeros_like(eps_grid, dtype=float)
    births = D[:, 0]
    deaths = np.minimum(D[:, 1], max_death)
    bc = np.zeros_like(eps_grid, dtype=float)
    for i, e in enumerate(eps_grid):
        bc[i] = float(np.sum((births <= e) & (e < deaths)))
    return bc


# -----------------------------
# Topology gate and metrics
# -----------------------------

@dataclass
class TopologyMetrics:
    bottleneck_H0: float
    bottleneck_H1: float
    wasserstein1_H0: float
    wasserstein1_H1: float
    divergence: float
    consistency_score: float
    accepted: bool


def compute_topology_metrics(
    diag_ref: Dict[int, PersistenceDiagram],
    diag_cand: Dict[int, PersistenceDiagram],
    weights: Tuple[float, float],
    tau: float
) -> TopologyMetrics:
    D0r = diag_ref[0].points
    D1r = diag_ref[1].points
    D0s = diag_cand[0].points
    D1s = diag_cand[1].points

    b0 = bottleneck_distance(D0r, D0s)
    b1 = bottleneck_distance(D1r, D1s)
    w0 = wasserstein_1_distance(D0r, D0s)
    w1 = wasserstein_1_distance(D1r, D1s)

    divergence = weights[0] * w0 + weights[1] * w1
    consistency = max(0.0, 1.0 - float(divergence) / float(tau + 1e-12))
    accepted = divergence <= tau

    return TopologyMetrics(
        bottleneck_H0=float(b0),
        bottleneck_H1=float(b1),
        wasserstein1_H0=float(w0),
        wasserstein1_H1=float(w1),
        divergence=float(divergence),
        consistency_score=float(consistency),
        accepted=bool(accepted),
    )


def diagram_divergence(diag_a: Dict[int, PersistenceDiagram], diag_b: Dict[int, PersistenceDiagram], weights: Tuple[float, float]) -> float:
    w0 = wasserstein_1_distance(diag_a[0].points, diag_b[0].points)
    w1 = wasserstein_1_distance(diag_a[1].points, diag_b[1].points)
    return float(weights[0] * w0 + weights[1] * w1)


def calibrate_tau_within_real(
    real_diagrams: List[Dict[int, PersistenceDiagram]],
    real_metas: List[ScenarioMeta],
    conditions: List[str],
    weights: Tuple[float, float],
    quantile: float = 0.95,
    max_pairs_per_condition: int = 25
) -> float:
    idx_by_cond: Dict[str, List[int]] = {c: [] for c in conditions}
    for i, meta in enumerate(real_metas):
        idx_by_cond[meta.condition].append(i)

    divergences = []
    for c in conditions:
        idxs = idx_by_cond[c]
        if len(idxs) < 3:
            continue
        num_pairs = min(max_pairs_per_condition, len(idxs) * (len(idxs) - 1) // 2)
        sampled = 0
        tried = set()
        while sampled < num_pairs:
            a, b = np.random.choice(idxs, size=2, replace=False)
            if (a, b) in tried or (b, a) in tried:
                continue
            tried.add((a, b))
            divergences.append(diagram_divergence(real_diagrams[a], real_diagrams[b], weights))
            sampled += 1

    if len(divergences) == 0:
        return 1.0

    tau = float(np.quantile(np.array(divergences, dtype=float), quantile))
    tau = float(1.10 * tau)
    return max(tau, 1e-6)


def compute_medoid_reference(
    diagrams: List[Dict[int, PersistenceDiagram]],
    weights: Tuple[float, float]
) -> Dict[int, PersistenceDiagram]:
    if len(diagrams) == 1:
        return diagrams[0]
    k = len(diagrams)
    dmat = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            d = diagram_divergence(diagrams[i], diagrams[j], weights)
            dmat[i, j] = d
            dmat[j, i] = d
    sums = dmat.sum(axis=1)
    med = int(np.argmin(sums))
    return diagrams[med]


# -----------------------------
# Feature extraction for AI task
# -----------------------------

def extract_features_from_diagrams(diag: Dict[int, PersistenceDiagram], eps_grid: np.ndarray) -> np.ndarray:
    D0 = diag[0].points
    D1 = diag[1].points

    max_death = 1.0
    if D0.size > 0:
        max_death = max(max_death, float(np.max(D0[:, 1])))
    if D1.size > 0:
        max_death = max(max_death, float(np.max(D1[:, 1])))

    H0_ent = persistent_entropy(D0)
    H1_ent = persistent_entropy(D1)

    bc0 = betti_curve_from_diagram(D0, eps_grid=eps_grid, max_death=max_death)
    bc1 = betti_curve_from_diagram(D1, eps_grid=eps_grid, max_death=max_death)

    bc0_mean = float(np.mean(bc0))
    bc1_mean = float(np.mean(bc1))
    bc0_max = float(np.max(bc0)) if bc0.size else 0.0
    bc1_max = float(np.max(bc1)) if bc1.size else 0.0
    bc0_auc = float(np.trapz(bc0, eps_grid))
    bc1_auc = float(np.trapz(bc1, eps_grid))

    n0 = float(D0.shape[0])
    n1 = float(D1.shape[0])
    pers0 = float(np.mean(D0[:, 1] - D0[:, 0])) if D0.shape[0] > 0 else 0.0
    pers1 = float(np.mean(D1[:, 1] - D1[:, 0])) if D1.shape[0] > 0 else 0.0

    return np.array([
        H0_ent, H1_ent,
        bc0_mean, bc1_mean,
        bc0_max, bc1_max,
        bc0_auc, bc1_auc,
        n0, n1,
        pers0, pers1
    ], dtype=float)


def compute_features_for_dataset(diagrams: List[Dict[int, PersistenceDiagram]], eps_grid: np.ndarray) -> np.ndarray:
    return np.vstack([extract_features_from_diagrams(d, eps_grid) for d in diagrams])


# -----------------------------
# Plotting utilities
# -----------------------------

def plot_divergence_boxplot(df: pd.DataFrame, out_path: str, tau: float) -> None:
    plt.figure(figsize=(10, 5))
    conditions = sorted(df["condition"].unique().tolist())
    data = [df.loc[df["condition"] == c, "divergence"].values for c in conditions]
    plt.boxplot(data, labels=conditions, showfliers=False)
    plt.axhline(tau, linestyle="--")
    plt.title("Topological divergence (weighted Wasserstein-1) by driving condition")
    plt.ylabel("Divergence (lower is better)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_acceptance_rate(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(9, 4.5))
    agg = df.groupby("condition")["accepted"].mean().reset_index()
    plt.bar(agg["condition"], agg["accepted"].values)
    plt.ylim(0, 1.0)
    plt.title("Topology-gate acceptance rate by driving condition")
    plt.ylabel("Acceptance rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_persistence_diagram(D: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(5.5, 5.0))
    if D.shape[0] > 0:
        plt.scatter(D[:, 0], D[:, 1], s=18)
        mx = float(np.max(D)) if np.isfinite(np.max(D)) else 1.0
        mx = max(mx, 1.0)
    else:
        mx = 1.0
    plt.plot([0, mx], [0, mx], linestyle="--")
    plt.title(title)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_f1_comparison(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(9, 4.5))
    pivot = df.pivot(index="regime", columns="test_type", values="f1")
    order = ["real_only", "synthetic_unfiltered", "synthetic_topology_gated"]
    pivot = pivot.loc[[r for r in order if r in pivot.index]]

    x = np.arange(len(pivot.index))
    width = 0.35
    cols = pivot.columns.tolist()

    if len(cols) == 1:
        plt.bar(x, pivot[cols[0]].values, width=0.6)
        plt.xticks(x, pivot.index, rotation=15)
    else:
        for k, col in enumerate(cols):
            plt.bar(x + (k - (len(cols)-1)/2)*width, pivot[col].values, width=width, label=col)
        plt.xticks(x, pivot.index, rotation=15)
        plt.legend()

    plt.title("Downstream F1 comparison across training regimes")
    plt.ylabel("F1 (higher is better)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hazard_fn_rate(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(9, 4.5))
    pivot = df[df["test_type"] == "clean"].set_index("regime")["hazard_false_negative_rate"]
    order = ["real_only", "synthetic_unfiltered", "synthetic_topology_gated"]
    pivot = pivot.loc[[r for r in order if r in pivot.index]]
    plt.bar(pivot.index, pivot.values)
    plt.ylim(0, 1.0)
    plt.title("Safety-relevant error pattern: Hazard false-negative rate (clean test)")
    plt.ylabel("False-negative rate (lower is better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_overhead_vs_size(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(9, 4.5))
    for regime in ["synthetic_unfiltered", "synthetic_topology_gated"]:
        sub = df[df["regime"] == regime].sort_values("n_total_train")
        if len(sub) == 0:
            continue
        plt.plot(sub["n_total_train"].values, sub["overhead_fraction"].values, marker="o", label=regime)
    plt.title("Topology overhead vs training-set size")
    plt.xlabel("Training set size (scenarios)")
    plt.ylabel("Overhead fraction = t_topology / (t_topology + t_train)")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cost_performance_tradeoff(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(9, 4.5))
    for regime in ["real_only", "synthetic_unfiltered", "synthetic_topology_gated"]:
        sub = df[df["regime"] == regime].sort_values("n_total_train")
        if len(sub) == 0:
            continue
        plt.scatter(sub["total_time_sec"].values, sub["f1_clean"].values, label=regime)
    plt.title("Cost–performance tradeoff (clean test)")
    plt.xlabel("Total pipeline time (sec)")
    plt.ylabel("F1 (higher is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_online_audit_latency_divergence(df_log: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df_log))
    plt.plot(x, df_log["latency_ms"].values, marker="o", linestyle="-", label="latency_ms")
    plt.plot(x, df_log["divergence"].values, marker="x", linestyle="--", label="divergence")
    plt.title("Online auditing: latency and divergence over incremental stream")
    plt.xlabel("Stream step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Downstream model helpers
# -----------------------------

def train_and_evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_estimators: int,
    max_depth: Optional[int]
) -> Dict[str, float]:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        max_depth=max_depth,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    if len(np.unique(y_train)) > 1:
        proba = clf.predict_proba(X_test)[:, 1]
    else:
        proba = np.zeros_like(y_test, dtype=float)
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    hazard_fnr = fn / (fn + tp + 1e-12)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(auc),
        "hazard_false_negative_rate": float(hazard_fnr),
    }


# -----------------------------
# Experiment configuration
# -----------------------------

@dataclass
class Config:
    seed: int = 7
    data_root: str = "data_root"
    output_root: Optional[str] = None

    # scenario generation (used only if no data on disk)
    conditions: Tuple[str, ...] = ("clear_day", "night", "rain", "fog")
    n_real_per_condition: int = 18

    # DISJOINT SPLITS
    frac_calibration: float = 0.20
    frac_development: float = 0.40
    frac_holdout: float = 0.40

    # synthetic generation (on DEVELOPMENT only)
    synthetic_per_real_dev: int = 1

    # TDA computation
    max_tda_points: int = 28
    weights_H0_H1: Tuple[float, float] = (0.45, 0.55)
    tau_quantile: float = 0.95

    # tau tuning on development
    tau_multiplier_candidates: Tuple[float, ...] = (0.90, 1.00, 1.10, 1.25)
    target_acceptance_rate_dev: float = 0.70

    # feature extraction
    eps_grid_k: int = 20

    # downstream model
    rf_n_estimators: int = 250
    rf_max_depth: Optional[int] = None

    # robustness test perturbation (on HOLD-OUT only)
    robustness_noise_sigma: float = 0.05
    robustness_dropout_rate: float = 0.15

    # scalability (Section 5.3)
    scaling_train_sizes: Tuple[int, ...] = (30, 60, 120)

    # online auditing (simulated)
    online_audit_stream_len: int = 40
    online_audit_buffer_size: int = 12


# -----------------------------
# Utility: subsampling and perturbation
# -----------------------------

def subsample_points(points: np.ndarray, k: int) -> np.ndarray:
    if points.shape[0] <= k:
        return points
    idx = np.random.choice(points.shape[0], size=k, replace=False)
    return points[idx]


def perturb_points(points: np.ndarray, noise_sigma: float, dropout_rate: float) -> np.ndarray:
    pts = points.copy()
    pts = pts + np.random.normal(0, noise_sigma, size=pts.shape)
    keep = np.random.rand(len(pts)) > dropout_rate
    pts = pts[keep]
    if pts.shape[0] < 5:
        pts = points[:min(len(points), 10)].copy()
    return pts


def try_load_real_dataset(data_root: str, conditions: List[str]) -> Optional[Tuple[List[np.ndarray], List[ScenarioMeta]]]:
    base = os.path.join(data_root, "real")
    if not os.path.isdir(base):
        return None

    all_points: List[np.ndarray] = []
    all_meta: List[ScenarioMeta] = []

    for c in conditions:
        cdir = os.path.join(base, c)
        if not os.path.isdir(cdir):
            continue
        files = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith((".npy", ".csv"))]
        files = sorted(files)
        for i, fp in enumerate(files):
            pts = load_point_cloud(fp)
            pts = pts.astype(float)
            pts = pts - pts.mean(axis=0, keepdims=True)
            scale = np.percentile(np.linalg.norm(pts, axis=1), 90) + 1e-9
            pts = pts / scale

            # hazard unknown if loaded; set False by default (user may extend with labels)
            meta = ScenarioMeta(
                scenario_id=f"real_{c}_{i:04d}",
                condition=c,
                is_real=True,
                is_hazard=False,
                generator="loaded_from_disk",
                noise_sigma=0.0,
                dropout_rate=0.0,
                occlusion_strength=0.0
            )
            all_points.append(pts)
            all_meta.append(meta)

    if len(all_points) == 0:
        return None
    return all_points, all_meta


def build_run_folders(cfg: Config) -> Tuple[str, str, str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = cfg.output_root or f"results_topological_efficiency_run_{ts}"
    ensure_dir(root)
    d51 = os.path.join(root, "5_1_topological_fidelity")
    d52 = os.path.join(root, "5_2_ai_tasks")
    d53 = os.path.join(root, "5_3_efficiency")
    ensure_dir(d51)
    ensure_dir(d52)
    ensure_dir(d53)
    return root, d51, d52, d53


# -----------------------------
# ROBUST DISJOINT SPLITS (cal/dev/holdout)
# -----------------------------

def _counts_by_label(labels: np.ndarray) -> Dict[Any, int]:
    vals, cnts = np.unique(labels, return_counts=True)
    return {v: int(c) for v, c in zip(vals.tolist(), cnts.tolist())}


def _feasible_stratification(labels: np.ndarray, n_test: int) -> bool:
    """
    Stratified split is feasible only if:
      - every class has at least 2 examples (so it can be split)
      - n_test is between [n_classes, n_total - n_classes]
    Additionally, each class must be able to allocate at least 1 to both sides.
    """
    if labels is None:
        return False
    labels = np.asarray(labels)
    n_total = labels.shape[0]
    counts = _counts_by_label(labels)
    n_classes = len(counts)
    if n_classes < 2:
        return False
    if min(counts.values()) < 2:
        return False
    if n_test < n_classes:
        return False
    if (n_total - n_test) < n_classes:
        return False
    return True


def _stratified_split_indices(
    idx: np.ndarray,
    labels: np.ndarray,
    n_test: int,
    seed: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Custom stratified split that avoids sklearn's failure on rare classes by:
      - enforcing at least 1 per class in both partitions
      - adjusting allocations to meet the exact n_test
    Returns None if impossible.
    """
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    labels = np.asarray(labels)

    n_total = len(idx)
    if n_test <= 0 or n_test >= n_total:
        return None

    if not _feasible_stratification(labels, n_test):
        return None

    # Group indices by class
    classes = np.unique(labels)
    by_class: Dict[Any, np.ndarray] = {}
    for c in classes:
        by_class[c] = idx[labels == c]

    # Initial allocation proportional to class size, at least 1, at most size-1
    frac = n_test / float(n_total)
    alloc: Dict[Any, int] = {}
    for c in classes:
        nc = len(by_class[c])
        a = int(round(nc * frac))
        a = max(1, a)
        a = min(a, nc - 1)
        alloc[c] = a

    # Adjust allocation to match n_test exactly
    total_alloc = sum(alloc.values())

    # Helper: candidates where we can decrement/increment while keeping >=1 on both sides
    def dec_candidates() -> List[Any]:
        out = []
        for c in classes:
            if alloc[c] > 1:
                out.append(c)
        return out

    def inc_candidates() -> List[Any]:
        out = []
        for c in classes:
            nc = len(by_class[c])
            if alloc[c] < (nc - 1):
                out.append(c)
        return out

    # If too many assigned to test, decrement
    while total_alloc > n_test:
        cand = dec_candidates()
        if len(cand) == 0:
            return None
        # decrement from the class with largest alloc share
        cand = sorted(cand, key=lambda c: alloc[c] / max(1, len(by_class[c])), reverse=True)
        c = cand[0]
        alloc[c] -= 1
        total_alloc -= 1

    # If too few assigned to test, increment
    while total_alloc < n_test:
        cand = inc_candidates()
        if len(cand) == 0:
            return None
        # increment in the class with largest remaining capacity
        cand = sorted(cand, key=lambda c: (len(by_class[c]) - 1 - alloc[c]), reverse=True)
        c = cand[0]
        alloc[c] += 1
        total_alloc += 1

    # Sample actual indices per class
    test_idx_parts = []
    train_idx_parts = []
    for c in classes:
        ids = by_class[c].copy()
        rng.shuffle(ids)
        a = alloc[c]
        test_idx_parts.append(ids[:a])
        train_idx_parts.append(ids[a:])

    test_idx = np.concatenate(test_idx_parts) if len(test_idx_parts) else np.array([], dtype=int)
    train_idx = np.concatenate(train_idx_parts) if len(train_idx_parts) else np.array([], dtype=int)

    # Final sanity checks
    if len(test_idx) != n_test:
        return None
    if len(np.unique(test_idx)) != len(test_idx):
        return None
    if len(np.intersect1d(train_idx, test_idx)) != 0:
        return None

    return train_idx, test_idx


def _random_split_indices(idx: np.ndarray, n_test: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx).copy()
    rng.shuffle(idx)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def _make_stratify_labels(metas: List[ScenarioMeta], mode: str) -> Optional[np.ndarray]:
    """
    mode:
      - "cond_hazard": condition|hazard combined label
      - "cond": condition only
    """
    if mode == "cond_hazard":
        return np.array([f"{m.condition}|{int(m.is_hazard)}" for m in metas], dtype=object)
    if mode == "cond":
        return np.array([m.condition for m in metas], dtype=object)
    return None


def _best_split(
    idx: np.ndarray,
    metas: List[ScenarioMeta],
    n_test: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Attempts stratified split in a robust way:
      1) condition|hazard
      2) condition
      3) random
    Returns (train_idx, test_idx, strategy_used).
    """
    if len(idx) < 3:
        # too small to be meaningful; force random
        tr, te = _random_split_indices(idx, n_test=max(1, min(n_test, len(idx)-1)), seed=seed)
        return tr, te, "random_forced_small_n"

    for mode in ["cond_hazard", "cond"]:
        labels_full = _make_stratify_labels(metas, mode=mode)
        if labels_full is None:
            continue
        labels = labels_full[np.searchsorted(np.arange(len(metas)), idx)]
        # The above relies on idx being from np.arange(len(metas)); in our use it is.
        # If idx is not aligned, we fallback to simpler mapping:
        if labels.shape[0] != idx.shape[0]:
            labels = np.array([labels_full[i] for i in idx], dtype=object)

        attempt = _stratified_split_indices(idx=idx, labels=labels, n_test=n_test, seed=seed)
        if attempt is not None:
            tr, te = attempt
            return tr, te, f"stratified_{mode}"

    tr, te = _random_split_indices(idx, n_test=n_test, seed=seed)
    return tr, te, "random"


def split_cal_dev_holdout(
    metas: List[ScenarioMeta],
    seed: int,
    frac_cal: float,
    frac_dev: float,
    frac_hold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Produces disjoint index arrays for calibration/development/holdout.
    Robustly attempts stratification; falls back safely when class counts are too small.

    Returns:
      idx_cal, idx_dev, idx_hold, split_info
    """
    n = len(metas)
    if n < 3:
        raise ValueError("Need at least 3 real samples to create calibration/development/holdout splits.")

    if not math.isclose(frac_cal + frac_dev + frac_hold, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("fractions must sum to 1.0")

    # Compute sizes with safeguards: at least 1 in each partition
    n_hold = int(round(n * frac_hold))
    n_cal = int(round(n * frac_cal))
    n_hold = max(1, min(n_hold, n - 2))
    n_cal = max(1, min(n_cal, n - n_hold - 1))
    n_dev = n - n_hold - n_cal
    if n_dev < 1:
        # rebalance if needed
        n_dev = 1
        if n_cal > 1:
            n_cal -= 1
        else:
            n_hold -= 1

    all_idx = np.arange(n)

    # Stage 1: HOLDOUT split
    idx_rest, idx_hold, strat1 = _best_split(
        idx=all_idx,
        metas=metas,
        n_test=n_hold,
        seed=seed
    )

    # Stage 2: CAL vs DEV split on remaining
    # We will treat "DEV" as the "test" portion of the remaining split for convenience.
    idx_rest = np.asarray(idx_rest)
    n_rest = len(idx_rest)
    if n_rest != (n - n_hold):
        raise RuntimeError("Internal split size mismatch.")

    # Ensure n_dev feasible
    n_dev = max(1, min(n_dev, n_rest - 1))
    idx_cal, idx_dev, strat2 = _best_split(
        idx=idx_rest,
        metas=metas,
        n_test=n_dev,
        seed=seed + 1
    )

    # Sanity: disjointness
    if len(np.intersect1d(idx_cal, idx_dev)) != 0 or len(np.intersect1d(idx_cal, idx_hold)) != 0 or len(np.intersect1d(idx_dev, idx_hold)) != 0:
        raise RuntimeError("Splits are not disjoint; this should never happen.")

    split_info = {
        "stage1_strategy": strat1,
        "stage2_strategy": strat2,
        "n_total": str(n),
        "n_cal": str(len(idx_cal)),
        "n_dev": str(len(idx_dev)),
        "n_hold": str(len(idx_hold)),
    }
    return np.array(idx_cal), np.array(idx_dev), np.array(idx_hold), split_info


# -----------------------------
# Tau tuning on development (no extra outputs)
# -----------------------------

def choose_tau_multiplier_on_development(
    tau_base: float,
    df_dev_syn: pd.DataFrame,
    candidates: Tuple[float, ...],
    target_accept: float
) -> Tuple[float, float]:
    best = None
    best_mult = 1.0
    best_tau = tau_base

    divs = df_dev_syn["divergence"].values.astype(float)
    if divs.size == 0:
        return 1.0, tau_base

    for m in candidates:
        tau = tau_base * float(m)
        acc = float(np.mean(divs <= tau))
        if np.sum(divs <= tau) > 0:
            mean_acc_div = float(np.mean(divs[divs <= tau]))
        else:
            mean_acc_div = float(np.mean(divs))

        permissive_penalty = 0.20 * float(m)
        score = abs(acc - target_accept) + 0.30 * (mean_acc_div / (tau + 1e-12)) + permissive_penalty

        if best is None or score < best:
            best = score
            best_mult = float(m)
            best_tau = float(tau)

    return best_mult, best_tau


# -----------------------------
# Online auditing simulation
# -----------------------------

def simulate_online_auditing(
    stream_diagrams: List[Dict[int, PersistenceDiagram]],
    stream_metas: List[ScenarioMeta],
    initial_ref_by_condition: Dict[str, Dict[int, PersistenceDiagram]],
    tau: float,
    weights: Tuple[float, float],
    buffer_size: int
) -> pd.DataFrame:
    ref_by_cond = {c: initial_ref_by_condition[c] for c in initial_ref_by_condition.keys()}
    buffers: Dict[str, List[Dict[int, PersistenceDiagram]]] = {c: [] for c in initial_ref_by_condition.keys()}

    rows = []
    for t, (d, m) in enumerate(zip(stream_diagrams, stream_metas)):
        cond = m.condition
        if cond not in ref_by_cond:
            ref_by_cond[cond] = d
            buffers[cond] = []

        t0 = time.perf_counter()
        tm = compute_topology_metrics(ref_by_cond[cond], d, weights=weights, tau=tau)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ref_updated = 0
        if tm.accepted:
            buffers[cond].append(d)
            if len(buffers[cond]) > buffer_size:
                buffers[cond] = buffers[cond][-buffer_size:]
            if len(buffers[cond]) >= 3:
                ref_by_cond[cond] = compute_medoid_reference(buffers[cond], weights=weights)
                ref_updated = 1

        rows.append({
            "step": int(t),
            "condition": cond,
            "candidate_id": m.scenario_id,
            "accepted": int(tm.accepted),
            "divergence": float(tm.divergence),
            "consistency_score": float(tm.consistency_score),
            "latency_ms": float(latency_ms),
            "reference_updated": int(ref_updated),
            "buffer_size_condition": int(len(buffers[cond])),
        })

    return pd.DataFrame(rows)


# -----------------------------
# Main experiment runner
# -----------------------------

def run_pipeline(cfg: Config) -> None:
    set_global_seed(cfg.seed)
    root, d51, d52, d53 = build_run_folders(cfg)
    conditions = list(cfg.conditions)
    weights = cfg.weights_H0_H1
    eps_grid = np.linspace(0.0, 1.0, cfg.eps_grid_k)

    # 1) Acquire or simulate real dataset
    t0_data = time.perf_counter()
    loaded = try_load_real_dataset(cfg.data_root, conditions)

    real_points: List[np.ndarray] = []
    real_metas: List[ScenarioMeta] = []

    if loaded is not None:
        real_points, real_metas = loaded
    else:
        sid = 0
        for c in conditions:
            for _ in range(cfg.n_real_per_condition):
                pts, meta = generate_real_scenario(c, base_id=sid)
                real_points.append(pts)
                real_metas.append(meta)
                sid += 1

    t_data = time.perf_counter() - t0_data

    # 2) Compute persistence diagrams for ALL real
    t0_tda_real = time.perf_counter()
    real_diagrams_all: List[Dict[int, PersistenceDiagram]] = []
    for pts in real_points:
        pts_sub = subsample_points(pts, cfg.max_tda_points)
        real_diagrams_all.append(compute_persistence_diagrams(pts_sub, max_dim=2))
    t_tda_real_all = time.perf_counter() - t0_tda_real

    # 3) DISJOINT splits: calibration / development / holdout (ROBUST FIX HERE)
    idx_cal, idx_dev, idx_hold, split_info = split_cal_dev_holdout(
        metas=real_metas,
        seed=cfg.seed,
        frac_cal=cfg.frac_calibration,
        frac_dev=cfg.frac_development,
        frac_hold=cfg.frac_holdout
    )

    cal_points = [real_points[i] for i in idx_cal.tolist()]
    dev_points = [real_points[i] for i in idx_dev.tolist()]
    hold_points = [real_points[i] for i in idx_hold.tolist()]

    cal_metas = [real_metas[i] for i in idx_cal.tolist()]
    dev_metas = [real_metas[i] for i in idx_dev.tolist()]
    hold_metas = [real_metas[i] for i in idx_hold.tolist()]

    cal_diagrams = [real_diagrams_all[i] for i in idx_cal.tolist()]
    dev_diagrams = [real_diagrams_all[i] for i in idx_dev.tolist()]
    hold_diagrams = [real_diagrams_all[i] for i in idx_hold.tolist()]

    # 4) Calibrate tau using ONLY CALIBRATION real set
    t0_cal = time.perf_counter()
    tau_base = calibrate_tau_within_real(
        real_diagrams=cal_diagrams,
        real_metas=cal_metas,
        conditions=conditions,
        weights=weights,
        quantile=cfg.tau_quantile
    )
    t_cal = time.perf_counter() - t0_cal

    # 5) Build per-condition references from calibration
    ref_by_cond: Dict[str, Dict[int, PersistenceDiagram]] = {}
    for c in conditions:
        diags_c = [d for d, m in zip(cal_diagrams, cal_metas) if m.condition == c]
        if len(diags_c) == 0:
            continue
        ref_by_cond[c] = compute_medoid_reference(diags_c, weights=weights)
    if len(ref_by_cond) == 0:
        # fallback to any calibration diagram
        ref_by_cond = {conditions[0]: cal_diagrams[0]}

    # 6) Generate synthetic candidates ONLY from DEVELOPMENT
    t0_syn = time.perf_counter()
    syn_points_dev: List[np.ndarray] = []
    syn_metas_dev: List[ScenarioMeta] = []
    syn_diagrams_dev: List[Dict[int, PersistenceDiagram]] = []

    for i, (rpts, rmeta) in enumerate(zip(dev_points, dev_metas)):
        for k in range(cfg.synthetic_per_real_dev):
            spts, smeta = generate_synthetic_candidate(
                real_points=rpts,
                condition=rmeta.condition,
                base_id=i * 10 + k,
                true_hazard=rmeta.is_hazard
            )
            syn_points_dev.append(spts)
            syn_metas_dev.append(smeta)

    t0_tda_syn = time.perf_counter()
    for spts in syn_points_dev:
        s_sub = subsample_points(spts, cfg.max_tda_points)
        syn_diagrams_dev.append(compute_persistence_diagrams(s_sub, max_dim=2))
    t_tda_syn_dev = time.perf_counter() - t0_tda_syn
    t_syn_total = time.perf_counter() - t0_syn

    # 7) Compute dev divergences vs calibration reference
    topo_rows_dev = []
    for sdiag, smeta in zip(syn_diagrams_dev, syn_metas_dev):
        ref = ref_by_cond.get(smeta.condition, next(iter(ref_by_cond.values())))
        tm_dev = compute_topology_metrics(ref, sdiag, weights=weights, tau=tau_base)
        topo_rows_dev.append({
            "synthetic_id": smeta.scenario_id,
            "condition": smeta.condition,
            "generator": smeta.generator,
            "divergence": tm_dev.divergence
        })
    df_dev_syn = pd.DataFrame(topo_rows_dev)

    # 8) Tune tau multiplier on DEVELOPMENT
    tau_mult, tau = choose_tau_multiplier_on_development(
        tau_base=tau_base,
        df_dev_syn=df_dev_syn,
        candidates=cfg.tau_multiplier_candidates,
        target_accept=cfg.target_acceptance_rate_dev
    )

    # 9) SECTION 5.1 outputs
    topo_rows_51 = []
    for sdiag, smeta in zip(syn_diagrams_dev, syn_metas_dev):
        ref = ref_by_cond.get(smeta.condition, next(iter(ref_by_cond.values())))
        tm = compute_topology_metrics(ref, sdiag, weights=weights, tau=tau)
        topo_rows_51.append({
            "synthetic_id": smeta.scenario_id,
            "condition": smeta.condition,
            "generator": smeta.generator,
            "bottleneck_H0": tm.bottleneck_H0,
            "bottleneck_H1": tm.bottleneck_H1,
            "wasserstein1_H0": tm.wasserstein1_H0,
            "wasserstein1_H1": tm.wasserstein1_H1,
            "divergence": tm.divergence,
            "consistency_score": tm.consistency_score,
            "accepted": int(tm.accepted),
        })
    df51 = pd.DataFrame(topo_rows_51)
    df51_path = os.path.join(d51, "table_5_1_topological_distances.csv")
    df51.to_csv(df51_path, index=False)

    fig51a = os.path.join(d51, "fig_5_1_divergence_boxplot.png")
    plot_divergence_boxplot(df51, fig51a, tau=tau)

    fig51b = os.path.join(d51, "fig_5_1_acceptance_rate.png")
    plot_acceptance_rate(df51, fig51b)

    ia = int(df51.index[df51["accepted"] == 1].tolist()[0]) if int(df51["accepted"].sum()) > 0 else 0
    fig51c = os.path.join(d51, "fig_5_1_example_persistence_diagrams_H1.png")
    plot_persistence_diagram(syn_diagrams_dev[ia][1].points, fig51c, title="Example H1 persistence diagram (synthetic; representative)")

    # 10) SECTION 5.2: downstream tasks (DEV train -> HOLD test)
    t0_feat = time.perf_counter()
    X_dev_real = compute_features_for_dataset(dev_diagrams, eps_grid=eps_grid)
    X_hold_real = compute_features_for_dataset(hold_diagrams, eps_grid=eps_grid)
    X_syn_dev = compute_features_for_dataset(syn_diagrams_dev, eps_grid=eps_grid) if len(syn_diagrams_dev) > 0 else np.zeros((0, X_dev_real.shape[1]), dtype=float)
    t_feat = time.perf_counter() - t0_feat

    y_dev_real = np.array([int(m.is_hazard) for m in dev_metas], dtype=int)
    y_hold_real = np.array([int(m.is_hazard) for m in hold_metas], dtype=int)
    y_syn_dev = np.array([int(m.is_hazard) for m in syn_metas_dev], dtype=int) if len(syn_metas_dev) > 0 else np.zeros((0,), dtype=int)

    # robustness set from HOLD-OUT
    t0_rob = time.perf_counter()
    pert_points = [perturb_points(p, cfg.robustness_noise_sigma, cfg.robustness_dropout_rate) for p in hold_points]
    pert_diags = []
    for p in pert_points:
        p_sub = subsample_points(p, cfg.max_tda_points)
        pert_diags.append(compute_persistence_diagrams(p_sub, max_dim=2))
    X_hold_pert = compute_features_for_dataset(pert_diags, eps_grid=eps_grid)
    t_rob = time.perf_counter() - t0_rob

    metrics_rows_52 = []

    # Regime A: real-only
    t0_trainA = time.perf_counter()
    metA_clean = train_and_evaluate_classifier(
        X_train=X_dev_real, y_train=y_dev_real,
        X_test=X_hold_real, y_test=y_hold_real,
        seed=cfg.seed,
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth
    )
    metA_pert = train_and_evaluate_classifier(
        X_train=X_dev_real, y_train=y_dev_real,
        X_test=X_hold_pert, y_test=y_hold_real,
        seed=cfg.seed,
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth
    )
    t_trainA = time.perf_counter() - t0_trainA
    metrics_rows_52.append({"regime": "real_only", "test_type": "clean", **metA_clean})
    metrics_rows_52.append({"regime": "real_only", "test_type": "perturbed", **metA_pert})

    # Regime B: synthetic unfiltered
    X_train_B = np.vstack([X_dev_real, X_syn_dev]) if X_syn_dev.shape[0] > 0 else X_dev_real.copy()
    y_train_B = np.concatenate([y_dev_real, y_syn_dev]) if y_syn_dev.size > 0 else y_dev_real.copy()

    t0_trainB = time.perf_counter()
    metB_clean = train_and_evaluate_classifier(
        X_train=X_train_B, y_train=y_train_B,
        X_test=X_hold_real, y_test=y_hold_real,
        seed=cfg.seed,
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth
    )
    metB_pert = train_and_evaluate_classifier(
        X_train=X_train_B, y_train=y_train_B,
        X_test=X_hold_pert, y_test=y_hold_real,
        seed=cfg.seed,
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth
    )
    t_trainB = time.perf_counter() - t0_trainB
    metrics_rows_52.append({"regime": "synthetic_unfiltered", "test_type": "clean", **metB_clean})
    metrics_rows_52.append({"regime": "synthetic_unfiltered", "test_type": "perturbed", **metB_pert})

    # Regime C: synthetic topology-gated
    accepted_mask = df51["accepted"].values.astype(int) == 1 if len(df51) else np.zeros((0,), dtype=bool)
    X_syn_g = X_syn_dev[accepted_mask] if X_syn_dev.shape[0] > 0 and accepted_mask.size > 0 else np.zeros((0, X_dev_real.shape[1]), dtype=float)
    y_syn_g = y_syn_dev[accepted_mask] if y_syn_dev.size > 0 and accepted_mask.size > 0 else np.zeros((0,), dtype=int)

    X_train_C = np.vstack([X_dev_real, X_syn_g]) if X_syn_g.shape[0] > 0 else X_dev_real.copy()
    y_train_C = np.concatenate([y_dev_real, y_syn_g]) if y_syn_g.size > 0 else y_dev_real.copy()

    t0_trainC = time.perf_counter()
    metC_clean = train_and_evaluate_classifier(
        X_train=X_train_C, y_train=y_train_C,
        X_test=X_hold_real, y_test=y_hold_real,
        seed=cfg.seed,
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth
    )
    metC_pert = train_and_evaluate_classifier(
        X_train=X_train_C, y_train=y_train_C,
        X_test=X_hold_pert, y_test=y_hold_real,
        seed=cfg.seed,
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth
    )
    t_trainC = time.perf_counter() - t0_trainC
    metrics_rows_52.append({"regime": "synthetic_topology_gated", "test_type": "clean", **metC_clean})
    metrics_rows_52.append({"regime": "synthetic_topology_gated", "test_type": "perturbed", **metC_pert})

    df52 = pd.DataFrame(metrics_rows_52)
    df52_path = os.path.join(d52, "table_5_2_downstream_metrics.csv")
    df52.to_csv(df52_path, index=False)

    fig52a = os.path.join(d52, "fig_5_2_f1_comparison.png")
    plot_f1_comparison(df52, fig52a)

    fig52b = os.path.join(d52, "fig_5_2_hazard_false_negative_rate.png")
    plot_hazard_fn_rate(df52, fig52b)

    # 11) SECTION 5.3: efficiency + scalability + online auditing
    def compute_efficiency_row(regime: str, n_train: int, f1_clean: float, total_time: float, t_topo: float, tau_val: float, mean_div: float) -> dict:
        overhead_frac = float(t_topo / (t_topo + (total_time - t_topo) + 1e-12))
        norm_div = float(mean_div / (tau_val + 1e-12))
        te = float(f1_clean / ((total_time + 1e-12) * (1.0 + norm_div)))
        return {
            "regime": regime,
            "n_total_train": int(n_train),
            "f1_clean": float(f1_clean),
            "total_time_sec": float(total_time),
            "t_topology_sec": float(t_topo),
            "overhead_fraction": float(overhead_frac),
            "mean_divergence": float(mean_div),
            "tau": float(tau_val),
            "topological_efficiency_index": float(te),
        }

    t_topo_base = t_tda_real_all + t_tda_syn_dev + t_feat

    f1A = float(df52[(df52["regime"] == "real_only") & (df52["test_type"] == "clean")]["f1"].iloc[0])
    f1B = float(df52[(df52["regime"] == "synthetic_unfiltered") & (df52["test_type"] == "clean")]["f1"].iloc[0])
    f1C = float(df52[(df52["regime"] == "synthetic_topology_gated") & (df52["test_type"] == "clean")]["f1"].iloc[0])

    mean_div_B = float(df51["divergence"].mean()) if len(df51) else 0.0
    mean_div_C = float(df51[df51["accepted"] == 1]["divergence"].mean()) if int(df51["accepted"].sum()) > 0 else mean_div_B

    total_time_A = t_data + t_tda_real_all + t_feat + t_trainA + t_rob + t_cal
    total_time_B = t_data + t_tda_real_all + t_tda_syn_dev + t_feat + t_trainB + t_rob + t_cal + t_syn_total
    total_time_C = t_data + t_tda_real_all + t_tda_syn_dev + t_feat + t_trainC + t_rob + t_cal + t_syn_total

    df53_rows = []
    df53_rows.append(compute_efficiency_row("real_only", n_train=len(X_dev_real), f1_clean=f1A, total_time=total_time_A, t_topo=(t_tda_real_all + t_feat), tau_val=tau, mean_div=0.0))
    df53_rows.append(compute_efficiency_row("synthetic_unfiltered", n_train=len(X_train_B), f1_clean=f1B, total_time=total_time_B, t_topo=t_topo_base, tau_val=tau, mean_div=mean_div_B))
    df53_rows.append(compute_efficiency_row("synthetic_topology_gated", n_train=len(X_train_C), f1_clean=f1C, total_time=total_time_C, t_topo=t_topo_base, tau_val=tau, mean_div=mean_div_C))

    for n_target in cfg.scaling_train_sizes:
        n_real_train = len(X_dev_real)
        n_need = max(0, n_target - n_real_train)
        n_need = min(n_need, X_syn_dev.shape[0])

        # unfiltered scaling
        X_syn_sub = X_syn_dev[:n_need]
        y_syn_sub = y_syn_dev[:n_need]
        X_train = np.vstack([X_dev_real, X_syn_sub]) if n_need > 0 else X_dev_real
        y_train = np.concatenate([y_dev_real, y_syn_sub]) if n_need > 0 else y_dev_real

        t0 = time.perf_counter()
        met = train_and_evaluate_classifier(
            X_train=X_train, y_train=y_train,
            X_test=X_hold_real, y_test=y_hold_real,
            seed=cfg.seed,
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth
        )
        t_train = time.perf_counter() - t0

        frac = (n_real_train + n_need) / max(1, (n_real_train + max(1, X_syn_dev.shape[0])))
        topo_time = (t_tda_real_all + frac * t_tda_syn_dev + t_feat)
        total_time = t_data + topo_time + t_train + t_cal

        df53_rows.append(compute_efficiency_row(
            "synthetic_unfiltered", n_train=len(X_train),
            f1_clean=float(met["f1"]),
            total_time=total_time,
            t_topo=topo_time,
            tau_val=tau,
            mean_div=mean_div_B
        ))

        # gated scaling
        X_syn_acc = X_syn_g
        y_syn_acc = y_syn_g
        n_need_g = max(0, n_target - n_real_train)
        n_need_g = min(n_need_g, X_syn_acc.shape[0])

        X_syn_sub_g = X_syn_acc[:n_need_g]
        y_syn_sub_g = y_syn_acc[:n_need_g]
        X_train_g = np.vstack([X_dev_real, X_syn_sub_g]) if n_need_g > 0 else X_dev_real
        y_train_g = np.concatenate([y_dev_real, y_syn_sub_g]) if n_need_g > 0 else y_dev_real

        t0 = time.perf_counter()
        metg = train_and_evaluate_classifier(
            X_train=X_train_g, y_train=y_train_g,
            X_test=X_hold_real, y_test=y_hold_real,
            seed=cfg.seed,
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth
        )
        t_train_g = time.perf_counter() - t0

        frac_g = (n_real_train + n_need_g) / max(1, (n_real_train + max(1, X_syn_dev.shape[0])))
        topo_time_g = (t_tda_real_all + frac_g * t_tda_syn_dev + t_feat)
        total_time_g = t_data + topo_time_g + t_train_g + t_cal

        df53_rows.append(compute_efficiency_row(
            "synthetic_topology_gated", n_train=len(X_train_g),
            f1_clean=float(metg["f1"]),
            total_time=total_time_g,
            t_topo=topo_time_g,
            tau_val=tau,
            mean_div=mean_div_C
        ))

    df53 = pd.DataFrame(df53_rows)
    df53_path = os.path.join(d53, "table_5_3_efficiency_scaling.csv")
    df53.to_csv(df53_path, index=False)

    fig53a = os.path.join(d53, "fig_5_3_overhead_vs_size.png")
    plot_overhead_vs_size(df53, fig53a)

    fig53b = os.path.join(d53, "fig_5_3_cost_performance_tradeoff.png")
    plot_cost_performance_tradeoff(df53, fig53b)

    # Online auditing stream
    stream_len = min(cfg.online_audit_stream_len, len(syn_diagrams_dev))
    if stream_len > 0:
        perm = np.random.permutation(len(syn_diagrams_dev))[:stream_len]
        stream_diags = [syn_diagrams_dev[i] for i in perm.tolist()]
        stream_metas = [syn_metas_dev[i] for i in perm.tolist()]

        df_audit = simulate_online_auditing(
            stream_diagrams=stream_diags,
            stream_metas=stream_metas,
            initial_ref_by_condition=ref_by_cond,
            tau=tau,
            weights=weights,
            buffer_size=cfg.online_audit_buffer_size
        )

        audit_csv = os.path.join(d53, "table_5_3_online_audit_log.csv")
        df_audit.to_csv(audit_csv, index=False)

        audit_fig = os.path.join(d53, "fig_5_3_online_audit_latency_divergence.png")
        plot_online_audit_latency_divergence(df_audit, audit_fig)
    else:
        audit_csv = None
        audit_fig = None

    # 12) Manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(cfg),
        "split_info": split_info,
        "splits": {
            "n_total_real": int(len(real_points)),
            "n_calibration": int(len(idx_cal)),
            "n_development": int(len(idx_dev)),
            "n_holdout": int(len(idx_hold)),
            "disjoint": True
        },
        "tau_calibration": {
            "tau_base": float(tau_base),
            "tau_multiplier_selected": float(tau_mult),
            "tau_final": float(tau)
        },
        "counts": {
            "n_dev_synthetic": int(len(syn_points_dev)),
            "n_dev_synthetic_accepted": int(int(df51["accepted"].sum()) if len(df51) else 0),
            "online_audit_stream_len": int(stream_len),
        },
        "timing_seconds": {
            "data_acquisition_or_generation": float(t_data),
            "tda_real_all": float(t_tda_real_all),
            "tda_synthetic_dev": float(t_tda_syn_dev),
            "tau_calibration_time": float(t_cal),
            "feature_extraction": float(t_feat),
            "robustness_precompute_holdout": float(t_rob),
            "train_time_real_only": float(t_trainA),
            "train_time_synthetic_unfiltered": float(t_trainB),
            "train_time_synthetic_topology_gated": float(t_trainC),
            "synthetic_generation_total_dev": float(t_syn_total),
        },
        "outputs": {
            "section_5_1_table": df51_path,
            "section_5_2_table": df52_path,
            "section_5_3_table": df53_path,
            "section_5_3_online_audit_log": audit_csv,
            "section_5_3_online_audit_figure": audit_fig,
        }
    }
    save_json(manifest, os.path.join(root, "run_manifest.json"))

    print("====================================================================")
    print("RESULTS GENERATED SUCCESSFULLY (robust disjoint splits fix applied)")
    print(f"Output root: {root}")
    print("Section 5.1:", d51)
    print("Section 5.2:", d52)
    print("Section 5.3:", d53)
    print(f"Tau: base={tau_base:.6f}, multiplier={tau_mult:.3f}, final={tau:.6f}")
    print(f"Splits: cal={len(idx_cal)}, dev={len(idx_dev)}, holdout={len(idx_hold)}")
    print("Split strategies:", split_info)
    print("====================================================================")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Generate Section 5 results (5.1–5.3) with robust disjoint splits + online auditing.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data_root", type=str, default="data_root")
    p.add_argument("--output_root", type=str, default=None)

    p.add_argument("--n_real_per_condition", type=int, default=18)

    p.add_argument("--frac_calibration", type=float, default=0.20)
    p.add_argument("--frac_development", type=float, default=0.40)
    p.add_argument("--frac_holdout", type=float, default=0.40)

    p.add_argument("--synthetic_per_real_dev", type=int, default=1)

    p.add_argument("--max_tda_points", type=int, default=28)
    p.add_argument("--tau_quantile", type=float, default=0.95)

    p.add_argument("--eps_grid_k", type=int, default=20)

    p.add_argument("--rf_n_estimators", type=int, default=250)
    p.add_argument("--rf_max_depth", type=int, default=None)

    p.add_argument("--robustness_noise_sigma", type=float, default=0.05)
    p.add_argument("--robustness_dropout_rate", type=float, default=0.15)

    p.add_argument("--online_audit_stream_len", type=int, default=40)
    p.add_argument("--online_audit_buffer_size", type=int, default=12)

    args = p.parse_args()

    return Config(
        seed=args.seed,
        data_root=args.data_root,
        output_root=args.output_root,
        n_real_per_condition=args.n_real_per_condition,
        frac_calibration=args.frac_calibration,
        frac_development=args.frac_development,
        frac_holdout=args.frac_holdout,
        synthetic_per_real_dev=args.synthetic_per_real_dev,
        max_tda_points=args.max_tda_points,
        tau_quantile=args.tau_quantile,
        eps_grid_k=args.eps_grid_k,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        robustness_noise_sigma=args.robustness_noise_sigma,
        robustness_dropout_rate=args.robustness_dropout_rate,
        online_audit_stream_len=args.online_audit_stream_len,
        online_audit_buffer_size=args.online_audit_buffer_size
    )


def main() -> None:
    cfg = parse_args()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()