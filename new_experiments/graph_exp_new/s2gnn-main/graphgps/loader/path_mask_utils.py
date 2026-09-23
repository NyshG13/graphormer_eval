"""Path-aware preprocessing utilities for Experiments 1–4.

Extends the existing ``dist_mask_utils`` pipeline with four additional
per-graph arrays computed once and pickled alongside the distance-mask cache:

dist_int  : (N, N) int16  — raw shortest-path distance (hop count, -1 = unreachable)
sigma     : (N, N) int32  — number of shortest paths between each pair
cnbr      : (N, N) int16  — common-neighbour count |N(i) ∩ N(j)|
path_mid  : (N, N) int16  — one BFS-intermediate node on a shortest path i→j
             (value = -1 when d(i,j) ≤ 1 or pair unreachable)

All arrays use small dtypes to keep cache files compact.
"""

from __future__ import annotations

import os
import pickle
from functools import partial
from typing import Dict

import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-graph computation
# ---------------------------------------------------------------------------

def _compute_path_features_single(adj: np.ndarray, max_hops: int = 40) -> Dict[str, np.ndarray]:
    """Compute all four path feature arrays for one graph.

    Parameters
    ----------
    adj : (N, N) float32 adjacency matrix (undirected, 0/1 weights)
    max_hops : int

    Returns
    -------
    dict with keys: dist_int, sigma, cnbr, path_mid
    """
    N = adj.shape[0]
    if N == 0:
        return {
            "dist_int": np.empty((0, 0), dtype=np.int16),
            "sigma":    np.empty((0, 0), dtype=np.int32),
            "cnbr":     np.empty((0, 0), dtype=np.int16),
            "path_mid": np.empty((0, 0), dtype=np.int16),
        }

    # ------------------------------------------------------------------
    # 1.  Shortest-path distances via Floyd-Warshall
    # ------------------------------------------------------------------
    dist_float = floyd_warshall(adj, directed=False, unweighted=True)
    dist_int = np.where(np.isfinite(dist_float) & (dist_float < max_hops),
                        dist_float, -1).astype(np.int16)

    # ------------------------------------------------------------------
    # 2.  Common-neighbour count  |N(i) ∩ N(j)| = (A @ A)[i,j]
    # ------------------------------------------------------------------
    cnbr = (adj @ adj).astype(np.int16)
    cnbr = np.clip(cnbr, 0, np.iinfo(np.int16).max).astype(np.int16)

    # ------------------------------------------------------------------
    # 3.  Number of shortest paths σ(i,j) &
    # 4.  One BFS-intermediate node path_mid for d(i,j) >= 2
    # ------------------------------------------------------------------
    sigma = np.zeros((N, N), dtype=np.int32)
    path_mid = np.full((N, N), -1, dtype=np.int16)

    # Pre-build neighbor lists for rapid BFS (avoiding inner-loop allocations)
    adj_list = [np.flatnonzero(adj[i]) for i in range(N)]

    for s in range(N):
        visited_dist = np.full(N, -1, dtype=np.int32)
        visited_dist[s] = 0
        sigma[s, s] = 1

        first_hop = np.full(N, -1, dtype=np.int16)
        queue = [s]
        head = 0

        while head < len(queue):
            u = queue[head]
            head += 1
            d_next = visited_dist[u] + 1

            for v in adj_list[u]:
                if visited_dist[v] == -1:
                    visited_dist[v] = d_next
                    queue.append(v)
                    if u == s:
                        first_hop[v] = v
                    else:
                        first_hop[v] = first_hop[u]

                if visited_dist[v] == d_next:
                    sigma[s, v] += sigma[s, u]

        for v in range(N):
            if visited_dist[v] >= 2:
                path_mid[s, v] = first_hop[v]

    return {
        "dist_int": dist_int,
        "sigma":    sigma,
        "cnbr":     cnbr,
        "path_mid": path_mid,
    }


# ---------------------------------------------------------------------------
# Dataset-level precomputation with pickle caching
# ---------------------------------------------------------------------------

def precompute_path_features(dataset, cache_dir: str, max_hops: int = 40,
                             num_workers: int = 8):
    """Compute and cache path feature arrays for every graph in *dataset*.

    The cache is keyed by ``max_hops``. If the cache already exists it is
    loaded directly without recomputation.

    Returns
    -------
    list of dict
        One dict per graph with keys: dist_int, sigma, cnbr, path_mid.
    """
    cache_path = os.path.join(cache_dir, f"path_features_max_hops_{max_hops}.pkl")
    if os.path.exists(cache_path):
        print(f"  Loading cached path features from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    _data   = getattr(dataset, '_data', None) or dataset.data
    _slices = dataset.slices
    x_slices  = _slices['x']
    all_ei    = _data.edge_index
    num_graphs = len(x_slices) - 1
    ei_slices  = _slices['edge_index']

    print(f"  Computing path features for {num_graphs} graphs "
          f"(max_hops={max_hops})...", flush=True)

    adjs = []
    for i in range(num_graphs):
        node_start = int(x_slices[i])
        node_end   = int(x_slices[i + 1])
        n = node_end - node_start

        ei_start = int(ei_slices[i])
        ei_end   = int(ei_slices[i + 1])
        ei = all_ei[:, ei_start:ei_end].numpy()

        if ei.size > 0 and np.min(ei) >= node_start and node_start > 0:
            ei = ei - node_start

        adj = np.zeros((n, n), dtype=np.float32)
        if ei.size > 0:
            adj[ei[0], ei[1]] = 1.0
            adj[ei[1], ei[0]] = 1.0   # undirected
        adjs.append(adj)

    features = []
    for adj in tqdm(adjs, desc="  [HopMasked] Path features", ncols=80):
        features.append(_compute_path_features_single(adj, max_hops=max_hops))

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(features, f)
    print(f"  Saved path features to {cache_path}", flush=True)
    return features


# ---------------------------------------------------------------------------
# Wrapper dataset that pairs each graph with BOTH dist_mask and path_features
# ---------------------------------------------------------------------------

class PathFeatureGraphDataset:
    """Wraps a PyG dataset to pair each graph with dist_mask + path features.

    ``__getitem__`` returns ``(Data, dist_mask, path_feat_dict)``.  Use with
    :func:`collate_path_features` as the DataLoader's collate_fn.
    """

    def __init__(self, pyg_dataset, dist_masks, path_features):
        assert len(pyg_dataset) == len(dist_masks) == len(path_features)
        self.pyg_dataset   = pyg_dataset
        self.dist_masks    = dist_masks
        self.path_features = path_features

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        return self.pyg_dataset[idx], self.dist_masks[idx], self.path_features[idx]


def collate_path_features(batch_list, max_hops=40):
    """Collate ``(Data, dist_mask, path_feat_dict)`` tuples into a PyG Batch.

    Attributes added to the batch
    -----------------------------
    dist_mask     : list of (K_i, N_i, N_i) bool arrays  (existing)
    path_features : list of dicts with keys dist_int / sigma / cnbr / path_mid
    """
    import torch_geometric.data

    graphs, dist_masks_list, path_feats_list = zip(*batch_list)
    pyg_batch = torch_geometric.data.Batch.from_data_list(list(graphs))
    pyg_batch.dist_mask     = list(dist_masks_list)
    pyg_batch.path_features = list(path_feats_list)
    return pyg_batch


# ---------------------------------------------------------------------------
# Dense padding helper  (called inside each network's forward pass)
# ---------------------------------------------------------------------------

def pad_path_features(batch, N_max: int, max_hops: int, device) -> dict:
    """Pad per-graph path feature arrays into dense (B, N_max, N_max) tensors.

    Returns a dict with tensors:
      dist_int : (B, N_max, N_max) int64   — hop distance (0 = self, -1 = missing)
      sigma    : (B, N_max, N_max) float32 — log1p(number of shortest paths)
      cnbr     : (B, N_max, N_max) float32 — log1p(common-neighbour count)
      path_mid : (B, N_max, N_max) int64   — intermediate node index (-1 = none)
    """
    import torch
    import numpy as np

    pf_list = batch.path_features
    B = batch.num_graphs
    ptr = batch.ptr  # (B+1,)

    dist_int = torch.full((B, N_max, N_max), -1,  dtype=torch.long,    device=device)
    sigma    = torch.zeros((B, N_max, N_max),      dtype=torch.float32, device=device)
    cnbr     = torch.zeros((B, N_max, N_max),      dtype=torch.float32, device=device)
    path_mid = torch.full((B, N_max, N_max), -1,  dtype=torch.long,    device=device)

    for i, pf in enumerate(pf_list):
        n = int(ptr[i + 1]) - int(ptr[i])

        di = pf["dist_int"]
        if isinstance(di, np.ndarray):
            di = torch.from_numpy(di.astype(np.int64))
        dist_int[i, :n, :n] = di.to(device)

        sg = pf["sigma"]
        if isinstance(sg, np.ndarray):
            sg = torch.from_numpy(sg.astype(np.float32))
        sigma[i, :n, :n] = torch.log1p(sg.float().to(device))

        cn = pf["cnbr"]
        if isinstance(cn, np.ndarray):
            cn = torch.from_numpy(cn.astype(np.float32))
        cnbr[i, :n, :n] = torch.log1p(cn.float().to(device))

        pm = pf["path_mid"]
        if isinstance(pm, np.ndarray):
            pm = torch.from_numpy(pm.astype(np.int64))
        # Shift local node indices to batch-global indices
        offset = int(ptr[i])
        valid  = pm >= 0
        pm_global = pm.clone()
        pm_global[valid] += offset
        path_mid[i, :n, :n] = pm_global.to(device)

    return {
        "dist_int": dist_int,
        "sigma":    sigma,
        "cnbr":     cnbr,
        "path_mid": path_mid,
    }
