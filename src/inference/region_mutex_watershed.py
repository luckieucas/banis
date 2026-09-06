"""Mutex Watershed on a sparse SDT-fragment region graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RegionMWSResult:
    label_lut: np.ndarray
    n_fragments: int
    n_clusters: int
    attractive_edges: int
    mutex_edges: int


def region_mutex_watershed_lut(
    fragment_ids: Sequence[int],
    edge_pairs: np.ndarray,
    edge_affinities: np.ndarray,
    beta: float,
) -> RegionMWSResult:
    """Cluster fragments with signed mean-affinity evidence.

    Affinities above ``beta`` become attractive edges and affinities below
    ``beta`` become mutex edges. Their priority is the absolute distance from
    ``beta``. Fragment ids are mapped densely for clustering and restored via a
    lookup table, so sparse watershed ids do not create unused graph nodes.
    """
    try:
        from affogato.segmentation import compute_mws_clustering
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise ImportError(
            "region_mutex_watershed_lut requires affogato; use the isolated affogato environment"
        ) from exc

    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError(f"beta must lie in [0, 1], got {beta}")
    ids = np.asarray(fragment_ids, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("fragment_ids must be a one-dimensional sequence of unique positive ids")
    ids = np.sort(ids)
    pairs = np.asarray(edge_pairs, dtype=np.int64)
    affinities = np.asarray(edge_affinities, dtype=np.float32)
    if pairs.shape != (len(affinities), 2):
        raise ValueError(f"Expected edge_pairs shape ({len(affinities)}, 2), got {pairs.shape}")
    if np.any(pairs <= 0) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("Every edge must join two distinct positive fragment ids")
    if not np.all(np.isfinite(affinities)) or np.any((affinities < 0) | (affinities > 1)):
        raise ValueError("edge_affinities must be finite probabilities in [0, 1]")

    max_id = int(ids[-1]) if len(ids) else 0
    lut = np.zeros(max_id + 1, dtype=np.uint64)
    if not len(ids):
        return RegionMWSResult(lut, 0, 0, 0, 0)

    id_to_node = {int(fragment): node for node, fragment in enumerate(ids)}
    kept_pairs = []
    kept_affinities = []
    seen = set()
    for pair, affinity in zip(pairs, affinities):
        left, right = sorted((int(pair[0]), int(pair[1])))
        if left not in id_to_node or right not in id_to_node:
            continue
        key = (left, right)
        if key in seen:
            raise ValueError(f"Duplicate fragment edge {key}")
        seen.add(key)
        kept_pairs.append((id_to_node[left], id_to_node[right]))
        kept_affinities.append(float(affinity))

    if not kept_pairs:
        lut[ids] = np.arange(1, len(ids) + 1, dtype=np.uint64)
        return RegionMWSResult(lut, len(ids), len(ids), 0, 0)

    dense_pairs = np.asarray(kept_pairs, dtype=np.uint64)
    values = np.asarray(kept_affinities, dtype=np.float32)
    signed = values - np.float32(beta)
    attractive = signed >= 0
    # Stable lexicographic input order makes equal-weight behavior reproducible.
    order = np.lexsort((dense_pairs[:, 1], dense_pairs[:, 0]))
    dense_pairs = dense_pairs[order]
    signed = signed[order]
    attractive = attractive[order]
    attractive_pairs = np.require(dense_pairs[attractive], dtype=np.uint64, requirements="C")
    mutex_pairs = np.require(dense_pairs[~attractive], dtype=np.uint64, requirements="C")
    attractive_weights = np.require(signed[attractive], dtype=np.float32, requirements="C")
    mutex_weights = np.require(-signed[~attractive], dtype=np.float32, requirements="C")

    cluster_labels = compute_mws_clustering(
        len(ids),
        attractive_pairs,
        mutex_pairs,
        attractive_weights,
        mutex_weights,
    )
    _, compact = np.unique(np.asarray(cluster_labels), return_inverse=True)
    lut[ids] = compact.astype(np.uint64) + 1
    n_clusters = int(compact.max(initial=-1) + 1)
    return RegionMWSResult(
        lut,
        len(ids),
        n_clusters,
        int(attractive.sum()),
        int((~attractive).sum()),
    )
