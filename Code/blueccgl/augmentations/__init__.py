"""Feed-aware and random graph augmentation strategies for contrastive learning.

Shared utilities used by all augmentation strategies:
    ``subgraph``          — extract a node-subset subgraph (vectorised).
    ``root_component``    — keep only the connected component containing the root.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch_geometric.data import Data


def subgraph(data: Data, keep_mask: torch.Tensor) -> Data:
    """Extract the subgraph induced by *keep_mask*.

    Args:
        data: PyG ``Data`` object (single graph, not batched).
        keep_mask: [N] bool tensor.  ``keep_mask[0]`` **must** be ``True``
                   (root is always retained).

    Returns:
        New ``Data`` with reindexed nodes/edges.  Graph-level attributes
        are copied unchanged.
    """
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    N_new = keep_idx.shape[0]

    # Old-index → new-index remapping
    remap = torch.full((data.x.shape[0],), -1, dtype=torch.long)
    remap[keep_idx] = torch.arange(N_new)

    # Filter edges whose *both* endpoints survive
    src, tgt = data.edge_index
    edge_mask = keep_mask[src] & keep_mask[tgt]

    return Data(
        x=data.x[keep_idx],
        edge_index=torch.stack([remap[src[edge_mask]], remap[tgt[edge_mask]]]),
        edge_type=data.edge_type[edge_mask],
        # Graph-level attrs (unchanged)
        sentiment_composition=data.sentiment_composition.clone(),
        follower_edge_ratio=data.follower_edge_ratio.clone(),
        event_window=data.event_window.clone(),
        cascade_depth=data.cascade_depth.clone(),
        cascade_breadth=data.cascade_breadth.clone(),
        wiener_index=data.wiener_index.clone(),
        cascade_size=data.cascade_size.clone(),
        cascade_id=data.cascade_id.clone(),
    )


def root_component(data: Data) -> Data:
    """Keep only the connected component that contains node 0 (root).

    Uses undirected BFS so that it works even if the graph has been
    made disconnected by edge or node removal.
    """
    N = data.x.shape[0]
    if N == 0:
        return data

    # Build undirected adjacency list
    adj: defaultdict[int, list[int]] = defaultdict(list)
    src, tgt = data.edge_index
    for s, t in zip(src.tolist(), tgt.tolist()):
        adj[s].append(t)
        adj[t].append(s)

    visited: set[int] = set()
    queue = [0]
    visited.add(0)
    while queue:
        node = queue.pop(0)
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    if len(visited) == N:
        return data  # already fully connected

    keep_mask = torch.zeros(N, dtype=torch.bool)
    for v in visited:
        keep_mask[v] = True
    return subgraph(data, keep_mask)
