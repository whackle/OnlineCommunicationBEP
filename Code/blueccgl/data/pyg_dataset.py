"""
Convert cascade edge lists to PyG Data objects.

Node features [N, 6]:
    0: temporal_delta  — log1p(post_id offset from root; proxy for temporal order)
    1: sent_label      — ordinal (0=neg, 1=neutral, 2=pos)
    2: sent_score      — float sentiment confidence
    3: depth           — hop distance from root in cascade tree
    4: in_degree       — within-cascade in-degree
    5: out_degree      — within-cascade out-degree

Graph-level attributes stored on each Data object:
    sentiment_composition [3]  — proportion neg/neutral/pos
    follower_edge_ratio        — fraction of FOLLOWER-typed edges
    event_window               — integer label (0=baseline, 1–6=E1–E6)
    cascade_depth              — max tree depth
    cascade_breadth            — max width at any depth level
    wiener_index               — structural virality (O(N) tree algorithm)
    cascade_size               — number of nodes
    cascade_id                 — original root post_id
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data

EVENT_TO_IDX: dict[str, int] = {
    "baseline": 0,
    "E1_Oct7_Attack": 1,
    "E2_AlAhli_Hospital": 2,
    "E3_Ceasefire": 3,
    "E4_ICJ_Ruling": 4,
    "E5_Bushnell": 5,
    "E6_Flour_Massacre": 6,
}


def edges_to_pyg(
    cascade_id: int,
    source_uids: np.ndarray,
    target_uids: np.ndarray,
    edge_types: np.ndarray,
    source_order: np.ndarray,
    target_order: np.ndarray,
    event_window: str,
    sentiment: dict[int, tuple[int, float]],
) -> Data:
    """Build a single PyG Data object from one cascade's edge list.

    Args:
        cascade_id: Root post ID.
        source_uids/target_uids: Parent/child user IDs per edge.
        edge_types: 0=FOLLOWER, 1=NON_FOLLOWER, 2=UNKNOWN.
        source_order/target_order: Temporal ordering values per edge (post_id
            used as proxy since repost timestamps in the dataset reflect the
            original post's creation time, not the repost time).
        event_window: Event string key (e.g. "E1", "baseline").
        sentiment: uid -> (sent_label, sent_score) lookup.
    """
    E = len(source_uids)

    # --- Collect unique nodes with their ordering values ---
    uid_order: dict[int, float] = {}
    for i in range(E):
        s, t = int(source_uids[i]), int(target_uids[i])
        if s not in uid_order:
            uid_order[s] = float(source_order[i])
        if t not in uid_order:
            uid_order[t] = float(target_order[i])

    # Root = source that never appears as target
    target_set = set(int(x) for x in target_uids)
    root_uid = next(
        (int(s) for s in source_uids if int(s) not in target_set),
        int(source_uids[0]),
    )
    root_order = uid_order[root_uid]

    # Contiguous reindexing: root = 0
    uid_to_idx: dict[int, int] = {root_uid: 0}
    for uid in uid_order:
        if uid not in uid_to_idx:
            uid_to_idx[uid] = len(uid_to_idx)
    N = len(uid_to_idx)

    # --- Edge tensors + adjacency ---
    edge_index = torch.empty(2, E, dtype=torch.long)
    edge_type_t = torch.empty(E, dtype=torch.long)
    children: defaultdict[int, list[int]] = defaultdict(list)

    for i in range(E):
        si = uid_to_idx[int(source_uids[i])]
        ti = uid_to_idx[int(target_uids[i])]
        edge_index[0, i] = si
        edge_index[1, i] = ti
        edge_type_t[i] = int(edge_types[i])
        children[si].append(ti)

    # --- Depth (BFS from root) ---
    depth_arr = [0] * N
    queue = [0]
    visited = {0}
    while queue:
        nxt: list[int] = []
        for nd in queue:
            for ch in children[nd]:
                if ch not in visited:
                    visited.add(ch)
                    depth_arr[ch] = depth_arr[nd] + 1
                    nxt.append(ch)
        queue = nxt

    # --- Degree ---
    in_deg = [0] * N
    out_deg = [0] * N
    for i in range(E):
        out_deg[edge_index[0, i].item()] += 1
        in_deg[edge_index[1, i].item()] += 1

    # --- Temporal delta: log1p(post_id offset from root) ---
    # The dataset's repost timestamps reflect the original post's creation
    # time, not the repost time. We use post_id as a temporal proxy instead
    # (88.6% of reposts have post_id > source post_id in this dataset).
    temporal = [0.0] * N
    for uid, idx in uid_to_idx.items():
        delta = max(0.0, uid_order[uid] - root_order)
        temporal[idx] = float(np.log1p(delta))

    # --- Sentiment ---
    sl = [1.0] * N   # default neutral
    ss = [0.5] * N
    for uid, idx in uid_to_idx.items():
        if uid in sentiment:
            sl[idx] = float(sentiment[uid][0])
            ss[idx] = float(sentiment[uid][1])

    # --- Assemble node feature matrix [N, 6] ---
    x = torch.tensor(
        list(zip(temporal, sl, ss, depth_arr, in_deg, out_deg)),
        dtype=torch.float,
    )

    # --- Graph-level attributes ---
    sent_comp = torch.zeros(3)
    for i in range(N):
        label = max(0, min(2, int(sl[i])))
        sent_comp[label] += 1
    sent_comp = sent_comp / N

    n_fol = (edge_type_t == 0).sum().item()
    fol_ratio = n_fol / E if E > 0 else 0.0

    # Breadth: max nodes at any single depth level
    depth_counts: dict[int, int] = {}
    for d in depth_arr:
        depth_counts[d] = depth_counts.get(d, 0) + 1
    breadth = max(depth_counts.values()) if depth_counts else 1

    max_depth = max(depth_arr) if depth_arr else 0
    wiener = _wiener_index(children, N)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type_t,
        sentiment_composition=sent_comp,
        follower_edge_ratio=torch.tensor(fol_ratio, dtype=torch.float),
        event_window=torch.tensor(EVENT_TO_IDX.get(event_window, 0), dtype=torch.long),
        cascade_depth=torch.tensor(max_depth, dtype=torch.long),
        cascade_breadth=torch.tensor(breadth, dtype=torch.long),
        wiener_index=torch.tensor(wiener, dtype=torch.float),
        cascade_size=torch.tensor(N, dtype=torch.long),
        cascade_id=torch.tensor(cascade_id, dtype=torch.long),
    )


def _wiener_index(children: dict[int, list[int]], N: int) -> float:
    """O(N) Wiener index for a rooted tree.

    W = sum over each non-root node v of: subtree_size(v) * (N - subtree_size(v)).
    This equals the sum of all pairwise shortest-path distances in the tree.
    """
    if N <= 1:
        return 0.0

    subtree = [1] * N

    # Post-order DFS to compute subtree sizes
    order: list[int] = []
    stack: list[tuple[int, bool]] = [(0, False)]
    seen = [False] * N
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if seen[node]:
            continue
        seen[node] = True
        stack.append((node, True))
        for ch in children.get(node, []):
            if not seen[ch]:
                stack.append((ch, False))

    for node in order:
        for ch in children.get(node, []):
            subtree[node] += subtree[ch]

    return float(sum(subtree[v] * (N - subtree[v]) for v in range(1, N)))
