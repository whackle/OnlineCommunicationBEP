"""
Core cascade construction logic.

Given a set of cascade root posts and their reposters, builds directed cascade
trees using the parent assignment heuristic:

  Priority 1: Reposter follows a prior reposter in the cascade → FOLLOWER edge
  Priority 2: Most recent prior reposter (temporal proximity) → NON_FOLLOWER edge
  Priority 3: Connect to root → UNKNOWN edge

Edge types:
  0 = FOLLOWER       (target follows source → organic spread)
  1 = NON_FOLLOWER   (both in follower graph, no edge → algorithmic/external)
  2 = UNKNOWN        (one/both users missing from follower graph)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blueccgl.data.follower_graph import FollowerGraph

EDGE_FOLLOWER     = 0
EDGE_NON_FOLLOWER = 1
EDGE_UNKNOWN      = 2


@dataclass
class CascadeNode:
    user_id: int
    timestamp: datetime
    sent_label: int = 1       # 0=neg, 1=neutral, 2=pos
    sent_score: float = 0.5
    post_id: int | None = None


@dataclass
class CascadeEdge:
    source_uid: int            # parent (exposer)
    target_uid: int            # child (reposter)
    edge_type: int             # 0/1/2
    source_ts: datetime | None = None
    target_ts: datetime | None = None


@dataclass
class Cascade:
    cascade_id: int            # = root post_id
    root: CascadeNode
    event_window: str
    root_text: str = ""
    nodes: list[CascadeNode] = field(default_factory=list)      # includes root at index 0
    edges: list[CascadeEdge] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        """Longest path from root. O(N) via BFS on the tree."""
        if not self.edges:
            return 0
        children: dict[int, list[int]] = {}
        for e in self.edges:
            children.setdefault(e.source_uid, []).append(e.target_uid)
        max_depth = 0
        stack = [(self.root.user_id, 0)]
        while stack:
            uid, d = stack.pop()
            max_depth = max(max_depth, d)
            for child in children.get(uid, []):
                stack.append((child, d + 1))
        return max_depth


def build_cascade_tree(
    root: CascadeNode,
    reposters: list[CascadeNode],
    follower_graph: FollowerGraph | None = None,
) -> tuple[list[CascadeEdge], dict[str, int]]:
    """
    Assign parents to each reposter using the priority heuristic.

    Args:
        root: The cascade root node.
        reposters: List of reposter nodes, in any order (will be sorted by timestamp).
        follower_graph: Optional FollowerGraph for follower lookups.
            If None, all edges are assigned UNKNOWN (degraded mode).

    Returns:
        (edges, stats) where stats counts priority assignments.
    """
    # Sort reposters by timestamp (chronological order)
    reposters = sorted(reposters, key=lambda n: n.timestamp)

    # Deduplicate by user_id (keep earliest repost per user)
    seen_users: set[int] = {root.user_id}
    unique_reposters: list[CascadeNode] = []
    for node in reposters:
        if node.user_id not in seen_users:
            seen_users.add(node.user_id)
            unique_reposters.append(node)
    reposters = unique_reposters

    edges: list[CascadeEdge] = []
    stats = {"priority_1": 0, "priority_2": 0, "priority_3": 0}

    # Build the cascade chronologically — earlier_nodes is the set of users
    # already in the cascade (available as potential parents)
    earlier_nodes: list[CascadeNode] = [root]

    for reposter in reposters:
        parent, edge_type, priority = _assign_parent(
            reposter, earlier_nodes, follower_graph
        )
        stats[f"priority_{priority}"] += 1

        edges.append(CascadeEdge(
            source_uid=parent.user_id,
            target_uid=reposter.user_id,
            edge_type=edge_type,
            source_ts=parent.timestamp,
            target_ts=reposter.timestamp,
        ))
        earlier_nodes.append(reposter)

    return edges, stats


def _assign_parent(
    reposter: CascadeNode,
    earlier_nodes: list[CascadeNode],
    follower_graph: FollowerGraph | None,
) -> tuple[CascadeNode, int, int]:
    """
    Returns (parent_node, edge_type, priority_level).

    Priority 1: reposter follows a prior cascade member → pick the most recent one.
    Priority 2: most recent prior reposter (temporal fallback).
    Priority 3: root node (last resort).
    """
    # Priority 1: Find followed prior members (if follower graph available)
    if follower_graph is not None:
        # Iterate from most recent to earliest (skip root for now, check it too)
        followed_priors: list[CascadeNode] = []
        for node in reversed(earlier_nodes):
            if follower_graph.follows(reposter.user_id, node.user_id):
                followed_priors.append(node)
                break  # take the most recent followed prior
        if followed_priors:
            parent = followed_priors[0]
            return parent, EDGE_FOLLOWER, 1

    # Priority 2: most recent prior reposter (not the root if there are others)
    if len(earlier_nodes) > 1:
        parent = earlier_nodes[-1]  # most recent (list is chronological)
        # Determine edge type
        if follower_graph is None:
            edge_type = EDGE_UNKNOWN
        elif follower_graph.follows(reposter.user_id, parent.user_id):
            # Actually a follower — upgrade to Priority 1 edge type
            edge_type = EDGE_FOLLOWER
        else:
            edge_type = EDGE_NON_FOLLOWER
        return parent, edge_type, 2

    # Priority 3: connect to root
    root = earlier_nodes[0]
    if follower_graph is None:
        edge_type = EDGE_UNKNOWN
    elif follower_graph.follows(reposter.user_id, root.user_id):
        edge_type = EDGE_FOLLOWER
    else:
        edge_type = EDGE_NON_FOLLOWER
    return root, edge_type, 3
