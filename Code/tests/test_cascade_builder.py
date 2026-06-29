"""Tests for the cascade builder parent assignment heuristic."""

from datetime import datetime, timedelta

from blueccgl.data.cascade_builder import (
    EDGE_FOLLOWER,
    EDGE_NON_FOLLOWER,
    EDGE_UNKNOWN,
    CascadeNode,
    build_cascade_tree,
)


def _ts(minutes: int) -> datetime:
    """Helper: create a timestamp offset from a base time."""
    return datetime(2023, 10, 7, 12, 0) + timedelta(minutes=minutes)


def _node(uid: int, minutes: int) -> CascadeNode:
    return CascadeNode(user_id=uid, timestamp=_ts(minutes))


class FakeFollowerGraph:
    """Stub follower graph for testing."""

    def __init__(self, edges: set[tuple[int, int]]):
        # edges = set of (follower, followed) tuples
        self._edges = edges

    def follows(self, follower_uid: int, followed_uid: int) -> bool:
        return (follower_uid, followed_uid) in self._edges


def test_degraded_mode_all_unknown():
    """Without a follower graph, all edges should be UNKNOWN."""
    root = _node(1, 0)
    reposters = [_node(2, 10), _node(3, 20), _node(4, 30)]

    edges, stats = build_cascade_tree(root, reposters, follower_graph=None)

    assert len(edges) == 3
    assert all(e.edge_type == EDGE_UNKNOWN for e in edges)
    assert stats["priority_3"] == 1  # first reposter → root (P3)
    assert stats["priority_2"] == 2  # subsequent → temporal (P2)


def test_priority_1_follower():
    """Reposter follows an earlier cascade member → Priority 1 FOLLOWER edge."""
    root = _node(1, 0)
    reposters = [_node(2, 10), _node(3, 20)]

    # User 3 follows user 2 (who is already in the cascade)
    fg = FakeFollowerGraph({(3, 2)})
    edges, stats = build_cascade_tree(root, reposters, follower_graph=fg)

    assert len(edges) == 2
    # Edge for user 3 should be P1 (follows user 2)
    edge_3 = [e for e in edges if e.target_uid == 3][0]
    assert edge_3.source_uid == 2
    assert edge_3.edge_type == EDGE_FOLLOWER
    assert stats["priority_1"] >= 1


def test_priority_1_picks_most_recent_followed():
    """When reposter follows multiple earlier members, pick the most recent."""
    root = _node(1, 0)
    reposters = [_node(2, 10), _node(3, 20), _node(4, 30)]

    # User 4 follows both user 2 and user 3
    fg = FakeFollowerGraph({(4, 2), (4, 3)})
    edges, stats = build_cascade_tree(root, reposters, follower_graph=fg)

    edge_4 = [e for e in edges if e.target_uid == 4][0]
    # Should pick user 3 (most recent)
    assert edge_4.source_uid == 3
    assert edge_4.edge_type == EDGE_FOLLOWER


def test_priority_2_temporal_fallback():
    """No follower link → falls back to most recent prior reposter."""
    root = _node(1, 0)
    reposters = [_node(2, 10), _node(3, 20)]

    # No follow relationships at all
    fg = FakeFollowerGraph(set())
    edges, stats = build_cascade_tree(root, reposters, follower_graph=fg)

    edge_3 = [e for e in edges if e.target_uid == 3][0]
    assert edge_3.source_uid == 2  # most recent prior
    assert edge_3.edge_type == EDGE_NON_FOLLOWER
    assert stats["priority_2"] >= 1


def test_priority_3_first_reposter_to_root():
    """First reposter always connects to root (only 1 earlier node = root)."""
    root = _node(1, 0)
    reposters = [_node(2, 10)]

    fg = FakeFollowerGraph(set())
    edges, stats = build_cascade_tree(root, reposters, follower_graph=fg)

    assert len(edges) == 1
    assert edges[0].source_uid == 1  # root
    assert edges[0].target_uid == 2
    assert stats["priority_3"] == 1


def test_deduplicates_by_user_id():
    """Same user reposting twice → keep only the earliest."""
    root = _node(1, 0)
    reposters = [_node(2, 10), _node(2, 20), _node(3, 30)]

    edges, stats = build_cascade_tree(root, reposters, follower_graph=None)

    # Should only have 2 edges (user 2 and user 3), not 3
    assert len(edges) == 2
    target_uids = {e.target_uid for e in edges}
    assert target_uids == {2, 3}


def test_unsorted_reposters_handled():
    """Reposters passed out of order should still be sorted by timestamp."""
    root = _node(1, 0)
    # Pass them out of order
    reposters = [_node(3, 30), _node(2, 10)]

    edges, stats = build_cascade_tree(root, reposters, follower_graph=None)

    # User 2 (t=10) should be first → connects to root
    # User 3 (t=30) → connects to user 2 (temporal proximity)
    edge_2 = [e for e in edges if e.target_uid == 2][0]
    edge_3 = [e for e in edges if e.target_uid == 3][0]
    assert edge_2.source_uid == 1  # root
    assert edge_3.source_uid == 2  # temporal
