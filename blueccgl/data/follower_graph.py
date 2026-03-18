"""
DuckDB-backed follower graph for edge-type lookups.

Loads followers.csv.gz into an indexed DuckDB database on first use.
Subsequent calls use the pre-built .duckdb file.

Usage:
    fg = FollowerGraph("Data/raw/followers.csv.gz", "Data/followers.duckdb")
    fg.follows(target_uid, source_uid)  # True if target follows source
    fg.close()
"""

from __future__ import annotations

from pathlib import Path

import duckdb


class FollowerGraph:
    def __init__(self, csv_gz_path: str | Path, db_path: str | Path):
        self.db_path = Path(db_path)
        self.csv_gz_path = Path(csv_gz_path)
        self._conn: duckdb.DuckDBPyConnection | None = None

    def _ensure_db(self) -> duckdb.DuckDBPyConnection:
        if self._conn is not None:
            return self._conn

        if self.db_path.exists():
            print(f"[follower_graph] Loading existing DuckDB: {self.db_path}")
            self._conn = duckdb.connect(str(self.db_path))
            return self._conn

        if not self.csv_gz_path.exists():
            raise FileNotFoundError(
                f"followers.csv.gz not found at {self.csv_gz_path} "
                f"and no pre-built DuckDB at {self.db_path}"
            )

        print(f"[follower_graph] Building DuckDB from {self.csv_gz_path}...")
        print(f"[follower_graph] This is a one-time operation (~144M edges). May take a few minutes.")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(self.db_path))

        # Infer schema from the CSV; expected columns: follower, followed (or similar)
        # First, peek at the header
        conn.execute(f"""
            CREATE TABLE followers AS
            SELECT * FROM read_csv_auto('{self.csv_gz_path}', compression='gzip')
            LIMIT 0
        """)
        cols = [row[0] for row in conn.execute("DESCRIBE followers").fetchall()]
        conn.execute("DROP TABLE followers")
        print(f"[follower_graph] Detected columns: {cols}")

        # Load the full CSV
        conn.execute(f"""
            CREATE TABLE followers AS
            SELECT * FROM read_csv_auto('{self.csv_gz_path}', compression='gzip')
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM followers").fetchone()[0]
        print(f"[follower_graph] Loaded {row_count:,} edges")

        # Create index for fast (follower, followed) lookups
        # Columns might be named differently — adapt to what we find
        if "follower" in cols and "followed" in cols:
            self._follower_col = "follower"
            self._followed_col = "followed"
        elif "source" in cols and "target" in cols:
            self._follower_col = "source"
            self._followed_col = "target"
        else:
            # Use positional: first col = follower, second col = followed
            self._follower_col = cols[0]
            self._followed_col = cols[1]
            print(f"[follower_graph] Using positional columns: {cols[0]}=follower, {cols[1]}=followed")

        conn.execute(f"""
            CREATE INDEX idx_follow
            ON followers ({self._follower_col}, {self._followed_col})
        """)
        print(f"[follower_graph] Index built. DB saved to {self.db_path}")

        self._conn = conn
        return self._conn

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return self._ensure_db()

    def preload_for_users(self, user_ids: set[int]) -> None:
        """
        Load all follower edges involving `user_ids` into an in-memory set.
        After calling this, `follows()` uses the in-memory cache — no more
        per-lookup DB queries. Call this once before bulk tree building.
        """
        self._ensure_columns()
        print(f"[follower_graph] Preloading edges for {len(user_ids):,} users...")
        conn = self.conn
        # Create temp table of user IDs, then join — much faster than a giant IN clause
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _cascade_users (uid BIGINT)")
        conn.execute("DELETE FROM _cascade_users")
        conn.executemany("INSERT INTO _cascade_users VALUES (?)", [[u] for u in user_ids])
        rows = conn.execute(f"""
            SELECT f.{self._follower_col}, f.{self._followed_col}
            FROM followers f
            JOIN _cascade_users u ON f.{self._follower_col} = u.uid
        """).fetchall()
        self._cache: set[tuple[int, int]] = set(rows)
        print(f"[follower_graph] Preloaded {len(self._cache):,} relevant edges into memory")

    def follows(self, follower_uid: int, followed_uid: int) -> bool:
        """Returns True if follower_uid follows followed_uid."""
        # Use in-memory cache if preloaded
        if hasattr(self, "_cache"):
            return (follower_uid, followed_uid) in self._cache

        self._ensure_columns()
        result = self.conn.execute(
            f"SELECT 1 FROM followers WHERE {self._follower_col} = ? AND {self._followed_col} = ? LIMIT 1",
            [follower_uid, followed_uid],
        ).fetchone()
        return result is not None

    def _ensure_columns(self) -> None:
        if hasattr(self, "_follower_col"):
            return
        cols = [row[0] for row in self.conn.execute("DESCRIBE followers").fetchall()]
        if "follower" in cols and "followed" in cols:
            self._follower_col = "follower"
            self._followed_col = "followed"
        elif "source" in cols and "target" in cols:
            self._follower_col = "source"
            self._followed_col = "target"
        else:
            self._follower_col = cols[0]
            self._followed_col = cols[1]

    def both_exist(self, uid_a: int, uid_b: int) -> bool:
        """Check if both users appear anywhere in the follower graph."""
        if hasattr(self, "_cache"):
            # Check cache: either user appears as follower in any edge
            return any(uid_a == f or uid_a == t for f, t in self._cache) and \
                   any(uid_b == f or uid_b == t for f, t in self._cache)
        self._ensure_columns()
        result = self.conn.execute(f"""
            SELECT 1 FROM followers
            WHERE {self._follower_col} = ? OR {self._followed_col} = ?
               OR {self._follower_col} = ? OR {self._followed_col} = ?
            LIMIT 1
        """, [uid_a, uid_a, uid_b, uid_b]).fetchone()
        return result is not None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
