import sqlite3
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from .base import StorageBackend


class SQLiteBackend(StorageBackend):
    """
    SQLite-backed exact (brute-force) vector search with optional FTS5 prefilter.

    Tables:
      vectors(id TEXT PRIMARY KEY, vec BLOB(float32), meta TEXT(JSON), text TEXT)
      docs_fts(id, content)  -- only if FTS5 is available

    Notes:
      - If your DB was created before adding the `text` column, delete the file
        (e.g., data/processed/vectors.sqlite) to let this schema apply.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.dim: Optional[int] = None
        self._fts_available: bool = False

    def name(self) -> str:
        return "sqlite_exact"

    def initialize(self, dim: int, **kwargs) -> None:
        """Open the DB and ensure schema exists."""
        self.dim = dim
        self.conn = sqlite3.connect(self.db_path)
        # Pragmas for better perf/stability on local experiments
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        cur = self.conn.cursor()

        # Main table with a `text` column (used for FTS mirroring + context)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors(
                id   TEXT PRIMARY KEY,
                vec  BLOB,
                meta TEXT,
                text TEXT
            )
            """
        )

        # Optional FTS5 table for keyword prefiltering
        try:
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(id, content)")
            self._fts_available = True
        except sqlite3.OperationalError:
            # FTS5 not available in this SQLite build; prefiltering will be skipped
            self._fts_available = False

        self.conn.commit()

    # ---------------- Ingestion ----------------

    def index_documents(
        self,
        ids: List[str],
        vectors,
        metadatas: List[Dict[str, Any]],
        texts: List[str],
    ) -> None:
        """
        Bulk (re)index documents.
        `vectors` can be a numpy array (N, dim) float32 or convertible.
        """
        if self.conn is None:
            raise RuntimeError("SQLiteBackend not initialized. Call initialize() first.")

        cur = self.conn.cursor()
        # Ensure ndarray of float32 for consistent storage
        vecs = np.asarray(vectors, dtype=np.float32)

        for _id, v, m, txt in zip(ids, vecs, metadatas, texts):
            cur.execute(
                "INSERT OR REPLACE INTO vectors(id, vec, meta, text) VALUES (?, ?, ?, ?)",
                (_id, v.tobytes(), json.dumps(m), txt),
            )
            if self._fts_available:
                cur.execute("INSERT INTO docs_fts(id, content) VALUES(?, ?)", (_id, txt))

        self.conn.commit()

    def add_documents(self, ids, vectors, metadatas, texts=None) -> None:
        """
        Insert/append docs. `texts` may be None (will store empty strings).
        """
        if texts is None:
            texts = [""] * len(ids)
        self.index_documents(ids, vectors, metadatas, texts)


    # ---------------- Querying ----------------

    def full_text_candidates(self, keywords: str, limit: int = 1000) -> list[str]:
        """
        Return a list of candidate IDs using FTS (if available).
        If you didn't create an FTS table, return [] to trigger brute-force fallback.
        """
        cur = self.conn.cursor()
        try:
            # Example if you created an FTS5 table 'docs_fts(content, doc_id UNINDEXED)'
            cur.execute("SELECT doc_id FROM docs_fts WHERE docs_fts MATCH ? LIMIT ?", (keywords, limit))
            rows = cur.fetchall()
            return [r[0] for r in rows]
        except Exception:
            # No FTS table configured — safe fallback
            return []

    def query_bruteforce_subset(self, query_vec: np.ndarray, subset_ids: list[str], k: int) -> list[tuple[str, float]]:
        """
        Exact cosine over a subset of doc IDs (candidate set from FTS).
        """
        if not subset_ids: return []
        cur = self.conn.cursor()
        qmarks = ",".join(["?"] * len(subset_ids))
        cur.execute(f"SELECT id, vec FROM vectors WHERE id IN ({qmarks})", subset_ids)
        rows = cur.fetchall()
        return self._cosine_topk_from_rows(query_vec, rows, k)

    def _cosine_topk_from_rows(
        self,
        query_vec: np.ndarray,
        rows: List[Tuple[str, bytes]],
        k: int,
    ) -> List[Tuple[str, float]]:
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-8
        sims: List[Tuple[str, float]] = []
        for _id, buf in rows:
            v = np.frombuffer(buf, dtype=np.float32)
            denom = qn * (float(np.linalg.norm(v)) + 1e-8)
            sims.append((_id, float(np.dot(q, v) / denom)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[: int(k)]

    def query(self, query_vec, k: int) -> List[Tuple[str, float]]:
        """
        Exact cosine top-k over the entire table (no prefilter).
        """
        if self.conn is None:
            raise RuntimeError("SQLiteBackend not initialized. Call initialize() first.")
        cur = self.conn.cursor()
        cur.execute("SELECT id, vec FROM vectors")
        rows = cur.fetchall()
        return self._cosine_topk_from_rows(query_vec, rows, k)

    # ---------------- Teardown ----------------

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.commit()
            finally:
                self.conn.close()

                self.conn = None
