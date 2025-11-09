from typing import List, Tuple, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from .base import StorageBackend

class ChromaBackend(StorageBackend):
    def __init__(self, path: str):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = None

    def name(self) -> str: return "chroma_hnsw"

    def initialize(self, dim: int, **kwargs) -> None:
        # Chroma manages HNSW underneath; presets from config are passed when creating collection
        self.collection = self.client.get_or_create_collection(name="docs")

    def index_documents(self, ids, vectors, metadatas, texts=None):
        # Chroma allows storing metadata + contents together
        # We store the text as part of metadata, because Chroma doesn't index raw text automatically
        if texts is not None:
            for i, m in enumerate(metadatas):
                m["text"] = texts[i]
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas
        )

    def add_documents(self, ids, vectors, metadatas, texts=None):
        self.index_documents(ids, vectors, metadatas, texts)

    def query(self, query_vec, k: int) -> List[Tuple[str, float]]:
        res = self.collection.query(query_embeddings=[query_vec], n_results=k, include=["distances"])
        ids = res["ids"][0]
        dists = res["distances"][0] if "distances" in res else [0.0]*len(ids)
        # Convert distance to similarity proxy (optional)
        sims = list(zip(ids, [1.0 - float(d) for d in dists]))
        return sims
    
    def multi_query_fuse(self, query_vecs, k: int, per_q: int = 20) -> List[Tuple[str, float]]:
        rank = {}
        for qv in query_vecs:
            res = self.collection.query(query_embeddings=[qv], n_results=per_q, include=["distances"])
            ids = res["ids"][0]
            for r, did in enumerate(ids):
                rank[did] = rank.get(did, 0.0) + 1.0 / (60 + r)  # RRF
        fused = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:k]
        # scores are fusion scores (not cosine); still fine for ranking
        return [(did, score) for did, score in fused]

    def close(self) -> None:
        pass
