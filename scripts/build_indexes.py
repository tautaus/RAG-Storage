import json
import argparse, os, numpy as np, yaml
from tqdm import tqdm  # for progress
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from backends.sqlite_backend import SQLiteBackend
from backends.chroma_backend import ChromaBackend

def chunks(iterable, size):
    for i in range(0, len(iterable), size):
        yield i, iterable[i:i+size]

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    processed = cfg["paths"]["processed_dir"]

    ids = np.load(os.path.join(processed, "doc_ids.npy"), allow_pickle=True)
    vecs = np.load(os.path.join(processed, "doc_embeddings.npy"))
    dim = vecs.shape[1]
    raw_docs_path = os.path.join(cfg["paths"]["raw_dir"], "docs.jsonl")
    id2text = {}
    with open(raw_docs_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            id2text[o["id"]] = o["text"]
    texts = [id2text.get(i, "") for i in ids]
    metas = [{"i": i} for i in range(len(ids))]

    batch_size = cfg.get("index", {}).get("batch_size", 1000)

    # --- SQLite (exact / brute force) ---
    sqlite = SQLiteBackend(os.path.join(processed, "vectors.sqlite"))
    sqlite.initialize(dim=dim)
    # SQLite can handle larger batches, but we’ll reuse batching for symmetry
    for i, idx in tqdm(chunks(list(range(len(ids))), batch_size), total=(len(ids)+batch_size-1)//batch_size, desc="SQLite indexing"):
        batch_ids = [ids[j] for j in idx]
        batch_vecs = vecs[idx]
        batch_meta = [metas[j] for j in idx]
        sqlite.index_documents(batch_ids, batch_vecs, batch_meta, texts)
    sqlite.close()

    # --- Chroma (HNSW ANN) ---
    chroma = ChromaBackend(path=os.path.join(processed, "chroma"))
    chroma.initialize(dim=dim)
    for i, idx in tqdm(chunks(list(range(len(ids))), batch_size), total=(len(ids)+batch_size-1)//batch_size, desc="Chroma indexing"):
        batch_ids = [ids[j] for j in idx]
        batch_vecs = vecs[idx]
        batch_meta = [metas[j] for j in idx]
        chroma.index_documents(batch_ids, batch_vecs, batch_meta, texts)
    chroma.close()

    print(f"Indexes built: {len(ids)} vectors → SQLite + Chroma")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
