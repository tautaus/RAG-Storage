import argparse, time, json, os, yaml, numpy as np, psutil, random
from tqdm import tqdm
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from backends.sqlite_backend import SQLiteBackend
from backends.chroma_backend import ChromaBackend

def cosine(a, b):  # for recall gold
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    processed = cfg["paths"]["processed_dir"]
    results_dir = os.path.join(cfg["paths"]["results_dir"], "logs")
    os.makedirs(results_dir, exist_ok=True)

    ids = np.load(os.path.join(processed, "doc_ids.npy"), allow_pickle=True)
    vecs = np.load(os.path.join(processed, "doc_embeddings.npy"))
    dim = vecs.shape[1]

    # toy query set = random docs; swap with BEIR queries later
    rng = random.Random(42)
    q_idx = rng.sample(range(len(ids)), k=min(cfg["query_count"], len(ids)))
    queries = vecs[q_idx]

    # gold (exact) neighbors @k using brute-force over full set
    ks = cfg["recall_k"]
    def gold_topk(q, k):
        sims = [cosine(q, v) for v in vecs]
        top = np.argsort(sims)[::-1][:k]
        return set(ids[i] for i in top)

    sqlite = SQLiteBackend("data/processed/vectors.sqlite"); sqlite.initialize(dim)
    chroma = ChromaBackend("data/processed/chroma"); chroma.initialize(dim)

    for backend in [sqlite, chroma]:
        for repeat in range(cfg["repeats"]):
            rows = []
            for q in tqdm(queries, desc=f"{backend.name()} r{repeat+1}"):
                for k in ks:
                    t0 = time.perf_counter()
                    res = backend.query(q, k)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    got_ids = set(x[0] for x in res)
                    recall = len(gold_topk(q, k) & got_ids) / float(k)
                    mem = psutil.Process().memory_info().rss / (1024*1024)
                    rows.append({"backend": backend.name(), "k": k, "lat_ms": elapsed, "recall": recall, "mem_mb": mem})
            out = os.path.join(results_dir, f"{backend.name()}_r{repeat+1}.json")
            json.dump(rows, open(out, "w"))
            print("Wrote", out)

    sqlite.close(); chroma.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
