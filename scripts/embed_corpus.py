import argparse, os, json, numpy as np, yaml, pathlib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_docs(raw_dir):
    docs_path = pathlib.Path(raw_dir) / "docs.jsonl"
    docs = []
    # Use explicit encoding to avoid Windows decode errors
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append((obj["id"], obj["text"]))
    return docs

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    raw, processed = cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"]
    os.makedirs(processed, exist_ok=True)

    # Optional: cap docs without changing default behavior
    doc_limit = cfg.get("embedding", {}).get("doc_limit", None)

    model = SentenceTransformer(cfg["embedding"]["model_name"])
    docs = load_docs(raw)
    if doc_limit is not None:
        docs = docs[:int(doc_limit)]

    ids, texts = zip(*docs)
    embs = model.encode(
        list(texts),
        batch_size=cfg["embedding"]["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=cfg["embedding"]["normalize"],
    )
    np.save(os.path.join(processed, "doc_ids.npy"), np.array(ids))
    np.save(os.path.join(processed, "doc_embeddings.npy"), embs.astype("float32"))
    print(f"Saved embeddings: {embs.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)