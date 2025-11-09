# scripts/prepare_data.py
import argparse, os, json, pathlib, sys
from typing import Dict, Any
from tqdm import tqdm
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def to_jsonl_corpus(corpus: Dict[str, Dict[str, Any]], limit: int | None):
    rows = []
    for i, (doc_id, fields) in enumerate(corpus.items()):
        if limit and i >= limit: break
        title = (fields.get("title") or "").strip()
        text  = (fields.get("text") or "").strip()
        body = (title + "\n\n" + text).strip() if title else text
        rows.append({"id": doc_id, "text": body})
    return rows

def to_jsonl_queries(queries: Dict[str, str], limit: int | None):
    rows = []
    for i, (qid, qtext) in enumerate(queries.items()):
        if limit and i >= limit: break
        rows.append({"id": qid, "text": qtext})
    return rows

def to_qrels_tsv(qrels: Dict[str, Dict[str, int]], out_path: str, query_limit: int | None):
    # TREC format: qid \t 0 \t docid \t rel
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (qid, docrels) in enumerate(qrels.items()):
            if query_limit and i >= query_limit: break
            for doc_id, rel in docrels.items():
                f.write(f"{qid}\t0\t{doc_id}\t{rel}\n")
            written += 1
    return written

# ---------- BEIR loader ----------
def load_beir_dataset(name: str, split: str):
    """
    name ∈ {fiqa, scifact, nfcorpus, arguana, trec-covid, climate-fever, cqadupstack, dbpedia-entity, fever, hotpotqa, msmarco, nq, quora, robust04, scidocs, scifact, trec-news, webis-touche2020}
    """

    url_map = {
        "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        # add more here if needed
    }
    if name not in url_map:
        raise ValueError(f"Unsupported BEIR dataset '{name}'. Try one of: {list(url_map.keys())}")

    out_dir = pathlib.Path("data") / "beir" / name
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    data_path = util.download_and_unzip(url_map[name], str(out_dir))

    # Some BEIR datasets don’t have all splits; 'test' is safest default.
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output folder for raw files (e.g., data/raw)")
    ap.add_argument("--source", required=True, help="beir-fiqa | beir-scifact | beir-nfcorpus")
    ap.add_argument("--split", default="test", help="BEIR split to load (default: test)")
    ap.add_argument("--doc-limit", type=int, default=None, help="Optional cap on number of docs")
    ap.add_argument("--query-limit", type=int, default=None, help="Optional cap on number of queries")
    args = ap.parse_args()

    ensure_dir(args.out)

    ds = args.source.split("-", 1)[1]
    print(f"[beir] downloading/loading '{ds}' split={args.split} ...")
    corpus, queries, qrels = load_beir_dataset(ds, args.split)
    # Convert to your raw format
    docs_rows   = to_jsonl_corpus(corpus, args.doc_limit)
    query_rows  = to_jsonl_queries(queries, args.query_limit)
    qrels_written = to_qrels_tsv(qrels, os.path.join(args.out, "qrels.tsv"), args.query_limit)

    write_jsonl(os.path.join(args.out, "docs.jsonl"), tqdm(docs_rows, desc="write docs"))
    write_jsonl(os.path.join(args.out, "queries.jsonl"), tqdm(query_rows, desc="write queries"))

    pathlib.Path(os.path.join(args.out, "README.txt")).write_text(
        f"BEIR dataset: {ds}, split={args.split}\n"
        f"docs={len(docs_rows)}, queries={len(query_rows)}, qrels(qids)~={qrels_written}\n",
        encoding="utf-8"
    )
    print(f"[beir] wrote docs/queries/qrels to {args.out}")

if __name__ == "__main__":
    main()