import csv
import os, argparse, json, yaml, time, pathlib, re, torch
import random
import numpy as np
from typing import List, Tuple

from sentence_transformers import SentenceTransformer # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(min(4, torch.get_num_threads() or 4))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# Backends
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from backends.sqlite_backend import SQLiteBackend
from backends.chroma_backend import ChromaBackend

def load_docs(raw_dir: str) -> dict[str, str]:
    """id -> text"""
    id_to_text = {}
    p = pathlib.Path(raw_dir) / "docs.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            id_to_text[o["id"]] = o["text"]
    return id_to_text

def load_queries(raw_dir: str) -> List[dict[str, str]]:
    p = pathlib.Path(raw_dir) / "queries.jsonl"
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def load_qrels(raw_dir: str) -> dict[str, str]:
    """
    qrels.tsv format: qid \t 0 \t docid \t rel
    Return mapping qid -> a single 'relevant' docid (first max-rel).
    If no qrels.tsv exists (synthetic), assume q{i} -> d{i}.
    """
    qrels_path = pathlib.Path(raw_dir) / "qrels.tsv"
    if not qrels_path.exists():
        return {}
    best = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4: 
                continue
            qid, _, did, rel_str = parts
            rel = int(rel_str)
            if qid not in best or rel > best[qid][1]:
                best[qid] = (did, rel)
    return {q: did for q, (did, _) in best.items()}


# ---------------- RAG bits ----------------

def make_prompt(question: str, context: str) -> str:
    return (
        "You are a precise assistant. Answer the question ONLY using the context. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

def build_context(id_to_text: dict[str, str], ids: List[str], max_chars: int) -> str:
    ctx = ""
    for i, did in enumerate(ids, 1):
        t = (id_to_text.get(did, "") or "").strip()
        if not t:
            continue
        chunk = f"\n[Doc {i} | id={did}]\n{t}\n"
        if len(ctx) + len(chunk) > max_chars:
            break
        ctx += chunk
    return ctx

def recall_at_k(hit_ids: List[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in set(hit_ids[:k]) else 0.0


# ---------------- Query expansion ----------------

_STOP = {"the","and","with","from","about","into","which","that","this","there","their","have","been","what","when","where","how","why"}

def simple_keywords(text: str, k: int = 6) -> str:
    """
    Cheap keyword extractor for FTS prefilter. Not an embedding.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    toks = [t for t in text.split() if len(t) > 3 and t not in _STOP]
    # frequency sort; break ties lexicographically for determinism
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:k]
    return " ".join([t for t, _ in terms]) if terms else " ".join(toks[:k])


def llm_paraphrases_local(q: str, n: int, tok, gen, temp: float = 0.7) -> list[str]:
    """
    Use an LLM (local causal LM) to generate n paraphrased search queries.
    No deduplication, no heuristic fallback — directly return decoded lines.
    """
    prompt = (
        f"Paraphrase the question into {n} short, distinct queries suitable for search.\n"
        f"Question: {q}\n"
        "Return as a numbered list.\n"
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True)
    out = gen.generate(
        **inputs,
        max_new_tokens=96,
        temperature=temp,
        do_sample=True,
        top_p=0.9,
    )

    txt = tok.decode(out[0], skip_special_tokens=True)

    # Extract lines after numbering: "1) xxx", "2. xxx", "3 - xxx"
    lines = [re.sub(r"^\s*\d+[\).\s-]*", "", L).strip() for L in txt.splitlines()]
    # Keep only non-empty lines
    lines = [L for L in lines if len(L) > 0]

    # Just return the first n — NO dedup, NO heuristic
    return lines[:n]

def expand_queries(question: str, cfg, tok=None, gen=None) -> list[str]:
    rcfg = cfg.get("retrieval", {})
    n = int(rcfg.get("n_query_expansions", 3))
    mode = rcfg.get("query_expander", "none").lower()
    if mode == "llm" and tok is not None and gen is not None:
        return llm_paraphrases_local(question, n, tok, gen)
    return [question]

# ---------------- Backend-specific retrieval using expansions ----------------

def retrieve_sqlite_rag(
    sqlite: SQLiteBackend,
    emb_model,
    question: str,
    cfg,
    q_emb_original=None  # <–– allow caller to pass existing embedding
) -> list[tuple[str, float]]:
    """
    Hybrid retrieval for SQLite:
      1) expand queries (LLM/heuristic)
      2) FTS candidate union
      3) exact cosine over subset using the ORIGINAL question embedding
         (keeps ranking semantics consistent with corpus embeddings)
      4) fallback: full brute-force if FTS yields nothing
    """
    k = int(cfg["recall_k"])
    sqlite_limit = int(cfg.get("retrieval", {}).get("sqlite_limit", 1000))

    # 1) expand queries
    qs = expand_queries(
        question,
        cfg,
        tok=cfg.get("_tok"),
        gen=cfg.get("_gen")
    )

    # 2) collect candidate IDs from FTS
    cand_ids, seen = [], set()
    for q in qs:
        kw = simple_keywords(q)  # cheap keyword extraction
        cands = sqlite.full_text_candidates(kw, limit=sqlite_limit)
        for cid in cands:
            if cid not in seen:
                seen.add(cid)
                cand_ids.append(cid)

    # 3) embed the original question ONCE — same config as doc embeddings
    if q_emb_original is None:
        q_emb_original = encode_query(emb_model, question, cfg)

    if cand_ids:
        return sqlite.query_bruteforce_subset(q_emb_original, subset_ids=cand_ids, k=k)

    # 4) fallback: full brute-force scan if FTS yields nothing
    return sqlite.query(q_emb_original, k=k)

def retrieve_chroma_rag(chroma: ChromaBackend, emb_model, question: str, cfg) -> list[tuple[str, float]]:
    """
    ANN retrieval for Chroma:
      1) expand queries (LLM/heuristic)
      2) embed each expansion
      3) HNSW search per expansion
      4) fuse with Reciprocal Rank Fusion
    """
    k = int(cfg["recall_k"])
    per_q = int(cfg.get("retrieval", {}).get("chroma_per_query", 20))

    qs = expand_queries(question, cfg, tok=cfg.get("_tok"), gen=cfg.get("_gen"))
    norm = cfg.get("embedding", {}).get("normalize", True)
    q_embs = emb_model.encode(qs, normalize_embeddings=norm).astype("float32")
    return chroma.multi_query_fuse(q_embs, k=k, per_q=per_q)


# ---------------- Main runner ----------------

def main(cfg_path: str, backend_name: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))

    raw_dir       = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    results_dir   = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Data & gold
    id_to_text = load_docs(raw_dir)
    queries = load_queries(raw_dir)
    qrels   = load_qrels(raw_dir)

    # Experiment slice
    Q = int(cfg["query_count"])
    random.seed(42)
    random.shuffle(queries)
    queries = queries[:Q]

    # Models
    emb_model = SentenceTransformer(cfg["embedding"]["model_name"])

    # Optional local LLM (used both for query-expansion and/or final generation)
    llm_cfg      = cfg.get("llm", {})
    enable_gen   = bool(llm_cfg.get("enable_generation", False))
    enable_qexp  = cfg.get("retrieval", {}).get("query_expander", "heuristic") in ("heuristic", "llm")

    tok = gen = None
    if enable_gen or (enable_qexp and llm_cfg.get("provider","gpt2-local") == "gpt2-local"):
        tok = AutoTokenizer.from_pretrained(llm_cfg.get("model", "gpt2"))
        gen = AutoModelForCausalLM.from_pretrained(llm_cfg.get("model", "gpt2"))
        # stash for access inside helpers (clean way is passing; this is explicit here)
        cfg["_tok"] = tok
        cfg["_gen"] = gen

    # Backends
    dim = int(np.load(os.path.join(processed_dir, "doc_embeddings.npy")).shape[1])
    if backend_name.lower().startswith("sqlite"):
        be = SQLiteBackend(os.path.join(processed_dir, "vectors.sqlite"))
        be.initialize(dim=dim)
        backend_label = "sqlite_exact"
    else:
        be = ChromaBackend(os.path.join(processed_dir, "chroma"))
        be.initialize(dim=dim)
        backend_label = "chroma_hnsw"

    # Metrics accumulators
    k = int(cfg["recall_k"])
    max_ctx = int(llm_cfg.get("max_context_chars", 2800))
    rows = []
    sum_retr_ms = 0.0
    sum_gen_ms  = 0.0
    sum_recall  = 0.0

    for qobj in queries:
        qid = qobj["id"]
        qtext = qobj["text"]

        # Gold mapping
        if qrels:
            gold = qrels.get(qid, None)
        else:
            gold = "d" + qid.lstrip("q")
        
        with torch.inference_mode():
            q_emb = emb_model.encode([qtext], normalize_embeddings=cfg["embedding"].get("normalize", True))[0].astype("float32")

            # Retrieval
            t0 = time.perf_counter()
            if backend_label == "sqlite_exact":
                hits = retrieve_sqlite_rag(be, emb_model, qtext, cfg, q_emb_original=q_emb)
            else:
                hits = retrieve_chroma_rag(be, emb_model, qtext, cfg)
            retr_ms = (time.perf_counter() - t0) * 1000.0

            hit_ids = [h[0] for h in hits]
            r_at_k  = recall_at_k(hit_ids, gold, k) if gold else 0.0

            # Optional end-to-end generation timing
            gen_ms = 0.0
            context = build_context(id_to_text, hit_ids, max_chars=max_ctx)
            prompt  = make_prompt(qtext, context)
            g0 = time.perf_counter()
            inputs  = tok(prompt, return_tensors="pt", truncation=True)
            _ = gen.generate(
                **inputs,
                max_new_tokens=int(llm_cfg.get("max_new_tokens", 200)),
                temperature=float(llm_cfg.get("temperature", 0.5)),
                do_sample=True
            )
            gen_ms = (time.perf_counter() - g0) * 1000.0
            # count prompt/context tokens (rough size proxy)
            c_tokens = len(inputs["input_ids"][0])

            rows.append({
                "backend": backend_label,
                "qid": qid,
                "question": qtext,
                "gold_id": gold,
                "hit_ids": hit_ids,
                "recall_at_k": r_at_k,
                "retrieval_ms": retr_ms,
                "generation_ms": gen_ms,
                "context_token": c_tokens,
            })
            sum_retr_ms += retr_ms
            sum_gen_ms  += gen_ms
            sum_recall  += r_at_k

    be.close()

        # --- Build and persist a compact run summary with tail latency & resources ---
    def dir_size_mb(path):
        if not os.path.exists(path): return 0.0
        if os.path.isfile(path): return os.path.getsize(path)/1e6
        total = 0
        for root, _, files in os.walk(path):
            for fn in files:
                total += os.path.getsize(os.path.join(root, fn))
        return total/1e6

    idx_mb = 0.0
    if backend_label == "sqlite_exact":
        idx_mb = dir_size_mb(os.path.join(processed_dir, "vectors.sqlite"))
    else:
        idx_mb = dir_size_mb(os.path.join(processed_dir, "chroma"))

    
    summary = {
        "backend": backend_label,
        "k": k,
        "queries_evaluated": len(rows),
        "latency_ms": {
            "mean": (sum(retr_latencies)/len(retr_latencies)) if retr_latencies else 0.0,
            "p50": pctl(retr_latencies, 50),
            "p90": pctl(retr_latencies, 90),
            "p95": pctl(retr_latencies, 95),
            "p99": pctl(retr_latencies, 99),
            "qps": (len(rows) / (sum(retr_latencies)/1000.0)) if retr_latencies else 0.0
        },
        "quality": {
            "recall_at_k_mean": (sum_recall/len(rows)) if rows else 0.0,
            "mrr_at_k_mean": (sum(mrr_vals)/len(mrr_vals)) if mrr_vals else 0.0,
            "precision_at_k_mean": (sum(prec_vals)/len(prec_vals)) if prec_vals else 0.0
        },
        "resources": {
            "index_mb": idx_mb,
            "peak_rss_mb": peak_rss_mb,
            "cpu_pct_avg": (sum(cpu_samples)/len(cpu_samples)) if cpu_samples else 0.0
        },
        "llm": {
            "used": bool(enable_gen),
            "gen_ms_mean": (sum(gen_latencies)/len(gen_latencies)) if gen_latencies else 0.0,
            "context_tokens_mean": (sum(ctx_tokens)/len(ctx_tokens)) if ctx_tokens else 0.0,
            "gold_in_context_rate": (gold_in_ctx/len(rows)) if rows else 0.0
        },
        "notes": {
            "n_query_expansions": cfg.get("retrieval",{}).get("n_query_expansions"),
            "normalize": cfg.get("embedding",{}).get("normalize"),
        }
    }

    summary_path = os.path.join(results_dir, f"rag_summary_{backend_label}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--backend", default="chroma", help="chroma | sqlite")
    args = ap.parse_args()
    main(args.cfg, args.backend)