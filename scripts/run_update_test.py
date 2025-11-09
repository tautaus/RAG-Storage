import argparse, os, time, json, yaml, numpy as np, uuid, pathlib, sys
from datetime import datetime

# Ensure package imports work when invoked as a script
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from backends.sqlite_backend import SQLiteBackend
from backends.chroma_backend import ChromaBackend


def _now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _unique_ids(base_count: int, prefix: str = "new") -> list[str]:
    """
    Create add_count unique IDs, avoiding collisions in stores that require unique IDs (e.g., Chroma).
    """
    seed = uuid.uuid4().hex[:6]
    return [f"{prefix}_{seed}_{i}" for i in range(base_count)]


def _batch_iter(ids, vecs, metas, batch_size: int):
    n = len(ids)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        yield ids[i:j], vecs[i:j], metas[i:j]


def run_one_backend(backend_name: str, backend, add_ids, add_vecs, metas, dim: int, results_dir: str, batch_size: int):
    """
    Returns a dict with timing details and error (if any).
    """
    log = {
        "backend": backend_name,
        "timestamp_utc": _now_iso(),
        "add_count": len(add_ids),
        "batch_size": batch_size,
        "per_batch_ms": [],
        "elapsed_s": None,
        "error": None,
    }

    try:
        backend.initialize(dim=dim)  # re-open existing index
        t0 = time.perf_counter()

        for bid, bvecs, bmetas in _batch_iter(add_ids, add_vecs, metas, batch_size):
            bstart = time.perf_counter()
            backend.add_documents(bid, bvecs, bmetas)
            blog = (time.perf_counter() - bstart) * 1000.0
            log["per_batch_ms"].append(blog)

        log["elapsed_s"] = time.perf_counter() - t0

    except Exception as e:
        log["error"] = f"{type(e).__name__}: {e}"

    finally:
        try:
            backend.close()
        except Exception as e:
            # don't override the primary error if it exists
            if log["error"] is None:
                log["error"] = f"close_{type(e).__name__}: {e}"

    out_path = os.path.join(results_dir, f"update_{backend_name}.json")
    os.makedirs(results_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[{backend_name}] add_count={len(add_ids)} "
          f"batches={len(log['per_batch_ms'])} "
          f"elapsed={None if log['elapsed_s'] is None else round(log['elapsed_s'], 3)}s "
          f"{'(ERROR)' if log['error'] else ''}")

    return log


def main(cfg_path: str, add_count_arg: int | None, batch_size_arg: int | None):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    processed = cfg["paths"]["processed_dir"]
    results_dir = os.path.join(cfg["paths"]["results_dir"], "logs")
    os.makedirs(results_dir, exist_ok=True)

    # --- Load existing embeddings (we’ll reuse the first N vectors as “new” docs) ---
    ids = np.load(os.path.join(processed, "doc_ids.npy"), allow_pickle=True).tolist()
    vecs = np.load(os.path.join(processed, "doc_embeddings.npy"))

    # --- Resolve parameters (CLI flag > config > defaults) ---
    default_add = int(cfg.get("updates", {}).get("add_count", 1000))
    default_batch = int(cfg.get("updates", {}).get("batch_size", 200))
    add_count = int(add_count_arg) if add_count_arg is not None else default_add
    batch_size = int(batch_size_arg) if batch_size_arg is not None else default_batch

    if add_count > len(ids):
        raise ValueError(f"add_count={add_count} exceeds available base vectors={len(ids)}")

    add_vecs = vecs[:add_count]
    # Create unique IDs to avoid collisions (Chroma requires unique ids)
    add_ids = _unique_ids(add_count, prefix="newdoc")
    metas = [{"added": True, "ts": _now_iso()} for _ in range(add_count)]

    dim = add_vecs.shape[1]

    # Paths configured from processed dir
    sqlite_path = os.path.join(processed, "vectors.sqlite")
    chroma_path = os.path.join(processed, "chroma")

    # --- Run both backends ---
    sqlite_log = run_one_backend(
        "sqlite",
        SQLiteBackend(sqlite_path),
        add_ids,
        add_vecs,
        metas,
        dim,
        results_dir,
        batch_size,
    )

    chroma_log = run_one_backend(
        "chroma",
        ChromaBackend(chroma_path),
        add_ids,
        add_vecs,
        metas,
        dim,
        results_dir,
        batch_size,
    )

    # Summary print for quick copy into notes
    print("\n== Update Summary ==")
    for log in [sqlite_log, chroma_log]:
        if log["error"]:
            print(f"{log['backend']}: ERROR -> {log['error']}")
        else:
            p50 = None
            if log["per_batch_ms"]:
                sorted_ms = sorted(log["per_batch_ms"])
                mid = len(sorted_ms) // 2
                p50 = sorted_ms[mid] if len(sorted_ms) % 2 else 0.5 * (sorted_ms[mid - 1] + sorted_ms[mid])
            print(f"{log['backend']}: total={round(log['elapsed_s'], 3)}s, "
                  f"batches={len(log['per_batch_ms'])}, "
                  f"median_batch={None if p50 is None else round(p50, 1)} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to config/experiment.yaml")
    ap.add_argument("--add-count", type=int, default=None, help="Docs to insert (overrides config.updates.add_count)")
    ap.add_argument("--batch-size", type=int, default=None, help="Batch size for insertion (overrides config.updates.batch_size)")
    args = ap.parse_args()
    main(args.cfg, args.add_count, args.batch_size)
