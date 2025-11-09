import argparse, os, csv, ast, pathlib
from collections import defaultdict
from radon.visitors import ComplexityVisitor
from radon.metrics import mi_visit
from radon.raw import analyze as raw_analyze

def py_files(dirs):
    for d in dirs:
        dpath = pathlib.Path(d)
        for p in dpath.rglob("*.py"):
            # skip __pycache__ or virtual envs
            if any(seg in {"__pycache__", ".venv", "venv"} for seg in p.parts):
                continue
            yield p

def read_text(path):
    return path.read_text(encoding="utf-8", errors="ignore")

def analyze_file(path: pathlib.Path):
    src = read_text(path)
    # Raw metrics (SLOC, comments, blanks, etc.)
    raw = raw_analyze(src)
    # Maintainability Index (radon MI: 0–100, higher is better)
    mi = mi_visit(src, multi=True)
    # Cyclomatic complexity per function/class
    cv = ComplexityVisitor.from_code(src)
    cc_values = [b.complexity for b in cv.functions + cv.classes]
    max_cc = max(cc_values) if cc_values else 0
    avg_cc = sum(cc_values)/len(cc_values) if cc_values else 0.0
    return {
        "file": str(path),
        "sloc": raw.sloc,
        "lloc": raw.lloc,
        "comments": raw.comments,
        "multi": raw.multi,
        "blank": raw.blank,
        "mi": round(mi, 2),
        "avg_cc": round(avg_cc, 2),
        "max_cc": int(max_cc),
        "blocks": len(cc_values)
    }

def module_name(root: pathlib.Path, file: pathlib.Path):
    rel = file.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    # convert path\to\mod.py -> path.to.mod
    return ".".join(parts)

def collect_imports(root_dirs, files):
    # Build a per-module import graph using ast (no extra deps)
    # Nodes: module names relative to each root dir
    edges = defaultdict(set)
    nodes = set()
    roots = [pathlib.Path(d).resolve() for d in root_dirs]

    # map file -> module name by first matching root
    def file_to_mod(f):
        f = f.resolve()
        for r in roots:
            if r in f.parents or f.parent == r:
                try:
                    return module_name(r, f)
                except ValueError:
                    continue
        # fallback to stem
        return f.stem

    for f in files:
        src = read_text(f)
        this_mod = file_to_mod(f)
        nodes.add(this_mod)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    edges[this_mod].add(a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    edges[this_mod].add(n.module.split(".")[0])

    # compute fan-out (outdegree) and fan-in (indegree within our nodes set)
    fan_out = {m: len(edges.get(m, set())) for m in nodes}
    fan_in = defaultdict(int)
    for src, tgts in edges.items():
        for t in tgts:
            # only count if target resembles an internal module prefix
            if t in nodes:
                fan_in[t] += 1

    # detect cycles (simple DFS)
    visited, stack = set(), set()
    has_cycle = set()

    def dfs(u):
        visited.add(u); stack.add(u)
        for v in edges.get(u, []):
            if v not in nodes:  # external import: skip
                continue
            if v not in visited:
                dfs(v)
            elif v in stack:
                has_cycle.add(u); has_cycle.add(v)
        stack.remove(u)

    for n in nodes:
        if n not in visited:
            dfs(n)

    return fan_in, fan_out, has_cycle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="Folders to analyze, e.g. backends scripts")
    ap.add_argument("--out", default="results/static_metrics.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    files = list(py_files(args.src))

    # per-file metrics
    rows = [analyze_file(f) for f in files]

    # coupling (per-module) metrics
    fan_in, fan_out, cyc = collect_imports(args.src, files)
    # attach coupling by matching filename→module heuristic
    # if a module name isn’t found, fall back to 0
    for r in rows:
        f = pathlib.Path(r["file"])
        # guess module name relative to first provided src root
        base = pathlib.Path(args.src[0]).resolve()
        try:
            mod = module_name(base, f.resolve())
        except Exception:
            mod = f.stem
        r["fan_in"] = fan_in.get(mod, 0)
        r["fan_out"] = fan_out.get(mod, 0)
        r["in_cycle"] = 1 if mod in cyc else 0

    # write CSV
    hdr = ["file","sloc","lloc","comments","multi","blank","mi","avg_cc","max_cc","blocks","fan_in","fan_out","in_cycle"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # quick summary printed to console
    import statistics as stats
    mi_vals = [r["mi"] for r in rows if r["mi"] >= 0]
    cc_vals = [r["avg_cc"] for r in rows]
    print(f"[summary] files={len(rows)}  MI(median)={stats.median(mi_vals):.1f}  "
          f"avgCC(mean)={stats.mean(cc_vals):.2f}  cycles={sum(r['in_cycle'] for r in rows)}")

if __name__ == "__main__":
    main()