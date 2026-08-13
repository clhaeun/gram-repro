"""[50] Fig 10/12 용 해 개수 분포.  sidecar 만 읽는다 (GPU·모델 불필요).

nq10 : data/nqueens-10/sidecar_10.json   inputs[hash] = {split, clues, solutions}
gc8/10: data/graph-coloring-{8,10}/sidecar_gc_{8,10}.json  graphs[hash] = {split, edges, colorings}

test split 기준으로 통일한다.  nq10 은 unique 입력 단위(행 단위 아님)이며
sidecar 의 키가 이미 unique 입력이므로 키당 1개씩 센다.

출력: {"counts": [...]} + 같은 이름의 .meta.json
"""
import argparse
import json
import os
from collections import Counter

SPECS = {
    "nq10": ("data/nqueens-10/sidecar_10.json", "inputs", "solutions"),
    "gc8": ("data/graph-coloring-8/sidecar_gc_8.json", "graphs", "colorings"),
    "gc10": ("data/graph-coloring-10/sidecar_gc_10.json", "graphs", "colorings"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--outdir", default="/workspace/fprm/results/figdata")
    ap.add_argument("--repo", default="/workspace/TinyRecursiveModels")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for name, (rel, container, field) in SPECS.items():
        path = os.path.join(args.repo, rel)
        doc = json.load(open(path))
        entries = doc[container]

        counts = [len(e[field]) for e in entries.values() if e["split"] == args.split]
        by_split = Counter(e["split"] for e in entries.values())

        out = os.path.join(args.outdir, f"{name}_solution_counts.json")
        with open(out, "w") as f:
            json.dump({"counts": counts}, f)

        hist = sorted(Counter(counts).items())
        meta = {
            "source": rel,
            "source_abspath": path,
            "sidecar_meta": doc.get("meta"),
            "split": args.split,
            "unit": "unique 입력 1개 = 1 항목 (sidecar 키 단위, 행 단위 아님)",
            "field": field,
            "n_items": len(counts),
            "n_by_split": dict(by_split),
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
            "mean": (sum(counts) / len(counts)) if counts else None,
            "histogram_head": hist[:10],
            "note": "논문 Fig 10/12 가 어느 split 인지 밝히지 않아 test 로 통일했다.",
        }
        with open(out.replace(".json", ".meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(
            f"[out] {out}  n={len(counts)}  min={meta['min']} max={meta['max']} "
            f"mean={meta['mean']:.3f}  {os.path.getsize(out):,} bytes  splits={dict(by_split)}"
        )


if __name__ == "__main__":
    main()
