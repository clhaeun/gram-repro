"""Build a 3-Graph-Coloring dataset in the TRM/GRAM ``PuzzleDataset`` format.

The model gets an undirected graph (the strict upper triangle of its adjacency
matrix, flattened in ``np.triu_indices(n, 1)`` order) and must output a proper
3-coloring.  Tokens: 0 = PAD/ignore, 1 = no edge, 2 = edge, 3/4/5 = the colors.
``seq_len = n*(n-1)/2``; the label holds the n node colors in the first n
positions and 0 (ignored by the loss) in the tail.

A proper coloring survives any permutation of the 3 colors, so only colorings in
*first-appearance normal form* are kept (``c[0] == 0`` and
``c[j] <= max(c[:j]) + 1``): exactly one member of each permutation orbit.

Erdos-Renyi graphs G(n, p) are sampled until ``num_graphs`` unique 3-colorable
ones are kept.  Rows are grouped by graph and the train/test split is over
*groups*, so no graph straddles the split.  Test keeps one row per graph (the
lexicographically first coloring), train keeps all -- as in the N-Queens builder.

``sidecar_gc_{n}.json`` maps each graph hash to its split, edge list, and the
FULL list of canonical proper colorings, for coverage@K style harnesses.  The
hash is sha1 of the flat uint8 upper-triangle token vector, [:16].
"""

from typing import Dict, List
import hashlib
import json
import os
import sys
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata


NO_EDGE, EDGE = 1, 2
COLOR0 = 3          # 3=red, 4=blue, 5=green

# Published dataset sizes (GRAM paper, appendix C.2.2).
#   n -> (train_pairs, test_pairs)
SIZE_ANCHORS = {8: (7002, 255), 10: (13465, 192)}
# Unique 3-colorable graphs implied by the anchors: test keeps 1 row per graph
# and the split is 85:15, so num_graphs = test_pairs / 0.15.
DEFAULT_NUM_GRAPHS = {8: 1700, 10: 1280}
# ``p`` is not given in the paper.  It is chosen so that the resulting dataset
# sizes reproduce the published anchors; the number of colorings per graph falls
# steeply with p, which is what pins it down.  Scanned at seed 0 in steps of
# 0.0005; these minimise |actual - anchor|:
#   n=8  p=0.5455 -> 6996 train pairs (anchor 7002, -6)
#   n=10 p=0.4250 -> 13461 train pairs (anchor 13465, -4)
# The count is a random variable: neighbouring p values and neighbouring seeds
# move it by +-200 (n=8) / +-1000 (n=10), so the residual is sampling noise, not
# a wrong p or a wrong 3-colorability filter.  Hitting the anchor exactly would
# mean fitting the seed to the noise, which is not evidence of anything.
DEFAULT_P = {8: 0.5455, 10: 0.4250}


cli = ArgParser()


class DataProcessConfig(BaseModel):
    n: int = 8
    output_dir: str = "data/graph-coloring-8"
    seed: int = 0
    p: float = -1.0             # <0 -> DEFAULT_P[n]
    num_graphs: int = -1        # <0 -> DEFAULT_NUM_GRAPHS[n]
    train_frac: float = 0.85
    self_check: bool = True
    check_samples: int = 1000


def canonical_colorings(n: int) -> np.ndarray:
    """All first-appearance-normalized color assignments, ``[M, n]`` in 0..2."""
    C = np.indices((3,) * n).reshape(n, -1).T.astype(np.int8)
    running_max = np.maximum.accumulate(C, axis=1)
    ok = C[:, 0] == 0
    for j in range(1, n):
        ok &= C[:, j] <= running_max[:, j - 1] + 1
    return np.ascontiguousarray(C[ok])


def edge_ok_table(Ccan: np.ndarray, iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """``D[e, m]`` = canonical coloring m gives different colors to edge e's ends."""
    return np.ascontiguousarray((Ccan[:, iu] != Ccan[:, iv]).T)


def proper_mask(bits: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Which canonical colorings are proper for the graph given by ``bits``."""
    e = np.nonzero(bits)[0]
    if len(e) == 0:
        return np.ones(D.shape[1], dtype=bool)
    return D[e].all(0)


def graph_hash(bits: np.ndarray) -> str:
    tokens = np.where(bits, EDGE, NO_EDGE).astype(np.uint8)
    return hashlib.sha1(tokens.tobytes()).hexdigest()[:16]


def encode_input(bits: np.ndarray) -> np.ndarray:
    return np.where(bits, EDGE, NO_EDGE).astype(np.int32)


def encode_label(coloring: np.ndarray, seq_len: int) -> np.ndarray:
    lab = np.zeros(seq_len, dtype=np.int32)          # 0 = ignore
    lab[:len(coloring)] = coloring.astype(np.int32) + COLOR0
    return lab


def sample_graphs(n: int, p: float, num_graphs: int, seed: int, D: np.ndarray,
                  P: int, batch: int = 4096, max_tries: int = 40_000_000):
    """Sample unique 3-colorable ER graphs. Returns (bits[G,P], colorings list)."""
    rng = np.random.default_rng(seed)
    seen: Dict[str, int] = {}
    bits_out: List[np.ndarray] = []
    cols_out: List[np.ndarray] = []
    tries = kept_dup = not3 = 0

    while len(bits_out) < num_graphs:
        if tries >= max_tries:
            raise RuntimeError(f"max_tries exceeded with {len(bits_out)}/{num_graphs} graphs")
        block = rng.random((batch, P)) < p
        tries += batch
        for bits in block:
            h = graph_hash(bits)
            if h in seen:
                kept_dup += 1
                continue
            m = proper_mask(bits, D)
            if not m.any():
                not3 += 1
                continue
            seen[h] = len(bits_out)
            bits_out.append(bits.copy())
            cols_out.append(np.nonzero(m)[0])
            if len(bits_out) >= num_graphs:
                break

    stats = dict(tries=tries, duplicates=kept_dup, not_3colorable=not3,
                 accept_rate=len(bits_out) / max(tries, 1))
    return np.stack(bits_out), cols_out, stats


def build_pairs(config: DataProcessConfig):
    """Sample the graphs, enumerate their canonical colorings, split by graph."""
    n = config.n
    p = config.p if config.p >= 0 else DEFAULT_P.get(n, 0.5)
    num_graphs = config.num_graphs if config.num_graphs > 0 else DEFAULT_NUM_GRAPHS.get(n, 1000)
    seq_len = n * (n - 1) // 2
    iu, iv = np.triu_indices(n, 1)
    P = len(iu)
    assert P == seq_len

    print(f"n={n}  p={p}  num_graphs={num_graphs}  seq_len={seq_len}  seed={config.seed}")

    Ccan = canonical_colorings(n)
    D = edge_ok_table(Ccan, iu, iv)
    print(f"canonical color assignments: {Ccan.shape[0]}  (all 3^n = {3**n})")

    bits, col_idx, stats = sample_graphs(n, p, num_graphs, config.seed, D, P)
    per_graph = np.array([len(c) for c in col_idx])
    print(f"sampled: tries={stats['tries']} duplicates={stats['duplicates']} "
          f"not_3colorable={stats['not_3colorable']} accept={stats['accept_rate']:.4f}")
    print(f"canonical colorings per graph: mean={per_graph.mean():.4f} "
          f"min={per_graph.min()} max={per_graph.max()}")
    print(f"edges per graph: mean={bits.sum(1).mean():.3f}")

    hashes = [graph_hash(b) for b in bits]
    assert len(set(hashes)) == len(hashes), "duplicate graph slipped through"

    # Split unique graphs 85:15
    perm = np.random.default_rng(config.seed).permutation(num_graphs)
    n_train = int(round(config.train_frac * num_graphs))
    train_ids = sorted(perm[:n_train].tolist())
    test_ids = sorted(perm[n_train:].tolist())
    assert not (set(train_ids) & set(test_ids))
    print(f"split: {len(train_ids)} train graphs / {len(test_ids)} test graphs")

    return Ccan, bits, col_idx, hashes, train_ids, test_ids, p, num_graphs


def convert_subset(set_name: str, ids: List[int], Ccan: np.ndarray, bits: np.ndarray,
                   col_idx: List[np.ndarray], config: DataProcessConfig):
    n = config.n
    seq_len = n * (n - 1) // 2

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for gi in ids:
        inp = encode_input(bits[gi])

        # One row per canonical proper coloring; test keeps only the first.
        # ``col_idx`` comes from ``np.nonzero``, so it is sorted and cs[0] is the
        # lexicographically first coloring.
        cs = col_idx[gi]
        if set_name == "test":
            cs = cs[:1]

        for ci in cs:
            results["inputs"].append(inp)
            results["labels"].append(encode_label(Ccan[ci], seq_len))
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    # To Numpy.  Tokens are already in vocab space (1/2 for the graph, 3/4/5 for
    # the colors, 0 = PAD/ignore in the label tail), so there is no +1 shift.
    results = {
        "inputs": np.vstack(results["inputs"]).astype(np.int32),
        "labels": np.vstack(results["labels"]).astype(np.int32),

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    assert results["inputs"].shape == results["labels"].shape
    assert set(np.unique(results["inputs"]).tolist()) <= {NO_EDGE, EDGE}
    assert set(np.unique(results["labels"]).tolist()) <= {0, 3, 4, 5}

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=6,  # PAD + no_edge + edge + 3 colors
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["inputs"]),
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    return results


def save_sidecar(Ccan, bits, col_idx, hashes, train_ids, test_ids, p, num_graphs,
                 config: DataProcessConfig):
    """Graph hash -> split, edge list, and every canonical proper coloring."""
    n = config.n
    iu, iv = np.triu_indices(n, 1)

    split_of = {gi: "train" for gi in train_ids}
    split_of.update({gi: "test" for gi in test_ids})

    sidecar = {"meta": {"n": n, "p": p, "num_graphs": num_graphs, "seed": config.seed,
                        "encoding": "upper_triangle", "canonical": "first_appearance",
                        "vocab": {"pad": 0, "no_edge": 1, "edge": 2,
                                  "red": 3, "blue": 4, "green": 5},
                        "hash": "sha1(upper-triangle token bytes).hexdigest()[:16]"},
               "graphs": {}}
    for gi in range(num_graphs):
        e = np.nonzero(bits[gi])[0]
        sidecar["graphs"][hashes[gi]] = {
            "split": split_of[gi],
            "edges": [[int(iu[k]), int(iv[k])] for k in e],
            "colorings": [Ccan[ci].tolist() for ci in col_idx[gi]],
        }

    with open(os.path.join(config.output_dir, f"sidecar_gc_{n}.json"), "w") as f:
        json.dump(sidecar, f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    # Unlike the other builders, both splits come from one sampling pass: the
    # split is over unique graphs, so it cannot be decided per-subset.
    Ccan, bits, col_idx, hashes, train_ids, test_ids, p, num_graphs = build_pairs(config)

    tr = convert_subset("train", train_ids, Ccan, bits, col_idx, config)
    te = convert_subset("test", test_ids, Ccan, bits, col_idx, config)

    tr_rows, te_rows = len(tr["inputs"]), len(te["inputs"])
    print(f"train rows={tr_rows} groups={len(tr['group_indices'])-1}   "
          f"test rows={te_rows} groups={len(te['group_indices'])-1}")
    anchor = SIZE_ANCHORS.get(config.n)
    if anchor is not None:
        print(f"[anchor] train pairs actual={tr_rows} expected={anchor[0]}  "
              f"({tr_rows-anchor[0]:+d})")
        print(f"[anchor] test  pairs actual={te_rows} expected={anchor[1]}  "
              f"({te_rows-anchor[1]:+d})")

    save_sidecar(Ccan, bits, col_idx, hashes, train_ids, test_ids, p, num_graphs, config)

    if config.self_check:
        run_self_check(config.n, tr["inputs"], tr["labels"], te["inputs"], te["labels"],
                       config.seed, config.check_samples)


def run_self_check(n, tr_x, tr_y, te_x, te_y, seed, samples):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluators.graph_coloring import verify_colorings  # noqa: E402

    print("\n" + "=" * 72)
    print(f"ACCEPTANCE SELF-CHECK  (graph coloring, n={n})")
    print("=" * 72)

    for name, x, y in (("train", tr_x, tr_y), ("test", te_x, te_y)):
        r = verify_colorings(y.astype(np.int64), x.astype(np.int64), n)
        print(f"[labels {name:5}] valid={r['valid'].mean():.6f}  "
              f"conflicts total={int(r['conflicts'].sum())}  "
              f"colors_ok={r['colors_ok'].mean():.6f}  rows={len(y)}")
        assert r["valid"].all(), f"{name} label is not a proper coloring"
        assert r["conflicts"].sum() == 0, f"{name} label has conflicting edges"

    def keyset(a):
        return set(map(bytes, np.ascontiguousarray(a.astype(np.int8))))
    ktr, kte = keyset(tr_x), keyset(te_x)
    print(f"[split ] distinct train x={len(ktr)}  test x={len(kte)}  "
          f"intersection={len(ktr & kte)}")
    assert not (ktr & kte), "train/test graph overlap -- leakage!"

    ptr = keyset(np.concatenate([tr_x, tr_y], 1))
    pte = keyset(np.concatenate([te_x, te_y], 1))
    print(f"[dup   ] train duplicate rows={len(tr_x)-len(ptr)}  test duplicate rows={len(te_x)-len(pte)}")
    assert len(ptr) == len(tr_x) and len(pte) == len(te_x), "duplicate (x,y) rows"

    # corrupting one node's color into a neighbor's color must break validity
    rng = np.random.default_rng(seed + 999)
    iu, iv = np.triu_indices(n, 1)
    m = min(samples, len(tr_x))
    sel = rng.choice(len(tr_x), size=m, replace=False)
    bad = tr_y[sel].copy()
    nbad = 0
    for r_i, row in enumerate(sel):
        e = np.nonzero(tr_x[row] == EDGE)[0]
        if len(e) == 0:
            continue
        k = e[rng.integers(len(e))]
        u, v = iu[k], iv[k]
        bad[r_i, u] = bad[r_i, v]          # force a conflict on edge (u,v)
        nbad += 1
    r = verify_colorings(bad.astype(np.int64), tr_x[sel].astype(np.int64), n)
    print(f"[negative] forced one conflicting edge on {nbad} rows -> "
          f"valid={r['valid'].sum()} (expect 0)  mean conflicts={r['conflicts'].mean():.3f}")
    assert r["valid"].sum() == 0, "verifier accepted a conflicting coloring"
    print("=" * 72)
    print("SELF-CHECK PASSED")


if __name__ == "__main__":
    cli()
