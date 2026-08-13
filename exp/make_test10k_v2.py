"""test10k-v2 생성 (eval 전용 10k test split)."""
import json
import os
import shutil

import numpy as np

SRC = "data/sudoku-extreme-1k-aug-1000"
DST = "data/sudoku-extreme-1k-aug-1000-test10k-v2"
K = 10000
SEED = 42


def main():
    assert not os.path.exists(DST), f"{DST} already exists"
    shutil.copytree(SRC, DST)

    d = os.path.join(DST, "test")

    inputs = np.load(os.path.join(d, "all__inputs.npy"))
    n = inputs.shape[0]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)[:K]

    for f in ["all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy"]:
        p = os.path.join(d, f)
        a = np.load(p)
        assert a.shape[0] == n, f"{f}: rows {a.shape[0]} != {n}"
        np.save(p, a[idx])

    for f in ["all__puzzle_indices.npy", "all__group_indices.npy"]:
        p = os.path.join(d, f)
        orig = np.load(p)
        assert np.array_equal(orig, np.arange(n + 1, dtype=orig.dtype)), (
            f"{f}: 원본이 arange(N+1) 오프셋이 아니다 — 구조 재확인 필요")
        np.save(p, np.arange(K + 1, dtype=orig.dtype))

    jp = os.path.join(d, "dataset.json")
    meta = json.load(open(jp))
    meta["total_groups"] = K
    meta["total_puzzles"] = K
    with open(jp, "w") as fh:
        json.dump(meta, fh)

    print(f"[done] {DST}  test rows {n} -> {K}")


if __name__ == "__main__":
    main()
