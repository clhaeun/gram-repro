"""test10k-v2 생성 — eval 전용 10k test split.

exp/make_test10k.py 의 결함 재현 방지판.  구 스크립트는 test/ 의 .npy 를 전부
같은 ``a[idx]`` 로 골라서 puzzle_indices / group_indices 까지 "원본 행 번호"로
덮어썼고 (오프셋 배열이 아니게 된다), dataset.json 은 .npy 가 아니라서
total_groups 422786 이 그대로 남았다.

여기서는
  · train/ 은 손대지 않고 그대로 복사한다 (학습은 원본과 비트 동일이어야 한다).
  · test/ 만 처리하며, 행 데이터(inputs/labels/puzzle_identifiers)에만 idx 를
    적용하고 puzzle_indices / group_indices 는 **원본 test 의 오프셋 구조를 그대로
    따라** 새 행 수로 다시 만든다.
    원본 test 실측: puzzle_indices == group_indices == arange(N+1), diff 전부 1
    (그래프 하나 = 퍼즐 하나 = 예제 하나).  따라서 v2 는 arange(10001).
  · dataset.json 의 total_groups / total_puzzles 만 10000 으로 고치고 나머지 키는
    보존한다.

seed 42 는 과거 run 과 같은 10k 를 유지하기 위한 것이다 — 바꾸지 말 것.
"""
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

    # 행 데이터만 idx 로 고른다.
    for f in ["all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy"]:
        p = os.path.join(d, f)
        a = np.load(p)
        assert a.shape[0] == n, f"{f}: rows {a.shape[0]} != {n}"
        np.save(p, a[idx])

    # 오프셋 배열은 원본 구조(arange(N+1))를 새 행 수로 재생성한다.
    for f in ["all__puzzle_indices.npy", "all__group_indices.npy"]:
        p = os.path.join(d, f)
        orig = np.load(p)
        assert np.array_equal(orig, np.arange(n + 1, dtype=orig.dtype)), (
            f"{f}: 원본이 arange(N+1) 오프셋이 아니다 — 구조 재확인 필요")
        np.save(p, np.arange(K + 1, dtype=orig.dtype))

    # dataset.json 은 그룹/퍼즐 수만 갱신, 나머지 키 보존.
    jp = os.path.join(d, "dataset.json")
    meta = json.load(open(jp))
    meta["total_groups"] = K
    meta["total_puzzles"] = K
    with open(jp, "w") as fh:
        json.dump(meta, fh)

    print(f"[done] {DST}  test rows {n} -> {K}")


if __name__ == "__main__":
    main()
