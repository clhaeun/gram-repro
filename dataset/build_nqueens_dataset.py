"""Build an N-Queens dataset in the TRM/GRAM ``PuzzleDataset`` format.

The model gets a *partial* board (a subset of the queens of some complete
solution) and must output a complete legal placement keeping every given queen.
Tokens: 0 = PAD (unused), 1 = empty, 2 = queen.  ``seq_len = n * n``, row-major.

The train/test split is over unique inputs (decided BEFORE pair expansion), so
no input straddles the split.  Both splits keep every (input, solution) row —
one row per consistent solution, and each row is its own group.

``sidecar_{n}.json`` maps each input hash to its split, clues, and the FULL set
of solutions consistent with it -- ``experiments/eval_harness.py`` (coverage@K)
and ``experiments/diag/diag_bucket.py`` read it.  Solutions are column encodings:
``solution[r] = c``.  The hash is sha1 of the flat uint8 token board, [:16].
"""

from typing import Optional
from collections import defaultdict
import math
import os
import json
import hashlib
import itertools
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


SOLUTION_ANCHORS = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200}
# 논문 p19 값이다 (우리가 고른 값이 아니다): 8x8 은 k in {5,6,7}, 10x10 은 k in {7,8,9} 를
# 제거한다.  removals 를 바꾸면 해-개수 분포가 통째로 달라진다 -- 10x10 에서 단서 3 개짜리
# (k=7)가 C(10,7)=120 으로 전체 인스턴스의 92% 를 차지하고 평균 해 2.18 개인 반면,
# 단서 1 개짜리(k=9)는 100 개뿐인데 평균 해 72.4 개다.  coverage 를 논문과 맞대려면
# 이 값이 일치해야 한다.
DEFAULT_REMOVALS = {8: (5, 6, 7), 10: (7, 8, 9)}
PAIR_ANCHORS = {8: 8464, 10: 126700}


cli = ArgParser()


class DataProcessConfig(BaseModel):
    n: int = 8
    output_dir: str = "data/nqueens-8"
    seed: int = 0
    train_frac: float = 0.85

    removals: Optional[str] = None


def build_pairs(config: DataProcessConfig):
    """Enumerate solutions, generate (input, label) pairs, split by input."""
    n = config.n
    assert config.removals or n in DEFAULT_REMOVALS, f"no default removals for n={n}; pass --removals"
    removals = tuple(int(x) for x in config.removals.split(",")) if config.removals else DEFAULT_REMOVALS[n]

    # All complete solutions, by backtracking
    sols = []      # 정답 체스판을 tuple 형식(= 각 행의 열 번호들)으로 계속해서 append
    cols = [0] * n   # cols[행 번호] = 열 번호
    used_c, used_d, used_a = [False] * n, [False] * (2 * n - 1), [False] * (2 * n - 1)  # column, diagonal, anti-diagonal

    def place(r):
        # 모든 행 다 채움
        if r == n:
            sols.append(tuple(cols))
            return
        # 모든 열 순회
        for c in range(n):
            if not (used_c[c] or used_d[r - c + n - 1] or used_a[r + c]):  # r-c+n-1, r+c이
                used_c[c] = used_d[r - c + n - 1] = used_a[r + c] = True
                cols[r] = c   # sols에 채울 cols 값
                place(r + 1)
                used_c[c] = used_d[r - c + n - 1] = used_a[r + c] = False  # 재귀의 끝이 r==n이 아니라 여기로 돌아온거면 어떤 행에서 실패한거니까, 직전꺼까지 False로 잡아주기

    place(0)
    sols = np.array(sorted(sols), dtype=np.int32)  # 왜 int32로 설정하는지?
    assert sols.shape[0] == SOLUTION_ANCHORS.get(n, sols.shape[0]), f"solver bug: {sols.shape[0]} solutions for n={n}"

    # Every way of deleting k of the n queens -> the survivors are the clues
    # 0부터 n-1까지의 전체 행 중에서, 버릴 행 k개를 뽑는 모든 조합 / 버리고 남은 걸 오름차순 정리, tuple로 묶기
    keep_sets = [tuple(sorted(set(range(n)) - set(drop)))
                 for k in removals for drop in itertools.combinations(range(n), k)]
    assert keep_sets, f"removals={removals} produce no clue sets for n={n}"

    # input: {"발급된 해시값(문제)": ((행 1, 열 2), (행 4, 열 5)...), : ~ } 형태로 문제의 이름표(해시)와 퀸의 단서 좌표 묶음을 1:1로 매핑
    key_to_clues = {}
    # label: {"발급된 해시값(문제)": [label 14, label 55], : ~ }
    rows_by_key = defaultdict(list)

    # label(정답 보드) 하나씩 꺼내기
    for si, sol in enumerate(tqdm(sols, desc="Generating pairs")):
        # 살려둘 퀸들만 찍기
        for keep in keep_sets:
            # 살아남기로 한 행(r)에 대해서만 정답 보드의 열 위치(sol[r]) 가져옴 -> 살아남는 (행, 열)
            clues = tuple((r, int(sol[r])) for r in keep)

            # 1로 채운 빈 보드판
            board = np.ones(n * n, dtype=np.uint8)
            # 빈 보드판에 퀸(단서) 찍기 -> hash 뽑기 위해서
            for r, c in clues:
                board[r * n + c] = 2
            # 해시 발급
            h = hashlib.sha1(board.tobytes()).hexdigest()[:16]

            # key_to_clues 딕셔너리에 저장. 같은 해시에 다른 clues 가 오면 sha1 충돌이므로 즉시 중단
            assert key_to_clues.setdefault(h, clues) == clues, f"sha1 collision on {h}"
            # 해시값 - label 매칭
            rows_by_key[h].append(si)

    num_pairs = sum(len(v) for v in rows_by_key.values())
    assert num_pairs == PAIR_ANCHORS.get(n, num_pairs), f"pair anchor mismatch: {num_pairs}"

    # Split unique inputs 85:15
    # train/test 데이터
    unique_keys = sorted(key_to_clues)
    perm = np.random.default_rng(config.seed).permutation(len(unique_keys))
    # train_frac = 0.85
    n_train = int(round(config.train_frac * len(unique_keys)))
    train_keys = [unique_keys[i] for i in perm[:n_train]]
    test_keys = [unique_keys[i] for i in perm[n_train:]]

    return sols, removals, key_to_clues, rows_by_key, train_keys, test_keys


# hash와 그에 따른 좌표 튜플들을 받아서 input/labels를 만듬
def convert_subset(set_name: str, keys, sols, key_to_clues, rows_by_key, config: DataProcessConfig):
    n = config.n
    grid_size = (n, n)

    # Generate dataset
    # group_indices = 데이터 행렬에서 몇번째 row까지가 같은 input에 대응하는 label인지 / puzzle_indices: 걍 1씩 커짐.
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    # (문제-정답) 쌍 count
    puzzle_id = 0
    example_id = 0

    # 경계선
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    # 각 unique input(hash)
    for h in tqdm(sorted(keys), desc=f"Processing {set_name}"):
        # NxN 크기의 1차원 input 배열을 0으로 초기화
        inp = np.zeros(n * n, dtype=np.int32)
        # 단서 퀸이 있는 자리를 1로
        for r, c in key_to_clues[h]:
            inp[r * n + c] = 1

        # One row per consistent solution, in BOTH splits (no test truncation).
        # 왜: 해가 k개인 입력은 k행(같은 입력, 서로 다른 라벨)으로 저장한다.  GRAM 은
        # 같은 입력에 여러 정답이 오는 것을 posterior/prior 로 모델링하는 구조인데,
        # 입력당 1라벨이면 그 신호가 학습에 존재하지 않는다.  다해 입력일수록 행이
        # 많이 소실되어(해 93개 입력이면 92행) train 에서 실효 비중이 눌린다.
        # GC 빌더는 처음부터 이 확장을 하고 있다(그래프당 색칠 평균 4.82개).
        # sols 는 정렬된 해 목록이고 sidxs 도 정렬 — 행 순서는 결정론적이다.
        sidxs = sorted(rows_by_key[h])

        # test/train에 해당하는 hash만 뽑아옴
        for si in sidxs:
            # NxN 크기의 1차원 label 배열을 0으로 초기화
            lab = np.zeros(n * n, dtype=np.int32)
            for r, c in enumerate(sols[si]):
                lab[r * n + c] = 1

            results["inputs"].append(inp)
            results["labels"].append(lab)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            # 행 하나 = group 하나 (사수님 로그 "Saved train: 107494 groups, 107494
            # puzzles" 기준).  group 을 입력 단위로 두면 epoch 당 입력마다 해 1개만
            # 샘플되어 1 epoch = unique 입력 수가 되고, 행 단위면 매 epoch 모든
            # (입력, 해) 행을 본다 — step 수 계산(groups/batch)이 여기 걸려 있다.
            results["group_indices"].append(puzzle_id)

    # To Numpy, shifting raw {0, 1} to tokens {1, 2} so PAD id 0 stays unused
    # 0/1 -> 1/2로.
    results = {
        "inputs": np.vstack(results["inputs"]).astype(np.int32) + 1,
        "labels": np.vstack(results["labels"]).astype(np.int32) + 1,

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),  # type: ignore
        vocab_size=3,  # PAD + empty + queen
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


def save_sidecar(sols, removals, key_to_clues, rows_by_key, train_keys, test_keys, config: DataProcessConfig):
    """Input hash -> split, clues, and every solution consistent with it."""
    split_of = {h: "train" for h in train_keys}
    split_of.update({h: "test" for h in test_keys})
    assert len(split_of) == len(train_keys) + len(test_keys), "train/test overlap -- leakage!"

    sidecar = {
        "meta": {
            "n": config.n,
            "encoding": "columns",
            "removals": list(removals),
            "num_solutions_total": int(sols.shape[0]),
            "num_unique_inputs": len(key_to_clues),
            "hash": "sha1(flat token board uint8 bytes).hexdigest()[:16]",
        },
        "inputs": {
            h: {
                "split": split_of[h],
                "clues": [[int(r), int(c)] for r, c in key_to_clues[h]],
                "solutions": [[int(x) for x in sols[si]] for si in sorted(rows_by_key[h])],
            }
            for h in sorted(key_to_clues)
        },
    }

    with open(os.path.join(config.output_dir, f"sidecar_{config.n}.json"), "w") as f:
        json.dump(sidecar, f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    # Unlike the other builders, both splits come from one enumeration pass:
    # the split is over unique inputs, so it cannot be decided per-subset.
    sols, removals, key_to_clues, rows_by_key, train_keys, test_keys = build_pairs(config)

    convert_subset("train", train_keys, sols, key_to_clues, rows_by_key, config)
    convert_subset("test", test_keys, sols, key_to_clues, rows_by_key, config)
    save_sidecar(sols, removals, key_to_clues, rows_by_key, train_keys, test_keys, config)


if __name__ == "__main__":
    cli()
