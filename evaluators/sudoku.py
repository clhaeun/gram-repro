"""Sudoku 제약 평가기.

`evaluators/nqueens.py` · `evaluators/graph_coloring.py` 와 같은 철학이다:
채점은 라벨 일치가 아니라 **제약 충족**으로 한다.  다만 Sudoku-Extreme 은
퍼즐당 해가 유일하므로 (제약 충족 + 단서 보존) ⇔ (라벨 일치) 이고, 그래서
`exact` 와 `valid` 가 원리상 같은 값이어야 한다.  두 축을 따로 내는 이유는
**불일치가 곧 버그 신호**이기 때문이다 — 어긋나면 인코딩이나 채점기를 의심한다.

★ `consistent`(단서 보존)를 빼면 안 된다.  행/열/박스 제약만 보면 "완전히 유효한
  스도쿠지만 주어진 문제와 다른 판" 이 통과한다.  N-Queens 의 clue 보존과 같은
  자리의 검사다 (GC 는 입력이 그래프 자체라 이 개념이 없다).

인코딩 (`dataset/build_sudoku_dataset.py`, dataset.json vocab_size=11):
    inputs : 81칸 row-major.  1 = blank, 2..10 = 단서 숫자 1..9
    labels : 81칸 완성 해.     2..10 = 숫자 1..9
    0 = pad/ignore (Sudoku 데이터에는 실제로 등장하지 않는다 -- 실측 확인)

반환 축
    valid       : 행·열·3x3 박스가 전부 1..9 순열   AND  digits_ok
    consistent  : ★단서 칸이 전부 보존됨★ (inputs != BLANK 인 칸에서 pred == input)
    solved      : valid AND consistent            <- 논문의 "accuracy" 에 해당
    exact       : pred == label 전 칸 일치         <- 라벨 일치 (해 유일성 때문에 solved 와 같아야)
    digits_ok   : 81칸 전부 2..10 로 디코딩됨
    violations  : 행 27개 그룹(행9 + 열9 + 박스9) 중 1..9 순열이 아닌 그룹 수
    clue_breaks : 단서 칸 중 값이 바뀐 칸 수
"""
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

BLANK = 1
DIGIT_MIN, DIGIT_MAX = 2, 10          # 토큰 2..10 = 숫자 1..9
N = 9
SEQ_LEN = N * N


def _group_index() -> np.ndarray:
    """[27, 9] -- 행 9 + 열 9 + 박스 9 각각이 담는 81칸 중 9개의 평면 인덱스."""
    idx = np.arange(SEQ_LEN).reshape(N, N)
    rows = idx                                              # [9, 9]
    cols = idx.T                                            # [9, 9]
    boxes = (idx.reshape(3, 3, 3, 3)                        # [3,3,3,3]
                .transpose(0, 2, 1, 3)                      # (박스행, 박스열, 칸행, 칸열)
                .reshape(9, 9))                             # [9, 9]
    return np.concatenate([rows, cols, boxes], axis=0)      # [27, 9]


GROUPS = _group_index()


def verify_boards(preds: np.ndarray, inputs: np.ndarray) -> Dict[str, np.ndarray]:
    """preds/inputs: [B, 81] int.  반환값은 전부 [B]."""
    preds = np.asarray(preds)
    inputs = np.asarray(inputs)
    assert preds.shape[-1] == SEQ_LEN, f"seq_len {preds.shape[-1]} != 81"

    digits_ok = ((preds >= DIGIT_MIN) & (preds <= DIGIT_MAX)).all(-1)       # [B]

    # 27개 그룹 각각이 1..9 순열인지.  범위 밖 토큰은 아래 bincount 에서 걸러지고
    # digits_ok 가 따로 잡으므로, 여기서는 유효 숫자만 세면 된다.
    g = preds[:, GROUPS]                                                    # [B, 27, 9]
    in_range = (g >= DIGIT_MIN) & (g <= DIGIT_MAX)
    # 각 그룹에서 숫자별 등장 횟수 -> 전부 1 이어야 순열
    shifted = np.where(in_range, g - DIGIT_MIN, N)                          # 범위 밖은 bin 9 로
    onehot = np.zeros(g.shape + (N + 1,), dtype=np.int16)                   # [B, 27, 9, 10]
    np.put_along_axis(onehot, shifted[..., None], 1, axis=-1)
    counts = onehot.sum(axis=2)[..., :N]                                    # [B, 27, 9]
    group_ok = (counts == 1).all(-1)                                        # [B, 27]
    violations = (~group_ok).sum(-1).astype(np.int64)                       # [B]
    valid = group_ok.all(-1) & digits_ok                                    # [B]

    clue = inputs != BLANK                                                  # [B, 81]
    clue_breaks = ((preds != inputs) & clue).sum(-1).astype(np.int64)       # [B]
    consistent = clue_breaks == 0                                           # [B]

    return {
        "valid": valid,
        "consistent": consistent,
        "solved": valid & consistent,
        "digits_ok": digits_ok,
        "violations": violations,
        "clue_breaks": clue_breaks,
        "n_clues": clue.sum(-1).astype(np.int64),
    }


def exact_match(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """[B] bool -- 81칸 전부 라벨과 일치."""
    return (np.asarray(preds) == np.asarray(labels)).all(-1)


class Sudoku:
    """in-loop evaluator.  사용법:  evaluators="[sudoku@Sudoku]"

    학습 launch 에는 붙이지 않았다 (sudoku_paper_34 는 evaluators=[] 로 기동됨).
    붙이면 in-loop eval 에 sudoku/* 축이 추가된다.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum: Dict[str, float] = {}
        self._n = 0

    def update(self, batch: Dict[str, torch.Tensor], preds: torch.Tensor,
               **kwargs: Any) -> None:
        p = preds.detach().cpu().numpy()
        x = batch["inputs"].detach().cpu().numpy()
        y = batch["labels"].detach().cpu().numpy()
        # pad 행(= inputs 전부 0)은 제외한다.  _collate_batch 가 pad_id 로 채운다.
        keep = (x != 0).any(-1)
        if not keep.any():
            return
        p, x, y = p[keep], x[keep], y[keep]
        r = verify_boards(p, x)
        acc = {
            "valid": float(r["valid"].mean()),
            "consistent": float(r["consistent"].mean()),
            "solved": float(r["solved"].mean()),
            "exact": float(exact_match(p, y).mean()),
            "violations_mean": float(r["violations"].mean()),
            "clue_breaks_mean": float(r["clue_breaks"].mean()),
        }
        b = len(p)
        for k, v in acc.items():
            self._sum[k] = self._sum.get(k, 0.0) + v * b
        self._n += b

    def compute(self) -> Optional[Dict[str, float]]:
        n = self._n
        out = dict(self._sum)
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([n] + [out[k] for k in sorted(out)], dtype=torch.float64,
                             device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            vals = t.cpu().numpy()
            n = float(vals[0])
            out = {k: float(v) for k, v in zip(sorted(out), vals[1:])}
        if not n:
            return None
        res = {f"sudoku/{k}": v / n for k, v in out.items()}
        res["sudoku/n_eval"] = float(n)
        return res
