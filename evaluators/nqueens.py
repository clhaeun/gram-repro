"""N-Queens constraint evaluator.

GRAM 논문 Table 1의 N-Queens "Accuracy"는 라벨 일치가 아니라 **제약 만족** 기준이다.
같은 input에 여러 개의 완성해가 대응하므로(multi-solution), in-loop의
exact_accuracy(라벨 일치)는 구조적으로 0에 가깝고 논문 수치와 대응하지 않는다.

이 evaluator는 eval 배치마다 예측 보드를 규칙으로 검증해 논문과 비교 가능한
single-sample accuracy를 찍는다.

  valid      : queen이 정확히 n개 + row/col/두 대각선에 중복 없음
  consistent : 입력에 주어진 queen을 모두 보존
  strict     : valid & consistent   <- 헤드라인

consistent는 논문에 명시되지 않은 조건이다. 이것이 없으면 입력을 무시하고
아무 valid 보드나 출력해도 만점이 되므로, strict를 헤드라인으로 쓰되
valid도 함께 로깅해 정의 차이를 사후에 분리할 수 있게 한다.

Coverage(Table 1)는 한 input당 N개 샘플이 필요하므로 여기서 계산하지 않는다.
in-loop eval은 input당 1샘플이다. coverage는 experiments/eval_harness.py 담당.

vocab: 0=pad, 1=empty, 2=queen

사용법 (launch 커맨드):
    evaluators="[nqueens@NQueens]"
"""
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

QUEEN = 2


# 학습 시 제공한 정답지와 얼마나 일치하는지 x
# 체스 룰을 제대로 지켰는지 rule-based로 처리
def verify_boards(preds: np.ndarray, inputs: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    """preds/inputs: [B, n*n] int. 반환: 각 [B] bool."""
    B = preds.shape[0]
    pq = preds.reshape(B, n, n) == QUEEN
    iq = inputs.reshape(B, n, n) == QUEEN

    count_ok = pq.reshape(B, -1).sum(-1) == n

    rows = np.arange(n)[None, :, None].repeat(B, 0)
    cols = np.arange(n)[None, None, :].repeat(B, 0)

    def _no_dup(index_grid: np.ndarray, num_bins: int) -> np.ndarray:
        """queen이 놓인 칸들의 index가 서로 겹치지 않는지."""
        out = np.ones(B, dtype=bool)
        for b in range(B):
            idx = index_grid[b][pq[b]]
            out[b] = len(np.unique(idx)) == len(idx)
        return out

    row_ok = _no_dup(np.broadcast_to(rows, (B, n, n)), n)
    col_ok = _no_dup(np.broadcast_to(cols, (B, n, n)), n)
    d1_ok = _no_dup(np.broadcast_to(rows - cols, (B, n, n)), 2 * n)
    d2_ok = _no_dup(np.broadcast_to(rows + cols, (B, n, n)), 2 * n)

    valid = count_ok & row_ok & col_ok & d1_ok & d2_ok
    consistent = (~iq | pq).reshape(B, -1).all(-1)  # 입력 queen ⊆ 출력 queen

    return {
        "valid": valid,
        "consistent": consistent,
        "strict": valid & consistent,
        "queen_count": pq.reshape(B, -1).sum(-1),
        # 실패 사유 분해용 하위 제약 (Task 5). 전부 [B] bool.
        "count_ok": count_ok,
        "row_ok": row_ok,
        "col_ok": col_ok,
        "d1_ok": d1_ok,
        "d2_ok": d2_ok,
    }


class NQueens:
    required_outputs = {"inputs", "preds"}

    def __init__(self, data_path: str, eval_metadata: Any, n: Optional[int] = None, **kwargs):
        seq_len = getattr(eval_metadata, "seq_len", None)
        if n is None:
            if seq_len is None:
                raise ValueError("board size n을 알 수 없다. evaluator 인자로 n을 넘겨라.")
            n = int(round(seq_len ** 0.5))
            if n * n != seq_len:
                raise ValueError(f"seq_len={seq_len}이 제곱수가 아니다. n을 명시해라.")
        self.n = n
        self._acc = None

    def begin_eval(self):
        # [n_total, n_valid, n_consistent, n_strict, queen_sum]
        self._acc = np.zeros(5, dtype=np.float64)

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        if self._acc is None:
            self.begin_eval()

        p = preds["preds"].detach().cpu().numpy()
        x = batch["inputs"].detach().cpu().numpy()

        # padding으로 채워진 행 제외 (마지막 배치)
        keep = (x != 0).any(-1)
        if not keep.any():
            return
        p, x = p[keep], x[keep]

        r = verify_boards(p, x, self.n)
        self._acc += np.array([
            len(p),
            r["valid"].sum(),
            r["consistent"].sum(),
            r["strict"].sum(),
            r["queen_count"].sum(),
        ], dtype=np.float64)

    def result(self, save_path, rank, world_size, group=None):
        acc = self._acc if self._acc is not None else np.zeros(5)
        if world_size > 1:
            t = torch.tensor(acc, dtype=torch.float64)
            dist.reduce(t, dst=0, op=dist.ReduceOp.SUM, group=group)
            acc = t.numpy()
        if rank != 0:
            return None
        total = max(acc[0], 1.0)
        return {
            # 논문 Table 1 "Accuracy" 대응: 제약 만족만, input 보존 안 봄
            "nqueens/accuracy": float(acc[1] / total),
            # 이하 우리 자체 진단 (논문에 없는 지표)
            "nqueens/strict_accuracy": float(acc[3] / total),
            "nqueens/consistent_rate": float(acc[2] / total),
            "nqueens/queen_count_mean": float(acc[4] / total),
            "nqueens/n_eval": float(acc[0]),
        }