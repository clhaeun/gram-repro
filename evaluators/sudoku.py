"""Sudoku 제약 평가기."""
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

BLANK = 1
DIGIT_MIN, DIGIT_MAX = 2, 10
N = 9
SEQ_LEN = N * N


def _group_index() -> np.ndarray:
    idx = np.arange(SEQ_LEN).reshape(N, N)
    rows = idx                                              # [9, 9]
    cols = idx.T                                            # [9, 9]
    boxes = (idx.reshape(3, 3, 3, 3)                        # [3,3,3,3]
                .transpose(0, 2, 1, 3)
                .reshape(9, 9))                             # [9, 9]
    return np.concatenate([rows, cols, boxes], axis=0)      # [27, 9]


GROUPS = _group_index()


def verify_boards(preds: np.ndarray, inputs: np.ndarray) -> Dict[str, np.ndarray]:
    preds = np.asarray(preds)
    inputs = np.asarray(inputs)
    assert preds.shape[-1] == SEQ_LEN, f"seq_len {preds.shape[-1]} != 81"

    digits_ok = ((preds >= DIGIT_MIN) & (preds <= DIGIT_MAX)).all(-1)       # [B]

    g = preds[:, GROUPS]                                                    # [B, 27, 9]
    in_range = (g >= DIGIT_MIN) & (g <= DIGIT_MAX)
    shifted = np.where(in_range, g - DIGIT_MIN, N)
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
    return (np.asarray(preds) == np.asarray(labels)).all(-1)


class Sudoku:

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
