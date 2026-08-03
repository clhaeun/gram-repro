"""3-Graph-Coloring constraint evaluator.

Same philosophy as ``evaluators/nqueens.py``: the paper's Graph Coloring
"Accuracy" is **constraint satisfaction**, not label match.  A graph usually has
many proper 3-colorings, so in-loop ``exact_accuracy`` (label match) is
structurally near zero and does not correspond to the paper's number.

  valid      : every edge joins two different colors, and all n node positions
               actually carry a color token (3/4/5)
  conflicts  : number of edges whose endpoints share a color  (0 <=> valid,
               given colors_ok)
  colors_ok  : all n node positions decoded to a legal color token

Unlike N-Queens there is no "consistent" notion: the input is the graph itself,
not a partial answer, so there is nothing from the input to preserve in the
output.  ``valid`` is therefore the headline.

Encoding (see ``dataset/build_graph_coloring_dataset.py``)
    inputs : upper triangle of the adjacency matrix, ``np.triu_indices(n, 1)``
             order, tokens 1 = no edge, 2 = edge.  seq_len = n(n-1)/2.
    labels : node colors in the first n positions, tokens 3/4/5; the rest is 0
             (ignored by the loss).

vocab: 0=pad/ignore, 1=no edge, 2=edge, 3=red, 4=blue, 5=green

사용법 (launch 커맨드):
    evaluators="[graph_coloring@GraphColoring]"
"""
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

EDGE = 2
COLOR_MIN, COLOR_MAX = 3, 5


def n_from_seq_len(seq_len: int) -> int:
    # 시퀀스 길이 받아서 -> 노드 개수 반환
    # Invert seq_len = n(n-1)/2.
    n = int(round((1 + (1 + 8 * seq_len) ** 0.5) / 2))
    if n * (n - 1) // 2 != seq_len:
        raise ValueError(f"seq_len={seq_len}이 n(n-1)/2 꼴이 아니다. n을 명시해라.")
    return n


def verify_colorings(preds: np.ndarray, inputs: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    # preds/inputs: [B, n(n-1)/2] int. 채점결과: 각 [B]
    B = preds.shape[0]
    # 가능한 간선들의 idx (출발-도착)
    iu, iv = np.triu_indices(n, 1)

    # 노드에 칠해진 색들 n개
    colors = preds[:, :n]                                   # [B, n]
    # 주어진 색으로 칠해진 게 맞는지 
    colors_ok = ((colors >= COLOR_MIN) & (colors <= COLOR_MAX)).all(-1)

    # 실제로 해당 문제에서 연결된 부분
    edge = inputs == EDGE                                   # [B, P]
    # 동일 간선에 연결된 노드 2개의 색이 같은지
    same = colors[:, iu] == colors[:, iv]                   # [B, P]
    # 연결됨 & 색 동일 -> conflict!임
    conflicts = (edge & same).sum(-1)

    valid = colors_ok & (conflicts == 0)

    # 사용한 색 개수 (진단용): 3색을 다 쓰는지, 한 색으로 뭉갰는지
    num_colors = np.array([len(np.unique(c[(c >= COLOR_MIN) & (c <= COLOR_MAX)]))
                           for c in colors], dtype=np.int64)

    return {
        "valid": valid,
        "conflicts": conflicts,
        "colors_ok": colors_ok,
        "num_edges": edge.sum(-1),
        "num_colors_used": num_colors,
    }


class GraphColoring:
    required_outputs = {"inputs", "preds"}

    def __init__(self, data_path: str, eval_metadata: Any, n: Optional[int] = None, **kwargs):
        seq_len = getattr(eval_metadata, "seq_len", None)
        if n is None:
            if seq_len is None:
                raise ValueError("노드 수 n을 알 수 없다. evaluator 인자로 n을 넘겨라.")
            n = n_from_seq_len(int(seq_len))
        self.n = n
        self._acc = None

    def begin_eval(self):
        # [n_total, n_valid, n_colors_ok, conflict_sum, colors_used_sum]
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

        r = verify_colorings(p, x, self.n)
        self._acc += np.array([
            len(p),
            r["valid"].sum(),
            r["colors_ok"].sum(),
            r["conflicts"].sum(),
            r["num_colors_used"].sum(),
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
            # 논문 Table 1 "Accuracy" 대응: 제약 만족
            "gc/accuracy": float(acc[1] / total),
            # 이하 자체 진단
            "gc/colors_ok_rate": float(acc[2] / total),
            "gc/conflicts_mean": float(acc[3] / total),
            "gc/num_colors_used_mean": float(acc[4] / total),
            "gc/n_eval": float(acc[0]),
        }
