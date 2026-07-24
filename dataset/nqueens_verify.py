"""Fully vectorized N-Queens verifier.

Token vocabulary (matches the dataset builder / PuzzleDataset convention):

    0 = PAD
    1 = empty cell
    2 = queen

Boards are flat row-major arrays of shape ``[B, n*n]`` (``[B, n, n]`` is also
accepted and reshaped).  Cell ``(r, c)`` lives at flat index ``r * n + c``.

The three metrics returned by :func:`verify`:

* ``valid``      -- the board is a complete, legal n-queens placement:
                    exactly ``n`` queens, no two sharing a row, a column,
                    a diagonal (r - c) or an anti-diagonal (r + c).
* ``consistent`` -- every queen given in ``inputs`` (the partial clue board)
                    is still a queen in ``boards``; i.e. input queens are a
                    subset of output queens.  Nothing is required of the
                    input's empty cells, which the model is free to fill.
* ``strict``     -- ``valid & consistent``.  This is the headline metric.

All reductions are performed over the batch axis with ``np.bincount``; there is
no Python-level loop over ``B``.
"""

from typing import Dict

import numpy as np


__all__ = ["verify", "boards_to_tokens", "tokens_to_columns"]


PAD_TOKEN = 0
EMPTY_TOKEN = 1
QUEEN_TOKEN = 2


def _as_flat(arr: np.ndarray, n: int) -> np.ndarray:
    """Coerce ``[B, n, n]`` or ``[B, n*n]`` (or a single unbatched board) to ``[B, n*n]``."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 3:
        assert arr.shape[1] == n and arr.shape[2] == n, (
            f"expected [B, {n}, {n}], got {arr.shape}"
        )
        arr = arr.reshape(arr.shape[0], n * n)
    assert arr.ndim == 2 and arr.shape[1] == n * n, (
        f"expected [B, {n * n}], got {arr.shape}"
    )
    return arr


def _no_collision(batch_idx: np.ndarray, key_idx: np.ndarray, num_keys: int, B: int) -> np.ndarray:
    """Return ``[B]`` bool: True where no key value is used more than once.

    ``batch_idx`` / ``key_idx`` are the (row, key) coordinates of every queen.
    Counting is done with a single flattened ``bincount``.
    """
    if B == 0:
        return np.zeros(0, dtype=bool)
    if batch_idx.size == 0:
        return np.ones(B, dtype=bool)

    flat = batch_idx.astype(np.int64) * num_keys + key_idx.astype(np.int64)
    counts = np.bincount(flat, minlength=B * num_keys).reshape(B, num_keys)
    return counts.max(axis=1) <= 1


def verify(boards: np.ndarray, inputs: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    """Verify a batch of candidate n-queens boards against their clue boards.

    Args:
        boards: ``[B, n*n]`` (or ``[B, n, n]``) integer tokens -- model output.
        inputs: ``[B, n*n]`` (or ``[B, n, n]``) integer tokens -- partial clues.
        n: board size.

    Returns:
        dict with ``valid``, ``consistent`` and ``strict``, each a ``[B]`` bool array.
    """
    boards = _as_flat(boards, n)
    inputs = _as_flat(inputs, n)
    assert boards.shape[0] == inputs.shape[0], "boards and inputs must share a batch size"

    B = boards.shape[0]
    if B == 0:
        empty = np.zeros(0, dtype=bool)
        return {"valid": empty, "consistent": empty.copy(), "strict": empty.copy()}

    queens = boards == QUEEN_TOKEN  # PAD (0) and empty (1) both count as "no queen"

    # --- exactly n queens ---------------------------------------------------
    count_ok = queens.sum(axis=1) == n

    # --- collision checks ---------------------------------------------------
    batch_idx, cell_idx = np.nonzero(queens)
    rows = cell_idx // n
    cols = cell_idx % n

    row_ok = _no_collision(batch_idx, rows, n, B)
    col_ok = _no_collision(batch_idx, cols, n, B)
    # r - c in [-(n-1), n-1]  ->  shift by n-1, giving 2n-1 buckets
    diag_ok = _no_collision(batch_idx, rows - cols + (n - 1), 2 * n - 1, B)
    # r + c in [0, 2n-2]
    anti_ok = _no_collision(batch_idx, rows + cols, 2 * n - 1, B)

    valid = count_ok & row_ok & col_ok & diag_ok & anti_ok

    # --- consistency with the clues ----------------------------------------
    # every input queen must still be a queen in the output
    violation = (inputs == QUEEN_TOKEN) & (boards != QUEEN_TOKEN)
    consistent = ~violation.any(axis=1)

    return {"valid": valid, "consistent": consistent, "strict": valid & consistent}


# ---------------------------------------------------------------------------
# Small helpers shared with the dataset builder / eval harnesses
# ---------------------------------------------------------------------------

def boards_to_tokens(columns, n: int) -> np.ndarray:
    """Column-index encoding -> flat token board.

    ``columns`` is ``[B, n]`` (or ``[n]``) where ``columns[b, r]`` is the column
    of the queen in row ``r``.  Returns ``[B, n*n]`` tokens with 1=empty, 2=queen.
    """
    columns = np.asarray(columns)
    single = columns.ndim == 1
    if single:
        columns = columns[None, :]
    B = columns.shape[0]
    out = np.full((B, n, n), EMPTY_TOKEN, dtype=np.int32)
    b = np.repeat(np.arange(B), n)
    r = np.tile(np.arange(n), B)
    out[b, r, columns.reshape(-1)] = QUEEN_TOKEN
    out = out.reshape(B, n * n)
    return out[0] if single else out


def tokens_to_columns(boards: np.ndarray, n: int) -> np.ndarray:
    """Flat token board -> ``[B, n]`` column indices.  Only meaningful for boards
    with exactly one queen per row (e.g. valid solutions)."""
    boards = _as_flat(boards, n).reshape(-1, n, n)
    return np.argmax(boards == QUEEN_TOKEN, axis=2)
