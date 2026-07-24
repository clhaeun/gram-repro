# make_test10k.py — 한 번만 실행
import numpy as np, shutil, os
src = "data/sudoku-extreme-1k-aug-1000"
dst = "data/sudoku-extreme-1k-aug-1000-test10k"
shutil.copytree(src, dst)
d = os.path.join(dst, "test")
n, k = None, 10000
rng = np.random.default_rng(42)
for f in os.listdir(d):
    if f.endswith(".npy"):
        a = np.load(os.path.join(d, f))
        if n is None:
            n = a.shape[0]; idx = rng.permutation(n)[:k]
        np.save(os.path.join(d, f), a[idx])