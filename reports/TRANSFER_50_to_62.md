# [50 → 62] Sudoku 데이터 + sudoku_paper_34 모델 전송

- 보내는 쪽: 50번 (`trm_chaeun` docker), repo `/workspace/TinyRecursiveModels`
- 받는 쪽: `chaeun@143.248.135.62`, repo 루트 `~/gram-repro`
- 이 컨테이너에 **rsync 없음** (`scp`/`ssh` 만 있음). rsync 를 쓰려면 아래 A-0 참조.
- 모델(111MB) 먼저, 데이터(1.3GB) 나중 — 62번이 격자를 먼저 시작할 수 있게 명령을 분리했다.
- 전송 대상에 `data/sudoku-extreme-1k-aug-1000-test10k-v2` 는 **포함하지 않는다**
  (62번이 `exp/make_test10k_v2.py` 로 로컬 생성, seed 42 고정).

---

## 0. 목적지 디렉토리 생성 (62번)

```bash
ssh chaeun@143.248.135.62 'mkdir -p \
  ~/gram-repro/data \
  ~/gram-repro/checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch'
```

## 1. 모델 전송 — 111MB, 먼저 보낸다

목적지(62번 repo 루트 기준):
`checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34/`

```bash
cd /workspace/TinyRecursiveModels
scp -rp checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34 \
  chaeun@143.248.135.62:~/gram-repro/checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/
```

## 2. 데이터 전송 — 1.28GiB

목적지(62번 repo 루트 기준): `data/sudoku-extreme-1k-aug-1000/`

```bash
cd /workspace/TinyRecursiveModels
scp -rp data/sudoku-extreme-1k-aug-1000 \
  chaeun@143.248.135.62:~/gram-repro/data/
```

### A-0. rsync 를 쓰고 싶다면 (중단 재개가 필요할 때)

```bash
# 50번 컨테이너에 설치 (root 로)
apt-get update && apt-get install -y rsync
# 이후 (as-chaeun 으로 실행)
cd /workspace/TinyRecursiveModels
as-chaeun rsync -avP --partial \
  checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34 \
  chaeun@143.248.135.62:~/gram-repro/checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/
as-chaeun rsync -avP --partial \
  data/sudoku-extreme-1k-aug-1000 \
  chaeun@143.248.135.62:~/gram-repro/data/
```

## 3. 62번에서 md5 검증

```bash
ssh chaeun@143.248.135.62 'cd ~/gram-repro && python3 - <<PY
import hashlib, json, os
import numpy as np
def fmd5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8<<20), b""): h.update(c)
    return h.hexdigest()
for root in ["data/sudoku-extreme-1k-aug-1000",
             "checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34"]:
    print("=====", root)
    for dp, _, ns in sorted(os.walk(root)):
        for n in sorted(ns):
            p = os.path.join(dp, n); rel = os.path.relpath(p, root)
            if n.endswith(".npy"):
                a = np.load(p)
                print(f"  {rel:38s} {os.path.getsize(p):>12,}  file={fmd5(p)}  tobytes12={hashlib.md5(a.tobytes()).hexdigest()[:12]}  shape={a.shape} {a.dtype}")
            else:
                print(f"  {rel:38s} {os.path.getsize(p):>12,}  file={fmd5(p)}")
PY'
```

---

## 50번 실측 기준값 (2026-08-08)

`tobytes12` = `md5(arr.tobytes())[:12]` — `reports/DATA_MD5.md` 관례.

### (a) `data/sudoku-extreme-1k-aug-1000/` — 1,378,874,503 bytes (1.28 GiB)

| 파일 | bytes | file md5 | tobytes12 | shape / dtype |
|---|---|---|---|---|
| `identifiers.json` | 11 | `d283417ab34dc25608e4b7a91f8c1a41` | — | — |
| `test/all__group_indices.npy` | 1,691,276 | `5b33a5c11e11fb9b5951055d9c98a630` | `da17db8f58aa` | (422787,) int32 |
| `test/all__inputs.npy` | 34,245,794 | `9285169ab9f7f6fbe264ef4635a1834f` | `c71fb061ea9f` | (422786, 81) uint8 |
| `test/all__labels.npy` | 34,245,794 | `4111b1b46b13ef867c18baaee822080f` | `3dacb0378e5a` | (422786, 81) uint8 |
| `test/all__puzzle_identifiers.npy` | 1,691,272 | `0988dbea3e5017c008fe149c3227f042` | `0d5808b83f7a` | (422786,) int32 |
| `test/all__puzzle_indices.npy` | 1,691,276 | `5b33a5c11e11fb9b5951055d9c98a630` | `da17db8f58aa` | (422787,) int32 |
| `test/dataset.json` | 218 | `a3134a1cb4afc8e07ca98afd88078a2b` | — | total_groups 422,786 |
| `train/all__group_indices.npy` | 4,132 | `92a43f8f94ac220de1dc68620fd93ed4` | `49e08ec68b01` | (1001,) int32 |
| `train/all__inputs.npy` | 648,648,128 | `9e73b7d98cf9fc407f2551eef8f9bdc8` | `ec629e0f6a1f` | (1001000, 81) int64 |
| `train/all__labels.npy` | 648,648,128 | `1d23f685dbacbf5c61fc33b489248c7b` | `0d6e12fee32c` | (1001000, 81) int64 |
| `train/all__puzzle_identifiers.npy` | 4,004,128 | `3172d6228e9ab1ef5daa5ac3ec7c2e55` | `d69ad371a1fa` | (1001000,) int32 |
| `train/all__puzzle_indices.npy` | 4,004,132 | `f0e6fe5f0eb556bd292517b36be6aad4` | `ce8f0fdea7d9` | (1001001,) int32 |
| `train/dataset.json` | 214 | `09d2756a700a460b84b094634f54dca1` | — | total_groups 1,000 |

`train` 은 1,001,000행 (`all__inputs.npy` shape), `test` 는 422,786행. 둘 다 포함되어 있다.

### (b) `checkpoints/.../sudoku_paper_34/` — 115,801,354 bytes (0.11 GiB)

| 파일 | bytes | file md5 |
|---|---|---|
| `all_config.yaml` | 1,235 | `be0f9f1c930d6c8f46a9006f98120947` |
| `gram.py` | 30,768 | `b801bb0deaa1cf5f03dcdcb2541e53c7` |
| `losses.py` | 5,315 | `d732d75b39ca95c67924a37adb775f13` |
| `step_65100` | 57,882,018 | `b050e90165c998abf79f55b3e27193c6` |
| `step_65100.live` | 57,882,018 | `4f6d93f31ed8bf917772f19e9b3fd9f2` |
