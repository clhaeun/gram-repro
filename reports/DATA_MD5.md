# 데이터셋 md5 기준값 대장

★ **이 파일의 존재 이유**: 62번에서 nqueens-10 데이터를 검산하려 했으나
**기준값이 repo 어디에도 기록돼 있지 않아 대조가 불가능했다.**
같은 사고를 막기 위해 여기에 서버별 실측값을 모은다.
새 서버에서 데이터를 쓰기 전 이 표와 대조하라.

## 검산 명령 (모든 서버 공통)

```bash
cd ~/gram-repro && python3 -c "
import numpy as np, hashlib, json
for sp in ['train','test']:
    a=np.load(f'data/nqueens-10/{sp}/all__inputs.npy')
    m=json.load(open(f'data/nqueens-10/{sp}/dataset.json'))
    print(sp, a.shape, m['total_groups'], hashlib.md5(a.tobytes()).hexdigest()[:12])
"
```

md5 는 `all__inputs.npy` 의 `tobytes()` 앞 12자리다.

## nqueens-10

| 서버 | split | shape | total_groups | md5(12) | 확인일 |
|---|---|---|---|---|---|
| **62번** (jurgen) | train | (107480, 100) | 107480 | `703026643e60` | 2026-08-06 |
| **62번** (jurgen) | test  | (19220, 100)  | 19220  | `9b6a03fe8c8a` | 2026-08-06 |
| **50번** | train | ❓ 미기록 | ❓ | ❓ **대조 필요** | — |
| **50번** | test  | ❓ 미기록 | ❓ | ❓ **대조 필요** | — |

62번 데이터 파일 타임스탬프: `2026-08-02 13:12` (이후 변경 없음, 재빌드 안 함).

### 상태
- shape 와 `total_groups` 는 **기대값과 정확히 일치** → 62번은 이 값으로 진행 승인됨(62번 지시).
- ★ **50번 md5 미대조.** 위 명령을 50번에서 실행해 이 표를 채울 것.
  일치하면 그대로 확정, 불일치하면 즉시 보고 — 두 서버 결과의 직접 비교가 무효가 된다.

## nqueens-8

| 서버 | split | shape | total_groups | md5(12) | 확인일 |
|---|---|---|---|---|---|
| — | — | 미기록 | — | — | — |

★ nq8 도 같은 문제가 있다. 기준값이 없다. 여유 생기면 채울 것.

## graph-coloring-8 / graph-coloring-10

측정: 50번(lecun), 2026-08-08. 파일 mtime — gc8 `2026-07-29 14:39`, gc10 `2026-07-27 14:30`
(이후 변경 없음, 재빌드 안 함). 빌더 `dataset/build_graph_coloring_dataset.py`
기본값 n/seed=0/p/num_graphs/train_frac=0.85 (62-70행), 난수는 전부 `default_rng(seed)`.

### all__inputs.npy — 위 검산 명령과 같은 눈금 (`tobytes()` md5 앞 12자리)

| 서버 | dataset/split | shape | total_groups | total_puzzles | md5(12) | 확인일 |
|---|---|---|---|---|---|---|
| **50번** | gc8/train | (6996, 28) | 1445 | 6996 | `3952113f6a83` | 2026-08-08 |
| **50번** | gc8/test | (255, 28) | 255 | 255 | `9fca612c9cc5` | 2026-08-08 |
| **50번** | gc10/train | (13461, 45) | 1088 | 13461 | `085538a60f49` | 2026-08-08 |
| **50번** | gc10/test | (192, 45) | 192 | 192 | `79fe0dad7d33` | 2026-08-08 |
| **34번** | 전 split | ❓ 미기록 | ❓ | ❓ | ❓ **전송 후 대조 필요** | — |

★ gc8/gc10 은 `total_groups < total_puzzles` 다 (그룹당 각각 4.842행 / 12.372행).
nqueens 는 pair 확장 후 `total_groups == total_puzzles` 다. 두 값이 다른 것 자체는
데이터 손상이 아니라 빌더 차이다 (`build_graph_coloring_dataset.py:227` 은 그래프 단위,
`build_nqueens_dataset.py:180` 은 행 단위로 group 을 push 한다).

### 나머지 npy — 전송 검산용 (같은 `tobytes()` 눈금)

| 파일 | gc8 | gc10 |
|---|---|---|
| train/all__labels.npy | `0d9a0eea7f49` | `25038f03a3ec` |
| train/all__puzzle_indices.npy | `ea9c2e90b804` | `d9f8310d2a2e` |
| train/all__group_indices.npy | `ecac91b3356b` | `977c43c632a2` |
| train/all__puzzle_identifiers.npy | `c42b40bd1e94` | `bbaafa92a4d7` |
| test/all__labels.npy | `60d868dd6fde` | `f4ca1ff5dd44` |
| test/all__puzzle_indices.npy | `bb45a329f7bb` | `0cdec98a5f9f` |
| test/all__group_indices.npy | `bb45a329f7bb` | `0cdec98a5f9f` |
| test/all__puzzle_identifiers.npy | `a6adb26ddc7d` | `33c250bf306b` |

(test 의 puzzle_indices 와 group_indices 는 값이 같아 md5 도 같다 — 그룹당 1행이라 정상.)

### 파일 단위 md5 (`md5sum`, 전송 무결성용)

```
c5739f360ea00fdfcff690b61c83d719  data/graph-coloring-8/train/all__inputs.npy
8933bc39b0035d9587e58b0b8e9d2145  data/graph-coloring-8/train/all__labels.npy
98c6447a1a115503659817c7f70b524f  data/graph-coloring-8/train/all__puzzle_indices.npy
ba7db9903e609836e61b3216c4859de7  data/graph-coloring-8/train/all__group_indices.npy
c7bf52f0cfffa300524227d2f48af4d8  data/graph-coloring-8/train/all__puzzle_identifiers.npy
8baeaee5e339aa2fb4e21990469cd5fe  data/graph-coloring-8/train/dataset.json
135a6b0954030e5c1de7f74bd130e9c6  data/graph-coloring-8/test/all__inputs.npy
84c165e30c7cf3b312713e673bdcda65  data/graph-coloring-8/test/all__labels.npy
e6f9e37dfcb07ec2feef916fa9ce819e  data/graph-coloring-8/test/all__puzzle_indices.npy
e6f9e37dfcb07ec2feef916fa9ce819e  data/graph-coloring-8/test/all__group_indices.npy
3c8c5b72f9242c0c47230dc825bdfbdc  data/graph-coloring-8/test/all__puzzle_identifiers.npy
5759e4833394a0bd523419447d95e110  data/graph-coloring-8/test/dataset.json
d283417ab34dc25608e4b7a91f8c1a41  data/graph-coloring-8/identifiers.json
61824301cd028eb177937aeb7eb482a6  data/graph-coloring-8/sidecar_gc_8.json
58cf962121cc8e6580b250e658d055fa  data/graph-coloring-10/train/all__inputs.npy
7b29231c3fbdd5c0e38bdf5053f7e9dd  data/graph-coloring-10/train/all__labels.npy
57857f6153bcbbfee816eb3d0c04be11  data/graph-coloring-10/train/all__puzzle_indices.npy
b00648f3c820621693ef598c91917775  data/graph-coloring-10/train/all__group_indices.npy
64cc634ebb45201a830123b3b0195b85  data/graph-coloring-10/train/all__puzzle_identifiers.npy
c69c02e7bd19bf29b39f904a235b3e82  data/graph-coloring-10/train/dataset.json
22e156728ddfc93e78eff414dd3c42ab  data/graph-coloring-10/test/all__inputs.npy
3165200cc782d7f86ea363035f11a99f  data/graph-coloring-10/test/all__labels.npy
7e6b7f64a634e26b781404b1ebff932d  data/graph-coloring-10/test/all__puzzle_indices.npy
7e6b7f64a634e26b781404b1ebff932d  data/graph-coloring-10/test/all__group_indices.npy
b066f92e299d1a32f6102c2e029a67bc  data/graph-coloring-10/test/all__puzzle_identifiers.npy
8435a7d50e633cfbf7d2c1060dd4ede0  data/graph-coloring-10/test/dataset.json
d283417ab34dc25608e4b7a91f8c1a41  data/graph-coloring-10/identifiers.json
8794c71fe7aaf7e12029c5abd7d164d9  data/graph-coloring-10/sidecar_gc_10.json
```

### sidecar 항목 수

| 파일 | 크기 | graphs 항목 수 | meta |
|---|---|---|---|
| data/graph-coloring-8/sidecar_gc_8.json | 504,310 B | 1700 | `{"n":8,"p":0.5455,"num_graphs":1700,"seed":0}` |
| data/graph-coloring-10/sidecar_gc_10.json | 763,637 B | 1280 | `{"n":10,"p":0.425,"num_graphs":1280,"seed":0}` |
