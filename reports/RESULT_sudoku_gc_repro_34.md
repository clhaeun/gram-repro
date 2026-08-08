# Sudoku / Graph-Coloring 재현 라운드 (34번) — 데이터 준비 + launch 기록

일시: 2026-08-08. 서버 34번 (`/home/chaeun/gram-repro`, RTX A6000 ×8).
지시: 34번 — LPRM 중단, Sudoku/GC 데이터 준비 → 재현 학습.

논문 규격 기준: GRAM Appendix B/C, Table 6·7.
β 눈금은 논문과 동일(sum)이다 — 논문 N-Queens 값 0.07(8×8)/0.045(10×10)이 우리
run 과 정확히 일치하므로 환산하지 않는다. Sudoku 0.1 / GC-8 0.5 / GC-10 **0.45**.

---

## ★ eval 축 구분 (필수 — 혼동 금지)

Sudoku run 은 **두 개의 test 축**을 쓴다. 섞어 인용하면 안 된다.

| 축 | 데이터 | 행 수 | 용도 |
|---|---|---|---|
| **in-loop eval** | `data/sudoku-extreme-1k-aug-1000-test10k-v2` | 10,000 | 학습 중 `eval_interval` 마다 도는 평가 **전용** |
| **최종 성능 harness** | `data/sudoku-extreme-1k-aug-1000` (원본 test) | **422,786** | 최종 성능 수치는 **여기서 별도 측정한다** |

train split 은 두 경로가 **동일**하다 (`md5(inputs)[:12] = ec629e0f6a1f`).
따라서 학습 자체는 원본 데이터 학습과 같고, 10k 는 in-loop eval 비용만 줄인다.
논문 대조표에 올릴 Sudoku 수치는 반드시 422,786 축으로 재측정한 값이어야 한다.

---

## 1. Sudoku 데이터 — train 재생성 금지, test10k 만 재생성

`dataset/build_sudoku_dataset.py` 는 seed 인자가 없고 전역 `np.random` 을 6곳
(`:29 :32 :36 :37 :40 :41` 증강 셔플, `:80` 서브샘플)에서 쓴다. 재생성하면 다른
데이터가 된다. **이 빌더는 고치지 않았다.**

GC 의 group 결함은 Sudoku 에 해당하지 않는다 — GC 의 group 당 다중 행은 서로
다른 해지만 Sudoku 는 같은 문제의 증강본이라 epoch 당 1개 샘플링이 표준이다
(실측: 과거 run 총 step 65,100 = 50000×1000/768).

### 1-1. 원본 검증 (`data/sudoku-extreme-1k-aug-1000`)

| 항목 | train | test |
|---|---|---|
| `total_groups` / `total_puzzles` | 1000 / 1000 | 422786 / 422786 |
| `vocab_size` / `seq_len` / `num_puzzle_identifiers` | 11 / 81 / 1 | 동일 |
| inputs / labels | (1001000, 81) int64 | (422786, 81) uint8 |
| `puzzle_indices` | (1001001,) int32 = `arange(N+1)` | (422787,) int32 = `arange(N+1)` |
| `group_indices` | (1001,) int32, diff 전부 **1001** | (422787,) int32 = `arange(N+1)` |
| `md5(inputs.tobytes())[:12]` | **`ec629e0f6a1f`** | **`c71fb061ea9f`** |

기대값 전부 일치. 오프셋 규약(0 시작, 길이 N+1) 준수.

### 1-2. 구 `test10k` 의 규약 위반 (29번·50번 실측 재확인)

| 파일 | 값 |
|---|---|
| `test/all__puzzle_indices.npy` | shape **(10000,)**, head5 `[258581 272999 55913 372697 348967]` |
| `test/all__group_indices.npy` | 위와 동일 내용 |
| `test/dataset.json` | `total_groups 422786` (미수정 잔존) |

오프셋 배열이 아니라 **원본 행 번호**다. 원인: `exp/make_test10k.py:9-14` 가
`for f in os.listdir(d)` 로 `.npy` 전부에 같은 `a[idx]` 를 적용하고, `dataset.json`
은 `.npy` 가 아니라 손대지 않는다.

참고 — `puzzle_dataset.py` 가 test split 에서 실제로 쓰는 것 (값만):
- test 경로는 `_iter_test` (`:168-199`). **`group_indices` 는 참조되지 않는다**
  (`group_indices` 참조는 `_iter_train` 의 `:209 :217` 뿐).
- `puzzle_indices` 는 `:183` `np.searchsorted(...)` 와 `:185-186` while 증가에만
  쓰이고, 그 결과는 `:193` `dataset["puzzle_identifiers"][puzzle_indices]` 인덱싱
  에만 쓰인다. 행 슬라이스는 `:191-192` 의 순차 인덱스다.
- Sudoku `all__puzzle_identifiers.npy` 는 전 원소 0 (`num_puzzle_identifiers=1`).
- `metadata.total_groups` 는 `pretrain.py:256` 에서 **train metadata 만** 읽는다.

### 1-3. `test10k-v2` 재생성

신규 스크립트 `exp/make_test10k_v2.py` (구 `exp/make_test10k.py` 는 미수정).
`rng = np.random.default_rng(42); idx = rng.permutation(422786)[:10000]` — seed 42
유지라 과거 run 과 같은 10k 다.

| 검증 항목 | 결과 |
|---|---|
| `test/all__inputs.npy` md5_12 | **`55016ae27206`** (기대값 일치) |
| `train/all__inputs.npy` md5_12 | **`ec629e0f6a1f`** (무손상) |
| test `puzzle_indices` / `group_indices` | (10001,) int32, `[0]=0`, `arange(10001)` |
| dtype 원본 일치 | 5/5 (inputs·labels uint8, ids·indices int32) |
| test `dataset.json` | `total_groups 10000, total_puzzles 10000`, 나머지 키 보존 |
| 구 test10k 와 행 동일성 | `inputs` 배열 완전 일치 |

---

## 2. GC 빌더 결함 수정 + 데이터 생성

커밋 `9def6ff`. `dataset/build_graph_coloring_dataset.py:227` 이
`group_indices.append` 를 색칠 루프 **밖**에서 호출해 group 이 그래프 단위였다.
sampler (`puzzle_dataset.py:25`) 가 그룹당 1행만 뽑으므로 epoch 당 gc8
20.7%(1445/6996) / gc10 8.1%(1088/13461) 만 학습된다.

주 근거: nqueens 빌더가 커밋 `05ba06c`(08-02)에서 같은 결함을 같은 방식으로 고쳤고
(`build_nqueens_dataset.py:176-180`), 그 뒤 nq8 99.4 / nq10 96.2 재현에 성공했다.
GC 빌더 최종은 07-30 이라 미적용이었다.

★ `05ba06c` 는 group_indices 단독 변경이 **아니다** — test 절단 제거
(`if set_name == "test": sidxs = sidxs[:1]` 삭제)도 함께 들어 있다. 그쪽은
nqueens 전용이고, GC 빌더의 test 절단 (`:214-215` `cs = cs[:1]`) 은 논문 앵커
test 255/192 와 정합하므로 **유지했다**. 따라서 GC 는 행 생성이 불변이고,
아래 md5 대조가 성립한다.

### 2-1. 생성 명령

```
python dataset/build_graph_coloring_dataset.py --n 8
python dataset/build_graph_coloring_dataset.py --n 10 --output-dir data/graph-coloring-10
```
나머지는 `DataProcessConfig(:62-70)` 기본값 — seed 0 / p −1(→`DEFAULT_P`) /
num_graphs −1(→`DEFAULT_NUM_GRAPHS`) / train_frac 0.85 / self_check True.

### 2-2. 검증

| 항목 | gc8 train | gc8 test | gc10 train | gc10 test |
|---|---|---|---|---|
| 행 수 | 6,996 (앵커 7,002, −6) | 255 (255, +0) | 13,461 (13,465, −4) | 192 (192, +0) |
| `total_groups == total_puzzles == 행 수` | ✅ 6,996 | ✅ 255 | ✅ 13,461 | ✅ 192 |
| `group_indices` 길이 = 행+1 | 6,997 | 256 | 13,462 | 193 |
| `md5(inputs)[:12]` | `3952113f6a83` | `9fca612c9cc5` | `085538a60f49` | `79fe0dad7d33` |

md5 4/4 가 **수정 전 데이터(50번 보유)의 값과 일치** → group_indices 단독 변경이
행 생성에 영향 없음이 실측 확인됐다.

### 2-2a. ★ 런타임 반영 근거 — eval 간격 2,277

정적 검증(총 step, group_indices 길이)은 파일만 본다. 수정이 **런타임 sampler 에
실제로 걸렸는지**는 in-loop eval 간격으로 갈린다:

| | eval_interval 250 epoch 이 도는 step 간격 | 산식 |
|---|---|---|
| **수정 후 (행 = group)** | **2,277** | 250 × **6996** / 768 |
| 수정 전 (그래프 = group) | 470 | 250 × **1445** / 768 |

`logs/gc8_paper_34.log` 실측: `[EVAL @ step 2277]`, 이후 4554 / 6831 / 9108 /
11385 / 13662 / 15939 / 18216 / 20493 — **2,277 등간격**. `gc/n_eval=255.0000`
(gc8 test 행 수와 일치).

즉 `pretrain.py:256` 의 `total_steps = epochs × total_groups × mean_puzzle_examples
/ global_batch_size` 와 `puzzle_dataset.py:209` 의 `rng.permutation(group_indices.size - 1)`
가 둘 다 6,996 을 보고 있다. 파일 수준 검증(§2-2)에 더해, **런타임에서도 epoch
당 전 (그래프, 색칠) 행을 본다**는 것이 이 값으로 확인된다.

- `seq_len` gc8 28 / gc10 45, `vocab_size` 6, `num_puzzle_identifiers` 1,
  `puzzle_identifiers` 전 원소 0
- `sidecar_gc_8.json` graphs 1700 (meta `p=0.5455 num_graphs=1700 seed=0`)
- `sidecar_gc_10.json` graphs 1280 (meta `p=0.425 num_graphs=1280 seed=0`)
- self_check: 양쪽 PASSED (labels valid 1.000000, conflicts 0, split intersection 0,
  dup 0, negative valid 0)

### 2-3. seq_len 구조 (논문 미명시 — 구조만 기술)

논문은 입력 n(n−1)/2 토큰, 출력 길이 N 이라 길이가 다르다. 우리 산출물의 실제 배치:

- `inputs.shape == labels.shape` (gc8 `(6996, 28)`, gc10 `(13461, 45)`), 둘 다 int32
- `inputs` 값 uniq `[1, 2]` — 28/45 토큰 전부 사용
- `labels` 값 uniq `[0, 3, 4, 5]`, 행당 비0 개수 gc8 **8** / gc10 **10**,
  비0 열 인덱스 = `arange(n)`
- 생성 코드 `dataset/build_graph_coloring_dataset.py:115-118`:
  ```python
  lab = np.zeros(seq_len, dtype=np.int32)   # 0 = ignore
  lab[:len(coloring)] = coloring.astype(np.int32) + COLOR0
  ```

즉 `dataset.json` 의 `seq_len` 은 **입력 기준** n(n−1)/2 이고, 출력(길이 N)은 같은
길이 배열의 **앞 N칸**에 놓이며 나머지 `seq_len − N` 칸은 0(=`ignore_label_id`)이다.

---

## 3. Launch 기록 (2026-08-08 14:38 KST)

| run | GPU | tmux | 총 step | 산식 |
|---|---|---|---|---|
| `sudoku_paper_34` | 0–3 | `sudoku_paper_34` | **65,104** | 50000 × 1000 / 768 |
| `gc8_paper_34` | 4–7 | `gc8_paper_34` | **45,546** | 5000 × 6996 / 768 |

로그: `logs/sudoku_paper_34.log`, `logs/gc8_paper_34.log`
(부팅 로그 `logs/_boot_*.log`). 기동 6분 시점 프로세스 22개 생존, GPU 8장 100%.

### 조립된 명령 (DRYRUN 확인분)

```
pretrain.py arch=gram data_paths=[data/sudoku-extreme-1k-aug-1000-test10k-v2]
  evaluators=[] epochs=50000 eval_interval=2000 seed=0
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
  lr_min_ratio=1.0 lr_warmup_steps=2000 global_batch_size=768
  arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2
  arch.H_cycles=3 arch.L_cycles=6 arch.halt_max_steps=16
  arch.kl_coef=0.1 arch.kl_balance=0.8 arch.no_ACT_continue=True
  ema=True ema_rate=0.9999
```
```
pretrain.py arch=gram_gc8 data_paths=[data/graph-coloring-8]
  evaluators=[{name: graph_coloring@GraphColoring}]
  epochs=5000 eval_interval=250 seed=0
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
  lr_min_ratio=1.0 lr_warmup_steps=500 global_batch_size=768
  arch.mlp_t=False arch.pos_encodings=rope arch.L_layers=2
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16
  arch.kl_coef=0.5 arch.noise_head_expansion=1.5
  ema=True ema_rate=0.9999
```
`kl_balance=0.8` 은 `config/arch/gram_gc8.yaml:39` 기본값이라 CLI 미명시.

### 대기 중

`gc10`: 위 둘 중 하나가 끝난 뒤 기동. `arch.kl_coef=0.45` (0.5 아님),
`data_paths=[data/graph-coloring-10]`, 총 step 5000 × 13461 / 768 = **87,637**.

---

## 3a. ★ 과거 Sudoku run 과의 축 비교 가능 여부 (값만)

`all_config.yaml` 의 `data_paths` 실측:

| run | data_paths | epochs | kl_coef |
|---|---|---|---|
| `gram_sudoku_paper` | `data/sudoku-extreme-1k-aug-1000-test10k` | 50000 | 0.1 |
| `gram_sudoku_halt_only` | `data/sudoku-extreme-1k-aug-1000-test10k` | 50000 | 0.1 |
| `gram_sudoku_lsum` | `data/sudoku-extreme-1k-aug-1000-test10k` | 50000 | 0.1 |
| `gram_sudoku_lsum_exp` | `data/sudoku-extreme-1k-aug-1000-test10k` | 50000 | 0.1 |
| `gram_sudoku_lmean_v2` | `data/sudoku-extreme-1k-aug-1000-test10k` | 50000 | 0.1 |
| `gram_sudoku_lmean_ctrl` | `data/sudoku-extreme-1k-aug-1000-test10k` | 50000 | 0.1 |
| `gram_sudoku_sigmabound` | `data/sudoku-extreme-1k-aug-1000-test10k` | 50000 | 0.1 |
| `gram_sudoku_sigma_fix` | `data/sudoku-extreme-1k-aug-1000-test10k` | 50000 | 0.1 |
| `gram_sudoku_long` | `data/sudoku-extreme-1k-aug-1000-test10k` | 100000 | 0.1 |
| `gram_sudoku_b005` | `data/sudoku-extreme-1k-aug-1000-test10k` | 100000 | 0.05 |
| `gram_mlp_t_sudoku` | `data/sudoku-extreme-1k-aug-1000` | 50000 | 0.1 |
| **`sudoku_paper_34` (현행)** | `data/sudoku-extreme-1k-aug-1000-test10k-v2` | 50000 | 0.1 |

판정 (값 기준):

| 대조쌍 | in-loop eval 행 집합 | 비교 가능 여부 |
|---|---|---|
| 과거 `test10k` run 10건 vs 현행 `test10k-v2` | 양쪽 다 10,000행, `inputs` 배열 **완전 일치** (§1-3 실측: 같은 seed 42·idx) | **가능** |
| `gram_mlp_t_sudoku` vs 현행 | 422,786행 vs 10,000행 | **불가** (행 집합 다름) |
| 임의 과거 run vs 최종 harness 축 | 10,000행 vs 422,786행 | **불가** (행 집합 다름) |

즉 손상됐던 것은 `puzzle_indices` / `group_indices` / `dataset.json` 이고,
`inputs` / `labels` 행 집합 자체는 구 `test10k` 와 `test10k-v2` 가 동일하다.
in-loop eval 행 집합 기준으로는 과거 `test10k` run 10건과 현행 run 이 같은 축이다.

---

## 4. 참고 확인 — puzzle embedding (값만, 설정 변경 없음)

논문 Table 6 은 Sudoku·N-Queens·GC 전부 "Puzzle Emb ✗" 인데 우리 config 는
`puzzle_emb_ndim 512 / puzzle_emb_len 16` 이다.

`num_puzzle_identifiers` 실측 — **전 데이터셋 `1`**:

| 데이터 | train | test |
|---|---|---|
| `sudoku-extreme-1k-aug-1000` | 1 | 1 |
| `sudoku-extreme-1k-aug-1000-test10k-v2` | 1 | 1 |
| `graph-coloring-8` | 1 | 1 |
| `graph-coloring-10` | 1 | 1 |
| `nqueens-10` | 1 | 1 |
| `nqueens-8` | 1 | 1 |

nq10 A0(96.22) = `nq10_repro_s0`
(`reports/configs/Nqueens-10-ACT-torch/nq10_repro_s0/all_config.yaml`):
`:23 puzzle_emb_len: 16`, `:24 puzzle_emb_ndim: 512` — 즉 A0 도 같은 설정이었고
해당 데이터의 `num_puzzle_identifiers` 는 1이다.

`num_puzzle_identifiers` 가 1이면 전 퍼즐 공용 학습 prefix 라 논문의
"task-specific puzzle embedding 없음" 과 모순되지 않을 수 있다. **설정은 바꾸지
않았다** — nq 재현과의 일관성이 우선이다.

---

## 5. 로깅 축 실측 (진행 보고용)

`scripts/progress_report_34.py` 가 eval 시점 표를 뽑는다. 축별 가용 여부:

| 축 | 가용 | 근거 |
|---|---|---|
| train acc / eval acc / kl_loss | ✅ stdout + wandb | `pretrain.py:672`, `:698` |
| **grad_norm** | ❌ **미기록** | `pretrain.py:364` 가 `clip_grad_norm_(..., 1.0)` 로 클립만 하고 반환값을 metrics 에 넣지 않는다. wandb 키 목록에도 없다 (실측). |
| `diag/z_h_norm` · `diag/eps_std_norm` · `diag/eps_mu_norm` | ✅ **wandb 만** | `gram.py:335-339` `_latent_diag` → `losses.py:113-114` → `pretrain.py:398`. stdout 은 `pretrain.py:671` 이 `'loss'`/`'accuracy'` 포함 키만 걸러 프린트하므로 로그 파일에는 안 나온다. |

전 diag 키 (run 당 기록): `train/diag/{z_h_norm, z_l_norm, eps_mu_norm,
eps_std_norm, train_kl_t1..t3, train_mu_absmean_t1..t3, train_std_mean_t1..t3,
train_std_dimstd_t1..t3}`.
