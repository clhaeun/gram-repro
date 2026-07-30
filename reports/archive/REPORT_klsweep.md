# REPORT_klsweep.md — 데이터 육안 확인 → 입력 전달 경로 → KL coef sweep

- 일시: 2026-07-27
- 브랜치 `main`, HEAD `4c847f1`
- **GPU: 4,5,6,7만 사용. 0~3은 건드리지 않았다** (launch 전후 `nvidia-smi`로 0~3이 35–50MiB 유휴임을 확인).
- 코드 수정 없음. `train_nqueens8_fix.sh`를 환경변수로만 구동.

---

## 1. Task 1 — 데이터 배치 육안 확인

`data/nqueens-8/{train,test}/all__{inputs,labels}.npy` 직접 로드.

### 6항목 판정 — **전부 통과**

| # | 항목 | train | test | 판정 |
|---|---|---|---|---|
| 1 | vocab이 {0,1,2}만인가 | `[1 2]` | `[1 2]` | ✅ 3 이상 없음 |
| 2 | pad(0) 있는가 | inputs 0개, labels 0개 | inputs 0개, labels 0개 | ✅ pad 없음 (seq_len=64=8²) |
| 3 | input queen 1~3인가 | min 1 max 3 | min 1 max 3 | ✅ |
| 4 | label queen 정확히 8인가 | 전 7194개가 8 | 전 772개가 8 | ✅ |
| 5 | input queen ⊆ label queen | **7194/7194 (100.00%)** | **772/772 (100.00%)** | ✅ clue 보존 |
| 6 | test split 동일 검사 | — | 위 전부 동일 | ✅ |

shape/dtype: train `(7194, 64) int32`, test `(772, 64) int32` (inputs/labels 동일).

input queen 개수 분포 (`8 − k`, `k ∈ {5,6,7}` 예상과 일치):

| queens | train | test |
|---|---|---|
| 1 | 650 | 6 |
| 2 | 2176 | 160 |
| 3 | 4368 | 606 |

**주의(이상 아님)**: vocab에 0이 **없다**. `evaluators/nqueens.py`의 주석은
`vocab: 0=pad, 1=empty, 2=queen`인데 N-Queens 데이터에는 pad가 쓰이지 않아
실제 등장 토큰은 {1,2}뿐이다. 항목 2의 기대("pad 없어야 정상")와 일치한다.

### sample 출력 (train 0~4, 좌 INPUT / 우 LABEL)

```
=== sample 0: input queens=3, label queens=8, input⊆label=True
   INPUT            LABEL
   ........         ....Q...
   ......Q.         ......Q.
   ........         .Q......
   ........         ...Q....
   ........         .......Q
   ........         Q.......
   ..Q.....         ..Q.....
   .....Q..         .....Q..
=== sample 1: input queens=3, label queens=8, input⊆label=True
   ....Q...         ....Q...
   ........         .Q......
   ........         ...Q....
   .....Q..         .....Q..
   ........         .......Q
   ........         ..Q.....
   Q.......         Q.......
   ........         ......Q.
=== sample 2: input queens=3, label queens=8, input⊆label=True
   ........         ....Q...
   ......Q.         ......Q.
   ........         .Q......
   ........         .....Q..
   ..Q.....         ..Q.....
   ........         Q.......
   ........         ...Q....
   .......Q         .......Q
=== sample 3: input queens=3, label queens=8, input⊆label=True
   ...Q....         ...Q....
   ........         .Q......
   ........         ......Q.
   ........         ..Q.....
   .....Q..         .....Q..
   ........         .......Q
   ........         ....Q...
   Q.......         Q.......
=== sample 4: input queens=2, label queens=8, input⊆label=True
   ..Q.....         ..Q.....
   ........         ....Q...
   ........         .Q......
   ........         .......Q
   ........         .....Q..
   ...Q....         ...Q....
   ........         ......Q.
   ........         Q.......
```
(`.` = 토큰 1 = empty, `Q` = 토큰 2 = queen. `?`(=0) 은 한 칸도 없었다.)

**Task 1 결론: 데이터 파이프라인 이상 없음. sweep 진행 조건 충족.**

---

## 2. Task 2 — 입력 전달 경로

### 2-1. 데이터셋 → batch dict 키

`puzzle_dataset.py:124-128` 이 npy를 그대로 세 키로 싣는다:
```python
124            "inputs": "r",
125            "labels": "r",
...
128            "puzzle_identifiers": None,
```
배치 슬라이싱도 키를 섞지 않는다 — `puzzle_dataset.py:191-193`(순차) / `:232-234`(샘플링):
```python
191                    "inputs": dataset["inputs"][local_start: local_end],
192                    "labels": dataset["labels"][local_start: local_end],
193                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
```

### 2-2. forward에서 inputs/labels가 뒤바뀌지 않았는가 — **바뀌지 않았다**

`gram.py:311` (labels → y_emb, posterior일 때만):
```python
311        y_emb = self._y_embeddings(batch["labels"]) if use_posterior else None
```
`gram.py:318` (inputs → input_embeddings, **무조건**):
```python
318        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
```
`_input_embeddings`는 `inputs`와 `puzzle_identifiers`만 받고(`gram.py:202-222`),
`_y_embeddings`는 `labels`만 받는다(`gram.py:225-232`). 인자 교차 없음.

### 2-3. prior 경로가 `batch["inputs"]`를 실제로 쓰는가 — **쓴다** (코드 + 실측)

**코드**: `input_embeddings`는 `use_posterior`와 무관하게 계산되고(318행),
두 경로 모두에서 `z_L` 갱신에 들어간다 —
`gram.py:329`(no_grad 루프) / `gram.py:354`(grad 경로):
```python
329                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
354            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
```
(주입 지점은 `z_L` 갱신뿐이다. `z_H = self.L_level(z_H, z_L, ...)`(331/356)에는
`input_embeddings`가 직접 들어가지 않고 `z_L`을 통해 전달된다. TRM `trm.py:211-216`과 동일 구조.)

**실측** (`gram_nq8_fix_b/step_17085`, test 128개, K=1): 입력만 바꿔 prior 출력이 변하는지.

| 입력 변형 | prior: 원본과 동일한 보드 | prior: 동일 토큰 |
|---|---|---|
| shuffled inputs (다른 퍼즐 clue) | **3.1%** | 94.06% |
| all-empty (clue 제거) | **0.8%** | 95.19% |

→ 입력을 바꾸면 prior 출력이 바뀐다. **prior가 고정 z0만으로 도는 것이 아니다.**
(토큰 일치율 94–95%가 높은 것은 64칸 중 queen이 ~10개뿐이라 빈칸이 기본적으로 겹치기 때문이다.
보드 단위 일치율이 판정 기준이다.)

**Task 2 결론: 배선 이상 없음. sweep 진행 조건 충족.**

### 2-4. 부수 발견 — posterior valid 100%는 **라벨 복사**다 (지시 항목 밖, 그러나 baseline 해석을 바꿈)

2-3의 대조군을 posterior 경로에도 돌리다가 나온 결과다. 지시된 6항목/3항목은 전부 통과했으므로
sweep을 막지는 않지만, 지금까지 "백본 정상"의 근거였던 수치의 의미가 달라진다.

test 128개, `step_17085`, K=1:

| run | posterior pred == label (exact board) | prior pred == label | posterior에 **틀린 라벨**을 넣으면 그 틀린 라벨을 재현 |
|---|---|---|---|
| gram_nq8_fix_b | **100.00%** | 0.00% (token 73.73%) | **99.22%** |
| gram_nq8_mean_nh512 | **100.00%** | 0.00% (token 45.63%) | **100.00%** |
| gram_nq8_mean1 | **100.00%** | 0.00% (token 85.77%) | **100.00%** |

그리고 posterior 출력은 **입력에 완전 불변**이다 (fix_b, 위와 같은 대조군):

| 입력 변형 | posterior: 원본과 동일한 보드 | posterior valid | posterior consistent |
|---|---|---|---|
| shuffled inputs | **100.0%** | 1.0000 | **0.0312** |
| all-empty | **100.0%** | 1.0000 | 1.0000 |

즉 posterior 경로는 **입력을 보지 않고 y를 그대로 복원**한다.
shuffled inputs에서 valid=1.0인데 consistent=0.0312인 것이 그 직접 증거다 —
출력이 여전히 *원래* 라벨이라, 바뀐 clue와는 맞지 않는다.

**관측된 것만 적으면**: "posterior valid 100%"는 제약을 풀어낸 결과가 아니라
y를 잠재로 통과시켜 재생한 결과다. 세 run 모두 동일. β sweep이 조절하려는 것이
바로 이 통과량이므로, 이 발견은 sweep을 막는 것이 아니라 sweep의 대상을 명확히 한다.

---

## 3. Task 3 — KL coef sweep

### 실행

`train_nqueens8_fix.sh`를 환경변수로만 구동(스크립트 미변경). grad 구조·y 결합·σ 전부 유지.

```bash
RUN=nq8_b03 BETA=0.3 GPUS=4 PORT=29631 bash train_nqueens8_fix.sh
RUN=nq8_b07 BETA=0.7 GPUS=5 PORT=29632 bash train_nqueens8_fix.sh
RUN=nq8_b15 BETA=1.5 GPUS=6 PORT=29633 bash train_nqueens8_fix.sh
RUN=nq8_b30 BETA=3.0 GPUS=7 PORT=29634 bash train_nqueens8_fix.sh
```
launch 전 점유 확인 통과(4개 GPU 전부 35MiB):
```
[preflight] GPUS=4 NPROC=1 PORT=29631 전부 유휴 -- launch
[preflight] GPUS=5 NPROC=1 PORT=29632 전부 유휴 -- launch
[preflight] GPUS=6 NPROC=1 PORT=29633 전부 유휴 -- launch
[preflight] GPUS=7 NPROC=1 PORT=29634 전부 유휴 -- launch
```

config dump 확인 (`checkpoints/Nqueens-8-ACT-torch/<RUN>/all_config.yaml`):

| run | kl_coef | kl_reduction | ema | lr_warmup_steps | noise_head_expansion |
|---|---|---|---|---|---|
| nq8_b03 | 0.3 | mean | false | 500 | 1.5 |
| nq8_b07 | 0.7 | mean | false | 500 | 1.5 |
| nq8_b15 | 1.5 | mean | false | 500 | 1.5 |
| nq8_b30 | 3.0 | mean | false | 500 | 1.5 |

β=0.07 지점은 **기존 `gram_nq8_fix_b`가 그대로 대응**한다(같은 스크립트, 같은 설정, β만 0.07).
따라서 5점 계열 0.07 / 0.3 / 0.7 / 1.5 / 3.0 이 된다.

완주: 4개 run 전부 step 17085 도달. 학습 중 NaN/OOM/crash 없음.
속도 2.76–2.80 it/s, run당 1시간 42–43분 (4개 병렬).

### Task A — checkpoint 선정 (in-loop `nqueens/accuracy` 곡선)

로그 `logs/<RUN>.log`의 `[EVAL @ step N]` 15줄에서 읽었다. **prior 경로**다
(`evaluate()`가 `model.eval()`로 부르고 `gram.py:303-304`가 `use_posterior=False`로 해석).

| step | β=0.3 | β=0.7 | β=1.5 | β=3.0 |
|---|---|---|---|---|
| 1139 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2278 | **0.0013** | 0.0052 | 0.0000 | 0.0000 |
| 3417 | 0.0000 | 0.0091 | 0.0000 | 0.0000 |
| 4556 | 0.0000 | **0.2111** | 0.0376 | 0.0000 |
| 5695 | 0.0000 | 0.1386 | 0.1269 | 0.0000 |
| 6834 | 0.0000 | 0.0907 | **0.3135** | 0.0000 |
| 7973 | 0.0000 | 0.0570 | 0.1904 | 0.0000 |
| 9112 | 0.0000 | 0.0039 | 0.1969 | 0.0000 |
| 10251 | 0.0000 | 0.0415 | 0.1943 | 0.0000 |
| 11390 | 0.0000 | 0.0013 | 0.1373 | 0.0000 |
| 12529 | 0.0000 | 0.0117 | 0.1917 | 0.0026 |
| 13668 | 0.0000 | 0.0000 | 0.0674 | 0.0104 |
| 14807 | 0.0000 | 0.0078 | 0.1373 | 0.0427 |
| 15946 | 0.0000 | 0.0117 | 0.1347 | 0.0855 |
| 17085 | 0.0000 | 0.0065 | 0.1425 | **0.0984** |

| β | peak step | peak nq_acc | final nq_acc |
|---|---|---|---|
| 0.07 (`gram_nq8_fix_b`) | — (전 구간 0) | 0.0000 | 0.0000 |
| 0.3 | 2278 | 0.0013 | 0.0000 |
| 0.7 | **4556** | **0.2111** | 0.0065 |
| 1.5 | **6834** | **0.3135** | 0.1425 |
| 3.0 | 17085 (= final) | 0.0984 | 0.0984 |

지시문이 언급한 "β=1.5는 ~7100, β=0.7은 ~5500"은 폴링 중 관측한 근사치였고,
정확한 정점은 **β=1.5 → step 6834**, **β=0.7 → step 4556**이다.

**β=3.0은 정점이 마지막 step이다 — 아직 상승 중이고 수렴하지 않았다**
(12529부터 0.0026 → 0.0104 → 0.0427 → 0.0855 → 0.0984로 단조 증가).
다른 셋과 달리 "학습을 더 하면 더 오를" 여지가 남아 있다.

### Task B — harness (prior N=20 + posterior, test 772 전수)

`experiments/sweep_harness.py` (신규). seed=0, sigma_scale=1.0, **valid 기준**.
`eval_harness.py`를 쓰지 않은 이유: 그쪽 `NQueensTask.verify`는
`strict`를 정답으로 반환한다(`eval_harness.py:120-126`). 지시는 valid 기준을 요구했다.
샘플링 RNG는 `gram_harness.rollout`과 동일(`seed*100003+k`).

결과 json: `experiments/results/sweep_<run>_<step>.json`

| β | ckpt | | prior valid (N=20 전체) | k=0 | consistent | strict | queens (min–max) | **posterior valid** | post queens |
|---|---|---|---|---|---|---|---|---|---|
| 0.07 | 17085 | final | 0.00% | 0.00% | 0.00% | 0.00% | 10.53 (4–17) | 99.74% | 8.00 |
| 0.3 | 17085 | final | 0.00% | 0.00% | 92.40% | 0.00% | 2.85 (1–6) | 100.00% | 8.00 |
| 0.3 | 2278 | peak | 0.05% | 0.13% | 99.10% | 0.05% | 7.76 (6–10) | 97.80% | 7.98 |
| 0.7 | 17085 | final | 0.91% | 0.91% | 99.48% | 0.91% | 8.10 (7–10) | 99.87% | 8.00 |
| **0.7** | **4556** | **peak** | **21.10%** | 20.34% | 99.68% | 21.04% | 7.92 (6–10) | **99.48%** | 8.00 |
| 1.5 | 17085 | final | 15.34% | 14.12% | 96.57% | 15.03% | 8.04 (6–11) | 100.00% | 8.00 |
| **1.5** | **6834** | **peak** | **30.29%** | 27.85% | 98.15% | 29.97% | 7.62 (6–10) | **99.35%** | 7.99 |
| 3.0 | 17085 | final=peak | 9.18% | 8.42% | 87.22% | 8.74% | 8.06 (6–10) | 97.28% | 8.03 |

selection 지표 (전부 valid 기준):

| β | ckpt | pass@1 | pass@5 | pass@10 | **pass@20** | majority@20 | verifier_sel@20 | distinct@20 | ≥1 valid인 퍼즐 |
|---|---|---|---|---|---|---|---|---|---|
| 0.07 | 17085 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 20.00 | 0.00% |
| 0.3 | 17085 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.77 | 0.00% |
| 0.3 | 2278 | 0.0% | 0.2% | 0.5% | 0.9% | 0.1% | 0.1% | 19.77 | 0.91% |
| 0.7 | 17085 | 0.9% | 4.4% | 8.4% | 15.5% | 1.6% | 0.5% | 19.60 | 15.54% |
| **0.7** | **4556** | 21.1% | 61.3% | 79.5% | **92.5%** | 43.3% | 21.9% | 11.80 | 92.49% |
| 1.5 | 17085 | 15.3% | 51.3% | 71.4% | 86.9% | 34.3% | 14.8% | 13.58 | 86.92% |
| **1.5** | **6834** | 30.3% | 73.2% | 87.2% | **94.2%** | 56.9% | 31.6% | 9.94 | 94.17% |
| 3.0 | 17085 | 9.2% | 34.9% | 53.5% | 72.8% | 23.1% | 10.2% | 16.27 | 72.80% |

지시가 특히 보라고 한 두 가지:

- **peak prior valid가 pass@20으로 얼마가 되는가**: β=1.5 peak는 single 30.29% → **pass@20 94.17%**,
  β=0.7 peak는 21.10% → **92.49%**. selection 여지가 크다.
  다만 **verifier_selected@20은 31.6% / 21.9%로 pass@20보다 훨씬 낮다** — LPRM 헤드가
  20개 중 valid한 것을 골라내지 못한다(β=1.5 peak 기준 94.2% 중 31.6%만 선택 성공).
  majority@20(56.9% / 43.3%)이 verifier보다 낫다.
- **같은 지점 posterior valid**: β=1.5 peak 99.35%, β=0.7 peak 99.48%.
  **posterior는 안 무너졌다.** gap이 좁아진 것은 prior가 올라와서지 posterior가 내려가서가 아니다.
  (β=3.0 final의 97.28%가 최저.)

### Task C — transition별 KL 변화

`experiments/diag_contradiction.py --skip 2a,2c,2d` (posterior 경로, B=64, sup_steps=16).
결과 json: `experiments/results/klgrid_<run>_<step>.json`,
β=0.07은 기존 `experiments/results/contradiction_gram_nq8_fix_b.json`.

**step을 맞춘 비교 (전부 `step_17085`)** — 이쪽이 β 효과를 보는 정본이다:

| β | sup0 trans0 | sup0 trans1 | sup0 trans2 | 정상상태 trans0 | 정상상태 trans2 |
|---|---|---|---|---|---|
| 0.07 | **13.8083** | 0.0025 | 0.0022 | 0.0023 | 0.0023 |
| 0.3 | **12.0747** | 0.0068 | 0.0013 | 0.0016 | 0.0016 |
| 0.7 | **1.3084** | 0.0107 | 0.0005 | 0.0041 | 0.0045 |
| 1.5 | **0.5053** | 0.0154 | 0.0001 | 0.0003 | 0.0002 |
| 3.0 | **0.1859** | 0.1035 | 0.0005 | 0.0006 | 0.0007 |

**trans0의 13.81이 β가 커지며 단조 감소한다: 13.81 → 12.07 → 1.31 → 0.51 → 0.19 (74× 감소).**
trans0/trans1은 `torch.no_grad()` 안이라 KL 페널티를 직접 받지 않는데도 줄었다.

참고로 **peak step에서 잰 같은 표**(step이 β마다 달라 비교의 정본은 아니다):

| β | 측정 step | sup0 trans0 | 정상상태 trans0 |
|---|---|---|---|
| 0.07 | 17085 | 13.8083 | 0.0023 |
| 0.3 | 2278 | 77.4701 | 0.0388 |
| 0.7 | 4556 | 4.6940 | 0.0051 |
| 1.5 | 6834 | 0.4591 | 0.0014 |
| 3.0 | 17085 | 0.1859 | 0.0006 |

β=0.3의 77.47은 β 효과가 아니라 **step 2278(학습 초기)이라서**다 — 같은 run의
step 17085에서는 12.07로 내려온다. 정본 표(step 맞춤)에서는 단조성이 깨지지 않는다.

### Task D — 종합표

| β | in-loop peak | peak step | prior valid (final) | prior valid (peak) | posterior valid (peak) | pass@20 (peak) | trans0 KL (step 17085) | queens (peak) |
|---|---|---|---|---|---|---|---|---|
| 0.07 | 0.0000 | — | 0.00% | 0.00% | 99.74% | 0.0% | 13.8083 | 10.53 |
| 0.3 | 0.0013 | 2278 | 0.00% | 0.05% | 97.80% | 0.9% | 12.0747 | 7.76 |
| 0.7 | 0.2111 | 4556 | 0.91% | **21.10%** | 99.48% | **92.5%** | 1.3084 | 7.92 |
| **1.5** | **0.3135** | **6834** | 15.34% | **30.29%** | 99.35% | **94.2%** | 0.5053 | 7.62 |
| 3.0 | 0.0984 | 17085 | 9.18% | 9.18% | 97.28% | 72.8% | 0.1859 | 8.06 |

(prior valid는 harness N=20 전체 평균, posterior는 K=1. queens는 prior 경로 mean.)

---

## 4. β–성능 관계

수치만 나열한다.

**prior valid (peak checkpoint, harness N=20 전체 평균)**
```
β=0.07 :  0.00%
β=0.3  :  0.05%
β=0.7  : 21.10%
β=1.5  : 30.29%   <- 최대
β=3.0  :  9.18%
```
양 끝(0.07, 3.0)이 낮고 1.5에서 최대. 0.07→1.5 구간 단조 증가, 1.5→3.0 감소.

**pass@20 (peak)**: 0.0% → 0.9% → 92.5% → **94.2%** → 72.8%. 같은 형태.

**posterior valid (peak)**: 99.74% / 97.80% / 99.48% / 99.35% / 97.28%.
β 전 구간에서 97% 아래로 안 내려간다. β에 대한 뚜렷한 추세 없음.

**trans0 KL (step 17085 고정)**: 13.8083 → 12.0747 → 1.3084 → 0.5053 → 0.1859.
β에 대해 **단조 감소**. prior valid와 달리 U자가 아니다.

**queen_count mean (prior, peak)**: 10.53 / 7.76 / 7.92 / 7.62 / 8.06 (정답 8).
β=0.07만 10.53으로 크게 벗어나고 나머지는 7.6–8.1.

**distinct@20 (peak)**: 20.00 / 19.77 / 11.80 / 9.94 / 16.27.
prior valid가 높은 지점일수록 낮다(β=1.5 peak에서 9.94).

**정점 후 하강** — peak 대비 final:

| β | peak | final | 감소폭 |
|---|---|---|---|
| 0.7 | 21.10% | 0.91% | −20.19%p |
| 1.5 | 30.29% | 15.34% | −14.95%p |
| 3.0 | 9.18% | 9.18% | 0 (아직 상승 중) |

β=0.7이 β=1.5보다 더 크게 무너진다. β=3.0은 하강 구간에 아직 도달하지 않았다.

**측정으로 답하지 못한 것**
- 1.5와 3.0 사이(2.0–2.5)는 안 돌렸다. peak가 1.5인지 그 사이인지 **확인 불가**.
- β=3.0이 17085에서 아직 상승 중이라, epoch를 늘렸을 때 어디까지 가는지 **확인 불가**.
- peak 이후 하강의 원인(과적합/KL 붕괴/기타)은 이번 측정으로 **분리하지 않았다**.
- run당 seed 1개(seed=0)뿐이라 β별 차이가 seed 변동을 넘는지 **확인 불가**.

---

## 5. 임의로 정한 것

1. **EMA를 껐다 (`ema=False`).** 지시문 산문은 "EMA 0.9999"라고 했으나 지시문이 준
   launch 커맨드 4개에 `EMA=` 지정이 없어 `train_nqueens8_fix.sh`의 기본값 `EMA=False`가 적용된다.
   커맨드를 그대로 따랐다. 근거 두 가지: (a) 앞선 측정에서 EMA on/off가 prior valid를
   바꾸지 않았다(fix_b EMA off 0%, nh512 EMA on 0%), (b) 이렇게 하면 기존 `gram_nq8_fix_b`가
   β=0.07 지점으로 그대로 편입돼 **β만 다른 5점 계열**이 된다.
   `ema_rate=0.9999`는 스크립트 기본값 그대로 config에 남아 있으나 `ema=false`라 사용되지 않는다.
2. **GPU 1장/run.** 지시문 커맨드대로 `GPUS=4`~`GPUS=7` 단일 GPU. 이 경우
   `global_batch_size=768`이 1장에 전부 올라간다(GPU당 14,011MiB 사용, 48GB 중).
   4-GPU 분산이던 fix_b(장당 192)와 **총 배치는 같고** 장당 부하만 다르다.
   속도는 2.88 it/s → run당 약 1시간 37분. 4개 병렬이라 벽시계도 ~1시간 40분
   (지시문의 "각 ~37분"은 4-GPU 기준 수치다).
3. **입력 전달 실측의 대조군**을 shuffled inputs(배열 역순 = 다른 퍼즐의 clue)과
   all-empty(전부 토큰 1)로 정했다. 지시에 없던 측정이다 — 2-3의 "prior가 input을
   실제로 쓰는가"를 코드 인용만으로 끝내지 않으려고 넣었다.
4. **2-4는 지시 항목 밖**이다. 3-1의 대조군을 posterior에도 돌린 부산물이고,
   Task 1·2의 지정 판정 항목은 전부 통과했으므로 sweep을 중단하지 않았다.
5. 2-3/2-4의 실측은 `gram_nq8_fix_b/step_17085` + test 앞 128개, K=1, seed=0 기준이다
   (2-4의 3-run 표만 세 checkpoint). Task B의 sweep harness는 test 772 전수다.
6. **`experiments/sweep_harness.py`를 새로 만들었다.** 기존 `eval_harness.py`를 안 쓴 이유는
   §Task B에 적었다(그쪽 정답 기준이 strict라 지시의 "valid 기준"과 다름).
   pass@k 불편추정량과 majority 정의는 `eval_harness.py:248-258`을 그대로 옮겼다.
7. **β=0.07 행은 기존 `gram_nq8_fix_b`를 재사용**했다. 이번에 같은 `sweep_harness.py`로
   다시 측정해 다른 행과 같은 기준으로 맞췄다(`experiments/results/sweep_gram_nq8_fix_b_step_17085.json`).
8. **β=0.3의 peak를 step 2278(nq_acc 0.0013)로 잡았다.** 사실상 0이라 "정점"이라 부르기 어렵지만
   기계적으로 최대값 지점을 택했다.
9. **Task C를 두 번 측정했다.** 처음엔 각 β의 peak step에서 쟀는데 step이 β마다 달라
   교란이 생겼다(β=0.3이 77.47로 튐). step_17085로 맞춘 표를 추가하고 그쪽을 정본으로 삼았다.
   지시에 없던 추가 측정이다.
10. Task C의 KL 그리드는 B=64, sup_steps=16, seed=0. `--num`은 valid 측정용이라
    이 경로에선 쓰이지 않는다.
11. GPU 배정: 학습 4/5/6/7 각 1장, harness는 4~7에 분산, KL 그리드는 7번과 5번.
    측정 도중 GPU 0~3에 타 작업(447–462MiB)이 올라왔으나 건드리지 않았다.
