# REPORT_expand_prep.md — 확장 태스크 데이터/인프라 준비

- 일시: 2026-07-27
- **GPU 미사용. 학습 없음.** 전부 CPU 작업.
- 8×8 one-hot run(GPU 4/5/6) 은 건드리지 않았다. 작업 종료 시점 확인:
  GPU 4/5/6 = 14083MiB, 97~100% 가동, `nq8_oh_b007` step 6900 / `nq8_oh_b15` 6600 /
  `nq8_ctrl_b15` 6650 (17093 중). GPU 7 은 계속 유휴, GPU 0~3 은 조회만 함.
- 새로 만든 파일: `dataset/build_graph_coloring_dataset.py`, `evaluators/graph_coloring.py`,
  `data/graph-coloring-8/`, `data/graph-coloring-10/`. **yaml 은 만들지 않았다.**

---

## 1. N-Queens 10×10 — 점검 결과 (만들지 않음, 이미 존재)

### 1-1. 생성 규칙 확인 (k∈{7,8,9})

`data/nqueens-10/sidecar_10.json` 의 meta 가 그대로 답이다:
```json
{"n": 10, "encoding": "columns", "removals": [7, 8, 9],
 "num_solutions_total": 724, "num_unique_inputs": 43420}
```
→ **k∈{7,8,9} 제거로 생성됨 확인.** 대조군으로 8×8 은 `removals: [5, 6, 7]` 로 확인.
`dataset/build_nqueens_dataset.py:85` 의 `DEFAULT_REMOVALS = {8: (5,6,7), 10: (7,8,9)}` 와도 일치.

### 1-2. 데이터 무결성

| 항목 | train | test |
|---|---|---|
| rows (쌍) | 107,480 | 6,513 |
| groups (unique x) | 36,907 | 6,513 |
| seq_len / vocab | 100 / 3 | 100 / 3 |
| 라벨 verifier `valid` | **1.000000** | **1.000000** |
| 라벨 `consistent` (입력 queen 보존) | **1.000000** | **1.000000** |
| 라벨 `strict` | **1.000000** | **1.000000** |
| queens/board | 정확히 10 (min=max=10) | 정확히 10 |
| 주어진 clue 수 | 1~3 (평균 2.633) | 1~3 (평균 2.912) |

- **해 개수**: 라벨의 distinct 해 = train 724 / test 715, 합집합 **724**.
  10-queens 전체 해가 724 이므로 **전체 해를 다 덮는다**.
- **쌍 개수**: 107,480 + 6,513 = 113,993. unique input 36,907+6,513 = **43,420** 으로
  sidecar 의 `num_unique_inputs` 와 정확히 일치.
- **leakage 0**: unique x 기준 train∩test = **0**. (x,y) 쌍 기준 교집합도 **0**.
  train/test 각각 내부 중복 쌍도 0.
- **x-split 확인**: 각 group 안의 x 가 전부 동일(불일치 group 0개). train group 크기
  평균 2.912 (max 93) = 한 입력의 모든 해를 담고, test group 크기는 전부 정확히 1
  = 입력당 해 1개. `build_nqueens_dataset.py` 의 규율(85:15 group split, test 는
  입력당 1행)과 일치.
- 인덱스 정합성: `total_groups`/`total_puzzles` 가 `group_indices`/`puzzle_indices`
  길이와 일치, 단조증가, `puzzle_indices[-1] == len(inputs)`.

주의(문제 아님): y(해) 자체는 train/test 가 715개 겹친다. x-split 이므로 구조상 불가피하다
(724개뿐인 해 집합을 나눌 수는 없다). leakage 판정 기준은 x 와 (x,y) 쌍이고 둘 다 0이다.

### 1-3. evaluator 의 n=10 동작

- `evaluators/nqueens.py:verify_boards(preds, inputs, n)` — **n 을 인자로 받는다** ✓
  (n 은 함수 내부에서 reshape/대각선 bin 계산에 그대로 쓰임)
- `NQueens.__init__` 은 `n` 미지정 시 `n = round(seq_len ** 0.5)` 로 추론하고
  `n*n != seq_len` 이면 예외 → **seq_len=100 에서 n=10 으로 정확히 추론됨** ✓
- 따라서 10×10 에서 evaluator 인자 변경 없이 `evaluators=[{name: nqueens@NQueens}]` 그대로 쓸 수 있다.
- 실측 검증: 10×10 라벨 113,993행 전부를 `verify_boards(..., n=10)` 로 통과시켜 valid=1.0 확인
  (위 표). 즉 n=10 경로가 실제로 도는 것을 확인했다.

### 1-4. yaml 은 만들지 않음 — 복제할 필드 목록만

`config/arch/gram_nqueens8.yaml` 를 복사해 `gram_nqueens10.yaml` 을 만들 때 손댈 것:

| 필드 | 8×8 현재값 | 10×10 에서 | 비고 |
|---|---|---|---|
| `kl_coef` (β) | 0.07 | **0.045** | 논문값. 단 one-hot/β 재검토 중이라 8×8 결론 후 확정 |
| `L_cycles` | 4 | **미확인** | 논문의 10×10 값을 확인 못 함. 확인 전엔 4 유지가 기본 |
| (epochs) | 3000 | **미확인** | yaml 아님 — sh CLI 인자 |
| (seq_len) | 64 | 100 | **yaml 필드 아님. 자동 주입** — 아래 참조 |
| (vocab_size) | 3 | 3 | 역시 자동. 변경 불필요 |
| (data_paths) | `[data/nqueens-8]` | `[data/nqueens-10]` | yaml 아님 — sh CLI 인자 |

**중요 — `seq_len` 은 arch yaml 에 없다.** `gram_nqueens8.yaml` 에 `seq_len` 항목이 없고,
`pretrain.py:118-126 create_model()` 이 `seq_len=train_metadata.seq_len` 으로 데이터
metadata 에서 주입한다(`vocab_size`, `num_puzzle_identifiers` 도 동일). 즉 10×10 은
`data_paths` 만 바꾸면 seq_len=100 이 자동으로 들어간다. **yaml 에 seq_len 을 적으면 안 된다**
(struct 에 없는 키라 hydra 가 거부하거나, 있어도 덮어써짐).

그대로 두는 필드(8×8과 동일): `H_cycles=3`, `H_layers=0`, `L_layers=2`, `hidden_size=512`,
`num_heads=8`, `expansion=4`, `puzzle_emb_len=16`, `puzzle_emb_ndim=${.hidden_size}`,
`mlp_t=False`, `pos_encodings=rope`, `forward_dtype=bfloat16`, `halt_max_steps=16`,
`halt_exploration_prob=0.1`, `no_ACT_continue=True`, `kl_balance=0.8`, `kl_reduction`,
`lprm_coef=1.0`, `noise_head_expansion=1.5`, `y_onehot*` 3필드.

참고 — 규모 차이가 커서 `epochs`/`eval_interval` 은 그대로 쓰면 안 될 가능성이 높다:

| | seq_len | train rows | train groups | test rows |
|---|---|---|---|---|
| 8×8 | 64 | 7,194 | 4,376 | 772 |
| 10×10 | 100 | **107,480** (14.9배) | 36,907 | 6,513 |

같은 `epochs=3000` 이면 10×10 은 step 수가 약 15배가 된다. 확정은 사수 판단.

---

## 2. Graph Coloring — 데이터 + verifier 구현 완료

### 2-1. 만든 것

- `dataset/build_graph_coloring_dataset.py` — 생성기 (ER + 3-colorable 필터 + 정규형 + split + 자체검증)
- `evaluators/graph_coloring.py` — `verify_colorings()` + `GraphColoring` evaluator
- `data/graph-coloring-8/`, `data/graph-coloring-10/` — 실제 데이터 + `sidecar_gc_{n}.json`

### 2-2. 생성 절차 (지시 스펙대로)

1. **ER**: n(n-1)/2 개 각 edge 를 확률 p 로 독립 샘플링, 대칭, self-loop 없음.
2. **3-colorable 필터**: canonical proper coloring 이 1개 이상인 그래프만 유지.
3. **정규형**: first-appearance normalization — 노드를 0..n-1 순서로 훑어 처음 나온 색을 0,
   다음 새 색을 1, 그 다음을 2. 조건식으로는 `c[0]==0` 이고 `c[j] <= max(c[:j])+1`.
   permutation orbit 당 정확히 1개만 남는다.
4. **중복 그래프 제거**: 상삼각 토큰 해시 기준. 데이터셋 내 모든 그래프가 유일.
5. **split**: unique 그래프 기준 85:15. train 은 그 그래프의 canonical coloring 전부,
   test 는 1개(사전순 첫번째). N-Queens 빌더와 동일 규율.

### 2-3. 인코딩 / vocab (지시 스펙 그대로)

```
seq_len = n(n-1)/2      # n=8 -> 28, n=10 -> 45
inputs  : 인접행렬 상삼각(대각 제외) flatten, np.triu_indices(n,1) 순서. 토큰 1/2 만.
labels  : 앞 n 칸 = 노드 색(3/4/5), 나머지 seq_len-n 칸 = 0
vocab_size = 6 :  0=PAD/ignore, 1=no edge, 2=edge, 3=red, 4=blue, 5=green
```
라벨 길이(N)가 seq_len 보다 짧은 문제는 **뒤를 `ignore_label_id=0` 으로 채워서** 해결했다.
`puzzle_dataset.py:152-153` 이 0 을 `IGNORE_LABEL_ID=-100` 으로 바꾸고
`models/losses.py:66` 이 마스킹하므로 padding 은 loss 에 기여하지 않는다.
실측으로 확인함(아래 2-6). **이 부분은 논문에 명시가 없어 내가 정한 것 — 4절 참조.**

### 2-4. p 결정과 데이터셋 크기

p 는 논문 미명시이므로 지시대로 **크기 앵커가 나오도록** 맞췄다. seed 0 고정, 0.0005 간격 스캔:

| n | 선택 p | edge/graph | train pairs | 앵커 | 차이 | test pairs | 앵커 |
|---|---|---|---|---|---|---|---|
| 8 | **0.5455** | 13.15 | **6,996** | 7,002 | **−6** (0.09%) | **255** | 255 ✓ **정확** |
| 10 | **0.4250** | 16.79 | **13,461** | 13,465 | **−4** (0.03%) | **192** | 192 ✓ **정확** |

- **test 는 정확히 일치**한다. test 행 수 = 그래프 수 × 0.15 로 결정론적이기 때문이다
  (255/0.15 = 1700, 192/0.15 = 1280 — 둘 다 정수로 떨어진다. 이게 unique 그래프 수를
  역산하는 근거였고, N-Queens 의 772 = 5148×0.15, 6513 = 43420×0.15 과 같은 규율이다).
- **train 은 −6 / −4 로 미달**이며 이게 최선이다. 이유: train 행 수는 "그래프당 canonical
  coloring 수"의 합이라 확률변수다. 실측 변동폭:
  - p 를 0.0005 씩 움직이면 n=8 은 6,776~7,283 사이를 진동(±250), n=10 은 12,948~13,704(±400).
  - seed 를 0~24 로 바꾸면 n=8 은 6,650~7,481(±400), n=10 은 11,426~15,258(±1,900).
  - **25개 seed 전부 시도했지만 정확히 앵커에 떨어지는 seed 는 없었다.** 최선이 seed 0 (−6/−4).
- 즉 잔차 −6/−4 는 **p 나 필터가 틀려서가 아니라 샘플링 노이즈**다. 오히려 노이즈 폭(±400)의
  1/60 수준으로 앵커에 붙었다는 것이 p·필터가 맞다는 근거다. 앵커에 정수까지 맞추려면
  seed 를 노이즈에 맞춰 고르는 수밖에 없는데, 그건 검증이 아니라 끼워맞추기라 하지 않았다.
  (원하시면 seed 를 더 넓게 뒤져 정확히 맞출 수 있다 — 말씀만 주시면 된다.)

### 2-5. 무결성 수치 (빌더 self-check, 두 n 모두 PASS)

| 항목 | n=8 | n=10 |
|---|---|---|
| 라벨 `valid` (train / test) | **1.000000 / 1.000000** | **1.000000 / 1.000000** |
| 라벨 conflict 총합 (train / test) | **0 / 0** | **0 / 0** |
| `colors_ok` | 1.000000 | 1.000000 |
| leakage: distinct x 의 train∩test | **0** | **0** |
| 중복 (x,y) 행 | train 0 / test 0 | train 0 / test 0 |
| x-split | train 1,445 / test 255 그래프, 교집합 0 | train 1,088 / test 192, 교집합 0 |
| 음성 대조: edge 하나를 강제로 충돌시킴 | valid=**0**/1000 (평균 conflict 1.947) | valid=**0**/1000 (평균 1.985) |
| 3-colorable 아님으로 버린 비율 | 3,134/8,192 | 1,683/4,096 |
| 그래프당 canonical coloring | 평균 4.82 (1~144) | 평균 12.38 (1~288) |

**정규형이 실제로 먹었는지 별도 검증** (이게 안 되면 같은 그래프가 최대 6개 라벨로 중복):

| | n=8 | n=10 |
|---|---|---|
| permutation-중복 라벨쌍 | **0** | **0** |
| (검사한 다중해 그래프 수) | 1,071 | 969 |
| 모든 라벨이 first-appearance 정규형 | **True** | **True** |

같은 그래프에 붙은 모든 라벨쌍에 대해 3!=6가지 색 치환을 전부 대입해 서로 옮겨지는 쌍이
있는지 봤고, **0쌍**이다. 색 치환 중복은 없다.

### 2-6. verifier / evaluator 동작 확인 (end-to-end, CPU)

`evaluators/graph_coloring.py`:
- `verify_colorings(preds, inputs, n)` → `valid`, **`conflicts`(위반 edge 수)**, `colors_ok`,
  `num_edges`, `num_colors_used`. 인접한 두 노드가 같은 색인 edge 를 세는 방식.
- `GraphColoring` evaluator: `nqueens@NQueens` 와 같은 API(`required_outputs`, `begin_eval`,
  `update_batch`, `result`). seq_len 에서 n 을 역산(`n(n-1)/2` 역함수: 28→8, 45→10, 검증됨).
- N-Queens 와 달리 `consistent` 개념이 없다. 입력이 "부분 답"이 아니라 그래프 자체라
  보존할 대상이 없기 때문. 그래서 `valid` 가 헤드라인이다.

실제 `PuzzleDataset` 로 로드해서 돌린 결과:

| 주입한 예측 | n=8 | n=10 |
|---|---|---|
| **정답 라벨** | `gc/accuracy=1.0`, conflicts_mean=0.0 | `gc/accuracy=1.0`, conflicts_mean=0.0 |
| **무작위 색** | `gc/accuracy=0.0039`, conflicts_mean=4.36 | `gc/accuracy=0.0`, conflicts_mean=5.77 |

또 확인한 것: metadata 가 `seq_len=28/45, vocab_size=6, ignore_label_id=0` 으로 읽히고,
배치의 라벨 뒤쪽 padding 이 전부 **-100** 으로 매핑되며 앞 n 칸은 전부 3~5 범위다.
→ 데이터·verifier·evaluator 가 학습 파이프라인에 그대로 물린다. **학습은 하지 않았다.**

---

## 3. ARC-AGI — 인프라 점검

### 3-1. 코드

| 항목 | 상태 |
|---|---|
| `dataset/build_arc_dataset.py` | **있음** (11,406 bytes) |
| `evaluators/arc.py` | **있음**, `class ARC` |

`evaluators/arc.py` 의 `ARC.__init__(data_path, eval_metadata, submission_K=2,
pass_Ks=(1,2,5,10,100,1000), aggregated_voting=True)`. NQueens 와 달리
`required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}` 로
**`puzzle_identifiers` 와 `q_halt_logits` 를 요구**한다. 또 `data_path` 안에
`identifiers.json` 과 **`test_puzzles.json`** 을 읽는다 — 둘 다 빌더가 만들어주는 파일이라,
빌드를 돌리기 전엔 evaluator 를 쓸 수 없다.

`build_arc_dataset.py` 가 실제로 찍는 metadata (`dataset/build_arc_dataset.py:297-302`):
```python
seq_len = ARCMaxGridSize * ARCMaxGridSize     # 30*30 = 900
vocab_size = 10 + 2                            # PAD + EOS + "0".."9" = 12
num_puzzle_identifiers = num_identifiers
```
→ **지시하신 seq 900, vocab 12(0=pad, 1=eos, 2-11=color) 와 일치 확인.**
30×30 pad + EOS 인코딩도 `np_grid_to_seq_translational_augment()` 주석
(`# PAD: 0, <eos>: 1, digits: 2 ... 11`)으로 확인.

CLI 인자: `input_file_prefix`, `output_dir`, `subsets`, `test_set_name`, `test_set_name2`,
`seed=42`, `num_aug=1000`, `puzzle_identifiers_start=1`.

### 3-2. 데이터 — **있음**

`kaggle/combined/` 에 전부 있다 (2026-07-08 자):
```
arc-agi_training_challenges.json     (1.27MB)   arc-agi_training_solutions.json     (185KB)
arc-agi_training2_challenges.json    (4.01MB)   arc-agi_training2_solutions.json    (659KB)
arc-agi_evaluation_challenges.json   (2.06MB)   arc-agi_evaluation_solutions.json   (313KB)
arc-agi_evaluation2_challenges.json  (985KB)    arc-agi_evaluation2_solutions.json  (224KB)
arc-agi_concept_challenges.json      (804KB)    arc-agi_concept_solutions.json      (210KB)
```
→ ARC-AGI-1(training/evaluation), ARC-AGI-2(training2/evaluation2), ConceptARC 모두 존재.
`input_file_prefix=kaggle/combined/arc-agi` 로 바로 빌드 가능한 배치다.
**지시대로 빌드는 돌리지 않았다** (`dataset/raw-data`, `data/arc*` 는 없음 = 아직 미빌드).

### 3-3. gram_arc.yaml 에 필요한 필드 (목록만, 생성 안 함)

`gram_nqueens8.yaml` 기준으로 ARC 용에 반드시 달라지는 것:

| 필드 | 값 | 근거/비고 |
|---|---|---|
| `puzzle_emb_ndim` | `${.hidden_size}` (활성) | **puzzle_emb 활성 필수.** 현재 8×8 도 이미 활성이라 형태는 같다 |
| `puzzle_emb_len` | 16 | 8×8과 동일. ARC 는 puzzle 별 임베딩이 실제로 의미를 가짐 |
| `kl_coef` (β) | **0.04 / 0.1** | 지시받은 값. 어느 쪽인지는 실험 arm 으로 갈림 |
| `L_cycles` (K) | **4** | 지시하신 K=4 |
| (seq_len 900) | 자동 | yaml 아님. 빌드 산출 metadata 에서 주입 |
| (vocab_size 12) | 자동 | 위와 동일 |
| (aug) | `num_aug=1000` | yaml 아님 — **빌드 시점** 인자 |
| `pos_encodings` | rope | 8×8과 동일 추정 |
| `halt_max_steps` | 16 | 8×8과 동일 추정 |

- `num_puzzle_identifiers` 는 빌드 결과(퍼즐 수)에 따라 정해지므로 미리 못 적는다. 자동 주입.
- evaluator 는 `evaluators=[{name: arc@ARC}]` 형태가 될 텐데, ARC 는 `q_halt_logits` 를
  요구하므로 모델 출력에 그게 실려야 한다. **GRAM 모델이 q_halt_logits 를 evaluator 로
  넘겨주는지는 확인 못 했다** (학습을 안 돌렸으므로 실측 불가).
- **yaml 생성·빌드·학습 전부 하지 않았다.** 200K epoch ≈ 5일이라 사수 승인 대상.

---

## 4. 임의로 정한 것 (전부 명시)

1. **GC 라벨의 tail padding 방식** — 논문에 "출력 길이 N" 만 있고 seq_len n(n-1)/2 와의
   길이 차이를 어떻게 채우는지는 지시/논문에 없었다. **앞 n 칸에 색, 나머지는
   `ignore_label_id=0`** 으로 정했다. 근거: 이 코드베이스가 이미 0 → -100 → loss 마스킹
   경로를 갖고 있어(`puzzle_dataset.py:152`, `models/losses.py:66`) 추가 코드 없이 동작하고,
   padding 이 학습 신호를 오염시키지 않는다. 실측으로 -100 매핑 확인했다.
   대안(색을 뒤 n 칸에 두기, 별도 seq_len=N 쓰기)도 가능하니 다르게 원하시면 바꾸겠다.
2. **unique 그래프 수 1700(n=8) / 1280(n=10)** — 논문이 준 건 train/test 쌍 수뿐이다.
   test 쌍 수 ÷ 0.15 로 역산했다(255/0.15=1700, 192/0.15=1280, 둘 다 정수로 떨어짐).
   N-Queens 빌더의 "test 는 입력당 1행" 규율과 일치해서 이 해석을 택했다.
3. **p = 0.5455 (n=8) / 0.4250 (n=10)** — seed 0 고정, 0.0005 간격 스캔에서 앵커에 가장
   가까운 값. 정확히 맞추려 seed 를 뒤지지는 않았다(2-4 참조).
4. **test 라벨은 사전순 첫 canonical coloring** — N-Queens 빌더가 test 에서 정렬된 해 중
   첫번째를 쓰는 것과 동일하게 맞췄다(무작위 대신 결정론적).
5. **중복 그래프 판정은 라벨된 상삼각 해시 기준** (그래프 동형(isomorphism)까지 보지 않음).
   논문에 언급이 없고, 입력 x 가 곧 상삼각 토큰이라 "unique input" 정의와 일치한다.
   동형까지 제거하면 그래프 수가 줄어 앵커가 안 맞는다.
6. **`gc/` 메트릭 접두사와 지표 구성** — `nqueens/` 관례를 따라 `gc/accuracy`(헤드라인),
   `gc/conflicts_mean`, `gc/colors_ok_rate`, `gc/num_colors_used_mean`, `gc/n_eval`.
7. **GC 에는 `consistent` 지표를 두지 않았다** — 입력이 부분 답이 아니라 그래프 자체라
   보존할 입력 원소가 없다. N-Queens 의 consistent 에 해당하는 개념이 성립하지 않는다.

## 5. 확인 못 한 것

- **10×10 의 `L_cycles`, `epochs` 논문값** — 확인 못 함. 웹 검색을 하지 말라고 하셨고
  논문 파일이 로컬에 없다. 지시로 받은 건 β=0.045 뿐이다.
- **GRAM 모델이 `q_halt_logits` 를 evaluator 로 노출하는지** — ARC evaluator 가 요구하는데,
  학습/빌드를 안 돌렸으므로 실측 확인 못 함.
- **GC 의 논문 p 값** — 애초에 논문 미명시라고 하셨고, 크기 앵커로 역산했다.
- **ARC `num_puzzle_identifiers`** — 빌드해야 정해지는 값이라 미확인.
