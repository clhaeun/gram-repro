# REPORT_sweep_50.md — N-Queens 8×8 β sweep, one-hot **ON** (50번 lecun docker)

- 일시: 2026-07-27
- 서버: 50번 lecun docker (`trm_chaeun`), `/workspace/TinyRecursiveModels`, root
- 사용 GPU: **4, 5, 6, 7** (run 당 1개). GPU 0~3 은 타 사용자 — 조회만 하고 일절 미사용.
- 커밋: `6a13eb6` (pull 시점 이미 최신)
- 짝: 29번이 **동일 β 를 one-hot OFF** 로 돌린다. 이 리포트는 **ON** 쪽 절반이다.

---

## 1. Task 0 — 사전 확인

```
$ git pull origin main
Already up to date.                     # HEAD = 6a13eb6

$ ls data/nqueens-8/train/
all__group_indices.npy  all__inputs.npy  all__labels.npy
all__puzzle_identifiers.npy  all__puzzle_indices.npy  dataset.json

$ grep -c y_onehot config/arch/gram_nqueens8.yaml
3                                       # 지시한 3필드 존재

$ nvidia-smi (launch 직전)
4, 15 MiB   5, 15 MiB   6, 15 MiB   7, 15 MiB      <- 4개 전부 유휴, 사용
0~3, 21737 MiB                                      <- 타 사용자. 미사용
```
`run_name` 중복 없음 — 네 이름 전부 checkpoint/log 모두 신규.

---

## 2. Task 1/2 — launch + 200 step 검증

tmux 세션 `sweep50`, 창 4개, 각 1 GPU. 로그는 `tee`(overwrite).

| run | β (kl_coef) | GPU | port |
|---|---|---|---|
| `nq8_on_b100` | 1.0 | 4 | 29711 |
| `nq8_on_b140` | 1.4 | 5 | 29712 |
| `nq8_on_b180` | 1.8 | 6 | 29713 |
| `nq8_on_b220` | 2.2 | 7 | 29714 |

### config dump 발췌 (`checkpoints/Nqueens-8-ACT-torch/<run>/all_config.yaml`)

```yaml
# nq8_on_b100                  # nq8_on_b140
  kl_coef: 1.0                   kl_coef: 1.4
  kl_reduction: mean             kl_reduction: mean
  y_onehot: true                 y_onehot: true
  y_onehot_proj: zeropad         y_onehot_proj: zeropad
  y_onehot_scale: 0.573          y_onehot_scale: 0.573
ema: false                     ema: false

# nq8_on_b180                  # nq8_on_b220
  kl_coef: 1.8                   kl_coef: 2.2
  kl_reduction: mean             kl_reduction: mean
  y_onehot: true                 y_onehot: true
  y_onehot_proj: zeropad         y_onehot_proj: zeropad
  y_onehot_scale: 0.573          y_onehot_scale: 0.573
ema: false                     ema: false
```

### 검증 결과

| 항목 | b100 | b140 | b180 | b220 |
|---|---|---|---|---|
| y_onehot | **true** ✓ | **true** ✓ | **true** ✓ | **true** ✓ |
| y_onehot_scale | 0.573 ✓ | 0.573 ✓ | 0.573 ✓ | 0.573 ✓ |
| kl_coef | 1.0 ✓ | 1.4 ✓ | 1.8 ✓ | 2.2 ✓ |
| ema | false ✓ | false ✓ | false ✓ | false ✓ |
| **num_params** | **9966085** ✓ | **9966085** ✓ | **9966085** ✓ | **9966085** ✓ |
| NaN | 없음 ✓ | 없음 ✓ | 없음 ✓ | 없음 ✓ |
| nqueens/accuracy 로그 | 찍힘 ✓ | 찍힘 ✓ | 찍힘 ✓ | 찍힘 ✓ |
| eval kl_loss | **0.0000** ✓ | **0.0000** ✓ | **0.0000** ✓ | **0.0000** ✓ |

num_params 는 `pretrain.py:624` 가 wandb 로만 보내므로 각 run 의 `wandb-summary.json` 에서
읽었다 — 네 run 전부 **9,966,085**, 예상치와 일치.

완주: 4개 전부 step 17085 도달, eval 15회, crash/NaN 0.

---

## 3. Task 3 — in-loop `nqueens/accuracy` 곡선 (prior 경로)

| step | β=1.0 | β=1.4 | β=1.8 | β=2.2 |
|---|---|---|---|---|
| 1139 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2278 | 0.0000 | 0.0091 | 0.0000 | 0.0000 |
| 3417 | 0.0738 | 0.0661 | 0.0142 | 0.0000 |
| 4556 | 0.2992 | 0.1580 | 0.0000 | 0.0039 |
| 5695 | 0.1697 | 0.1192 | 0.2098 | 0.0000 |
| 6834 | 0.2150 | 0.1166 | 0.1982 | 0.0000 |
| 7973 | 0.2746 | 0.2150 | 0.2254 | 0.0039 |
| 9112 | 0.0531 | 0.1516 | 0.2591 | 0.0389 |
| 10251 | 0.1412 | 0.0984 | 0.3070 | 0.1127 |
| 11390 | 0.1360 | 0.2824 | 0.4028 | 0.1736 |
| 12529 | 0.1425 | 0.2902 | 0.3096 | 0.0829 |
| 13668 | 0.1865 | 0.1205 | 0.2267 | 0.2280 |
| 14807 | 0.2383 | 0.1593 | **0.6244** | 0.2319 |
| 15946 | 0.1308 | 0.1373 | 0.4702 | 0.3588 |
| 17085 | **0.5285** | **0.3666** | 0.0492 | **0.4093** |

| run | β | peak step | peak | final | 하강 시작 |
|---|---|---|---|---|---|
| `nq8_on_b100` | 1.0 | **17085 (=final)** | 0.5285 | 0.5285 | **없음 — 마지막까지 상승** |
| `nq8_on_b140` | 1.4 | **17085 (=final)** | 0.3666 | 0.3666 | **없음 — 마지막까지 상승** |
| `nq8_on_b180` | 1.8 | **14807** | **0.6244** | 0.0492 | 15946 (이후 붕괴) |
| `nq8_on_b220` | 2.2 | **17085 (=final)** | 0.4093 | 0.4093 | **없음 — 마지막까지 상승** |

**4개 중 3개가 마지막 step 에서 정점이다 = 아직 수렴하지 않았다.**
epochs=3000 이 이 β 대역에는 부족하다는 뜻이고, 더 돌리면 순위가 바뀔 수 있다.
곡선 자체도 단조롭지 않다(β=1.0 은 4556 에서 0.2992 → 9112 에서 0.0531 로 떨어졌다가 다시 상승).

---

## 4. Task 3 — harness (prior N=20 + posterior, test 772 전수, harness seed 2개)

peak checkpoint 기준. β=1.8 만 peak≠final 이라 붕괴 확인차 final 도 같이 돌렸다.

| β | run | step | prior valid (s0/s1) | **평균** | posterior valid | pass@1 | pass@20 | distinct@20 | prior queens |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | `nq8_on_b100` | 17085 | 0.5151 / 0.5180 | **0.5165** | 0.9826 | 0.5165 | 0.994 | 7.95 | 8.04 (4–11) |
| 1.4 | `nq8_on_b140` | 17085 | 0.3365 / 0.3276 | **0.3320** | 0.9987 | 0.3320 | 0.970 | 9.08 | 7.81 (6–9) |
| **1.8** | `nq8_on_b180` | **14807** | 0.6260 / 0.6276 | **0.6268** | 0.9813 | 0.6268 | **0.996** | 6.92 | 7.96 (2–15) |
| 1.8 | `nq8_on_b180` | 17085 | 0.0669 / 0.0706 | 0.0688 | 0.8873 | 0.0688 | 0.600 | 17.71 | 7.85 (4–11) |
| 2.2 | `nq8_on_b220` | 17085 | 0.4152 / 0.4146 | **0.4149** | 0.9864 | 0.4149 | 0.986 | 8.66 | 7.96 (5–11) |

harness seed 편차는 전부 **≤ 0.0089** (최대: β=1.4 의 0.3365↔0.3276). prior 샘플링
노이즈는 무시할 수준 — 위 수치는 1 draw 우연이 아니다.

---

## 5. 판정 재료 (해석은 사용자)

### 5-1. 이번 sweep 의 prior valid (peak 기준)

```
β=1.8  0.6268   <- 최고 (step 14807)
β=1.0  0.5165
β=2.2  0.4149
β=1.4  0.3320
```

### 5-2. 기존 8×8 수치와의 대조 (전부 같은 harness, 같은 test 772)

| 설정 | one-hot | prior valid | posterior valid |
|---|---|---|---|
| β=0.07 `gram_nq8_fix_b` (final) | ✗ | 0.0000 | 0.9974 |
| β=0.07 `nq8_oh_b007` (peak/final) | ✓ | **0.0000** | 1.0000 |
| β=1.5 `nq8_b15` (peak 6834) | ✗ | 0.3029 | 0.9935 |
| β=1.5 `nq8_ctrl_b15` (peak 12529) | ✗ | 0.0571 | 0.9994 |
| β=1.5 `nq8_oh_b15` (peak 11390) | ✓ | 0.2948 | 1.0000 |
| **β=1.8 `nq8_on_b180` (peak 14807)** | ✓ | **0.6268** | 0.9813 |
| **β=1.0 `nq8_on_b100` (final)** | ✓ | **0.5165** | 0.9826 |
| **β=2.2 `nq8_on_b220` (final)** | ✓ | **0.4149** | 0.9864 |
| **β=1.4 `nq8_on_b140` (final)** | ✓ | **0.3320** | 0.9987 |

- 이번 4개 β 전부가 기존 최고치(β=1.5 one-hot 없음 0.3029)를 넘거나 맞먹는다.
  β=1.8 의 **0.6268 은 8×8 에서 지금까지 본 prior valid 중 최고**다(기존 최고의 2.07배).
- **posterior 는 전 구간 0.98~1.00 으로 유지**된다. one-hot 이 posterior 를 누른 흔적은
  이번에도 없다 — 개선은 전부 prior 쪽이다. 예외는 β=1.8 **final**(0.8873)인데, 이건
  붕괴 후 상태다.
- β=0.07 은 one-hot 유무와 무관하게 prior 0.0000 — 이전 결론 그대로.

### 5-3. 이번 데이터로 **말할 수 없는** 것

- **β 순위(1.8 > 1.0 > 2.2 > 1.4)는 확정이 아니다.** β 당 학습 seed 1개뿐이다.
  이 코드베이스에서 **같은 config·같은 seed 인 두 run 이 prior valid 0.3029 vs 0.0571
  (5.3배)** 로 갈린 전례가 있다(`nq8_b15` vs `nq8_ctrl_b15`, REPORT_onehot.md 5-4).
  1.4(0.3320)와 2.2(0.4149)의 차이는 그 편차폭 안에 충분히 들어간다.
- **one-hot 의 순수 효과도 이 리포트만으로는 못 낸다.** 29번의 one-hot OFF 결과와
  β 별로 짝지어야 한다. 그때도 arm 당 seed 1개면 위와 같은 한계가 그대로 남는다.
- **3개 run 이 미수렴**(마지막 step 이 정점)이라 위 수치는 하한에 가깝다.

### 5-4. β=1.8 의 붕괴 (설계에 쓸 재료)

step 14807 에서 prior valid **0.6268** → 17085 에서 **0.0688** (−0.558, 1/9 토막).
동시에 일어난 것:
- posterior valid 0.9813 → 0.8873 (posterior 까지 같이 무너졌다)
- distinct@20 6.92 → 17.71 (샘플 다양성 급증 = 답이 흩어짐)
- pass@20 0.996 → 0.600

in-loop 곡선도 14807(0.6244) → 15946(0.4702) → 17085(0.0492) 로 2 eval 연속 하락했다.
**early stop 이 없으면 최고 checkpoint 를 그냥 잃는다.** β=1.8 을 다시 돌린다면
eval 마다 prior valid 를 보고 하락 2회에서 멈추는 장치가 필요하다.

---

## 6. 임의로 정한 것

1. **harness 를 peak checkpoint 에 돌렸고, β=1.8 만 final 도 추가로 돌렸다.** 나머지 3개는
   peak==final(17085) 이라 한 번이면 충분하다. β=1.8 은 peak 과 final 이 달라 붕괴 폭을
   수치로 남기려고 둘 다 했다.
2. **harness seed 2개(0, 1).** 이번 지시엔 명시가 없었지만 직전 지시의 "1 draw 로 단정 금지"
   방침을 그대로 적용했다. 편차는 전부 ≤0.0089 로 작았다.
3. **"하강 시작점" 정의** = peak 다음 eval 부터 연속 하락이 시작되는 step. 3개 run 은
   정점이 마지막 step 이라 하강 구간 자체가 없어 "없음"으로 적었다.
4. **`experiments/run_sweep50_harness.sh` 를 새로 만들었다** (기존 `run_onehot_harness.sh`
   는 peak+final 고정이라 step 목록을 인자로 받게 고친 판). 커밋은 안 했다.
5. **GPU 4→b100, 5→b140, 6→b180, 7→b220** 으로 고정 배정. 학습·harness 모두 동일 배정.

## 7. 확인 못 한 것

- **29번(one-hot OFF)의 같은 β 결과** — 이 서버에 없다. 짝 비교는 그쪽 수치가 와야 가능하다.
- **β 순위의 유의성** — 학습 seed 1개씩이라 미확정 (5-3).
- **수렴 후 값** — 3개 run 이 미수렴 상태로 끝났다. epochs 를 늘린 결과는 확인 못 함.
