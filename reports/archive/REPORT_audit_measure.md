# REPORT_audit_measure.md — 측정 방법론 감사 (29번 hinton)

- 일시: 2026-07-28
- 범위: 읽기·검증만. GPU 미사용, 도는 학습 미접촉 (가중치 비교는 CPU `map_location="cpu"`).
- 대상 코드: `pretrain.py`, `models/ema.py`, `experiments/sweep_harness.py`, `experiments/gram_harness.py`

---

## 0. 감사 중 발견한 것 (지시 범위 밖이지만 1~4의 전제를 바꾸므로 먼저 적는다)

**같은 seed 가 같은 학습 궤적을 만들지 않는다.** 이 서버에 같은 조건(β=2.2, one-hot OFF, seed 0,
epochs 3000)으로 돌린 run 이 두 개 있다 — `nq8_off_b220`(EMA off)과 `nq8_off_b220_ema_s0`(EMA on).
EMA 는 아래 1-3 에서 보듯 학습에 개입하지 않는 순수 사본이므로, 두 run 의 **raw 가중치는
같아야 한다**. 실제로 비교하면 다르다.

| 비교 대상 | 상대 L2 차이 | 다른 텐서 |
|---|---|---|
| `nq8_off_b220/step_1139` vs `nq8_off_b220_ema_s0/step_1139.live` | **9.15%** | 25/27 |
| `nq8_off_b220/step_17085` vs `nq8_off_b220_ema_s0/step_17085.live` | **55.04%** | 25/27 |

**step 1139 은 eval 이 한 번도 일어나기 전 지점이다** (iteration 0 = 200 epoch = 1139 step,
eval 은 그 뒤에 온다). 즉 "eval 이 EMA 가중치로 도느라 RNG 소비가 달라졌다"로는 설명되지 않는다.
추가로 두 run 모두 전 eval 에서 `all/steps=16.0000` 으로 동일해 ACT 분기에 의한 RNG 소비 차이도 없다.

결정성 설정은 코드·sh 어디에도 없다 (`use_deterministic_algorithms`, `cudnn.deterministic`,
`CUBLAS_WORKSPACE_CONFIG` 전부 부재 → 비결정적 커널 허용이 기본).

**확인 못 함**: 두 run 은 물리적으로 다른 GPU(3번 vs 6번, 같은 RTX 3090)에서 돌았다.
같은 GPU 에서 두 번 돌려 비교하지 않았으므로, 이 차이가 순수 커널 비결정성인지
장치 차이가 섞인 것인지는 분리하지 못했다.

**의미**: 지금까지 "seed 편차"로 불러 온 것은 **seed 가 통제 변수인 편차가 아니다.**
seed 를 고정해도 run 마다 다른 모델이 나온다. 따라서 "같은 config 30% vs 6%" 같은 관측은
seed 라벨이 아니라 **run 반복 수**로 기술해야 한다.

---

## 1. 무엇이 저장되고 무엇이 평가되는가

### 1-1. checkpoint 저장 — `pretrain.py:687-692`

```python
if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
    save_train_state(config, train_state_eval)                       # step_N = (bias-corrected) EMA
    if config.ema:
        # also persist the raw live (non-EMA) SGD weights, so EMA
        # effects can be separated from path effects post-hoc.
        save_train_state(config, train_state, suffix=".live")        # step_N.live
```

- `step_N` 은 `train_state_eval` 을 저장한다. `train_state_eval` 이 무엇인지는 1-2 에서 갈린다.
  - **EMA on → `step_N` 은 bias-corrected EMA 가중치** (raw 아님).
  - **EMA off → `train_state_eval = train_state` 이므로 `step_N` 이 곧 raw.**
- `step_N.live` 는 `train_state`(raw SGD 가중치)를 그대로 저장하며 **EMA on 일 때만 생긴다.**

실측으로도 일치한다: EMA=True run 은 `.live` 가 checkpoint 수만큼 있고, EMA=False run 은 `.live` 가 0개다.

### 1-2. in-loop eval 이 쓰는 가중치 — `pretrain.py:655-662`

```python
if config.ema:
    print("SWITCH TO EMA (bias-corrected)")
    train_state_eval = copy.deepcopy(train_state)
    # bias-corrected EMA: removes the surviving mu^step * w0 init term
    train_state_eval.model = ema_helper.ema_copy(train_state_eval.model, step=train_state.step)   # :659
else:
    train_state_eval = train_state
train_state_eval.model.eval()                                                                     # :662
```

**EMA on 이면 in-loop eval 은 bias-corrected EMA 사본으로 돈다.** 따라서 로그의
`nqueens/accuracy` 곡선은 EMA 가중치의 곡선이며 raw 모델의 성능이 아니다.

부수 확인: `train_state.model.train()` 이 매 iteration 머리(`pretrain.py:638`)에서 호출되므로
`:662` 의 `.eval()` 이 다음 학습 구간으로 새지 않는다. EMA off 일 때 `.eval()` 이 live 모델에
직접 걸리지만 다음 iteration 에서 복구된다. **이 경로에 EMA on/off 비대칭은 없다.**

### 1-3. EMA 가 학습에 개입하는가 — **개입하지 않는다 (순수 평가용 사본)**

`models/ema.py`:

```python
def update(self, module):
    ...
    for name, param in module.named_parameters():
        if param.requires_grad:
            self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data
```

`update()` 는 `param.data` 를 **읽기만** 하고 `self.shadow` 에만 쓴다. 모델 파라미터를 수정하지 않는다.

```python
def ema_copy(self, module, step=None):
    module_copy = copy.deepcopy(module)      # 사본에만 기록
    ...
```

`ema_copy()` 는 `deepcopy` 후 사본에 써 넣는다. 호출부(`pretrain.py:657-659`)도
`copy.deepcopy(train_state)` 를 먼저 뜨고 사본의 `.model` 만 교체한 뒤 `del train_state_eval`(`:694`) 한다.

호출 지점은 `pretrain.py:649` (`ema_helper.update(train_state.model)`) 하나뿐이고, optimizer step
이후 순수 관측이다. **gradient·optimizer 경로에 EMA 는 없다.**

→ **코드상 EMA on/off 는 같은 seed 에서 같은 궤적을 내야 한다.** 그러나 0절에서 보듯 실제로는
다르다. 코드 경로가 아닌 **실행 비결정성**이 원인이라는 뜻이다(장치 차이 confound 는 미분리).

### 1-4. bias correction 의 크기 — `models/ema.py:_debiased`

```python
def _debiased(self, name, step):
    mt = self.mu ** step
    denom = 1.0 - mt
    if denom < 1e-8:
        return self.shadow[name].data
    return (self.shadow[name].data - mt * self.w0[name].data) / denom
```

`mu=0.9999` 에서 step 별 계수 (직접 계산):

| step | `mu^t` (init 잔존) | `denom = 1-mu^t` | 증폭 `1/denom` |
|---|---|---|---|
| 1139 (첫 eval) | **89.23%** | 0.1077 | **9.29×** |
| 2278 | 79.63% | 0.2037 | 4.91× |
| 4556 | 63.41% | 0.3659 | 2.73× |
| 7973 | 45.05% | 0.5495 | 1.82× |
| 13668 | 25.49% | 0.7451 | 1.34× |
| 17085 (3000ep 끝) | 18.11% | 0.8189 | 1.22× |
| 34187 (6000ep 끝) | 3.27% | 0.9673 | 1.03× |

**초기 step 에서 bias correction 은 9배 이상 증폭한다.** 이는 첫 eval(step 1139)의
EMA 가중치가 all-empty(queens 0.00)로 나오고 같은 step `.live` 는 정상(queens 2.78)이었던
관측과 방향이 맞는다.

**단, 후기 step 의 EMA/live 괴리는 이것으로 설명되지 않는다.** step 17085 의 증폭은 1.22× 에
불과한데 관측된 괴리는 3.1배(β=1.8 s0) ~ 395배(50번 β=1.6 s1)다. **후기 괴리의 원인은
이 감사에서 확인하지 못했다.**

### 1-5. harness 가 로드하는 가중치 — **명시 지정이 필수**

`experiments/gram_harness.py:70-81`:
```python
def load_model(ckpt_dir, step):
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "all_config.yaml"))
    cfg.load_checkpoint = os.path.join(ckpt_dir, step)
    cfg.ema = False                  # checkpoint is already the EMA copy
```

`experiments/sweep_harness.py:52`: `ap.add_argument("--step", required=True)`

- harness 는 **`--step` 으로 준 파일명을 그대로** 로드한다. 기본값이 없다.
- `--step step_N` → EMA on run 이면 **EMA 가중치**를 잰다.
- raw 를 재려면 **`--step step_N.live` 를 명시해야 한다.**
- `cfg.ema = False` 는 "checkpoint 가 이미 EMA 사본이므로 다시 감싸지 말라"는 뜻이지
  raw 를 로드한다는 뜻이 아니다.

---

## 2. 지금까지 낸 수치의 출처 감사

이 서버 24개 run 을 `all_config.yaml` 의 `ema` 값으로 분류했다.

### 2-1. EMA 편향 **없음** — `step_N` 이 곧 raw, 곡선도 raw

| REPORT | run | β | 비고 |
|---|---|---|---|
| `REPORT_sweep_29.md` | `nq8_off_b100/140/180/220` | 1.0/1.4/1.8/2.2 | EMA=false |
| `REPORT_finesweep_29.md` | `nq8_on_b160/170/190/200` | 1.6/1.7/1.9/2.0 | EMA=false |
| `REPORT_ema_seed_29.md` | `nq8_on_b180_s3` | 1.8 | EMA=false |
| (kill 됨) | `nq8_off_b260/320/400/500` | 2.6~5.0 | EMA=false, step 1139 1점만 |

이 run 들은 **곡선·peak 선택·harness 전부 raw 기준**이라 재해석이 필요 없다.

### 2-2. EMA 편향 **있음** — 보고한 값이 EMA 가중치이고, peak 도 EMA 곡선으로 골랐다

| REPORT | run | β | seed | 보고한 prior valid | 그 값의 가중치 | peak 선택 곡선 |
|---|---|---|---|---|---|---|
| `REPORT_ema_seed_29.md` | `nq8_on_b180_ema_s0` | 1.8 | 0 | 9.60% (step 13668) | **EMA** | **EMA 곡선** |
| `REPORT_ema_seed_29.md` | `nq8_on_b180_ema_s1` | 1.8 | 1 | 0.12% (step 7973) | **EMA** | **EMA 곡선** |
| `REPORT_ema_seed_29.md` | `nq8_off_b220_ema_s0` | 2.2 | 0 | **63.12%** (step 17085) | **EMA** | EMA 곡선 (peak=final) |
| `REPORT_grid_upper_29.md`(작성 전) | `nq8_on_b140_ema/b200_ema/b450_ema/b560_ema` | 1.4/2.0/4.5/5.6 | 0 | 미보고 | — | — |
| `REPORT_off_line_29.md`(작성 전) | `nq8_off_b220_ema6k_s0/s1`, `b240_ema`, `b260_ema` | 2.2/2.2/2.4/2.6 | 0/1/0/0 | 미보고 | — | — |

**"EMA 곡선으로 peak 를 고르고 그 step 의 값을 보고했다" 에 해당하는 run (= 보고값이 `.live` 정점이 아닐 수 있음)**:
- `nq8_on_b180_ema_s0` — 곡선 peak 13668. 같은 step `.live` 는 **29.71%** 로 EMA(9.60%)의 3.1배.
  `.live` 의 정점이 13668 인지는 **측정하지 않았다 → 모른다.**
- `nq8_on_b180_ema_s1` — 곡선 peak 7973. 그런데 이 run 의 EMA 곡선은 전 구간 최대 0.0052 로
  사실상 바닥이라, 7973 은 **노이즈 최댓값**이지 의미 있는 peak 가 아니다. 같은 step `.live` 0.08%.
  `.live` 정점 위치·값 **모름**.
- `nq8_off_b220_ema_s0` — 곡선 peak 가 마지막 step(17085)이라 선택 자체는 문제가 덜하다.
  다만 이 run 은 **끝까지 상승 중(미수렴)** 이었으므로 3000 epoch 안에 정점이 없다.
  같은 step `.live` 48.66% < EMA 63.12% 로 **여기서는 EMA 가 더 높다**(방향이 반대).

### 2-3. 이미 측정된 EMA vs `.live` 대조 (전부 harness seed 0/1 평균)

| run | step | EMA | `.live` | 방향 |
|---|---|---|---|---|
| `nq8_off_b220_ema_s0` | 17085 | **63.12%** | 48.66% | EMA 우위 +14.46%p |
| `nq8_on_b180_ema_s0` | 13668 | 9.60% | **29.71%** | live 우위 3.1배 |
| `nq8_on_b180_ema_s1` | 7973 | 0.12% | 0.08% | 둘 다 바닥 |
| `nq8_on_b180_ema_s0` | 1139 | queens 0.00 | queens 2.78 | live 우위 (초기 bias correction 구간) |

**EMA/`.live` 우열은 조건에 따라 뒤집힌다.** "EMA 사본이 깨진다" 는 β=1.8 ON 에서는 맞지만
β=2.2 OFF 에서는 반대다. 한 방향으로 일반화할 근거는 이 데이터에 없다.

---

## 3. 재측정 필요 목록

### 3-1. 최우선 — `.live` 전체 곡선(15포인트) 스캔이 필요한 run

`.live` 정점의 위치와 값을 모르는 run. 2-2 의 세 개 + 진행 중 8개.

| 우선순위 | run | 이유 | step 수 |
|---|---|---|---|
| 1 | `nq8_on_b180_ema_s1` | EMA 곡선이 바닥이라 peak 선택이 무의미. 50번 β=1.6 s1(395배)과 같은 성격인지 판정 | 15 |
| 2 | `nq8_on_b180_ema_s0` | live 가 EMA 의 3.1배. 13668 이 live 정점인지 미확인 | 15 |
| 3 | `nq8_off_b220_ema_s0` | 이 라인은 EMA 우위. live 정점이 따로 있는지 | 15 |
| 4 | `nq8_on_b140_ema`, `nq8_on_b200_ema`, `nq8_on_b450_ema`, `nq8_on_b560_ema` | 격자 4점, 아직 미보고 → 처음부터 live 기준으로 | 15 ×4 |
| 5 | `nq8_off_b220_ema6k_s0/s1`, `nq8_off_b240_ema`, `nq8_off_b260_ema` | OFF 라인 4점, 아직 미보고 | 30/30/15/15 |

**이미 실행 중**: 우선순위 1(`nq8_on_b180_ema_s1`, 15포인트, harness seed 0)은 별건 지시로
GPU 0~3 반납 대기 큐에 걸어 뒀다 (`tmux livescan`).

### 3-2. 명령 (GPU 반납 후 실행)

```bash
# .live 전체 곡선 스캔 (한 run, 15포인트, harness seed 0, GPU 4장 분산)
RUN=nq8_on_b180_ema_s0        # 대상 run 만 바꾼다
STEPS=(1139 2278 3417 4556 5695 6834 7973 9112 10251 11390 12529 13668 14807 15946 17085)
for i in "${!STEPS[@]}"; do
  st=${STEPS[$i]}; gpu=$((i % 4))
  CUDA_VISIBLE_DEVICES=$gpu python experiments/sweep_harness.py \
    --ckpt-dir checkpoints/Nqueens-8-ACT-torch/$RUN \
    --step step_${st}.live --seed 0 \
    --out experiments/results/live_${RUN}_step_${st}_live_seed0.json &
  [ $gpu -eq 3 ] && wait
done; wait
```
6000-epoch run(`*_ema6k_*`)은 `STEPS` 를 30개(1139×1~30)로 바꾼다.

`.live` 정점을 찾은 뒤, 그 step 에 대해서만 harness seed 1 을 추가로 돌려 편차를 확인한다.

### 3-3. 재측정 **불필요**

2-1 의 EMA=false run 전부. 이미 raw 기준이다.

### 3-4. 별도 검증 제안 (0절 때문에 필요해진 것)

같은 GPU·같은 seed 로 같은 config 를 2회 돌려 `.live` 궤적이 일치하는지 확인.
0절의 9.15% 괴리가 커널 비결정성인지 장치 차이인지 분리하려면 이것이 필요하다.
지금 데이터로는 **분리 불가**.

---

## 4. 앞으로의 판정 규약 (제안 — 구현하지 않음)

### 4-1. 판정 기준 가중치

- **`.live`(raw) 를 1차 판정 기준으로 삼는다.** 근거: raw 는 EMA on/off 와 무관하게 존재하는
  유일한 공통 축이라 EMA=false run 과 직접 비교된다. EMA 가중치는 `mu`·step 에 의존하는 파생물이다.
- **EMA 값도 함께 보고한다.** 2-3 에서 우열이 뒤집히므로 어느 한쪽만 적으면 정보가 소실된다.
- 표기: `prior valid = 48.66% (.live) / 63.12% (EMA)` 처럼 **둘 다, 라벨과 함께**.
- EMA=false run 은 `step_N` 이 곧 raw 이므로 `(.live)` 와 동등하게 취급하되, `.live` 파일이
  없다는 점을 각주로 남긴다.

### 4-2. peak step 선택

- **`.live` 곡선에서 고른다.** 현재 in-loop 곡선은 EMA on 이면 EMA 곡선이라(1-2) peak 선택에
  쓸 수 없다.
- in-loop 곡선은 **이륙 여부·수렴 여부의 참고 지표로만** 쓰고, 수치 판정에는 쓰지 않는다.
- `.live` 곡선이 없으면 "peak 미확정" 으로 명시하고 final step 값을 보고한다
  (peak 라고 부르지 않는다).

### 4-3. REPORT 필수 명시 항목

1. 각 수치의 **가중치 종류** (`.live` / EMA / EMA=false 의 raw)
2. **peak step 과 그것을 고른 곡선** (`.live` 곡선 / EMA 곡선 / 미확정)
3. **수렴 여부** — 마지막 step 이 peak 면 "미수렴, 정점 미관측" 명시
4. **harness seed 개수와 편차 폭**
5. **run 반복 수** — seed 라벨이 아니라 반복 수로 (0절)

### 4-4. 코드 변경이 필요한 범위 (제안만, 구현 안 함)

| 제안 | 범위 | 효과 | 비용 |
|---|---|---|---|
| A. in-loop eval 에 raw 도 함께 로깅 | `pretrain.py:655-670`. EMA on 일 때 `train_state`(raw)로 eval 을 한 번 더 돌려 `eval_live/*` 로 기록 | `.live` 곡선이 학습 중에 바로 생겨 3-1 의 사후 스캔(run 당 15~30 job)이 불필요해진다 | **eval 시간 2배**. eval 은 15회뿐이라 전체 학습시간 증가는 작지만 측정 필요 |
| B. 결정성 옵션 | `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` | 0절의 same-seed 재현성 확보 | 속도 저하. 일부 커널은 미지원이라 에러 가능 — **적용 가능 여부 확인 못 함** |
| C. `sweep_harness.py` 에 `--weights {live,ema,both}` | harness CLI. `both` 면 `step_N` 과 `step_N.live` 를 한 번에 재고 같은 json 에 기록 | 대조 누락 방지, 파일명 규칙 통일 | 작음 |
| D. EMA bias correction 초기 구간 가드 | `models/ema.py:_debiased`. `denom` 이 임계 미만이면 EMA 평가를 건너뛰거나 경고 | 첫 eval 의 9.29× 증폭으로 인한 무의미한 곡선 구간 제거 | 작음. 다만 후기 괴리(1-4)는 해결 못 함 |

**권고 순서**: C(저비용, 즉시 효과) → A(측정 정확도의 핵심) → D → B(효과는 크나 적용 가능 여부 불명).

---

## 5. 확인하지 못한 것 (명시)

1. **후기 step 의 EMA/`.live` 괴리 원인.** step 17085 의 bias correction 증폭은 1.22× 인데
   관측 괴리는 3.1~395배다. 1-4 로 설명되지 않으며 다른 원인을 특정하지 못했다.
2. **0절 괴리의 원인 분리.** 커널 비결정성인지, 두 run 이 다른 물리 GPU(3번/6번)에서 돈 탓인지
   분리하지 못했다. 3-4 의 대조 실험이 필요하다.
3. **50번 수치(β=1.6 s1: EMA 0.0007 / live 0.2771)는 직접 검증하지 않았다.** 이 서버에
   해당 checkpoint 가 없다. 배경으로만 인용했다.
4. **`.live` 정점의 위치·값** — 2-2 의 세 run 전부 미측정 (3-1 에서 재측정 대상).
5. **결정성 옵션(B)의 적용 가능 여부** — 이 모델의 커널들이 deterministic 구현을 갖는지
   확인하지 않았다.
