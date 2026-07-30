# REPORT_audit_measure.md — 측정 방법론 감사 (무엇을 재고 있었는가)

- 일시: 2026-07-28
- **GPU 미사용. 학습 없음.** 읽기·검증만. (감사 시작 시점에 이 서버에서 도는 학습 없음:
  `pretrain.py` 프로세스 0개, `nvidia-smi --query-compute-apps` 결과 없음, GPU 4~7 각 15MiB.
  GPU 0~3 의 21737MiB 는 타 컨테이너 소유라 우리 쪽에서 보이지 않는다.)

> **감사 결론 요약**
> 1. `step_N` 은 EMA on 일 때 **EMA(bias-corrected) 사본**, `.live` 가 raw. EMA off 면 `step_N` 이 곧 raw.
> 2. in-loop eval 은 **EMA 사본**을 본다. harness 는 **파일명으로 지정한 것**을 본다(기본 `step_N`).
> 3. **EMA 는 순수 평가용 사본이다 — 학습 gradient 에 개입하지 않는다** (코드 근거 3-1).
> 4. **이 서버가 낸 수치 중 EMA 편향이 걸린 것은 `REPORT_emasweep_50.md` 4개 run 뿐이다.**
>    나머지 15개 run 은 전부 `ema=false` 라 checkpoint 자체가 raw — 편향 없음.
> 5. 다만 그 4개는 **peak step 을 EMA 곡선으로 골랐다.** `.live` 정점은 다른 step 일 수 있어
>    재측정이 필요하다(Task 3).

---

## 1. Task 1 — 무엇이 저장되고 무엇이 평가되는가

### 1-1. checkpoint 저장 (`pretrain.py:686-692`)

```python
if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
    save_train_state(config, train_state_eval)                       # step_N = (bias-corrected) EMA
    if config.ema:
        # also persist the raw live (non-EMA) SGD weights, so EMA
        # effects can be separated from path effects post-hoc.
        save_train_state(config, train_state, suffix=".live")        # step_N.live
```
`save_train_state` (`pretrain.py:238-245`) 는 `step_{step}{suffix}` 이름으로
`train_state.model.state_dict()` 를 그대로 쓴다.

**따라서:**

| config.ema | `step_N` 의 내용 | `step_N.live` |
|---|---|---|
| **True** | `train_state_eval` = **EMA(bias-corrected) 사본** | `train_state` = **raw live SGD 가중치** |
| **False** | `train_state_eval` **is** `train_state` (`pretrain.py:661`) = **raw** | **생성 안 됨** |

`ema=False` 일 때 `train_state_eval = train_state` (같은 객체)이므로 `step_N` 이 곧 raw 다.
실측으로도 `ema=false` run 19개 중 `.live` 파일이 있는 run 은 0개, `ema=true` run 4개는
각각 15개씩 있다.

### 1-2. in-loop eval 이 쓰는 가중치 (`pretrain.py:655-661`)

```python
if config.ema:
    print("SWITCH TO EMA (bias-corrected)")
    train_state_eval = copy.deepcopy(train_state)
    # bias-corrected EMA: removes the surviving mu^step * w0 init term
    train_state_eval.model = ema_helper.ema_copy(train_state_eval.model, step=train_state.step)
else:
    train_state_eval = train_state
train_state_eval.model.eval()
metrics = evaluate(config, train_state_eval, ...)
```

**`ema=True` 면 in-loop eval(= `[EVAL @ step N]` 줄의 `nqueens/accuracy`)은 EMA 사본을 평가한다.**
`ema=False` 면 raw 를 평가한다. 즉 로그의 곡선이 무엇을 그린 것인지가 `ema` 플래그에 따라 다르다.

### 1-3. EMA 갱신 시점 · bias correction · **gradient 개입 여부**

갱신 시점 (`pretrain.py:648-649`) — train batch 마다, optimizer step 뒤:
```python
if config.ema:
    ema_helper.update(train_state.model)
```

갱신 식 (`models/ema.py:38-44`):
```python
def update(self, module):
    self.num_updates += 1
    for name, param in module.named_parameters():
        if param.requires_grad:
            self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data
```

**gradient 개입 여부 — 개입하지 않는다.** 근거 셋:
1. `update()` 는 **`self.shadow` 에만 쓴다.** `param.data` 는 읽기만 한다(우변). 학습 가중치를
   덮어쓰는 문장이 없다.
2. 가중치를 덮어쓰는 두 함수 `ema()` (`ema.py:52`) 와 `ema_bias_corrected()` (`ema.py:66`) 는
   **`ema_copy()` 를 통해서만 호출**되는데, `ema_copy` 는 첫 줄에서
   `module_copy = copy.deepcopy(module)` (`ema.py:70`) 로 **사본을 뜬 뒤** 그 사본에만 쓴다.
3. 호출부(`pretrain.py:657-659`)도 `copy.deepcopy(train_state)` 한 사본에 적용하고,
   평가 후 `del train_state_eval` (`pretrain.py:694`) 로 버린다.

**따라서 EMA 는 순수 평가용 그림자다. 같은 seed 라면 `ema=True` 와 `ema=False` 의 학습 궤적
(= `.live` 가중치의 궤적)은 CUDA 비결정성을 빼면 동일해야 한다.**
→ 지시하신 "이게 중요하다" 항목의 답: **순수 사본이 맞다.**

*단, 이 동치성을 이 서버에서 실측 검증하지는 못했다.* 검증하려면 같은 β·같은 seed 로
`ema=True`/`False` 짝을 돌려 `.live` 를 비교해야 하는데, 이 서버의 EMA-on 4 run(β 1.2/1.6/2.4)과
EMA-off run(β 1.0~2.2 중 1.5/1.8/2.1/…)에 **β 가 겹치는 짝이 없다.** → **확인 못 함.**

bias correction (`models/ema.py:7-22` docstring + `ema.py:54-59`):
```
shadow_t = (1 - mu^t) * W_avg  +  mu^t * w0        # 초기 가중치 w0 가 mu^t 만큼 살아남음
W_avg    = (shadow_t - mu^t * w0) / (1 - mu^t)     # 올바른 debias (w0 를 빼고 나눔)
```
docstring 이 직접 경고한다: *"At mu=0.9999 that is 0.15% after 65k steps (Sudoku) but **18% after
17k steps (N-Queens)**"*, 그리고 Adam 식 `shadow/(1-mu^t)` 는 **틀렸다**(init 을 빼는 게 아니라
더하게 된다)고 명시. 코드는 올바른 쪽(`_debiased`, `ema.py:54-59`)을 쓴다.

**주목**: 우리 N-Queens 는 17,085 step, `mu=0.9999` → `mu^t ≈ 0.181`. docstring 이 말한
"18% 가 init" 구간이 정확히 우리 설정이다. bias correction 이 그 18% 를 빼주지만,
남는 `W_avg` 는 여전히 **시정수 ~10,000 step 짜리 장기 평균**이라 학습 후반의 좋은 가중치
비중이 낮다. (이것이 EMA<live 의 원인인지는 **미검증** — REPORT_emasweep_50.md 8절과 동일.)

### 1-4. harness 가 로드하는 가중치 (`experiments/gram_harness.py:70-78`)

```python
def load_model(ckpt_dir, step):
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "all_config.yaml"))
    cfg.load_checkpoint = os.path.join(ckpt_dir, step)
    cfg.ema = False                  # checkpoint is already the EMA copy
```

- **파일명을 문자열로 그대로 이어붙인다.** 즉 `--step step_17085` 면 EMA 사본(ema=True run 기준),
  `--step step_17085.live` 면 raw 를 읽는다.
- `cfg.ema = False` 는 **harness 안에서 EMA 를 새로 돌리지 않겠다**는 뜻이지, raw 를 읽는다는
  뜻이 **아니다**. 주석 `# checkpoint is already the EMA copy` 가 그 의도를 밝힌다.
- **`.live` 는 반드시 명시해야 한다.** 기본값(`step_N`)은 EMA-on run 에서 EMA 사본이다.
  `sweep_harness.py:51-59` 의 `--step` 은 required 인자라 자동 선택 로직이 없다.

---

## 2. Task 2 — 지금까지 낸 수치의 출처 감사

이 서버 run 19개의 `all_config.yaml` 을 전수 확인했다.

| REPORT | run | ema | seed | 보고한 prior valid 의 출처 | 편향 |
|---|---|---|---|---|---|
| REPORT_onehot | `nq8_oh_b007` | false | 0 | `step_N` = **raw** | 없음 |
| REPORT_onehot | `nq8_oh_b15` | false | 0 | `step_N` = **raw** | 없음 |
| REPORT_onehot | `nq8_ctrl_b15` | false | 0 | `step_N` = **raw** | 없음 |
| REPORT_sweep_50 | `nq8_on_b100` | false | 0 | `step_N` = **raw** | 없음 |
| REPORT_sweep_50 | `nq8_on_b140` | false | 0 | `step_N` = **raw** | 없음 |
| REPORT_sweep_50 | `nq8_on_b180` | false | 0 | `step_N` = **raw** | 없음 |
| REPORT_sweep_50 | `nq8_on_b220` | false | 0 | `step_N` = **raw** | 없음 |
| REPORT_finesweep_50 | `nq8_on_b150` | false | 0 | `step_N` = **raw** | 없음 |
| REPORT_finesweep_50 | `nq8_on_b210` | false | 0 | `step_N` = **raw** | 없음 |
| REPORT_finesweep_50 | `nq8_on_b180_s1` | false | 1 | `step_N` = **raw** | 없음 |
| REPORT_finesweep_50 | `nq8_on_b180_s2` | false | 2 | `step_N` = **raw** | 없음 |
| **REPORT_emasweep_50** | `nq8_on_b120_ema` | **true** | 0 | **EMA / .live 둘 다 보고** | **peak 선택 편향** |
| **REPORT_emasweep_50** | `nq8_on_b240_ema` | **true** | 0 | **EMA / .live 둘 다 보고** | **peak 선택 편향** |
| **REPORT_emasweep_50** | `nq8_on_b160_ema_s0` | **true** | 0 | **EMA / .live 둘 다 보고** | **peak 선택 편향** |
| **REPORT_emasweep_50** | `nq8_on_b160_ema_s1` | **true** | 1 | **EMA / .live 둘 다 보고** | **peak 선택 편향** |
| (미완주, kill됨) | `nq8_off_b600/750/900`, `nq8_off_b220_s1` | false | 0,0,0,1 | 수치 보고 안 함 | — |

### 2-1. 편향 없음이 확인된 것 (11 run)

`ema=false` 인 run 은 `train_state_eval is train_state` (`pretrain.py:661`)라 **`step_N` 이 곧 raw** 다.
`.live` 파일도 0개(생성 자체가 `if config.ema` 안에 있음). 따라서:

- **REPORT_onehot.md 의 모든 수치** (run1 prior 0.0000, run2 peak 0.2948 등)
- **REPORT_sweep_50.md 의 모든 수치** (β=1.8 peak 0.6268 등)
- **REPORT_finesweep_50.md 의 모든 수치** (β=1.8 seed 3개 0.6268/0.0014/0.0008 등)

→ **전부 raw 가중치 기준이다. EMA 편향 없음. 재측정 불필요.**
특히 **β=1.8 seed 783배 갈림은 EMA 착시가 아니다** — 이미 raw 로 잰 값이다.

### 2-2. 편향이 걸린 것 (4 run, REPORT_emasweep_50.md)

값 자체는 EMA/`.live` 양쪽 다 보고했으므로 "EMA 값을 raw 인 줄 알고 보고한" 오류는 **없다.**
문제는 **step 선택**이다:

| run | peak step 을 고른 곡선 | harness 를 돌린 step | `.live` 정점 여부 |
|---|---|---|---|
| `nq8_on_b120_ema` | **EMA in-loop** (peak 5695) | 5695, 17085 (양쪽 EMA+live) | **미확인** |
| `nq8_on_b240_ema` | **EMA in-loop** (peak=final 17085) | 17085 (EMA+live) | **미확인** |
| `nq8_on_b160_ema_s0` | **EMA in-loop** (peak=final 17085) | 17085 (EMA+live) | **미확인** |
| `nq8_on_b160_ema_s1` | EMA 곡선이 전 구간 0 → peak 정의 불가, **final 로 대체** | 17085 (EMA+live) | **미확인** |

**즉 REPORT_emasweep_50.md 에 보고한 `.live` 수치(0.0111 / 0.1210 / 0.4069 / 0.2771)는
"EMA 곡선이 정점이라고 지목한 step 에서의 `.live` 값"이지, `.live` 곡선의 정점이 아니다.**
`.live` 가 다른 step 에서 더 높을 수 있다. 특히 `nq8_on_b160_ema_s1` 은 EMA 곡선이 전 구간
0 이라 step 선택 근거가 사실상 없었다(final 을 임의 채택) — 이 run 의 0.2771 이 가장 신뢰도가 낮다.

또한 b120 의 EMA/live 역전(step 5695 에서 EMA 0.0101 > live 0.0000, step 17085 에서 EMA 0.0015
< live 0.0111)은 두 곡선의 **정점 위치가 서로 다르다**는 직접 증거다.

---

## 3. Task 3 — 재측정 필요 목록 + 준비된 명령

### 3-1. 필요한 것

**EMA-on 4 run 의 `.live` 전체 곡선(15 포인트) 스캔.** in-loop 로그에는 `.live` 곡선이
아예 없으므로(로그는 EMA 사본만 평가), harness 로 15개 checkpoint 를 각각 재는 수밖에 없다.

| run | 스캔 대상 | checkpoint 존재 |
|---|---|---|
| `nq8_on_b120_ema` | `step_{1139..17085}.live` 15개 | 확인함 (각 15개) |
| `nq8_on_b240_ema` | 〃 | 확인함 |
| `nq8_on_b160_ema_s0` | 〃 | 확인함 |
| `nq8_on_b160_ema_s1` | 〃 | 확인함 |

EMA-off run 11개는 재측정 **불필요** (2-1).

### 3-2. 비용

harness 1회 ≈ 3~4분(prior N=20 × test 772 + posterior). 15 step × 4 run = **60회**.
GPU 4개 병렬 → run 당 15회 순차 ≈ **50~60분**.

비용을 줄이려면 `--N 5` 로 스캔해 정점 후보를 찾고, 후보 step 만 `--N 20` 으로 확정하는
2단계가 가능하다(≈ 20분). 다만 N=5 는 prior valid 추정 분산이 커진다. **어느 쪽을 쓸지는
사용자 결정** — 아래 명령은 기본(N=20) 기준이다.

### 3-3. 준비된 명령 (GPU 반납 후 실행. **아직 실행하지 않았다**)

```bash
cd /workspace/TinyRecursiveModels
STEPS="1139 2278 3417 4556 5695 6834 7973 9112 10251 11390 12529 13668 14807 15946 17085"

scan () {   # $1=run  $2=gpu
  for s in $STEPS; do
    CUDA_VISIBLE_DEVICES=$2 python experiments/sweep_harness.py \
      --ckpt-dir checkpoints/Nqueens-8-ACT-torch/$1 \
      --step step_${s}.live --seed 0 \
      --tag "$1 step_${s}.live livescan" \
      --out experiments/results/livescan_$1_step_${s}.live_s0.json
  done
}

nohup bash -c "$(declare -f scan); STEPS=\"$STEPS\"; scan nq8_on_b120_ema    4" > logs/livescan_b120.log   2>&1 &
nohup bash -c "$(declare -f scan); STEPS=\"$STEPS\"; scan nq8_on_b240_ema    5" > logs/livescan_b240.log   2>&1 &
nohup bash -c "$(declare -f scan); STEPS=\"$STEPS\"; scan nq8_on_b160_ema_s0 6" > logs/livescan_b160s0.log 2>&1 &
nohup bash -c "$(declare -f scan); STEPS=\"$STEPS\"; scan nq8_on_b160_ema_s1 7" > logs/livescan_b160s1.log 2>&1 &
```
정점이 정해지면 그 step 만 harness seed 0/1 로 재확인(기존 `run_sweep50_harness.sh` 사용).

**GPU 4~7 만 쓴다. launch 전 `nvidia-smi` 로 유휴 확인 필요.**

---

## 4. Task 4 — 판정 규약 제안 (구현 안 함, 제안만)

### 4-1. 판정 기준 가중치

**제안: `.live`(raw)를 1차 판정 기준으로, EMA 는 보조로 병기.**

근거:
- EMA 사본은 `mu^t` 만큼 init 이 섞이고(우리 설정 17k step 에서 18%, `ema.py:12-14`),
  bias correction 후에도 시정수 ~10,000 step 짜리 장기 평균이다. 학습 후반 개선을 늦게 반영한다.
- 실측으로 EMA 가 raw 를 최대 395배 과소평가했다(`nq8_on_b160_ema_s1`: 0.0007 vs 0.2771).
  같은 run 의 EMA posterior 는 0.4145 로 **망가져 있는 반면** live posterior 는 0.9994 였다 —
  posterior 가 0.4 로 무너지는 것은 가중치 자체가 나쁘다는 신호이지 모델이 못 배웠다는 신호가 아니다.
- `.live` 는 EMA on/off 와 무관하게 항상 "그 시점 실제 학습 가중치"라 **run 간 비교 축이 하나로 통일된다.**
  (EMA-off run 의 `step_N` 도 raw 이므로 같은 축이다.)

단, EMA 를 버리자는 뜻은 아니다. 논문 B.2 가 EMA 를 "For stability" 로 명시하므로 **최종 산출
모델**로 EMA 를 쓸지는 별개 문제다. 제안은 **"β/seed 비교 같은 진단 판정은 `.live` 로 한다"** 이다.

### 4-2. peak step 선택 기준

**제안: 판정에 쓰는 곡선과 peak 를 고르는 곡선을 반드시 일치시킨다.**
`.live` 로 판정한다면 peak 도 `.live` 곡선에서 골라야 한다. 지금은 in-loop 에 `.live` 곡선이
없어서 이게 불가능하다 → 4-4 의 코드 변경이 필요한 이유.

과도기 규약: `.live` 곡선이 없는 EMA-on run 은 **"EMA 곡선 정점"이 아니라 `.live` 스캔 정점**을
쓰고, 스캔 전이면 **"peak 미확정"으로 표기**한다. EMA 곡선 정점의 `.live` 값을 peak 라고
부르지 않는다.

### 4-3. REPORT 필수 명시 항목

모든 수치 표에 다음을 붙인다:
1. **가중치 종류** — `EMA` / `live` / `raw(ema=off)` 중 무엇인가
2. **peak 선택 근거** — 어느 곡선의 정점인가, 아니면 final 인가, 아니면 미확정인가
3. **`ema` 플래그와 `ema_rate`** — config dump 에서 확인한 값
4. **harness seed 와 그 편차** — 샘플링 잡음 폭
5. **학습 seed 수** — 1개면 "순위 미확정" 명시 (기존 방침 유지)

### 4-4. 코드 변경 범위 평가 (구현 안 함)

**변경안: EMA on 일 때 in-loop eval 을 EMA/live 두 번 돌려 둘 다 로깅.**

- 위치: `pretrain.py:655-684`. 현재 `train_state_eval` 하나만 평가한다.
  `config.ema` 면 `evaluate()` 를 raw(`train_state`)에도 한 번 더 호출하고, 결과를
  `live/*` 접두사로 wandb·stdout 에 찍으면 된다.
- **비용: eval 시간 2배.** 현재 eval 은 test 772 × halt_max_steps 16 이라 run 당 15회 발생한다.
  실측 eval 소요는 따로 재지 않았다 — **확인 못 함**. 다만 학습 전체가 ~2h07m 이고 eval 이
  그중 일부이므로, 2배가 되면 총 시간이 유의미하게 늘어난다.
- **부작용 주의**: `evaluate()` 안에서 모델이 `reparameterize`(`gram.py:298-301`)로
  `torch.randn_like` 를 뽑는다. eval 을 한 번 더 돌리면 **전역 CUDA RNG 소비량이 달라져
  이후 학습 궤적이 바뀐다.** 즉 이 변경을 넣은 run 은 넣기 전 run 과 같은 seed 라도
  bit-identical 하지 않다. 비교 기준선이 끊긴다는 뜻이므로, **넣는다면 새 비교군부터
  일괄 적용해야 한다.** (이 부작용은 코드를 읽고 추론한 것이며, 실측 검증은 **안 했다**.)
- 더 싼 대안: eval 을 2배로 늘리는 대신 **`.live` checkpoint 를 사후 스캔**(Task 3 방식)으로
  처리한다. 학습 코드를 안 건드리므로 기준선이 유지되고, 필요한 run 에만 비용을 쓴다.
  **이쪽을 권한다.**

---

## 5. 확인 못 한 것

- **`ema=True`/`False` 의 `.live` 궤적 동치성** — 코드상 EMA 는 순수 평가 사본이라 동일해야
  하지만, 이 서버에 β 가 겹치는 짝이 없어 실측 검증 못 함.
- **EMA 가중치가 나빠지는 원인** — `mu=0.9999` 시정수 가설은 여전히 미검증.
  `ema_rate` 를 바꿔 돌려야 확인된다.
- **EMA-on 4 run 의 `.live` 정점 위치와 값** — Task 3 스캔 전까지 미확정.
- **eval 1회 소요 시간** — 측정 안 함. 4-4 의 "2배" 비용 평가가 절대시간으로 얼마인지 모른다.
- **29번 쪽 run 들의 ema 플래그와 측정 가중치** — 이 서버에서 확인 불가. 29번도 같은 감사를
  해야 두 서버 수치를 같은 축에서 합칠 수 있다.
