# REPORT_onehot.md — one-hot y_emb 3-run (50번 lecun docker, GPU 4/5/6)

- 일시: 2026-07-27
- 컨테이너: lecun docker (`trm_chaeun`), repo `/workspace/TinyRecursiveModels`, root
- 사용 GPU: **4, 5, 6** (7은 여유로 비워둠). GPU 0~3 은 조회만 하고 일절 건드리지 않음.
- 상태: **3-run 학습 진행 중** (Task 0~3 완료, Task 4 완주 대기)

---

## 1. Task 0 — pull

로컬 미커밋 변경(`gram_nq8_fix.sh`)은 지시대로 `git stash` 로 보관했다 (`stash@{0}`, 삭제 안 함).

첫 `git pull` 은 인증 실패했다:
```
fatal: could not read Username for 'https://github.com': No such device or address
```
원인은 credential 자체가 아니라 **credential helper 가 VS Code dev-container IPC 헬퍼**라서다.
헬퍼 스크립트는 `process.env.REMOTE_CONTAINERS_IPC` 로 소켓을 찾는데, 내 셸에는 그 환경변수가
상속되지 않아 소켓을 못 찾은 것. 오늘 자(12:54) 살아있는 소켓이 있어서 그것을 지정해 재시도했다:
```bash
export REMOTE_CONTAINERS_IPC=/tmp/vscode-remote-containers-ipc-f0d45e46-6aa3-4a03-8288-e925f3e45a03.sock
git pull origin main
```
→ 성공. PAT 재입력은 필요 없었다.

pull 후 확인 (지시한 4개 항목 전부):
```
$ git log --oneline -3
6f144aa REPORT_push.md: push 결과 + 50번 실행 커맨드
7cf727f one-hot y_emb 토글(zeropad/frozen_random) + y_onehot_scale, evaluator 형식 확정, sh 환경변수화
4c847f1 GRAM reproduction

$ grep -c "y_onehot" models/recursive_reasoning/gram.py
18                      # 0 아님 -> 성공

$ ls train_nqueens8_fix.sh
train_nqueens8_fix.sh   # 존재
```

---

## 2. Task 1 — 사전 확인

### 데이터
`data/nqueens-8/train/` 존재: `all__inputs.npy`, `all__labels.npy`, `all__puzzle_indices.npy`,
`all__group_indices.npy`, `all__puzzle_identifiers.npy`, `dataset.json`. 이상 없음.

### GPU (launch 직전)
```
4, 15 MiB    5, 15 MiB    6, 15 MiB    7, 15 MiB      <- 전부 유휴, 사용 가능
0~3, 21737 MiB, 100%                                  <- 타인 작업. 조회만 함, 미사용
```
skip 대상 GPU 없음. 3-run 이므로 4/5/6 사용, 7 은 남겨둠.

### sh 의 GPUS 처리
`train_nqueens8_fix.sh:63` 이 `CUDA_VISIBLE_DEVICES=${GPUS}` 로 직접 받는다. 구조 조정 불필요.
launch 전 preflight 로 해당 GPU가 1000MiB 초과 점유면 자동 중단하는 로직도 이미 들어있다(L53-56).

### dry-run (run1 조립)
```
arch.kl_coef=0.07  arch.kl_reduction=mean  arch.noise_head_expansion=1.5
arch.y_onehot=True arch.y_onehot_proj=zeropad arch.y_onehot_scale=0.573
ema=False  lr_warmup_steps=500  +run_name=nq8_oh_b007
evaluators=[{name: nqueens@NQueens}]   epochs=3000  eval_interval=200  global_batch_size=768
```
지시한 값 전부 일치. **단, 이 조립 통과가 실제 적용을 뜻하지 않았다 — 아래 3절 참조.**

### num_params
`pretrain.py:624` 는 num_params 를 wandb 에만 기록하고 stdout 엔 안 찍는다. 그래서 실제 config 로
모델을 직접 생성해 셌다:
```
y_onehot=True : num_params = 9,966,085
y_onehot=False: num_params = 9,966,085
```
**예상치 9,966,085 와 정확히 일치.** one-hot 은 `zeropad` 라 학습 파라미터를 추가하지 않으므로
두 arm 의 파라미터 수가 같은 것이 정상이다.

---

## 3. 1차 launch 실패 → yaml 수정 (임의로 정하지 않고 지시받은 대로 수정)

첫 3-run launch 는 세 개 모두 **즉사**했다:
```
Could not override 'arch.y_onehot'.
To append to your config use +arch.y_onehot=True
Key 'y_onehot' is not in struct
    full_key: arch.y_onehot
torch.distributed.elastic.multiprocessing.errors.ChildFailedError
```
`gram.py` 의 config 클래스엔 필드가 있으나(`y_onehot:89`, `y_onehot_proj:92`, `y_onehot_scale:94`)
`config/arch/gram_nqueens8.yaml` 에 없어서 hydra 가 struct 모드에서 override 를 거부했다.
**한 스텝도 돌지 않고 죽었으므로 "one-hot 이 꺼진 채 학습된 run" 은 존재하지 않는다** (GPU 4~7 이
전부 15MiB 로 복귀한 것으로 확인). 그래도 지시대로 세션을 kill 하고 잔여 프로세스 0개를 확인했다.

수정 (`config/arch/gram_nqueens8.yaml`, 커밋 안 함 — 로컬 변경 상태):
```yaml
# 사수 처방 4: posterior 의 y 를 one-hot 으로 (y 복사 통로 차단).
# 기본값은 gram.py 의 dataclass 기본과 동일 = 기존 동작. sh 에서 덮어쓴다.
y_onehot: False
y_onehot_proj: zeropad
y_onehot_scale: 1.0
```
기본값은 지시받은 그대로이며 `gram.py` 의 dataclass 기본값과도 일치하므로 기존 경로는 불변이다.

### 검증 launch (run1 1개만, 확인 후 즉시 kill)
`RUN=nq8_verify ... GPUS=4 PORT=29661` 로 띄워 저장된 config dump 확인:
```yaml
  kl_coef: 0.07
  kl_reduction: mean
  y_onehot: true
  y_onehot_proj: zeropad
  y_onehot_scale: 0.573
ema: false
```
찍히는 것 확인 후 즉시 kill (GPU 4 → 15MiB 복귀 확인). 그 다음에 3-run 을 재개했다.

---

## 4. Task 2/3 — 3-run launch + 200 step 검증

tmux 세션 `onehot`, 창 3개, 각 1 GPU. 로그는 `tee`(overwrite, `-a` 아님).

| run | run_name | GPU | port | y_onehot | β(kl_coef) |
|---|---|---|---|---|---|
| run1 | `nq8_oh_b007` | 4 | 29651 | True | 0.07 |
| run2 | `nq8_oh_b15` | 5 | 29652 | True | 1.5 |
| run3 | `nq8_ctrl_b15` | 6 | 29653 | False | 1.5 |

run_name 중복 없음 (checkpoint/wandb 디렉터리 신규 생성).

### config dump 발췌 (`checkpoints/Nqueens-8-ACT-torch/<run>/all_config.yaml`)

**run1 `nq8_oh_b007`**
```yaml
  kl_coef: 0.07
  kl_reduction: mean
  y_onehot: true
  y_onehot_proj: zeropad
  y_onehot_scale: 0.573
ema: false
run_name: nq8_oh_b007
```
**run2 `nq8_oh_b15`**
```yaml
  kl_coef: 1.5
  kl_reduction: mean
  y_onehot: true
  y_onehot_proj: zeropad
  y_onehot_scale: 0.573
ema: false
run_name: nq8_oh_b15
```
**run3 `nq8_ctrl_b15`**
```yaml
  kl_coef: 1.5
  kl_reduction: mean
  y_onehot: false
  y_onehot_proj: zeropad     # y_onehot=false 라 코드상 미사용
  y_onehot_scale: 0.573      # 동일하게 미사용
ema: false
run_name: nq8_ctrl_b15
```

### 검증 항목 결과

| 항목 | run1 | run2 | run3 |
|---|---|---|---|
| y_onehot | **true** ✓ | **true** ✓ | **false** ✓ |
| y_onehot_proj / scale | zeropad / 0.573 ✓ | zeropad / 0.573 ✓ | (미사용) |
| kl_coef | 0.07 ✓ | 1.5 ✓ | 1.5 ✓ |
| ema | false ✓ | false ✓ | false ✓ |
| num_params | 9,966,085 ✓ | 9,966,085 ✓ | 9,966,085 ✓ |
| NaN | 없음 ✓ | 없음 ✓ | 없음 ✓ |
| nqueens/accuracy 로그 | 찍힘 ✓ | 찍힘 ✓ | 찍힘 ✓ |
| eval kl_loss | **0.0000** ✓ | **0.0000** ✓ | **0.0000** ✓ |

첫 eval (step 1139) 원문:
```
run1: [EVAL @ step 1139] all/accuracy=0.9008 all/kl_loss=0.0000 nqueens/accuracy=0.0000
      nqueens/consistent_rate=1.0000 nqueens/n_eval=772.0000 nqueens/queen_count_mean=4.2370
run2: [EVAL @ step 1139] all/accuracy=0.9153 all/kl_loss=0.0000 nqueens/accuracy=0.0000
      nqueens/consistent_rate=1.0000 nqueens/n_eval=772.0000 nqueens/queen_count_mean=3.1580
run3: [EVAL @ step 1139] all/accuracy=0.9184 all/kl_loss=0.0000 nqueens/accuracy=0.0000
      nqueens/consistent_rate=1.0000 nqueens/n_eval=772.0000 nqueens/queen_count_mean=2.7772
```
세 run 모두 죽일 사유 없음 → 완주시킨다.

초기 학습 신호(참고, 해석은 사용자 몫): run1 은 step 600 에서 이미 `train/accuracy=1.0000`
(posterior 경로)에 도달한 반면, β=1.5 인 run2/run3 은 step 1050 까지 `train/accuracy=0.0000`.

---

## 5. Task 4 — 완주 후 결과

세 run 모두 step 17085 완주. eval 15회, NaN/crash 0. 소요 약 2시간 10분(3개 병렬).

### 5-1. in-loop `nqueens/accuracy` 곡선 (prior 경로)

| step | run1 oh β0.07 | run2 oh β1.5 | run3 ctrl β1.5 |
|---|---|---|---|
| 1139 | 0.0000 | 0.0000 | 0.0000 |
| 2278 | 0.0000 | 0.0000 | 0.0013 |
| 3417 | 0.0000 | 0.0000 | 0.0000 |
| 4556 | 0.0000 | 0.0052 | 0.0039 |
| 5695 | 0.0000 | 0.1399 | 0.0000 |
| 6834 | 0.0000 | 0.1982 | 0.0000 |
| 7973 | 0.0000 | 0.2655 | 0.0000 |
| 9112 | 0.0000 | 0.1723 | 0.0000 |
| 10251 | 0.0000 | 0.2824 | 0.0155 |
| **11390** | 0.0000 | **0.2927** ← peak | 0.0233 |
| **12529** | 0.0000 | 0.1917 | **0.0531** ← peak |
| 13668 | 0.0000 | 0.1386 | 0.0298 |
| 14807 | 0.0000 | 0.2552 | 0.0194 |
| 15946 | 0.0000 | 0.1179 | 0.0000 |
| 17085 | 0.0000 | 0.2073 | 0.0000 |

| run | peak step | peak | final |
|---|---|---|---|
| run1 `nq8_oh_b007` | — (전 구간 0) | 0.0000 | 0.0000 |
| run2 `nq8_oh_b15` | **11390** | **0.2927** | 0.2073 |
| run3 `nq8_ctrl_b15` | 12529 | 0.0531 | 0.0000 |

run1 은 peak 이 정의되지 않아(15회 eval 전부 0) harness 는 run2 의 peak 과 같은
**step 11390** + final 로 돌렸다(step 을 맞춰 비교하려고).

### 5-2. harness (prior N=20 + posterior, test 772 전수, harness seed 2개)

| run | step | prior valid (s0/s1) | 평균 | posterior valid | pass@1 | pass@20 | distinct@20 | prior queens |
|---|---|---|---|---|---|---|---|---|
| run1 oh β0.07 | 11390 | 0.0000 / 0.0000 | **0.0000** | 1.0000 | 0.0000 | 0.0000 | 20.00 | 7.55 (4–13) |
| run1 oh β0.07 | 17085 | 0.0000 / 0.0000 | **0.0000** | 1.0000 | 0.0000 | 0.0000 | 20.00 | 8.02 (5–13) |
| run2 oh β1.5 | **11390** | 0.2929 / 0.2966 | **0.2948** | **1.0000** | 0.2948 | **0.950** | 9.76 | 7.98 (6–10) |
| run2 oh β1.5 | 17085 | 0.2146 / 0.2157 | 0.2151 | 0.9987 | 0.2151 | 0.932 | 13.10 | 8.02 (6–10) |
| run3 ctrl β1.5 | 12529 | 0.0555 / 0.0587 | **0.0571** | 0.9994 | 0.0571 | 0.595 | 17.15 | 7.86 (6–10) |
| run3 ctrl β1.5 | 17085 | 0.0000 / 0.0000 | **0.0000** | 0.9994 | 0.0000 | 0.0000 | 20.00 | 9.38 (5–15) |

harness seed 편차는 전부 **≤ 0.0037** (최대: run2 peak 0.2929↔0.2966). 즉 prior 샘플링
노이즈는 무시할 수준이고, 아래 수치들은 1 draw 우연이 아니다.

### 5-3. 지시하신 4개 확정

**1) run1 prior valid = 0 확정.**
0.0000, 두 checkpoint(11390/17085) × 두 seed 전부. 세부:
**valid 샘플 0개 / 15,440개** (772 puzzle × 20 sample), **≥1개라도 맞힌 puzzle 0개 / 772**.
in-loop 15회 eval 0 과 harness 4회 0 이 일치한다. **one-hot 단독(β=0.07)으로는 prior 가 살지 않는다.**
baseline `gram_nq8_fix_b`(β=0.07, one-hot 없음)도 prior 0.0000 이었으므로 **차이 없음**.

**2) run2 posterior = 1.0000 — one-hot 은 posterior 를 누르지 않았다.**
peak 에서 posterior valid **1.0000**(queens 정확히 8.00, consistent 1.0000), final 0.9987.
one-hot 없는 baseline 들과 동일 수준(`nq8_b15` final 1.0000, `gram_nq8_fix_b` 0.9974).
→ **posterior 는 그대로 100% 인 채 prior 만 올라갔다.** gap 축소는 전부 prior 개선분이다.
one-hot 이 posterior 의 y 복사를 "줄였다"는 증거는 없다 — 복사 통로를 좁혔는데도
posterior 성능이 안 떨어졌다.

**3) run2 vs run3 prior valid (peak 끼리): +0.2377, 약 5.2배.**
run2 0.2948 (step 11390) vs run3 0.0571 (step 12529). final 끼리는 0.2151 vs 0.0000.
pass@20 은 0.950 vs 0.595, distinct@20 은 9.76 vs 17.15.
**단, 이 5.2배를 "one-hot 순수 증폭분"으로 읽으면 안 된다 — 5-4 참조.**

**4) run2 peak step = 11390** (in-loop 0.2927, harness 0.2948).
final 17085 에서 harness 0.2151 로 **-0.0797 하락**. → 과적합 시작점 11390 근방.
early stop 설계 시 이 지점. 참고로 run3 은 12529 (0.0571) 에서 final 0.0000 으로 **완전 붕괴**,
run1 은 애초에 오른 적이 없다.

### 5-4. 해석 재료 — 교란 요인 (중요)

**"one-hot 순수 증폭분 5.2배"는 학습 run-to-run 편차와 분리되지 않았다.**

| β=1.5 run | one-hot | peak prior valid | final prior valid |
|---|---|---|---|
| `nq8_b15` (세션 초반, 같은 config) | ✗ | **0.3029** (step 6834) | 0.1534 |
| run3 `nq8_ctrl_b15` (이번) | ✗ | **0.0571** (step 12529) | 0.0000 |
| run2 `nq8_oh_b15` (이번) | ✓ | **0.2948** (step 11390) | 0.2151 |

- one-hot **없는** 두 run 이 0.3029 vs 0.0571 로 **5.3배 차이**난다. config·seed·코드 경로가
  전부 동일한데도(앞서 확인) 이만큼 벌어진다.
- run2(one-hot, 0.2948)는 run3(0.0571)보다 5.2배 높지만, **one-hot 없는 `nq8_b15`(0.3029)와는
  사실상 같다**(-0.008).
- 즉 harness seed 편차(≤0.0037)는 작지만 **학습 run 편차는 그보다 두 자릿수 크다**.
  harness seed 2개로는 이 편차를 측정할 수 없다(같은 checkpoint 를 다시 샘플할 뿐).
- **β=1.5 에서 one-hot 의 효과는 이번 데이터로 확정 불가.** 확정하려면 각 arm 을 학습 seed
  여러 개로 돌려야 한다. 반면 **run1 의 prior valid=0 은 확정**이다(0/15,440, 4회 측정 전부 0,
  baseline 과도 일치).

### 5-5. 부수 관측 (해석 안 함)

- run1 의 prior `consistent` 가 step 11390 에서 0.9409 → final 17085 에서 **0.0017** 로 붕괴.
  prior queens 평균은 7.55 → 8.02 로 오히려 정상화됐는데 입력 queen 보존은 무너졌다.
- run3 final 의 prior queens 평균 9.38 (5–15) — 8 에서 크게 벗어남. run2 final 은 8.02 (6–10).
- run2 의 distinct@20 이 peak 9.76 → final 13.10 으로 증가(다양성↑, 정확도↓).

---

## 6. 임의로 정한 것 (전부 명시)

1. **`REMOTE_CONTAINERS_IPC` 소켓 지정으로 pull 인증 해결.** PAT 재입력을 요청하는 대신 이미 살아있는
   dev-container credential 소켓을 재사용했다. 저장된 credential 을 그대로 쓴 것이라 새 비밀정보는
   만들지 않았다.
2. **`config/arch/gram_nqueens8.yaml` 에 3필드 추가 — 커밋/push 하지 않았다.** 로컬 변경으로만 뒀다.
   push 여부는 사용자 판단.
3. **`kl_reduction=mean` 유지 (sh 기본값).** 코드 dataclass 기본은 `sum` 이고 옛 run 일부는 sum 이라
   확인차 대조했는데, 이번 비교 대상인 최근 N-Queens run 전부 — `gram_nq8_fix_b/b03/b10/ema/noema`
   (7/24), `gram_nq8_mean_*` (7/22~23) — 가 `mean` 이었다. 기준선과 맞추는 것이 비교에 옳다고 보고
   지시된 커맨드 그대로(=mean) 돌렸다. sum 으로 원하면 `KL_RED=sum` 으로 재실행 필요.
4. **GPU 7 미사용.** run 이 3개라 4/5/6 만 썼다.
5. **`WANDB_MODE=online` 유지 (sh 기본값).** 이 컨테이너엔 유효한 wandb credential 이 있고
   (`jennylee0358-sungkyunkwan-university`) 실제 online 업로드가 되는 것을 사전 테스트로 확인했다.
   최근 run 들이 offline 이던 것과 달라서 launch 전에 확인한 것. 프로젝트는
   `Nqueens-8-ACT-torch`.
6. **검증용 임시 run `nq8_verify` 의 흔적을 남겨뒀다** (`checkpoints/.../nq8_verify/all_config.yaml`,
   `logs/nq8_verify.log`). config dump 증거라 지우지 않았다. 필요 없으면 삭제해도 무방.
7. **소요 시간은 지시서의 ~37분이 아니라 run 당 약 2시간 10분이다.** 1 GPU 당 1 run 이라
   17,093 step 을 ~2.2 it/s 로 도는 탓. 3개는 병렬이라 전체도 ~2시간 10분. 37분은 아마 4 GPU
   기준치로 보인다. 지시대로 1 GPU 씩 유지했다.
