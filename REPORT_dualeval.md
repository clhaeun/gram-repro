# REPORT_dualeval.md — in-loop eval에 raw 평가 병행 추가

- 일시: 2026-07-28
- GPU: 검증에만 **GPU 7** 단독 사용(유휴 확인 후). GPU 0~6은 조회만 함, 도는 학습 미접촉.
- 변경 파일 3개: `pretrain.py`, `config/cfg_pretrain.yaml`, `train_nqueens8_fix.sh`
  (`+55 / -1`)

---

## 1. 무엇을 했나

`ema=True` run은 in-loop eval이 EMA 사본만 본다(`pretrain.py:659`의 `ema_helper.ema_copy`).
그래서 곡선이 live SGD 가중치의 성능을 말해주지 않고, 그걸 알려면 사후 `.live` 스캔에
run당 1~2시간이 더 든다.

`dual_eval=True`면 **같은 step에서 raw 모델로 eval을 1회 더** 돌리고 `_raw` 접미사로
로깅한다. 학습이 끝나는 즉시 EMA/raw 두 곡선이 함께 완성된다.

```python
if config.ema and config.dual_eval:
    raw_config = config.model_copy(update={"checkpoint_path": None, "eval_save_outputs": []})
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    train_state.model.eval()
    metrics_raw = evaluate(raw_config, train_state, ...)
    torch.set_rng_state(rng_cpu); torch.cuda.set_rng_state_all(rng_cuda)
```

**로깅 키**: 기존 `nqueens/accuracy`, `all/accuracy` 등은 그대로 EMA(호환성 유지).
raw는 `nqueens/accuracy_raw`, `all/accuracy_raw` 형태로 전부 `_raw` 접미사. wandb와
stdout 둘 다. wandb에는 기존 관례대로 `eval/*` 미러와 `token_accuracy_raw` 별칭도 같이 올린다.

**토글**: `dual_eval: bool = False` 기본. `cfg_pretrain.yaml`에도 필드를 추가해
hydra struct 모드에서 yaml 미동기화 사고가 재발하지 않게 했다.
sh는 `DUALEVAL="${DUALEVAL:-False}"` → `dual_eval=${DUALEVAL}` 주입.

```bash
RUN=... EMA=True DUALEVAL=True bash train_nqueens8_fix.sh
```

---

## 2. 설계상 주의한 것 2가지

### 2-1. raw pass는 파일을 건드리지 않는다

`evaluate()`는 `config.eval_save_outputs`가 있으면 `step_N_all_preds.{rank}`를,
`config.checkpoint_path`가 있으면 `evaluator_{cls}_step_N/`을 쓴다. 그대로 두 번
호출하면 **EMA pass가 방금 쓴 산출물을 raw가 덮어쓴다.** 그래서 raw pass에는
`checkpoint_path=None`, `eval_save_outputs=[]`인 config 사본을 넘겨 부작용을 없앴다.
raw는 지표만 낸다. `.live` 저장은 기존 코드 그대로다.

### 2-2. RNG 스트림 — 처음 구현이 틀렸고 검증에서 잡혔다

첫 구현은 RNG 복원이 없었다. 검증 결과 `dual_eval=True`가 **첫 eval 이후 train 로그를
바꿨다**:

```
dual=False  [step 50] train/kl_loss=12771.1380 train/lm_loss=0.6152 train/q_halt_loss=2.8616
dual=True   [step 50] train/kl_loss=12732.7370 train/lm_loss=0.6153 train/q_halt_loss=2.8874
```

원인은 GRAM이 forward마다 noise를 뽑기 때문에 **추가 forward가 torch CUDA RNG를 소모**해
이후 학습 난수열이 밀린 것이다. gradient/EMA/checkpoint 로직은 그대로였지만 궤적이
달라지므로 "학습 경로 무변화" 요건 위반이다.

raw eval 전후로 CPU/CUDA RNG 상태를 스냅샷·복원해 해결했다. eval loader 쪽은 문제
없다 — `puzzle_dataset.py:168`의 `_iter_test`는 순차 이터레이션이라 RNG를 쓰지 않는다.

---

## 3. 검증 (Task 2)

전부 GPU 7, `WANDB_MODE=disabled`, `checkpoint_path`는 scratchpad로 돌려
`checkpoints/`에 테스트 산출물이 섞이지 않게 했다. 확인 완료.

### 3-1. 조립 (DRYRUN)

`DUALEVAL` 미지정 시 sh 조립 결과가 기존과 **`dual_eval=False` 한 줄만** 다르다.

```
32a33
> dual_eval=False          # DUALEVAL 미지정
> dual_eval=True           # DUALEVAL=True
```

### 3-2. 기존 동작 불변 — 구코드(worktree `26fa387`)와 직접 대조

동일 인자(`epochs=4 eval_interval=2 seed=0 ema=True`)로 구코드 run과
신코드 `dual_eval=False` run을 돌려 비교:

| 비교 | 결과 |
|---|---|
| `[EVAL @ ...]` 지표 전체 | **완전 일치** |
| 저장된 `step_90` 가중치 | **bit-identical** (27 tensors) |
| `num_params` | **9,967,621** 로 동일 |

### 3-3. dual_eval=True가 학습을 바꾸지 않는가 (RNG 수정 후)

| 비교 (`dual=False` vs `dual=True`) | 결과 |
|---|---|
| `[step N]` train 로그 전체 | **완전 일치** |
| `[EVAL @ ...]` EMA 지표 | **완전 일치** |
| 저장된 `step_90` 가중치 | **bit-identical** |
| `step_45.live`, `step_90.live` | 정상 저장 |

즉 토글을 켜도 학습 궤적·EMA 곡선·체크포인트가 모두 그대로고, `_raw` 지표만 추가된다.

### 3-4. `_raw` 키가 실제로 찍히고 EMA와 다른 값인가

```
[EVAL @     step 45] all/accuracy=0.3074 ... nqueens/queen_count_mean=13.9326
[EVAL RAW @ step 45] all/accuracy_raw=0.2884 ... nqueens/queen_count_mean_raw=16.0583
[EVAL @     step 90] all/accuracy=0.3332 ... nqueens/queen_count_mean=15.8174
[EVAL RAW @ step 90] all/accuracy_raw=0.8750 ... nqueens/queen_count_mean_raw=0.0000
```

step 90에서 token accuracy가 EMA 0.3332 / raw 0.8750으로 **2.6배** 벌어진다.
문제 제기한 왜곡이 90 step짜리 초단기 run에서도 바로 보인다.

### 3-5. 기존 checkpoint 로드

`checkpoints/Nqueens-8-ACT-torch/nq8_off_b220_ema6k_s0/step_9112`를
`load_checkpoint`로 물려 `dual_eval=True`로 실행 → **exit 0**, 로드 성공.

```
[EVAL @     step 45] all/accuracy=0.8883 all/exact_accuracy=0.0078
[EVAL RAW @ step 45] all/accuracy_raw=0.8859 all/exact_accuracy_raw=0.0026
```

원본 체크포인트 md5 변경 없음.

> 참고: 처음엔 로드가 실패했는데 테스트 속도용으로 준 `DISABLE_COMPILE=1` 때문이었다.
> 체크포인트는 `torch.compile` 모델에서 저장돼 `_orig_mod.` 접두사가 있고, compile을
> 끄면 키가 안 맞는다. 이번 변경과 무관한 기존 성질이며, compile을 켜니 정상 로드된다.

---

## 4. 비용과 한계

- **eval 시간 2배**. 학습 시간은 그대로다. `eval_interval=200`, 전체 3000 epoch 기준
  eval은 15회이므로 run 전체로는 미미하지만, eval이 무거운 설정에서는 감안해야 한다.
- `ema=False`면 in-loop eval이 애초에 raw를 보므로 `dual_eval`은 무시된다. 이 조합으로
  실행하면 시작 시 안내 문구를 한 번 출력한다.
- raw pass는 preds/evaluator 산출물을 저장하지 않는다(2-1). raw 예측 덤프가 필요하면
  기존대로 `.live` 체크포인트를 사후 스캔해야 한다.
- 검증은 `arch=gram_nqueens8` 경로에서만 했다. ARC/Sudoku도 같은 코드 경로를 타지만
  실행해보진 않았다 — 기본값 `False`라 해당 run들의 동작은 정의상 불변이다.

---

## 5. 재현 커맨드

```bash
# 조립만 확인
DUALEVAL=True DRYRUN=1 bash train_nqueens8_fix.sh

# 실제 사용
RUN=nq8_dual_test EMA=True DUALEVAL=True GPUS=6,7 bash train_nqueens8_fix.sh

# 이 리포트의 검증 run (GPU 7, wandb 미기록, 산출물 scratchpad)
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=disabled python pretrain.py \
  arch=gram_nqueens8 data_paths="[data/nqueens-8]" \
  'evaluators=[{name: nqueens@NQueens}]' \
  epochs=4 eval_interval=2 seed=0 global_batch_size=192 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=False arch.pos_encodings=rope arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  arch.kl_coef=0.07 arch.kl_reduction=mean arch.noise_head_expansion=1.5 \
  ema=True ema_rate=0.9999 dual_eval=True lr_warmup_steps=500 \
  +run_name=dual_smoke +checkpoint_path=/tmp/dual_smoke
```
