# 플래그 신설 시 상설 검증 절차 (2026-08-08 확정, 50번)

새 config 필드·런처 손잡이를 추가할 때 (a)~(d) 를 전부 통과해야 push 한다.
**(d) 는 2026-08-08 SS 교착 사고로 추가됐다 — 아래 "왜 (d) 인가" 참조.**

## (a) DRYRUN 조립

```bash
DRYRUN=1 RUN=dry ... bash scripts/train_nqueens10_<arm>.sh 2>&1 | tr ' ' '\n' | grep <필드>
```
hydra arch config 는 struct 다. **pydantic 선언만으로는 부족하고
`config/arch/*.yaml` (arch 필드) 또는 `config/cfg_pretrain.yaml` (top-level 필드)에도
선언해야** 오버라이드가 `Key not in struct` 로 즉사하지 않는다.

## (b) harness md5 회귀

기본값에서 기존 checkpoint 의 harness 산출이 **비트 동일**해야 한다.

```bash
as-chaeun env CUDA_VISIBLE_DEVICES=<g> python3 experiments/sweep_harness.py \
  --ckpt-dir checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s0 --step step_67428 \
  --data data/nqueens-8/test --n 8 --N 20 --seed 0 \
  --sidecar data/nqueens-8/sidecar_8.json --out <out>.json
```
기준 md5 (2026-08-08 현재):
- nq8  `nq8_final_add_s0/step_67428`, N=20 → `eb7835e6a48f46f6b946157a5e0de2f9`
- nq10 `nq10_repro_s0/step_138534`, N=4  → `83f14a076ce5cff380b34264757cb42a`
- nq10 `nq10_repro_s0/step_138534`, N=20, `--halt-max-steps 16`
  → `71bd1818294327030a3e4c7740f1bfeb`
  (= 커밋된 `results/halt_sweep_20260806/nq10_repro_s0_final_ema_h0_hms16.json`)

★ 기준 JSON 의 **스키마가 현행과 같은지 먼저 확인**하라. 출력 필드가 늘면 기준선이
무효가 된다 (과거 `halt_max_steps` 키 추가로 8/4 기준선이 무효화된 전례).

## (c) 1GPU eager 스모크 — loss 비트 동일

`DISABLE_COMPILE=1` 로 돌린다. **compile 경로는 비결정적이다** — 동일 설정 2회
대조군(ctl1/ctl2, 같은 GPU 순차)도 step 1 만 일치하고 step 10 부터 6번째
유효숫자가 갈린다. eager 에서는 419 step 5,868 행이 비트 동일하다.

```bash
as-chaeun env WANDB_MODE=offline WANDB_DIR=<d> CUDA_VISIBLE_DEVICES=<g> DISABLE_COMPILE=1 \
  torchrun --nproc-per-node 1 ... pretrain.py arch=gram_nqueens10 ... \
  epochs=1 eval_interval=1 min_eval_interval=99 global_batch_size=256 ...
```
비교: `scripts/parse_wandb_offline.py` 로 CSV 덤프 후 `sort` 하고 md5.
★ CSV 는 **정렬 후** 비교하라 — 키 행 순서가 실행마다 다르다(값은 같다).
★ eager 는 `global_batch_size=768` 이 24 GiB 에 안 들어간다(23.50 GiB). 256 을 쓰고
  A/B 양쪽 동일하게 낮췄음을 보고에 명기하라.

## (d) ★2GPU 스모크 200 step — 플래그를 켠 상태로★

```bash
timeout 900 as-chaeun env WANDB_MODE=offline WANDB_DIR=<d> CUDA_VISIBLE_DEVICES=0,1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  torchrun --nproc-per-node 2 ... pretrain.py ... <플래그를 켠 값으로> ...
```
확인 항목:
1. **교착 없이 200 step 이상 도달**하는가 (timeout exit 124 면 교착)
2. rank 별 **분기 선택이 동일**한가 (분기가 있는 플래그일 때)
3. rank 간 **`grad=None` 파라미터 개수가 같은가**

`grad=None` 개수는 아래로 잰다:
```python
none = [n for n, p in model.named_parameters() if p.grad is None]
```

### 왜 (d) 인가 — 2026-08-08 SS 교착 사고

`arch.ss_prob` 을 (a)(b)(c) 로만 검증하고 push 했다. **1GPU eager 419 step 을
NaN 없이 통과**했으나 62번의 다중 GPU 실행이 4~6 step 에서 교착·SIGABRT 했다.

원인은 두 개였고 **둘 다 1GPU 에서는 원리적으로 드러나지 않는다**:

1. `pretrain.py:607` 이 `torch.random.manual_seed(config.seed + RANK)` 로 rank 마다
   전역 시드를 다르게 준다. SS 분기를 전역 RNG 로 뽑아 rank 별로 갈렸다.
2. `pretrain.py` 는 DDP 가 아니라 **수동 all_reduce** 이고 `if param.grad is not None`
   가드가 있었다. 마지막 transition 이 prior 로 뽑히면 posterior head 5개
   (`post_mu` ×2, `post_logvar` ×2, `post_logvar_bias`) 가 grad=None 이 되어
   rank 마다 collective 호출 횟수가 달라졌다.
   실측: `ss_prob=0.0` → grad=None 0/24개, `ss_prob=0.5` → 5/24개.

★ 결함 2 는 SS 와 무관하게 잠재돼 있던 버그다. SS 가 먼저 밟았을 뿐이다.

수정: `gram.py` 의 `_ss_gen` (rank 무관 CPU generator) + `pretrain.py` 에서
grad=None 을 `zeros_like` 로 채워 all_reduce 대상 집합을 고정.

## 하지 말 것

**플래그별로 `gram.py` 를 복제하지 마라.** 위 결함 2 는 SS 와 무관한 공용 경로의
버그였고, 파일을 갈랐다면 못 찾고 나중에 터졌다. 갈라진 파일은 이후 모든 수정을
두 곳에 해야 하고 조용히 어긋난다 (`layers.py` / `fp_layers.py` 가 이미 110줄
중복 상태다). 기본값 + bit-identical 회귀 방식은 `eps_scale`, `damping_eta`,
`diag_kl_per_transition`, `ss_prob` 에서 기존 실험을 한 번도 깨지 않았다.

## 실행 환경 주의

- GPU 작업은 `as-chaeun <command>` 로 감싼다 (호스트에서 점유자 식별).
- 산출 경로는 chaeun 이 쓸 수 있어야 한다 — `reports/INCIDENT_20260807_perm.md` 참조.
