# CHECKLIST.md — 반복해서 당한 것만 모아둔다

두 서버(29번 hinton / 50번 lecun)가 같은 repo 를 공유한다. 여기 있는 항목은 **실제로
사고가 난 뒤에** 추가된 것이고, 같은 사고가 재발한 항목은 횟수를 적어둔다.

---

## 1. pydantic 필드 추가 = yaml 필드 추가 (★ 3회 발생)

> **pydantic 필드 추가 = yaml 필드 추가. 둘 중 하나만 하면 hydra 가 조용히 거부하거나
> (struct) 조용히 무시한다(extra).**

`PretrainConfig`(`pretrain.py`) 나 arch Config(`gram.py` 의 `*_ACTV1Config`) 에 필드를
추가하면 **대응 yaml 에도 반드시 같이 넣는다.**

| 대상 | 코드 | yaml |
|---|---|---|
| pretrain 레벨 | `pretrain.py` `PretrainConfig` | `config/cfg_pretrain.yaml` |
| arch 레벨 | `models/recursive_reasoning/gram.py` `*Config` | `config/arch/gram_*.yaml` |

**두 가지 실패 모드가 있고 증상이 정반대다:**

- **struct 거부 (요란하게 죽는다)** — yaml 에 키가 없는데 CLI 로 `key=value` 를 주면
  hydra 가 즉시 중단한다. 그나마 이건 눈에 보인다.
  ```
  Could not override 'nodecay_marked'.
  Key 'nodecay_marked' is not in struct
  ```
- **extra 무시 (조용히 틀린 값으로 돈다)** — yaml 에는 있는데 pydantic Config 에 필드가
  없으면 값이 `extra` 로 흘러 **기본값인 채로 학습이 끝까지 돈다.** 로그에 아무 표시도
  안 난다. 이게 훨씬 위험하다.

**전례 3건:**

| 필드 | 커밋/시점 | 증상 |
|---|---|---|
| `y_onehot` | — | arch yaml 쪽 |
| `logvar_bias_init` | — | arch yaml 쪽 |
| `nodecay_marked` | `f95b698` (50번) → `cfg_pretrain.yaml` 누락 | 2026-08-03 29번에서 A3′/A1 launch 가 preflight 통과 후 **즉사**. `pretrain.py:98` 에는 필드가 있었다 |

**launch 후 확인법**: `checkpoints/<proj>/<run>/all_config.yaml` 에서 그 키의 **실효값**을
직접 읽어라. CLI 에 뭘 줬는지가 아니라 저장된 값이 진실이다. 키가 아예 없으면 extra 로
흘러간 것이다.

---

## 2. yaml 파일 끝 개행 (2026-08-03)

`config/cfg_pretrain.yaml` 이 개행 없이 끝나 있어서, 아래에 줄을 추가하면 **직전 줄에
이어붙는다.** 추가 전에 `tail -c 1 <file> | xxd` 로 `0a` 인지 확인해라.

현재 개행 없이 끝나는 추적 파일(17개, 손대지 않았다 — 건드리면 공유 repo 에 diff 소음):
`config/arch/gram_nqueens{8,10}.yaml`, `config/arch/{hrm,trm,trm_hier6,trm_singlez,transformers_baseline}.yaml`,
`evaluators/{arc,nqueens}.py`, `exp/make_test10k.py`, `experiments/diag/{diag_posterior,trace_kl}.py`,
`models/{fp_layers,fp_transformer}.py`, `train_nqueens8.sh`, `train_nqueens8_1.5.sh`

---

## 3. launch 규율

- **launch 전 `nvidia-smi` 로 대상 GPU 점유 확인. >1000MiB 면 중단.** 다중 사용자 박스다.
  (`train_*.sh` 에 preflight 가 들어 있지만 `DRYRUN=1` 이면 건너뛴다)
- **preflight 통과 ≠ 살아 있음.** 60초 뒤 `ps` 로 생존을 다시 확인해라 —
  hydra config 오류는 preflight **뒤에** 터진다(§1 전례).
- `run_name` 중복 확인: 같은 이름으로 다시 띄우면 checkpoint 와 로그가 **덮여 날아간다.**
- tmux 창 분리, 로그는 `tee` (`-a` 금지), 포트 중복 금지.
- 학습 sh 앞에 conda 활성 필요: `source ~/miniconda3/bin/activate gram &&`

## 4. epochs 는 eval_interval 의 배수여야 한다

`pretrain.py:657`
```python
assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."
```
`train_nqueens8_v2.sh` 는 `eval_interval=200` 이 **하드코딩**이다. 8×8 에서 EPOCHS 는
200 의 배수여야 한다. 10×10 은 `EVALINT` 환경변수라 조정 가능.

step 수: `total_steps = epochs × total_groups × mean_puzzle_examples / global_batch_size`
(`pretrain.py:279`). **`total_groups` 에 비례**하므로 데이터 group 정의가 바뀌면 같은
epochs 라도 step 수가 달라진다 (pair 확장 때 nq8 1.64배 / nq10 2.91배).

## 5. 출력 파일명에 step 을 넣어라

`band_harness.sh` 에서 step 을 빼먹어 **정점 결과가 최종 결과에 덮여 날아간 적이 있다.**
harness 출력은 `<tag>_<run>_step<STEP>_<ema|live>_s<seed>.json` 형식을 지킨다.
