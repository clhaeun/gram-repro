# scripts/ — 런처 정본과 은퇴본

2026-08-05 재배치. `scripts/` 는 **현행 정본만**, `scripts/archive/` 는 은퇴본이다.
**삭제한 파일은 없다.** 전부 `git mv` 이므로 이력은 `git log --follow <경로>` 로 그대로 따라간다.

## 정본 (scripts/)

| 스크립트 | 용도 |
|---|---|
| `train_nqueens8_v2.sh` | nq8 기본 arm (`arch=gram_nqueens8`) |
| `train_nqueens8_fp.sh` | nq8 FPRM arm. `ARM=A1\|A2\|A3p` 필수 |
| `train_nqueens8_stageA.sh` | nq8 Stage A. `RESIDUAL_SCALE`/`ALPHA1`/`ALPHA2` 필수 |
| `train_nqueens10_v2.sh` | nq10 기본 arm |
| `train_nqueens10_fp.sh` | nq10 FPRM arm. `ARM` 필수 |
| `run_repro_harness.sh` | 현행 harness 드라이버 |

제자리에 남긴 것 (이동하면 다른 서버의 실행 경로가 깨진다):

| 경로 | 사유 |
|---|---|
| `experiments/run_harness_nq10_29.sh` | 29번이 쓰는 중 (`6fd6b72`) |
| `experiments/run_final_harness.sh` | DEPRECATED 표시만. 현행 코드로 실행 불가 |

## ★ 호출 경로가 바뀌었다

정본 5종은 예전에 repo 루트에 있었다. **이제 `scripts/` 를 붙여야 한다.**

```bash
# 예전
RUN=nq8_x ARM=A3p GPUS=0 bash train_nqueens8_fp.sh
# 지금
RUN=nq8_x ARM=A3p GPUS=0 bash scripts/train_nqueens8_fp.sh
```

`cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"` 로 깊이를 보정했으므로
**어느 디렉토리에서 호출해도 repo 루트를 잡는다.** `scripts/archive/` 는 `/../..` 다.
`cd` 가 없는 은퇴본 13종(`gram_*.sh`, `exp/*.sh`, `train_nqueens8{,_1.5}.sh`)은
예전부터 호출자 cwd 에 의존했다 — repo 루트에서 실행할 것. 이동해도 동작은 같다.

## ★ 아카이브된 REPORT 의 명령줄은 고치지 않았다

`train_nqueens8_fix.sh` 는 인용 18건인데 **전부 `reports/archive/` 안의
과거 실행 기록**(`RUN=... bash train_nqueens8_fix.sh`)이고, 마크다운 링크는
repo 전체에 0건이라 깨질 링크가 없다. 그 줄들은 *그때 실제로 실행한 명령*이므로
경로를 덧붙이면 기록이 거짓이 된다. 대신 아래 매핑을 정본으로 둔다.

## 이동 매핑 (32건)

### -> scripts/ (6)
```
train_nqueens8_v2.sh              -> scripts/train_nqueens8_v2.sh
train_nqueens8_fp.sh              -> scripts/train_nqueens8_fp.sh
train_nqueens8_stageA.sh          -> scripts/train_nqueens8_stageA.sh
train_nqueens10_v2.sh             -> scripts/train_nqueens10_v2.sh
train_nqueens10_fp.sh             -> scripts/train_nqueens10_fp.sh
experiments/run_repro_harness.sh  -> scripts/run_repro_harness.sh
```

### -> scripts/archive/ (26)
```
train_nqueens8.sh                         -> scripts/archive/train_nqueens8.sh
train_nqueens8_1.5.sh                     -> scripts/archive/train_nqueens8_1.5.sh
train_nqueens8_decay.sh                   -> scripts/archive/train_nqueens8_decay.sh
train_nqueens8_fix.sh                     -> scripts/archive/train_nqueens8_fix.sh
train_nqueens8_probe.sh                   -> scripts/archive/train_nqueens8_probe.sh
train_nqueens10.sh                        -> scripts/archive/train_nqueens10.sh
train_gc8_v2.sh                           -> scripts/archive/train_gc8_v2.sh
gram_nq8_fix.sh                           -> scripts/archive/gram_nq8_fix.sh
gram_nq8_sweep.sh                         -> scripts/archive/gram_nq8_sweep.sh
gram_sudoku_halt_only.sh                  -> scripts/archive/gram_sudoku_halt_only.sh
gram_sudoku_lmean_v2.sh                   -> scripts/archive/gram_sudoku_lmean_v2.sh
exp/gram_sudoku_b005.sh                   -> scripts/archive/gram_sudoku_b005.sh
exp/gram_sudoku_clip50k.sh                -> scripts/archive/gram_sudoku_clip50k.sh
exp/gram_sudoku_long.sh                   -> scripts/archive/gram_sudoku_long.sh
exp/gram_sudoku_lsum.sh                   -> scripts/archive/gram_sudoku_lsum.sh
exp/gram_sudoku_paper.sh                  -> scripts/archive/gram_sudoku_paper.sh
exp/gram_sudoku_sigmabound.sh             -> scripts/archive/gram_sudoku_sigmabound.sh
exp/gram_sudoku_sigma_fix.sh              -> scripts/archive/gram_sudoku_sigma_fix.sh
exp/run_eval_final.sh                     -> scripts/archive/run_eval_final.sh
exp/run_experiments_e1234.sh              -> scripts/archive/run_experiments_e1234.sh
exp/run_sudoku_mlp_t_4gpu.sh              -> scripts/archive/run_sudoku_mlp_t_4gpu.sh
experiments/legacy/run_onehot_harness.sh  -> scripts/archive/run_onehot_harness.sh
experiments/legacy/run_sweep50_harness.sh -> scripts/archive/run_sweep50_harness.sh
experiments/legacy/run_v5_harness.sh      -> scripts/archive/run_v5_harness.sh
experiments/legacy/run_v6_harness.sh      -> scripts/archive/run_v6_harness.sh
```
(`exp/` 는 .sh 가 전부 빠졌고 .py 3종만 남았다: `analyze_sudoku_tta.py`,
 `diag_sigma.py`, `make_test10k.py`)
