# experiments/legacy/ — 은퇴한 스크립트 (과거 결론의 근거)

**지우지 않는다.** 현재 결과 산출 경로에서는 빠졌지만, 아래 REPORT 수치들의 근거라
재실행 가능해야 한다. 2026-08-03 harness 정리 때 루트에서 옮겼다 (`git mv`, 파일명 유지).
현역 경로는 experiments/ 루트의 gram_harness → sweep_harness/gc_harness → run_final_harness.sh
+ aggregate_final.py, 그리고 coverage_curve.py / make_paper_figures.py / wandb_harness_upload.py 다.

## 파일 → 어느 REPORT 수치를 만들었나

| 파일 | 근거 |
|---|---|
| `run_v5_harness.sh` | v5 라운드 harness 32건 (`results/v5h_*.json`) — REPORT_pairexpand_50.md 의 v5 표. 2026-07-31 실행이 마지막. |
| `run_v6_harness.sh` | v6 (29번 학습분) harness 24건 (`results/v6h_*.json`) — y_onehot ON/OFF 를 harness 축으로 맞춘 표. |
| `run_sweep50_harness.sh` | sweep50 라운드 42건 (`results/sweep50_nq8_*.json`) — REPORT_FINAL.md 의 step sweep. |
| `run_onehot_harness.sh` | one-hot 처방 판정 12건 (`results/onehot_nq8_*.json`) — reports/archive/REPORT_push.md. |
| `coverage_pair_weight.py` | 기존 harness JSON 의 coverage 를 pair 가중으로 재집계 — reports/REPORT_pair_expand_29.md §coverage 두 축. 이후 pair 축은 sweep_harness 의 per_row 로 직접 산출한다. |
| `decompose_y_proj.py` | y_concat_proj 를 축소 성분(alpha)과 방향 잔차로 분해 — "concat 은 add 의 축소판" 판정 근거 (REPORT_INDEX.md concat 항). 산출 JSON 없이 stdout 로 인용됨. |
| `wandb_make_report.py` | wandb report 패널 구성 스크립트. offline 환경이라 dry-run 위주로 쓰였다. |
| `wandb_tag_runs.py` | wandb run 일괄 태깅(probe/final 구분). 2026-07-30 정리 때 사용. |
| `diag/` | 진단 스크립트 20종. 각 파일 → REPORT 매핑은 [diag/README.md](diag/README.md) 에 있다. |
| `gram_legacy.py` | 2026-08-03 정리 직전의 `models/recursive_reasoning/gram.py` 원본 스냅샷 (cp). 정리 diff 의 대조본. |
| `pretrain_legacy.py` | 같은 시점의 `pretrain.py` 원본 스냅샷 (cp). |
| `eval_harness.py` | 구 harness (strict 축). **verify_nqueens_dataset.py [12] 가 지금도 import 하는 contract 라이브러리** -- 지우면 검증이 조용히 SKIP. |
| `aggregate_final.py` | pair final 라운드 집계 (`../results/fin_*.json` 구 포맷 전용, pass_at 필요). REPORT_final_pair_50.md 산출. |
| `make_paper_figures.py` | 논문 Figure 10/12/15 + Table 1 생성 (7/30 산출분이 reports/figures/ 에 있음). |
| `wandb_harness_upload.py` | 구 포맷 결과 JSON 의 wandb offline 일괄 업로드. |
| `eval_checkpoint.py` | (구 repo 루트) checkpoint 에 in-loop evaluate() 재실행 -- harness 축 아님. |

## 실행

repo root 에서 실행한다. 이동하면서 diag/ 의 sys.path 깊이(+1)와
`experiments.diag.*` → `experiments.legacy.diag.*` 상호참조를 고쳤고,
26개 전부 `--help` 로 import 통과를 확인했다 (2026-08-03).

주의: 참조 checkpoint 상당수가 이미 삭제됐다 (`REPORT_FINAL.md` §0 주의 2).
재실행 전 `reports/cleanup/keep_steps.txt` 로 대상 step 생존 확인.
