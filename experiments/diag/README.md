# experiments/diag/ — 진단 스크립트 (과거 결론의 근거)

**지우지 않는다.** 아래 스크립트들이 `reports/archive/` 의 수치를 만들었다. 결론을 다시 확인해야
할 때 재실행 가능해야 하므로 남긴다. 2026-07-30 에 `experiments/` 루트에서 여기로 옮겼다
(`git mv`). 루트에는 상시 쓰는 harness 7개만 남겼다.

**옮기면서 고친 것** — repo root 계산이 한 단계 깊어져 전부 깨졌었다:
- `sys.path.insert` 의 `dirname` 을 하나 더 씌웠다 (16개 파일)
- 스크립트 간 참조 `from experiments.diag_posterior_prior import` → `experiments.diag.…` (4개)
- `e1~e4` 의 `sys.path[:0]` 에 repo root 를 추가 (4개)
- 검증: 26개 전부 `--help` 로 import 통과. `../gram_harness.py` 는 `__main__` 이 없는
  라이브러리라 단독 실행 대상이 아니다(이동 전에도 동일).

## 스크립트 → 근거 리포트

| 스크립트 | 어느 REPORT 의 수치를 만들었나 |
|---|---|
| `diag_posterior_prior.py` | [REPORT_diag.md](../../reports/archive/REPORT_diag.md) — **"백본은 결백하다"** posterior valid 100.00% / 99.87% vs prior 0.00% |
| `diag_posterior.py` | [REPORT_contradiction.md](../../reports/archive/REPORT_contradiction.md) §1 — posterior/prior 를 같은 스크립트에서 나란히 측정 |
| `diag_contradiction.py` | [REPORT_contradiction.md](../../reports/archive/REPORT_contradiction.md) §2 — ε=0 ablation, transition 별 KL, μ_q 의 y 의존성 |
| `diag_bucket.py` | [REPORT_diag.md](../../reports/archive/REPORT_diag.md) — 해 개수 버킷 분해 (sidecar 의 per-x 해 개수로 층화) |
| `diag_depth_sigma.py` | [REPORT_diag2.md](../../reports/archive/REPORT_diag2.md) §3 — depth×σ 4셀 전부 prior 0%, 실패 사유는 queen 개수 |
| `diag_qhalt.py` | [REPORT_diag2.md](../../reports/archive/REPORT_diag2.md) — q_halt 발화율 (L-sum 0.0/depth16 vs L-mean 1.0/depth1) |
| `diag_task2_remeasure.py` | [REPORT_diag2.md](../../reports/archive/REPORT_diag2.md) §2 — SEQ_LEN 수정 전/후 재측정 (효과 크기 0) |
| `diag_guide.py` | [REPORT_guide_diag.md](../../reports/archive/REPORT_guide_diag.md) — 3모드 harness(full/μ=0/σ=0) + μ·σ 크기 실측, 논문 Table 3b 대조 |
| `diag_yemb.py` | [REPORT_yemb.md](../../reports/archive/REPORT_yemb.md) — y_emb vs z_H 노름, Sudoku 대조 → **"y_emb ≫ z_H 가설 기각"** |
| `diag_ema_contamination.py` | [REPORT_ema_test.md](../../reports/archive/REPORT_ema_test.md) — EMA 사본이 측정을 오염시키는지 (`logs/ema_contam_*.log.gz`) |
| `diag_tracekl.py` | [REPORT_contradiction.md](../../reports/archive/REPORT_contradiction.md) §2b — transition 단위 KL 추적 (grad/no_grad 경로 분리) |
| `trace_kl.py` | [REPORT_nograd.md](../../reports/archive/REPORT_nograd.md) — `gram.py` 의 `no_grad` 4개 분류, KL 이 어느 경로로 흐르는지 |
| `sigma_probe.py` | [REPORT_diag.md](../../reports/archive/REPORT_diag.md) §A6 — 최종 checkpoint 의 σ 실측 (nqueens test 256샘플) |
| `test_kl_reduction.py` | [REPORT_night2.md](../../reports/archive/REPORT_night2.md) §1.5 — **런타임 테스트로 L-MEAN 확정.** `kl_loss/kl_raw` 비가 L 인지 1 인지로 판정한다 |
| `verify_yonehot.py` | [REPORT_prep.md](../../reports/archive/REPORT_prep.md) §1.4, [REPORT_push.md](../../reports/archive/REPORT_push.md) §2 — one-hot y_emb 정적 검증 (기본 경로 bit-identical 확인) |
| `make_raw_ema.py` | [REPORT_ema_test.md](../../reports/archive/REPORT_ema_test.md) — EMA shadow 에서 raw 가중치를 복원 (`.live` 저장 이전 run 대상) |
| `e1_failure_decomp.py` | [REPORT_diag2.md](../../reports/archive/REPORT_diag2.md) — 실패 분해. `logs/e1_*.log.gz` |
| `e2_gap.py` | 〃 — posterior/prior 격차. `logs/e2_*.log.gz` |
| `e3_lprm.py` | 〃 — LPRM(verifier) 계열. `logs/e3_*.log.gz` |
| `e4_sigma.py` | 〃 — σ 계열. `logs/e4_*.log.gz` |

`e1`~`e4` 는 `gram_harness.rollout` 을 쓰기 때문에 **SEQ_LEN 버그의 영향을 받은 유일한 경로**다
([REPORT_diag2.md](../../reports/archive/REPORT_diag2.md) Task 1). 수정 후 회귀 앵커를 통과했다.

## 실행

repo root 에서 실행한다 (스크립트가 스스로 sys.path 를 잡는다):

    python experiments/diag/diag_posterior_prior.py --help
    python experiments/diag/sigma_probe.py --help

**주의**: 이 스크립트들이 참조하던 checkpoint 중 상당수가 이미 없다. 결과 JSON 561건 중 309건이
없는 checkpoint 를 가리킨다(`REPORT_FINAL.md` §0 주의 2). 재실행 전에 대상 step 이 남아 있는지
`reports/cleanup/keep_steps.txt` 로 확인할 것.
