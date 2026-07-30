# REPORT_INDEX.md — 리포트 29건 목차

작성 2026-07-30. 리포트 본문은 **삭제하지 않았다** — 29건 전부 `reports/archive/` 에 있고
이 문서와 [REPORT_FINAL.md](REPORT_FINAL.md) 만 루트에 둔다. 각 리포트가 근거 파일이다.

원본을 남기는 이유: FINAL 초안을 쓰는 과정에서 실측 대조로 수치 3건이 정정됐다
(10×10 측정 주체, v4 harness 미측정, `logvar_bias` 예산별 값). 원본이 있어야 사후 검증이 된다.

**정본 관리**: `REPORT_FINAL.md` 는 50번(lecun docker)이 관리한다. 29번은 데이터(결과 JSON)만 올린다.

- **일시**는 리포트 본문의 `일시:` 헤더 값이다(git 커밋일과 다를 수 있음 — `grid_upper_29` 는 본문에
  뒤늦게 작성했다고 명시돼 있다).
- **서버** 표기: `50` = lecun docker (`trm_chaeun`, `/workspace/TinyRecursiveModels`),
  `29` = hinton (`/home/chaeun/gram-repro`). 두 박스가 β 대역을 나눠 담당했다.

---

## 1. 진단기 (2026-07-21 ~ 07-24) — 실패 원인 규명

| 리포트 | 일시 | 다루는 run | 핵심 결론 1줄 |
|---|---|---|---|
| [REPORT_night2.md](reports/archive/REPORT_night2.md) | 07-21 | `gram_sudoku_lmean*`, nq8 v2 3-run | KL reduction 이 L-mean 으로 돌고 있었다 — 명시 키로 고쳐 v2 3개 완주했으나 **N-Queens 는 여전히 실패**(자명해 고착 + posterior collapse). |
| [REPORT_diag.md](reports/archive/REPORT_diag.md) | 07-23~24 | `gram_nq8_mean1`, `gram_nq8_mean_nh512` | **백본은 결백하다** — posterior valid 100.00% / 99.87%, prior 만 0.00%. 문제는 KL/σ(prior) 층. 단 EMA 사본 기준이라는 단서 붙음. |
| [REPORT_diag2.md](reports/archive/REPORT_diag2.md) | 07-24 | 위 2 run + harness | SEQ_LEN 버그는 실재하나 `eval_harness` 에는 영향 없음. depth×σ 4셀 전부 prior 0%, 실패 사유는 **queen 개수**(배치 충돌 아님). L-sum/L-mean 실패 모드 2종 분리. |

## 2. 감사기 (2026-07-26 ~ 07-28) — 코드·측정 신뢰성

| 리포트 | 일시 | 다루는 run | 핵심 결론 1줄 |
|---|---|---|---|
| [REPORT_audit.md](reports/archive/REPORT_audit.md) | 07-26 | 전체 (코드) | 논문 대조 전수 감사 — KL reduction/노이즈 head/gradient 격리/KL 위치/TTS 5항목 **[일치]**, EMA bias correction 은 논문 미기재 항목이라 **[이탈]** 판정. |
| [REPORT_nograd.md](reports/archive/REPORT_nograd.md) | 07-27 | 없음 (코드) | `gram.py` 의 `no_grad` 4개를 TRM 원본과 diff — recursive 루프의 그것은 **TRM 상속**이고 우리 추가가 아니다. |
| [REPORT_yemb.md](reports/archive/REPORT_yemb.md) | 07-27 | `gram_nq8_mean_nh512`, `gram_sudoku_halt_only` | **y_emb ≫ z_H 가설 기각** — 두 임베딩은 같은 스케일이고 Sudoku 와도 사실상 동일. 스케일 조정은 처방이 아니다. |
| [REPORT_klsweep.md](reports/archive/REPORT_klsweep.md) | 07-27 | β sweep 4-run | 데이터 6항목 전수 통과(clue 보존 100%), 입력 전달 경로 정상. **부수 발견: posterior valid 100% 는 라벨 복사다.** |
| [REPORT_contradiction.md](reports/archive/REPORT_contradiction.md) | 07-27 | `gram_nq8_fix_b`, `mean_nh512`, `mean1` | "KL≈0 인데 posterior valid=100%" 는 단일 run 안에서 동시 성립 — 모순이 아니라 **posterior 가 y 를 우회 복사**하고 있었다. |
| [REPORT_ema_test.md](reports/archive/REPORT_ema_test.md) | 07-27 | `gram_nq8_fix_b` (EMA off 재학습) | EMA bias correction 코드 검증 + EMA off 재학습. `step_N`/`.live` 의 의미를 확정. |
| [REPORT_audit_measure.md](reports/archive/REPORT_audit_measure.md) | 07-28 | `nq8_off_b220` vs `_ema_s0` | **같은 seed 가 같은 궤적을 만들지 않는다** — raw 가중치 L2 차이 9.15%(step 1139) → 55.04%(17085). 결정성 설정이 코드에 아예 없다. |
| [REPORT_audit_measure_50.md](reports/archive/REPORT_audit_measure_50.md) | 07-28 | 19 run 출처 감사 | `step_N`=EMA 사본, `.live`=raw. in-loop eval 은 EMA 를 본다. **EMA 편향이 걸린 수치는 `emasweep_50` 4 run 뿐**이고 나머지 15 run 은 raw. |
| [REPORT_dualeval.md](reports/archive/REPORT_dualeval.md) | 07-28 | 없음 (기능 추가) | `dual_eval=True` 로 같은 step 에서 raw eval 을 병행 — 사후 `.live` 스캔 1~2시간/run 을 없앴다. RNG 스트림 보존을 검증에서 잡아 수정. |

## 3. β sweep — one-hot OFF / ON 격자 (2026-07-27 ~ 07-28)

| 리포트 | 일시 | 다루는 run | 핵심 결론 1줄 |
|---|---|---|---|
| [REPORT_prep.md](reports/archive/REPORT_prep.md) | 07-27 | 없음 (구현) | one-hot y_emb 토글 구현 + 정적 검증. 기본 경로는 bit-identical 유지. |
| [REPORT_push.md](reports/archive/REPORT_push.md) | 07-27 | 없음 | one-hot 커밋 `55ef8c2` push + 50번 실행 커맨드. VS Code `GIT_ASKPASS` 로 인증이 통과한 경위 기록. |
| [REPORT_onehot.md](reports/archive/REPORT_onehot.md) | 07-27 | `nq8_oh_b007`, `nq8_oh_b15`, `nq8_ctrl_b15` | one-hot 3-run launch. dev-container IPC 소켓을 직접 지정해 pull 인증을 통과시킨 경위 포함. |
| [REPORT_sweep_50.md](reports/archive/REPORT_sweep_50.md) | 07-27 | `nq8_on_b{100,140,180,220}` | one-hot **ON** β 4점. |
| [REPORT_sweep_29.md](reports/archive/REPORT_sweep_29.md) | 07-28 | `nq8_off_b{100,140,180,220}` | one-hot **OFF** β 4점 — prior valid 단조 증가(0.61%→56.15%), **최댓값이 경계 β=2.2** 라 peak 을 감싸지 못했다. |
| [REPORT_finesweep_29.md](reports/archive/REPORT_finesweep_29.md) | 07-28 | `nq8_on_b{160,170,190,200}` | ON 0.1 간격 미세 sweep — **단조가 아니고 β=1.9 가 이웃의 1/5 로 급락**. 이 서버만으로는 peak 특정 불가. |
| [REPORT_finesweep_50.md](reports/archive/REPORT_finesweep_50.md) | 07-28 | `nq8_on_b{150,210}`, `b180_s1/s2` | **β=1.8 을 seed 만 바꿔 3번 = 0.6268 / 0.0014 / 0.0008.** seed 편차가 β 간 최대 차이의 1.9배 → **0.01 미세 sweep 은 무의미.** |
| [REPORT_grid_upper_29.md](reports/archive/REPORT_grid_upper_29.md) | 07-28 | `nq8_on_b{140,200,450,560}_ema` | 논문 환산 후보 대역(β 4.5/5.6)이 이 설정에서 작동 안 함(0.06%/0.00%), **EMA 아티팩트 아님**. β=1.4 는 EMA 0.0052 vs raw 9.21% — **EMA 곡선으로 판정했으면 오분류**. |
| [REPORT_offup_50.md](reports/archive/REPORT_offup_50.md) | 07-28 | `nq8_off_b{280,300,320,340}_ema` | OFF 상행은 β 단조 감소(4.08%→0.55%) — **상행 대역에 정점 없음.** EMA in-loop 은 4 run 전부 0.0000 이었으나 `.live` 는 살아 있었다. |
| [REPORT_off_line_29.md](reports/archive/REPORT_off_line_29.md) | 07-28 | `nq8_off_b240_ema`, `b260_ema` | `.live` 정점이 곡선 내부에 잡힘(45.52%@14807, 40.28%@6834) → **OFF 최적점은 β=2.4 왼쪽.** EMA 로 골랐다면 4.2배 틀렸다. |
| [REPORT_peak_50.md](reports/archive/REPORT_peak_50.md) | 07-28 | `nq8_on_b160_ema6k`, `off_b210/b230_ema6k`, `off_b220_ema6k_s2` | 헤드라인 ON 1.6(6k) 0.7121. **그러나 같은 설정 OFF 2.2 3회가 0.7869/0.6267/0.3146(2.5배)** — 4점 간격이 이 폭 안에 전부 들어간다. |
| [REPORT_emasweep_50.md](reports/archive/REPORT_emasweep_50.md) | 07-28 | `nq8_on_b120_ema`, `b240_ema`, `b160_ema_s0/s1` | EMA 는 seed 갈림을 막지 못했고, 그 "이륙 실패"는 상당 부분 **EMA 가 만든 착시**였다 — 같은 s1 의 raw 는 0.2771, EMA 는 0.0007(**395배**). |
| [REPORT_ema_seed_29.md](reports/archive/REPORT_ema_seed_29.md) | 07-28 | `nq8_off_b220_ema6k_s0`, `nq8_on_b180_ema_s0/s1` | EMA on 도 seed 편차를 못 잡음(9.60% vs 0.38%, 25배). **EMA vs `.live` 우열은 조건에 따라 뒤집힌다.** |
| [REPORT_guide_diag.md](reports/archive/REPORT_guide_diag.md) | 07-28 | `nq8_off_b220_ema6k_s0`, `nq8_on_b200_ema` | 논문 Table 3b ablation 대조 — 논문은 guide only(σ=0)에서 0.00 인데 우리는 **82.90% / 23.58%**. 노이즈를 빼도 죽지 않는다. |
| [REPORT_b45_band_29.md](reports/archive/REPORT_b45_band_29.md) | 07-29 | β 4.0~5.6 8-run (`nq8_{on,off}_b{400..560}_ema6k`) | **8×8 최고 수치 대역.** 단일시도 valid 98.82% vs 논문 99.7±0.3, coverage 99.37% vs 논문 90.3±1.9. 5축 측정 프로토콜을 여기서 확정. |

## 4. 확장 태스크 (2026-07-27 ~ 07-30)

| 리포트 | 일시 | 다루는 run | 핵심 결론 1줄 |
|---|---|---|---|
| [REPORT_expand_prep.md](reports/archive/REPORT_expand_prep.md) | 07-27 | 없음 (데이터) | 10×10 데이터는 이미 존재(k∈{7,8,9}, 43,420 입력)하고 무결성 통과. **Graph Coloring 데이터 + verifier 신규 구현.** |
| [REPORT_gc8_50.md](reports/archive/REPORT_gc8_50.md) | 07-29~30 | `gc8_v5_{add_b0,add_bm4,catnd}_s*` 4-run | **GC 8-vertex 첫 캠페인 — conflict 0.046±0.025 vs 논문 2.7±2.1, coverage 0.842±0.026 vs 0.858±0.005.** bias −4 는 ~24.5k step 단색 붕괴 후 탈출. |

---

## 리포트가 아직 없는 캠페인 (REPORT_FINAL 작성 시점)

결과 JSON 과 로그는 있으나 전용 리포트가 없는 run 이 있다. `REPORT_FINAL.md` 의 수치는 이들에 대해
원본 JSON/로그를 직접 인용한다.

| run 군 | 산출물 | 상태 |
|---|---|---|
| `nq8_v2_sum007_{add,catnd}_s0/s1` | `experiments/results/v2_*.json` 24건 | harness 완료, 리포트 없음 |
| `nq8_v4_{add,catnd}_b007_s2/s3` | 로그만 (`logs/nq8_v4_*.log.gz`) | **in-loop 완주, harness 미측정** |
| `nq8_on_b500_{s1,s2,decay}`, `nq8_off_b100_forcehalt` | `experiments/results/b500_*.json` 16건 | harness 완료, 리포트 없음 |
| 10×10 (`nqueens-10`) | 이 서버: `logs/failed_nq10_2129/` 실패 기록만 | **29번에서 측정됨** (`nq10_on_b520_s0/s1/s2`, `nq10_on_b450_s0`) — JSON 수령 대기, 잠정값은 `REPORT_FINAL.md` §1.2 |
