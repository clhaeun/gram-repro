# [50] residual probe + PCA 궤적 + depth utilization

- 실행: 2026-08-11, GPU 0-7, 전부 추론 전용 (재학습 없음, `models/` 무수정)
- 스크립트: `experiments/residual_probe.py` (GRAM), `/workspace/fprm/experiments/residual_probe.py` (FPRM),
  `experiments/depth_util_summary.py`, `scripts/run_residual_probe_50.sh`, `scripts/run_depth_util_50.sh`
- 산출: `results/figdata/` (= `/workspace/fprm/results/figdata` 심볼릭 링크)
- gc10 은 34번 담당이라 제외.

---

## ① u_t 훅

`z_H = u + eps` 이므로 eps 주입 전 u 를 f_H 출력 직후에 잡아야 한다.  두 모델 다 기존
지점을 그대로 썼고 모델 코드는 한 줄도 고치지 않았다.

| | 포착 지점 |
|---|---|
| GRAM | `inner.prior_mu` 의 forward-pre-hook 입력. `gram.py:490` / `gram.py:512` 에서 `mu_p = self.prior_mu(z_H)` 의 인자가 곧 `u_t` 이고, eps 는 그 뒤 `z_H = z_H + eps_scale*eps` 에서 더해진다. eps 는 `inner.reparameterize` 를 갈아끼워 포착 (선례: `experiments/eps_propagation_probe.py`). |
| FPRM | `L_optimizer.step(state, y)` 진입 시 `state['y']` = h_(t-1), 인자 `y` = u_t, 복귀 후 `state['y']` = h_t, `state['residues']` = FPRM 자체 잔차. 주입은 refine 창 앞에서 돌므로 `L_optimizer.inject` 도 함께 감쌌다. |

- `h_t = u_t + eps_scale*eps_t` 로 재구성하고, 48번째 transition 을 `carry.inner_carry.z_H` 와
  대조하는 assert 를 통과시켰다 — **8 셀 전부 max|Δ| = 0.0** (비트 동일).
- `h_0 = H_init` (reset_carry 초기값). `is_sup_boundary` = supervision step 의 마지막 transition.
- forward_dtype = **float32**. bf16 은 상대오차 ~3e-3 이라 `d` 의 바닥을 반올림이 만든다.
- **forward 는 셀당 1회**. batch = `[test 앞 256행]` ++ `[퍼즐 idx0 × N 복제]` 로 합쳐
  잔차 통계와 궤적을 같은 패스에서 뽑았다. 앞 256행의 난수열은 배치 합성과 무관하게 같다.

## ②③ residual — `results/figdata/residual_{key}.csv`

컬럼 `t, a, d, n, r, is_sup_boundary` (FPRM 은 `fprm_res, stepsize` 2컬럼 추가).
토큰별 L2 → 비율 → 배치·토큰 평균 (float64 누산). 본 파일은 전 위치, `token_content/`
사본은 puzzle_emb 접두 16 제외.

| key | ckpt / step | T | a 범위 | d(t=1) | d(t=8) | d(t=16) | d(t=48) | d48/d16 |
|---|---|---|---|---|---|---|---|---|
| nq10_sig1 | nq10_repro_s0 / 138534 | 48 | 31.65–32.53 | 1.1990 | 0.5464 | 0.5351 | 0.5226 | 0.977 |
| nq10_sig0 | 〃 | 48 | 22.47–23.86 | 1.1990 | 0.1580 | 0.1032 | 0.0894 | 0.866 |
| nq8_sig1 | nq8_final_add_s0 / 67428 | 48 | 31.76–32.55 | 1.2013 | 0.5569 | 0.5521 | 0.5515 | 0.999 |
| nq8_sig0 | 〃 | 48 | 22.61–23.71 | 1.2013 | 0.1037 | 0.0625 | 0.0610 | 0.976 |
| gc8_sig1 | gc8_paper_34 / 45540 | 48 | 30.71–32.00 | 1.0109 | 0.4137 | 0.4015 | 0.3984 | 0.992 |
| gc8_sig0 | 〃 | 48 | 20.87–22.81 | 1.0109 | 0.1354 | 0.0765 | 0.0601 | 0.785 |
| sudoku_sig1 | sudoku_paper_34 / 65100 | 48 | 32.16–34.36 | 1.3406 | 0.5906 | 0.5865 | 0.5905 | 1.007 |
| sudoku_sig0 | 〃 | 48 | 23.69–26.46 | 1.3406 | 0.2482 | 0.2258 | 0.2224 | 0.985 |
| fprm_s0p | nq10_fprm_S0p_nonoise_s0 / 138534 | 929 | 22.59–22.63 | 0.9920 | 0.1490 | 0.0475 | (t=929) 0.0004 | — |
| fprm_s2 | nq10_fprm_S2_conv3_s0 / 138534 | 1000 | 22.62–22.63 | 0.9948 | 0.0918 | 0.1690 | (t=1000) 0.0001 | — |

### ③ 분류 (주어진 규칙 적용, 인과 주장 없음)

| 셀 | a | d | 분류 |
|---|---|---|---|
| GRAM 8 셀 전부 | 유계 (변동폭 ≤ 2.8, 발산 없음) | t≈8 이후 평탄 (d48/d16 = 0.79–1.01) | 안정하나 수렴 없음 |
| fprm_s0p, fprm_s2 | 유계 (변동폭 ≤ 0.04) | 단조 감소, 1e-4 대까지 | 수렴 + 안정 |

σ=0 셀(노이즈 off, 순수 결정론 재귀)에서도 d 의 평탄이 유지된다 —
nq8 0.0610, gc8 0.0601, nq10 0.0894, sudoku 0.2224 에서 각각 정체.

## ④ 궤적 — `results/figdata/traj_{key}.npz`

- `h` (N, T, seq_all*D) float16 — 전 위치 flatten. `loss` (N, T) float32 — 각 h_t 를
  `lm_head` 에 통과시킨 softmax CE (유효토큰 평균, per-sample).
- 부가: `loss_stablemax` (학습 손실 정의), `h_mean` (N, T, 512; puzzle_emb 제외 토큰평균,
  구 `traj_dump` 규약), `t_index`.
- N=50 은 같은 퍼즐(test idx 0)에 노이즈 시드만 다르게. σ=0 셀은 N=1.
- **FPRM 은 N=50 을 유지했다**: `fixed_init=False`(init_std 6.0) 라 nonoise 셀도 y 초기값이
  샘플마다 난수여서 궤적이 하나로 붕괴하지 않는다.
- 지형 격자는 계산하지 않았다 (로컬 K-D tree 용 원자료만).

## ⑤ FPRM 자체 잔차

`fprm_res = ||z - f(z)||_inf / (||f(z)||_inf + eps)` (seq 축 max) — 우리 `d`(L2, 토큰별,
분모 ||h_(t-1)||)와 정의가 달라 **별도 컬럼**으로 분리했다.
fprm_s2: t1 5.288 → 마지막 0.00091. 주입은 1000 iteration 중 98 회 발화 (n 최대 0.446, t=9).

T 정의도 다르다: GRAM 은 `H_cycles(3) × halt_max_steps(16) = 48` 고정, FPRM 은 eval halting
(fp_thresh 0.1 / stepsize<1e-3, 배치 전체 기준, cap `max_iter_eval`=1000) 까지 가변이라
실제 iteration 수를 meta 의 `T`/`T_halt_reason` 에 적었다 (s0p 929, s2 1000=cap).

## ⑥ depth utilization

`sudoku_paper_34` / EMA / N=20 / `data/sudoku-extreme-1k-aug-1000/test` 앞 20,000행,
sigma 1.0 / mu 1.0 / seed 0 — 62번 격자와 같은 조건. hms **128, 320** 만 추가한다
(8–64 는 62번에 있음). 20,000행을 GPU 0-7 에 연속블록 8샤드로 나눈다.
곡선 데이터: `results/figdata/depth_util_sudoku.csv`
(`effective_layers = H_cycles×(L_cycles+1)×L_layers × hms = 42 × hms`; FPRM 4.4.2 의
Sudoku 640/1040 축과 같은 환산).

실행: hms=128 02:37→04:54 (2h17m), hms=320 04:54→10:39 (5h45m), 8샤드 × 2,500행.

| hms | eff layers | seed | prior exact | pass@20 | maj@20 | vsel@20 | prior valid | viol mean |
|---|---|---|---|---|---|---|---|---|
| 16 | 672 | 0 | 0.5933 | 0.8225 | 0.7694 | 0.8160 | 0.5965 | 7.662 |
| 16 | 672 | 1 | 0.5939 | 0.8213 | 0.7705 | 0.8156 | 0.5970 | 7.660 |
| **128** | **5376** | 0 | **0.7363** | **0.8885** | **0.8544** | **0.8842** | 0.7396 | 5.059 |
| **320** | **13440** | 0 | **0.7826** | **0.9091** | **0.8821** | **0.9045** | 0.7860 | 4.175 |

depth 2배당 증가폭 (log2 정규화):

| 구간 | prior | pass@20 | maj | vsel |
|---|---|---|---|---|
| 16→128 (3.00 doubling) | +0.0477 | +0.0220 | +0.0283 | +0.0227 |
| 128→320 (1.32 doubling) | +0.0350 | +0.0155 | +0.0210 | +0.0154 |

- 13,440 유효 레이어(FPRM 이 보고한 post-norm 포화점 640 의 21배)까지 단조 증가하며,
  2배당 증가폭이 줄되 0 이 되지 않는다. 측정 범위 안에서 포화점은 관측되지 않았다.
- `consistent` 0.9965/0.9966/0.9965, `clue_breaks_mean` 0.0035/0.0034/0.0035 로 깊이와 무관하게 불변.
  `violations_mean` 은 7.66 → 5.06 → 4.18.
- seed 재현성(hms16 h0 vs h1): prior 차 0.0006, pass@20 차 0.0012.
- ★ 128/320 은 학습 깊이(halt_max_steps=16) 밖의 외삽이다.
- hms 8/32/64 는 62번 격자에 있다. 같은 스키마로 붙이면 곡선이 완성된다
  (`results/figdata/depth_util_sudoku.meta.json` 의 `hms_missing` 참조).

## 표본·메타

- probe 256행 (gc8 은 test 전체가 255행이라 255), 궤적 퍼즐 idx 0, ⑥ 은 20,000행.
- 각 산출물 옆에 `*.meta.json` — ckpt·step·weight(EMA)·data·T·D·seq_len·puzzle_emb_len·
  정의·재구성 검증값 기록.
