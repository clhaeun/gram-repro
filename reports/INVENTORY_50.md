# INVENTORY_50.md — 50번(lecun docker) 재고 조사

- 조사 일시: 2026-08-11 06:35 UTC · 읽기 전용 (코드 수정·학습·평가 없음, GPU 미사용)
- 대상: `/workspace/TinyRecursiveModels` (= gram-repro), `/workspace/fprm`
- checkpoints 를 가진 디렉터리는 이 둘뿐 (`find /workspace -maxdepth 3 -type d -name checkpoints` 실측)
- run 총 **78건**

## 기준·주의

- `계획 total_steps` = `int(epochs × total_groups × mean_puzzle_examples / global_batch_size)`,
  `pretrain.py:256`(TRM) / `pretrain.py:105`(fprm) 의 식에 **이 박스의** `data/<data_paths[0]>/train/dataset.json` 을 대입한 값.
- 저장 눈금이 eval 간격이라 완주해도 최종 step 이 계획보다 몇 step 작다 (예: 60,192 vs 60,208).
  → `완주` 판정은 최종 step ≥ 계획×0.999 로 잡았다.
- `gc8_paper_34` / `gc10_paper_34` 는 34번에서 전송받은 checkpoint 라 로컬 데이터로 계산한 계획값과 눈금이 다르다 → `판정불가`.
- `ema` 열 = `step_N` (EMA 사본), `live` 열 = `step_N.live` (raw). 근거: `reports/archive/REPORT_audit_measure_50.md`.
- fprm 쪽은 `.live` 대신 `step_N_train_state.pt` 를 함께 저장한다.

---

## ① run 목록 — gram-repro (TinyRecursiveModels)

### `Graph-coloring-10-ACT-torch` — task `gc10` (1건)

| run_name | 최종 step | 계획 total | 완료 | ema 파일 | live 파일 | 크기(ema) | mtime(ema) | step 파일수 | run dir 합계 |
|---|---|---|---|---|---|---|---|---|---|
| `gc10_paper_34` | 87,620 | 7,083 | 판정불가 (로컬 데이터 기준 plan 7,083 ≠ 최종 87,620) | `step_87620` | **없음** | 38.0 MB | 08-09 15:52 | 1 | 38.1 MB |

| run_name | epochs | lr | kl_coef | logvar_bias_init | kl_balance | ema_rate | L_cycles | halt_max_steps | seed | data_paths |
|---|---|---|---|---|---|---|---|---|---|---|
| `gc10_paper_34` | 5000 | 0.0001 | 0.45 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/graph-coloring-10` |

### `Graph-coloring-8-ACT-torch` — task `gc8` (16건)

| run_name | 최종 step | 계획 total | 완료 | ema 파일 | live 파일 | 크기(ema) | mtime(ema) | step 파일수 | run dir 합계 |
|---|---|---|---|---|---|---|---|---|---|
| `gc8_paper_34` | 45,540 | 9,407 | 판정불가 (로컬 데이터 기준 plan 9,407 ≠ 최종 45,540) | `step_45540` | `step_45540.live` | 38.0 MB | 08-08 06:48 | 1 | 76.1 MB |
| `gc8_repro_s0` | 9,400 | 9,407 | 완주 | `step_9400` | `step_9400.live` | 38.0 MB | 08-04 07:53 | 20 | 1.5 GB |
| `gc8_repro_s1` | 9,400 | 9,407 | 완주 | `step_9400` | `step_9400.live` | 38.0 MB | 08-04 12:14 | 20 | 1.5 GB |
| `gc8_repro_s2` | 9,400 | 9,407 | 완주 | `step_9400` | `step_9400.live` | 38.0 MB | 08-04 12:32 | 20 | 1.5 GB |
| `gc8_v5_add_b0_s0` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 38.0 MB | 07-29 19:38 | 4 | 266.3 MB |
| `gc8_v5_add_b0_s1` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 38.0 MB | 07-30 18:57 | 32 | 2.4 GB |
| `gc8_v5_add_b0_s2` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 38.0 MB | 07-30 19:01 | 32 | 2.4 GB |
| `gc8_v5_add_bm4_s0` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 38.0 MB | 07-29 19:23 | 3 | 190.3 MB |
| `gc8_v5_catnd_s0` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 40.0 MB | 07-29 19:26 | 3 | 200.3 MB |
| `gc8_v5_catnd_s1` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 40.0 MB | 07-29 19:31 | 3 | 200.3 MB |
| `gc8_v5_catnd_s2` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 40.0 MB | 07-30 09:24 | 32 | 2.5 GB |
| `gc8_v5_catnd_s3` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 40.0 MB | 07-30 09:11 | 32 | 2.5 GB |
| `gc8_v5_catnd_s4` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 40.0 MB | 07-30 13:46 | 32 | 2.5 GB |
| `gc8_v6_catnd_noOH_s0` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 40.0 MB | 07-30 19:14 | 32 | 2.5 GB |
| `gc8_v6_catnd_noOH_s1` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 40.0 MB | 07-31 06:49 | 32 | 2.5 GB |
| `gc8_v6_catnd_noOH_s2` | 60,192 | 60,208 | 완주 | `step_60192` | `step_60192.live` | 40.0 MB | 07-31 10:24 | 32 | 2.5 GB |

| run_name | epochs | lr | kl_coef | logvar_bias_init | kl_balance | ema_rate | L_cycles | halt_max_steps | seed | data_paths |
|---|---|---|---|---|---|---|---|---|---|---|
| `gc8_paper_34` | 5000 | 0.0001 | 0.5 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/graph-coloring-8` |
| `gc8_repro_s0` | 5000 | 0.0001 | 0.5 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/graph-coloring-8` |
| `gc8_repro_s1` | 5000 | 0.0001 | 0.5 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 1 | `data/graph-coloring-8` |
| `gc8_repro_s2` | 5000 | 0.0001 | 0.5 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 2 | `data/graph-coloring-8` |
| `gc8_v5_add_b0_s0` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/graph-coloring-8` |
| `gc8_v5_add_b0_s1` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 1 | `data/graph-coloring-8` |
| `gc8_v5_add_b0_s2` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 2 | `data/graph-coloring-8` |
| `gc8_v5_add_bm4_s0` | 32000 | 0.0001 | 0.5 | -4.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/graph-coloring-8` |
| `gc8_v5_catnd_s0` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/graph-coloring-8` |
| `gc8_v5_catnd_s1` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 1 | `data/graph-coloring-8` |
| `gc8_v5_catnd_s2` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 2 | `data/graph-coloring-8` |
| `gc8_v5_catnd_s3` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 3 | `data/graph-coloring-8` |
| `gc8_v5_catnd_s4` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 4 | `data/graph-coloring-8` |
| `gc8_v6_catnd_noOH_s0` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/graph-coloring-8` |
| `gc8_v6_catnd_noOH_s1` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 1 | `data/graph-coloring-8` |
| `gc8_v6_catnd_noOH_s2` | 32000 | 0.0001 | 0.5 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 2 | `data/graph-coloring-8` |

### `Nqueens-10-ACT-torch` — task `nq10` (23건)

| run_name | 최종 step | 계획 total | 완료 | ema 파일 | live 파일 | 크기(ema) | mtime(ema) | step 파일수 | run dir 합계 |
|---|---|---|---|---|---|---|---|---|---|
| `nq10_A0_beta10x_s0` | 19,586 | 19,592 | 완주 | `step_19586` | `step_19586.live` | 38.0 MB | 08-08 07:24 | 7 | 532.5 MB |
| `nq10_A0_lr3e4` | 138,534 | 138,548 | 완주 | `step_138534` | `step_138534.live` | 38.0 MB | 08-05 23:53 | 33 | 2.5 GB |
| `nq10_A0_ndlvb_s0` | 19,586 | 19,592 | 완주 | `step_19586` | `step_19586.live` | 38.0 MB | 08-08 07:29 | 7 | 532.5 MB |
| `nq10_A0_ndlvbh_s0` | 19,586 | 19,592 | 완주 | `step_19586` | `step_19586.live` | 38.0 MB | 08-08 07:28 | 7 | 532.5 MB |
| `nq10_A0_sigdiag_ctrl_s0` | 19,586 | 19,592 | 완주 | `step_19586` | `step_19586.live` | 38.0 MB | 08-08 07:24 | 7 | 532.5 MB |
| `nq10_A3p_ii_a7525_lr2e4` | 96,554 | 138,548 | 중단 (69.7%) | `step_96554` | `step_96554.live` | 64.0 MB | 08-06 10:01 | 23 | 2.9 GB |
| `nq10_final_add_s0` | 58,772 | 58,778 | 완주 | `step_58772` | `step_58772.live` | 38.0 MB | 08-02 16:10 | 14 | 1.0 GB |
| `nq10_final_add_s1` | 58,772 | 58,778 | 완주 | `step_58772` | `step_58772.live` | 38.0 MB | 08-02 16:25 | 14 | 1.0 GB |
| `nq10_final_add_s2` | 58,772 | 58,778 | 완주 | `step_58772` | `step_58772.live` | 38.0 MB | 08-02 16:50 | 14 | 1.0 GB |
| `nq10_final_catnd_s0` | 58,772 | 58,778 | 완주 | `step_58772` | `step_58772.live` | 40.0 MB | 08-02 16:22 | 14 | 1.1 GB |
| `nq10_final_catnd_s1` | 58,772 | 58,778 | 완주 | `step_58772` | `step_58772.live` | 40.0 MB | 08-02 16:16 | 14 | 1.1 GB |
| `nq10_final_catnd_s2` | 58,772 | 58,778 | 완주 | `step_58772` | `step_58772.live` | 40.0 MB | 08-02 16:36 | 14 | 1.1 GB |
| `nq10_repro_s0` | 138,534 | 138,548 | 완주 | `step_138534` | `step_138534.live` | 38.0 MB | 08-03 20:14 | 33 | 2.5 GB |
| `nq10_repro_s1` | 138,534 | 138,548 | 완주 | `step_138534` | `step_138534.live` | 38.0 MB | 08-03 20:12 | 33 | 2.5 GB |
| `nq10_repro_s2` | 16,792 | 138,548 | 중단 (12.1%) | `step_16792` | `step_16792.live` | 38.0 MB | 08-04 07:25 | 4 | 304.3 MB |
| `nq10_repro_s2_crashed0804` | 58,772 | 138,548 | 중단 (42.4%) | `step_58772` | `step_58772.live` | 38.0 MB | 08-04 05:19 | 14 | 1.0 GB |
| `nq10_v5_add_bm4_s0` | 60,522 | 176,334 | 중단 (34.3%) | `step_60522` | `step_60522.live` | 38.0 MB | 07-30 16:37 | 42 | 3.1 GB |
| `nq10_v5_catnd_s0` | 60,522 | 176,334 | 중단 (34.3%) | `step_60522` | `step_60522.live` | 40.0 MB | 07-30 16:39 | 42 | 3.3 GB |
| `nq10_v5_catnd_s1` | 60,522 | 176,334 | 중단 (34.3%) | `step_60522` | `step_60522.live` | 40.0 MB | 07-30 16:59 | 42 | 3.3 GB |
| `nq10_v5_catnd_s2` | 60,522 | 176,334 | 중단 (34.3%) | `step_60522` | `step_60522.live` | 40.0 MB | 07-30 16:47 | 42 | 3.3 GB |
| `nq10_v6_add_b0_s3` | 60,522 | 176,334 | 중단 (34.3%) | `step_60522` | `step_60522.live` | 38.0 MB | 07-31 02:35 | 42 | 3.1 GB |
| `nq10_v7_catnd_e2010_s3` | 96,547 | 281,295 | 중단 (34.3%) | `step_96547` | `step_96547.live` | 40.0 MB | 07-31 22:22 | 67 | 5.2 GB |
| `nq10_v7_catnd_e2010_s4` | 96,547 | 281,295 | 중단 (34.3%) | `step_96547` | `step_96547.live` | 40.0 MB | 07-31 21:56 | 67 | 5.2 GB |

| run_name | epochs | lr | kl_coef | logvar_bias_init | kl_balance | ema_rate | L_cycles | halt_max_steps | seed | data_paths |
|---|---|---|---|---|---|---|---|---|---|---|
| `nq10_A0_beta10x_s0` | 140 | 0.0001 | 0.0045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_A0_lr3e4` | 990 | 0.0003 | 0.045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_A0_ndlvb_s0` | 140 | 0.0001 | 0.045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_A0_ndlvbh_s0` | 140 | 0.0001 | 0.045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_A0_sigdiag_ctrl_s0` | 140 | 0.0001 | 0.045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_A3p_ii_a7525_lr2e4` | 990 | 0.0002 | 0.045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_final_add_s0` | 420 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_final_add_s1` | 420 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 1 | `data/nqueens-10` |
| `nq10_final_add_s2` | 420 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 2 | `data/nqueens-10` |
| `nq10_final_catnd_s0` | 420 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_final_catnd_s1` | 420 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 1 | `data/nqueens-10` |
| `nq10_final_catnd_s2` | 420 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 2 | `data/nqueens-10` |
| `nq10_repro_s0` | 990 | 0.0001 | 0.045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_repro_s1` | 990 | 0.0001 | 0.045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 1 | `data/nqueens-10` |
| `nq10_repro_s2` | 990 | 0.0001 | 0.045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 2 | `data/nqueens-10` |
| `nq10_repro_s2_crashed0804` | 990 | 0.0001 | 0.045 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 2 | `data/nqueens-10` |
| `nq10_v5_add_bm4_s0` | 1260 | 0.0001 | 0.045 | -4.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_v5_catnd_s0` | 1260 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-10` |
| `nq10_v5_catnd_s1` | 1260 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 1 | `data/nqueens-10` |
| `nq10_v5_catnd_s2` | 1260 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 2 | `data/nqueens-10` |
| `nq10_v6_add_b0_s3` | 1260 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 3 | `data/nqueens-10` |
| `nq10_v7_catnd_e2010_s3` | 2010 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 3 | `data/nqueens-10` |
| `nq10_v7_catnd_e2010_s4` | 2010 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 4 | `data/nqueens-10` |

### `Nqueens-8-ACT-torch` — task `nq8` (17건)

| run_name | 최종 step | 계획 total | 완료 | ema 파일 | live 파일 | 크기(ema) | mtime(ema) | step 파일수 | run dir 합계 |
|---|---|---|---|---|---|---|---|---|---|
| `fp_a3p_nq8_s0` | 67,428 | 67,443 | 완주 | `step_67428` | `step_67428.live` | 64.0 MB | 08-03 01:56 | 2 | 256.1 MB |
| `gram_nq8_mean_nh512` | 17,085 | 28,101 | 중단 (60.8%) | `step_17085` | **없음** | 38.0 MB | 07-23 05:05 | 15 | 570.5 MB |
| `nq8_A0_e3000_s0` | 28,095 | 28,101 | 완주 | `step_28095` | `step_28095.live` | 38.0 MB | 08-04 12:41 | 15 | 1.1 GB |
| `nq8_A0_e3000_s1` | 28,095 | 28,101 | 완주 | `step_28095` | `step_28095.live` | 38.0 MB | 08-04 16:51 | 15 | 1.1 GB |
| `nq8_A0_e3000_w2000` | 9,365 | 28,101 | 중단 (33.3%) | `step_9365` | `step_9365.live` | 38.0 MB | 08-04 16:57 | 5 | 380.4 MB |
| `nq8_A3p_e3000_s0` | 28,095 | 28,101 | 완주 | `step_28095` | `step_28095.live` | 64.0 MB | 08-04 12:46 | 15 | 1.9 GB |
| `nq8_A3p_fx_a5050_s0` | 28,095 | 28,101 | 완주 | `step_28095` | `step_28095.live` | 64.0 MB | 08-04 16:51 | 15 | 1.9 GB |
| `nq8_A3p_fx_a7550_s0` | 28,095 | 28,101 | 완주 | `step_28095` | `step_28095.live` | 64.0 MB | 08-04 16:56 | 15 | 1.9 GB |
| `nq8_A3p_ii_a7550_s0` | 28,095 | 28,101 | 완주 | `step_28095` | `step_28095.live` | 64.0 MB | 08-04 16:58 | 15 | 1.9 GB |
| `nq8_final_add_s0` | 67,428 | 67,443 | 완주 | `step_67428` | `step_67428.live` | 38.0 MB | 08-02 13:18 | 36 | 2.7 GB |
| `nq8_final_add_s1` | 67,428 | 67,443 | 완주 | `step_67428` | `step_67428.live` | 38.0 MB | 08-02 13:17 | 36 | 2.7 GB |
| `nq8_on_b500_s1` | 34,170 | 56,203 | 중단 (60.8%) | `step_34170` | `step_34170.live` | 38.0 MB | 07-28 22:40 | 3 | 228.2 MB |
| `nq8_v2_sum007_add_s0` | 39,865 | 65,570 | 중단 (60.8%) | `step_39865` | `step_39865.live` | 38.0 MB | 07-29 10:50 | 4 | 266.3 MB |
| `nq8_v2_sum007_catnd_s0` | 39,865 | 65,570 | 중단 (60.8%) | `step_39865` | `step_39865.live` | 40.0 MB | 07-29 10:34 | 3 | 200.2 MB |
| `nq8_verify` | — | 28,101 | step 파일 0 | **없음** | **없음** | — | — | 0 | 29.7 KB |
| `trm_nq8_base_s0` | — | 28,101 | step 파일 0 | **없음** | **없음** | — | — | 0 | 19.0 KB |
| `trm_nq8_base_s0_crashed0804` | 11,238 | 28,101 | 중단 (40.0%) | `step_11238` | `step_11238.live` | 26.0 MB | 08-04 16:29 | 6 | 312.3 MB |

| run_name | epochs | lr | kl_coef | logvar_bias_init | kl_balance | ema_rate | L_cycles | halt_max_steps | seed | data_paths |
|---|---|---|---|---|---|---|---|---|---|---|
| `fp_a3p_nq8_s0` | 7200 | 0.0001 | 0.07 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `gram_nq8_mean_nh512` | 3000 | 0.0001 | 0.07 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_A0_e3000_s0` | 3000 | 0.0001 | 0.07 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_A0_e3000_s1` | 3000 | 0.0001 | 0.07 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 1 | `data/nqueens-8` |
| `nq8_A0_e3000_w2000` | 3000 | 0.0001 | 0.07 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_A3p_e3000_s0` | 3000 | 0.0001 | 0.07 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_A3p_fx_a5050_s0` | 3000 | 0.0001 | 0.07 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_A3p_fx_a7550_s0` | 3000 | 0.0001 | 0.07 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_A3p_ii_a7550_s0` | 3000 | 0.0001 | 0.07 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_final_add_s0` | 7200 | 0.0001 | 0.07 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_final_add_s1` | 7200 | 0.0001 | 0.07 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 1 | `data/nqueens-8` |
| `nq8_on_b500_s1` | 6000 | 0.0001 | 5.0 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 1 | `data/nqueens-8` |
| `nq8_v2_sum007_add_s0` | 7000 | 0.0001 | 0.07 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_v2_sum007_catnd_s0` | 7000 | 0.0001 | 0.07 | 0.0 | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `nq8_verify` | 3000 | 0.0001 | 0.07 | (키 없음) | 0.8 | 0.9999 | 4 | 16 | 0 | `data/nqueens-8` |
| `trm_nq8_base_s0` | 3000 | 0.0001 | (키 없음) | (키 없음) | (키 없음) | 0.9999 | 6 | 16 | 0 | `data/nqueens-8` |
| `trm_nq8_base_s0_crashed0804` | 3000 | 0.0001 | (키 없음) | (키 없음) | (키 없음) | 0.9999 | 6 | 16 | 0 | `data/nqueens-8` |

### `Sudoku-extreme-1k-aug-1000-ACT-torch` — task `sudoku` (1건)

| run_name | 최종 step | 계획 total | 완료 | ema 파일 | live 파일 | 크기(ema) | mtime(ema) | step 파일수 | run dir 합계 |
|---|---|---|---|---|---|---|---|---|---|
| `gram_mlp_t_sudoku` | 6,510 | 65,104 | 중단 (10.0%) | `step_6510` | **없음** | 55.2 MB | 07-16 07:56 | 1 | 55.2 MB |

| run_name | epochs | lr | kl_coef | logvar_bias_init | kl_balance | ema_rate | L_cycles | halt_max_steps | seed | data_paths |
|---|---|---|---|---|---|---|---|---|---|---|
| `gram_mlp_t_sudoku` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000` |

### `Sudoku-extreme-1k-aug-1000-test10k-ACT-torch` — task `sudoku` (10건)

| run_name | 최종 step | 계획 total | 완료 | ema 파일 | live 파일 | 크기(ema) | mtime(ema) | step 파일수 | run dir 합계 |
|---|---|---|---|---|---|---|---|---|---|
| `gram_sudoku_b005` | 67,704 | 130,208 | 중단 (52.0%) | `step_67704` | **없음** | 55.2 MB | 07-19 22:43 | 2 | 110.4 MB |
| `gram_sudoku_halt_only` | 65,100 | 65,104 | 완주 | `step_65100` | **없음** | 55.2 MB | 07-20 13:14 | 1 | 55.2 MB |
| `gram_sudoku_lmean_ctrl` | 2,604 | 65,104 | 중단 (4.0%) | `step_2604` | **없음** | 55.2 MB | 07-21 07:01 | 1 | 55.2 MB |
| `gram_sudoku_lmean_v2` | 65,100 | 65,104 | 완주 | `step_65100` | **없음** | 55.2 MB | 07-21 12:45 | 2 | 110.4 MB |
| `gram_sudoku_long` | 130,200 | 130,208 | 완주 | `step_130200` | **없음** | 55.2 MB | 07-20 05:50 | 2 | 110.4 MB |
| `gram_sudoku_lsum` | 65,100 | 65,104 | 완주 | `step_65100` | **없음** | 55.2 MB | 07-16 15:40 | 2 | 110.4 MB |
| `gram_sudoku_lsum_exp` | 65,100 | 65,104 | 완주 | `step_65100` | **없음** | 55.2 MB | 07-18 16:02 | 2 | 110.4 MB |
| `gram_sudoku_paper` | 65,100 | 65,104 | 완주 | `step_65100` | **없음** | 55.2 MB | 07-20 15:20 | 1 | 55.2 MB |
| `gram_sudoku_sigma_fix` | 10,416 | 65,104 | 중단 (16.0%) | `step_10416` | **없음** | 55.2 MB | 07-20 18:02 | 2 | 110.4 MB |
| `gram_sudoku_sigmabound` | 65,100 | 65,104 | 완주 | `step_65100` | **없음** | 55.2 MB | 07-20 23:45 | 1 | 55.2 MB |

| run_name | epochs | lr | kl_coef | logvar_bias_init | kl_balance | ema_rate | L_cycles | halt_max_steps | seed | data_paths |
|---|---|---|---|---|---|---|---|---|---|---|
| `gram_sudoku_b005` | 100000 | 0.0001 | 0.05 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |
| `gram_sudoku_halt_only` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |
| `gram_sudoku_lmean_ctrl` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |
| `gram_sudoku_lmean_v2` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |
| `gram_sudoku_long` | 100000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |
| `gram_sudoku_lsum` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |
| `gram_sudoku_lsum_exp` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |
| `gram_sudoku_paper` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |
| `gram_sudoku_sigma_fix` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |
| `gram_sudoku_sigmabound` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k` |

### `Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch` — task `sudoku` (3건)

| run_name | 최종 step | 계획 total | 완료 | ema 파일 | live 파일 | 크기(ema) | mtime(ema) | step 파일수 | run dir 합계 |
|---|---|---|---|---|---|---|---|---|---|
| `sudoku_paper_34` | 65,100 | 65,104 | 완주 | `step_65100` | `step_65100.live` | 55.2 MB | 08-08 11:10 | 1 | 110.4 MB |
| `sudoku_paper_34_eval50` | 65,100 | 65,104 | 해당없음 (링크 래퍼) | `step_65100` (symlink) | `step_65100.live` (symlink) | 55.2 MB | 08-08 11:10 | 1 | 110.4 MB |
| `sudoku_paper_34_origdata` | 65,100 | 65,104 | 해당없음 (링크 래퍼) | `step_65100` (symlink) | `step_65100.live` (symlink) | 55.2 MB | 08-08 11:10 | 1 | 110.4 MB |

| run_name | epochs | lr | kl_coef | logvar_bias_init | kl_balance | ema_rate | L_cycles | halt_max_steps | seed | data_paths |
|---|---|---|---|---|---|---|---|---|---|---|
| `sudoku_paper_34` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000-test10k-v2` |
| `sudoku_paper_34_eval50` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000` |
| `sudoku_paper_34_origdata` | 50000 | 0.0001 | 0.1 | (키 없음) | 0.8 | 0.9999 | 6 | 16 | 0 | `data/sudoku-extreme-1k-aug-1000` |

## ① run 목록 — fprm

### `fprm-nqueens10` — task `fprm_nq10` (6건)

| run_name | 최종 step | 계획 total | 완료 | ema 파일 | live 파일 | 크기(ema) | mtime(ema) | step 파일수 | run dir 합계 |
|---|---|---|---|---|---|---|---|---|---|
| `nq10_fprm_S0p_nonoise_s0` | 138,534 | 138,548 | 완주 | `step_138534` | `step_138534_train_state.pt` | 52.0 MB | 08-09 17:14 | 33 | 5.9 GB |
| `nq10_fprm_S2_conv3_s0` | 138,534 | 138,548 | 완주 | `step_138534` | `step_138534_train_state.pt` | 124.1 MB | 08-09 08:33 | 33 | 14.0 GB |
| `nq10_fprm_S2_convnone_s0` | 138,534 | 138,548 | 완주 | `step_138534` | `step_138534_train_state.pt` | 124.0 MB | 08-10 04:44 | 33 | 14.0 GB |
| `nq10_fprm_S3_conv16_s0` | 138,534 | 138,548 | 완주 | `step_138534` | `step_138534_train_state.pt` | 124.1 MB | 08-09 09:19 | 33 | 14.0 GB |
| `nq10_fprm_fixed_s0` | 83,960 | 138,548 | 중단 (60.6%) | `step_83960` | `step_83960_train_state.pt` | 52.0 MB | 08-08 20:38 | 20 | 3.6 GB |
| `nq10_fprm_stock_s0` | — | 138,548 | step 파일 0 | **없음** | **없음** | — | — | 0 | 1.9 KB |

| run_name | epochs | lr | kl_coef | logvar_bias_init | kl_balance | ema_rate | L_cycles | halt_max_steps | seed | data_paths |
|---|---|---|---|---|---|---|---|---|---|---|
| `nq10_fprm_S0p_nonoise_s0` | 990 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.999 | 6 | (키 없음) | 0 | `data/nqueens-10` |
| `nq10_fprm_S2_conv3_s0` | 990 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.999 | 6 | (키 없음) | 0 | `data/nqueens-10` |
| `nq10_fprm_S2_convnone_s0` | 990 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.999 | 6 | (키 없음) | 0 | `data/nqueens-10` |
| `nq10_fprm_S3_conv16_s0` | 990 | 0.0001 | 0.045 | 0.0 | 0.8 | 0.999 | 6 | (키 없음) | 0 | `data/nqueens-10` |
| `nq10_fprm_fixed_s0` | 990 | 0.0001 | (키 없음) | (키 없음) | (키 없음) | 0.999 | 6 | (키 없음) | 0 | `data/nqueens-10` |
| `nq10_fprm_stock_s0` | 990 | 0.0001 | (키 없음) | (키 없음) | (키 없음) | 0.999 | 6 | (키 없음) | 0 | `data/nqueens-10` |

### `fprm-sudoku` — task `fprm_sudoku` (1건)

| run_name | 최종 step | 계획 total | 완료 | ema 파일 | live 파일 | 크기(ema) | mtime(ema) | step 파일수 | run dir 합계 |
|---|---|---|---|---|---|---|---|---|---|
| `sudoku_fprm_D0_s0` | 6,510 | 78,125 | 중단 (8.3%) | `step_6510` | **없음** | 5.7 MB | 08-10 13:28 | 1 | 5.8 MB |

| run_name | epochs | lr | kl_coef | logvar_bias_init | kl_balance | ema_rate | L_cycles | halt_max_steps | seed | data_paths |
|---|---|---|---|---|---|---|---|---|---|---|
| `sudoku_fprm_D0_s0` | 60000 | 0.001 | (키 없음) | (키 없음) | (키 없음) | 0.999 | 6 | (키 없음) | 0 | `data/sudoku-extreme-1k-aug-1000` |

---

## ② run 목적 (근거 있는 것만. 없으면 `불명`)

| run_name | 목적 | 근거 |
|---|---|---|
| `nq10_fprm_S0p_nonoise_s0` | 불명 — 학습 목적 문장 없음. 결과 리포트에서 fprm_s0p 축으로 사용 (reports/RESULT_residual_probe_50.md) | — |
| `nq10_fprm_S2_conv3_s0` | injection_mode=converged 고정, max_injections 3 셀 | `fprm/reports/RESULT_nq10_fprm_S2_S3_injection_axis.md` |
| `nq10_fprm_S2_convnone_s0` | 불명 — 리포트 없음. config/arch/fprm_nqueens10_S2_convnone.yaml 은 아직 untracked | — |
| `nq10_fprm_S3_conv16_s0` | injection_mode=converged 고정, max_injections 16 셀 | `fprm/reports/RESULT_nq10_fprm_S2_S3_injection_axis.md` |
| `nq10_fprm_fixed_s0` | residual_scale: input-independent → fixed 한 축 | `fprm/reports/RESULT_nq10_fprm_fixed_s0.md` |
| `nq10_fprm_stock_s0` | 29번 baseline (stepsize_decay_train 0.9). 이 박스에는 all_config.yaml 만 있고 step 파일 없음 | `fprm/reports/RESULT_nq10_fprm_fixed_s0.md` |
| `sudoku_fprm_D0_s0` | 불명 — 리포트 없음 | — |
| `gc10_paper_34` | 34번에서 전송받은 GC-10 논문 재현 checkpoint | `reports/TRANSFER_34_to_50_gc10paper.md` |
| `gc8_paper_34` | 34번에서 전송받은 GC-8 논문 재현 checkpoint — 50번 ε probe 가 쓰던 gc8_repro_s0(붕괴 셀) 대체용 | `reports/TRANSFER_34_to_50_gc8paper.md` |
| `gc8_repro_s0` | GC-8 논문 예산 5K 재현 (Table 7 대조). s0 은 사수측 launch(root 소유) | `REPORT_final_pair_50.md §5` |
| `gc8_repro_s1` | GC-8 논문 예산 5K 재현 (Table 7 대조). 50번 launch | `REPORT_final_pair_50.md §5` |
| `gc8_repro_s2` | GC-8 논문 예산 5K 재현 (Table 7 대조). 50번 launch | `REPORT_final_pair_50.md §5` |
| `gc8_v5_add_b0_s0` | GC-8 1차 캠페인 4-run 중 add / logvar_bias_init 0.0 셀 | `reports/archive/REPORT_gc8_50.md §2` |
| `gc8_v5_add_b0_s1` | 불명 — in-loop 스냅샷에만 등장 (REPORT_gc8_50 4-run 구성에 없음) | — |
| `gc8_v5_add_b0_s2` | 불명 — in-loop 스냅샷에만 등장 | — |
| `gc8_v5_add_bm4_s0` | GC-8 1차 캠페인 4-run 중 add / logvar_bias_init −4.0 셀 | `reports/archive/REPORT_gc8_50.md §2` |
| `gc8_v5_catnd_s0` | GC-8 1차 캠페인 4-run 중 concat/nodecay_y_proj=true 셀 | `reports/archive/REPORT_gc8_50.md §2` |
| `gc8_v5_catnd_s1` | GC-8 1차 캠페인 4-run 중 concat 셀 seed 1 | `reports/archive/REPORT_gc8_50.md §2` |
| `gc8_v5_catnd_s2` | 불명 — in-loop 스냅샷에만 등장 | — |
| `gc8_v5_catnd_s3` | 불명 — in-loop 스냅샷에만 등장 | — |
| `gc8_v5_catnd_s4` | 불명 — in-loop 스냅샷에만 등장 | — |
| `gc8_v6_catnd_noOH_s0` | 불명 — in-loop 스냅샷에만 등장 | — |
| `gc8_v6_catnd_noOH_s1` | 불명 — in-loop 스냅샷에만 등장 | — |
| `gc8_v6_catnd_noOH_s2` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq10_A0_beta10x_s0` | 불명 — 리포트 없음. nq10_repro_s0 대비 config 차이: kl_coef 0.045→0.0045, epochs 990→140. 08-08 04:51 GPU4,5 launch | — |
| `nq10_A0_lr3e4` | 불명 — 리포트 없음. config 차이: lr·puzzle_emb_lr 1e-4→3e-4 | — |
| `nq10_A0_ndlvb_s0` | 불명 — 리포트 없음. config 차이: nodecay_logvar_bias=true, epochs 140. 08-08 04:51 GPU0,1 | — |
| `nq10_A0_ndlvbh_s0` | 불명 — 리포트 없음. config 차이: nodecay_logvar_bias=true + nodecay_logvar_head=true, epochs 140 | — |
| `nq10_A0_sigdiag_ctrl_s0` | 불명 — 리포트 없음. config 차이: 두 nodecay 플래그 모두 false, epochs 140. 08-08 04:51 GPU6,7 | — |
| `nq10_A3p_ii_a7525_lr2e4` | 불명 — 리포트 없음. config 차이: A3′ ii(alpha_1/2=0.75/0.25), lr 2e-4, nodecay_marked=true | — |
| `nq10_final_add_s0` | 구 A0 기준 3-seed (420ep) — §2 논문예산 run 으로 대체됨 | `REPORT_final_pair_50.md §2a` |
| `nq10_final_add_s1` | 구 A0 기준 3-seed (420ep) — §2 로 대체됨 | `REPORT_final_pair_50.md §2a` |
| `nq10_final_add_s2` | 구 A0 기준 3-seed (420ep) — §2 로 대체됨 | `REPORT_final_pair_50.md §2a` |
| `nq10_final_catnd_s0` | y_inject=concat(no-decay) 탐색 arm — 최종 설정 미채택, 논문 대비 제외 | `REPORT_final_pair_50.md §2b` |
| `nq10_final_catnd_s1` | y_inject=concat(no-decay) 탐색 arm — 논문 대비 제외 | `REPORT_final_pair_50.md §2b` |
| `nq10_final_catnd_s2` | y_inject=concat(no-decay) 탐색 arm — 논문 대비 제외 | `REPORT_final_pair_50.md §2b` |
| `nq10_repro_s0` | 논문 예산 재현(990ep, 138,534) — A0 기준값 (사수 지시 2026-08-04) | `REPORT_final_pair_50.md §2` |
| `nq10_repro_s1` | 논문 예산 재현(990ep) — A0 기준값 seed 1 | `REPORT_final_pair_50.md §2` |
| `nq10_repro_s2` | 논문 예산 재현(990ep) — A0 기준값 seed 2 | `REPORT_final_pair_50.md §2` |
| `nq10_repro_s2_crashed0804` | 08-04 crash 분 보존본 | `REPORT_final_pair_50.md / snapshot_lecun_20260805 항목명` |
| `nq10_v5_add_bm4_s0` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq10_v5_catnd_s0` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq10_v5_catnd_s1` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq10_v5_catnd_s2` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq10_v6_add_b0_s3` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq10_v7_catnd_e2010_s3` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq10_v7_catnd_e2010_s4` | 불명 — in-loop 스냅샷에만 등장 | — |
| `fp_a3p_nq8_s0` | G3 게이트 A3′ nq8 live 격자 셀 (mu=1 sg=0.5) | `REPORT_final_pair_50.md §5b / reports/g3/` |
| `gram_nq8_mean_nh512` | 진단기 — 백본 결백 확인(posterior valid 99.87% vs prior 0%) | `reports/archive/REPORT_diag.md` |
| `nq8_A0_e3000_s0` | 불명 — 목적 문장 없음. reports/RESULT_eps_probe_50.md 에서 측정 대상으로만 등장 | — |
| `nq8_A0_e3000_s1` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq8_A0_e3000_w2000` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq8_A3p_e3000_s0` | 불명 — 리포트 없음 | — |
| `nq8_A3p_fx_a5050_s0` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq8_A3p_fx_a7550_s0` | 불명 — in-loop 스냅샷에만 등장 | — |
| `nq8_A3p_ii_a7550_s0` | 불명 — 리포트 없음 | — |
| `nq8_final_add_s0` | pair 확장 최종 라운드 8×8 add arm | `REPORT_final_pair_50.md §1` |
| `nq8_final_add_s1` | pair 확장 최종 라운드 8×8 add arm seed 1 | `REPORT_final_pair_50.md §1` |
| `nq8_on_b500_s1` | 불명 — REPORT_INDEX.md 에 'harness 완료, 리포트 없음' (b500_*.json 16건) | — |
| `nq8_v2_sum007_add_s0` | 불명 — REPORT_INDEX.md 에 'harness 완료, 리포트 없음' (v2_*.json 24건) | — |
| `nq8_v2_sum007_catnd_s0` | 불명 — REPORT_INDEX.md 에 'harness 완료, 리포트 없음' (v2_*.json 24건) | — |
| `nq8_verify` | 저장된 config dump 확인용 (step 파일 없음) | `reports/archive/REPORT_onehot.md` |
| `trm_nq8_base_s0` | 불명 — snapshot 에 .launch 항목만. step 파일 없음(trm.py/losses.py/all_config.yaml 뿐) | — |
| `trm_nq8_base_s0_crashed0804` | 불명 — 08-04 crash 보존본, 리포트 없음 | — |
| `gram_mlp_t_sudoku` | 불명 — RESULT_sudoku_gc_repro_34.md §3a 표에 data/epochs/kl 만 | — |
| `gram_sudoku_b005` | 불명 — §3a 표에만 (kl_coef 0.05, 100000ep) | — |
| `gram_sudoku_halt_only` | 불명 — 목적 문장 없음. REPORT_yemb.md 에서 측정 대상으로 등장 | — |
| `gram_sudoku_lmean_ctrl` | 불명 — §3a 표에만 | — |
| `gram_sudoku_lmean_v2` | KL reduction L-mean 을 명시 키로 고쳐 재완주 | `reports/archive/REPORT_night2.md` |
| `gram_sudoku_long` | 불명 — §3a 표에만 (100000ep) | — |
| `gram_sudoku_lsum` | KL reduction L-sum 축 | `reports/archive/REPORT_night2.md / REPORT_diag2.md` |
| `gram_sudoku_lsum_exp` | 불명 — §3a 표에만 | — |
| `gram_sudoku_paper` | 불명 — §3a 표에만 | — |
| `gram_sudoku_sigma_fix` | 불명 — §3a 표에만 | — |
| `gram_sudoku_sigmabound` | 불명 — §3a 표에만 | — |
| `sudoku_paper_34` | 34번 Sudoku 논문 재현 정준 run (test10k-v2). 현행 기준 checkpoint | `reports/RESULT_sudoku_gc_repro_34.md` |
| `sudoku_paper_34_eval50` | ★학습 run 아님★ — sudoku_paper_34 의 step 파일을 가리키는 심볼릭 링크 래퍼. all_config.yaml 의 data_paths 만 sudoku-extreme-1k-aug-1000 로 교체 | `디렉터리 실측 + scripts/run_depth_util_50.sh 주석` |
| `sudoku_paper_34_origdata` | ★학습 run 아님★ — 같은 형태의 심볼릭 링크 래퍼 (data_paths 만 원본 데이터로 교체) | `디렉터리 실측` |

근거 있는 run **33건** / 불명 **45건**

---

## ③ results/ 아래 결과 파일

csv/json/npz **1116개**, 합계 **1.9 GB**. 심볼릭 링크는 제외했다
(`TinyRecursiveModels/results/figdata` → `fprm/results/figdata` 링크라 figdata 는 fprm 쪽으로만 1회 계수).

### fprm/results/figdata/ (= TinyRecursiveModels/results/figdata 심볼릭 링크 대상) — 66개, 1.9 GB

| 경로 | 크기 | 행수/shape | mtime(UTC) |
|---|---|---|---|
| `figdata/depth_util_sudoku.csv` | 339 B | rows=2 | 08-11 02:34 |
| `figdata/depth_util_sudoku.meta.json` | 1.8 KB | dict[29] | 08-11 02:40 |
| `figdata/gc10_solution_counts.json` | 656 B | dict[1] | 08-10 06:48 |
| `figdata/gc10_solution_counts.meta.json` | 1.8 KB | dict[25] | 08-11 02:40 |
| `figdata/gc8_solution_counts.json` | 803 B | dict[1] | 08-10 06:48 |
| `figdata/gc8_solution_counts.meta.json` | 1.8 KB | dict[25] | 08-11 02:40 |
| `figdata/nq10_perpuzzle.csv` | 83.1 KB | rows=6513 | 08-10 09:26 |
| `figdata/nq10_perpuzzle.meta.json` | 2.0 KB | dict[32] | 08-11 02:40 |
| `figdata/nq10_solution_counts.json` | 19.4 KB | dict[1] | 08-10 06:48 |
| `figdata/nq10_solution_counts.meta.json` | 1.7 KB | dict[25] | 08-11 02:40 |
| `figdata/residual_fprm_s0p.csv` | 62.0 KB | rows=929 | 08-11 02:33 |
| `figdata/residual_fprm_s0p.meta.json` | 3.7 KB | dict[47] | 08-11 02:40 |
| `figdata/residual_fprm_s2.csv` | 68.5 KB | rows=1000 | 08-11 02:34 |
| `figdata/residual_fprm_s2.meta.json` | 3.7 KB | dict[47] | 08-11 02:40 |
| `figdata/residual_gc8_sig0.csv` | 2.4 KB | rows=48 | 08-11 02:32 |
| `figdata/residual_gc8_sig0.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/residual_gc8_sig1.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/residual_gc8_sig1.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/residual_nq10_sig0.csv` | 2.4 KB | rows=48 | 08-11 02:32 |
| `figdata/residual_nq10_sig0.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/residual_nq10_sig1.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/residual_nq10_sig1.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/residual_nq8_sig0.csv` | 2.5 KB | rows=48 | 08-11 02:32 |
| `figdata/residual_nq8_sig0.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/residual_nq8_sig1.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/residual_nq8_sig1.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/residual_sudoku_sig0.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/residual_sudoku_sig0.meta.json` | 2.6 KB | dict[44] | 08-11 02:40 |
| `figdata/residual_sudoku_sig1.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/residual_sudoku_sig1.meta.json` | 2.6 KB | dict[44] | 08-11 02:40 |
| `figdata/token_content/residual_fprm_s0p.csv` | 62.0 KB | rows=929 | 08-11 02:33 |
| `figdata/token_content/residual_fprm_s2.csv` | 68.4 KB | rows=1000 | 08-11 02:34 |
| `figdata/token_content/residual_gc8_sig0.csv` | 2.4 KB | rows=48 | 08-11 02:32 |
| `figdata/token_content/residual_gc8_sig1.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/token_content/residual_nq10_sig0.csv` | 2.4 KB | rows=48 | 08-11 02:32 |
| `figdata/token_content/residual_nq10_sig1.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/token_content/residual_nq8_sig0.csv` | 2.5 KB | rows=48 | 08-11 02:32 |
| `figdata/token_content/residual_nq8_sig1.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/token_content/residual_sudoku_sig0.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/token_content/residual_sudoku_sig1.csv` | 2.3 KB | rows=48 | 08-11 02:32 |
| `figdata/traj_fprm_s0p.meta.json` | 3.7 KB | dict[47] | 08-11 02:40 |
| `figdata/traj_fprm_s0p.npz` | 535.9 MB | h(50, 93, 59392)<f2 loss(50, 93)<f4 loss_stablemax(50, 93)<f4 h_mean(50, 93, 512)<f4 t_index(93,)<i4 | 08-11 02:33 |
| `figdata/traj_fprm_s2.meta.json` | 3.7 KB | dict[47] | 08-11 02:40 |
| `figdata/traj_fprm_s2.npz` | 524.4 MB | h(50, 91, 59392)<f2 loss(50, 91)<f4 loss_stablemax(50, 91)<f4 h_mean(50, 91, 512)<f4 t_index(91,)<i4 | 08-11 02:34 |
| `figdata/traj_gc8_sig0.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/traj_gc8_sig0.npz` | 2.2 MB | h(1, 48, 22528)<f2 loss(1, 48)<f4 loss_stablemax(1, 48)<f4 h_mean(1, 48, 512)<f4 t_index(48,)<i4 | 08-11 02:32 |
| `figdata/traj_gc8_sig1.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/traj_gc8_sig1.npz` | 107.8 MB | h(50, 48, 22528)<f2 loss(50, 48)<f4 loss_stablemax(50, 48)<f4 h_mean(50, 48, 512)<f4 t_index(48,)<i4 | 08-11 02:32 |
| `figdata/traj_gram_nq10.meta.json` | 1.4 KB | dict[27] | 08-11 02:40 |
| `figdata/traj_gram_nq10.npz` | 4.7 MB | h(50, 48, 512)<f4 | 08-10 06:47 |
| `figdata/traj_nq10_sig0.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/traj_nq10_sig0.npz` | 5.5 MB | h(1, 48, 59392)<f2 loss(1, 48)<f4 loss_stablemax(1, 48)<f4 h_mean(1, 48, 512)<f4 t_index(48,)<i4 | 08-11 02:32 |
| `figdata/traj_nq10_sig1.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/traj_nq10_sig1.npz` | 276.6 MB | h(50, 48, 59392)<f2 loss(50, 48)<f4 loss_stablemax(50, 48)<f4 h_mean(50, 48, 512)<f4 t_index(48,)<i4 | 08-11 02:32 |
| `figdata/traj_nq8_sig0.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/traj_nq8_sig0.npz` | 3.8 MB | h(1, 48, 40960)<f2 loss(1, 48)<f4 loss_stablemax(1, 48)<f4 h_mean(1, 48, 512)<f4 t_index(48,)<i4 | 08-11 02:32 |
| `figdata/traj_nq8_sig1.meta.json` | 2.5 KB | dict[44] | 08-11 02:40 |
| `figdata/traj_nq8_sig1.npz` | 192.2 MB | h(50, 48, 40960)<f2 loss(50, 48)<f4 loss_stablemax(50, 48)<f4 h_mean(50, 48, 512)<f4 t_index(48,)<i4 | 08-11 02:32 |
| `figdata/traj_s0p_nq10.meta.json` | 1.5 KB | dict[27] | 08-11 02:40 |
| `figdata/traj_s0p_nq10.npz` | 9.4 MB | h(50, 96, 512)<f4 | 08-10 06:45 |
| `figdata/traj_s2_nq10.meta.json` | 1.5 KB | dict[27] | 08-11 02:40 |
| `figdata/traj_s2_nq10.npz` | 9.4 MB | h(50, 96, 512)<f4 | 08-10 06:45 |
| `figdata/traj_sudoku_sig0.meta.json` | 2.6 KB | dict[44] | 08-11 02:40 |
| `figdata/traj_sudoku_sig0.npz` | 4.6 MB | h(1, 48, 49664)<f2 loss(1, 48)<f4 loss_stablemax(1, 48)<f4 h_mean(1, 48, 512)<f4 t_index(48,)<i4 | 08-11 02:32 |
| `figdata/traj_sudoku_sig1.meta.json` | 2.6 KB | dict[44] | 08-11 02:40 |
| `figdata/traj_sudoku_sig1.npz` | 232.1 MB | h(50, 48, 49664)<f2 loss(50, 48)<f4 loss_stablemax(50, 48)<f4 h_mean(50, 48, 512)<f4 t_index(48,)<i4 | 08-11 02:32 |

### fprm/results/fprm_harness/ — 40개, 7.5 MB

| 경로 | 크기 | 행수/shape | mtime(UTC) |
|---|---|---|---|
| `fprm_harness/CN_ema.json` | 10.0 KB | dict[24] | 08-10 06:11 |
| `fprm_harness/CN_ema_shard0of4.json` | 247.6 KB | dict[25] | 08-10 06:10 |
| `fprm_harness/CN_ema_shard1of4.json` | 247.3 KB | dict[25] | 08-10 06:10 |
| `fprm_harness/CN_ema_shard2of4.json` | 247.5 KB | dict[25] | 08-10 06:11 |
| `fprm_harness/CN_ema_shard3of4.json` | 247.5 KB | dict[25] | 08-10 06:11 |
| `fprm_harness/CN_live.json` | 9.6 KB | dict[24] | 08-10 08:39 |
| `fprm_harness/CN_live_shard0of4.json` | 247.4 KB | dict[25] | 08-10 08:39 |
| `fprm_harness/CN_live_shard1of4.json` | 247.1 KB | dict[25] | 08-10 07:35 |
| `fprm_harness/CN_live_shard2of4.json` | 247.3 KB | dict[25] | 08-10 07:31 |
| `fprm_harness/CN_live_shard3of4.json` | 247.4 KB | dict[25] | 08-10 07:35 |
| `fprm_harness/S0p_ema.json` | 1.9 KB | dict[21] | 08-09 18:37 |
| `fprm_harness/S0p_ema_shard0of4.json` | 238.1 KB | dict[23] | 08-09 18:27 |
| `fprm_harness/S0p_ema_shard1of4.json` | 237.8 KB | dict[23] | 08-09 18:33 |
| `fprm_harness/S0p_ema_shard2of4.json` | 238.0 KB | dict[23] | 08-09 18:37 |
| `fprm_harness/S0p_ema_shard3of4.json` | 238.1 KB | dict[23] | 08-09 18:35 |
| `fprm_harness/S0p_live.json` | 2.0 KB | dict[21] | 08-10 00:54 |
| `fprm_harness/S0p_live_shard0of4.json` | 238.1 KB | dict[23] | 08-10 00:53 |
| `fprm_harness/S0p_live_shard1of4.json` | 237.8 KB | dict[23] | 08-10 00:52 |
| `fprm_harness/S0p_live_shard2of4.json` | 238.0 KB | dict[23] | 08-10 00:54 |
| `fprm_harness/S0p_live_shard3of4.json` | 238.0 KB | dict[23] | 08-10 00:51 |
| `fprm_harness/S2_ema.json` | 2.0 KB | dict[21] | 08-09 14:18 |
| `fprm_harness/S2_ema_shard0of4.json` | 238.1 KB | dict[23] | 08-09 14:15 |
| `fprm_harness/S2_ema_shard1of4.json` | 237.8 KB | dict[23] | 08-09 14:08 |
| `fprm_harness/S2_ema_shard2of4.json` | 238.0 KB | dict[23] | 08-09 14:18 |
| `fprm_harness/S2_ema_shard3of4.json` | 238.1 KB | dict[23] | 08-09 14:12 |
| `fprm_harness/S2_live.json` | 2.0 KB | dict[21] | 08-09 20:58 |
| `fprm_harness/S2_live_shard0of4.json` | 238.1 KB | dict[23] | 08-09 20:57 |
| `fprm_harness/S2_live_shard1of4.json` | 237.9 KB | dict[23] | 08-09 20:58 |
| `fprm_harness/S2_live_shard2of4.json` | 238.0 KB | dict[23] | 08-09 20:56 |
| `fprm_harness/S2_live_shard3of4.json` | 238.1 KB | dict[23] | 08-09 20:58 |
| `fprm_harness/S3_ema.json` | 2.0 KB | dict[21] | 08-09 17:42 |
| `fprm_harness/S3_ema_shard0of4.json` | 238.1 KB | dict[23] | 08-09 17:38 |
| `fprm_harness/S3_ema_shard1of4.json` | 237.8 KB | dict[23] | 08-09 17:42 |
| `fprm_harness/S3_ema_shard2of4.json` | 238.0 KB | dict[23] | 08-09 17:40 |
| `fprm_harness/S3_ema_shard3of4.json` | 238.1 KB | dict[23] | 08-09 17:40 |
| `fprm_harness/S3_live.json` | 2.0 KB | dict[21] | 08-10 00:20 |
| `fprm_harness/S3_live_shard0of4.json` | 238.1 KB | dict[23] | 08-10 00:18 |
| `fprm_harness/S3_live_shard1of4.json` | 237.8 KB | dict[23] | 08-10 00:20 |
| `fprm_harness/S3_live_shard2of4.json` | 238.0 KB | dict[23] | 08-10 00:19 |
| `fprm_harness/S3_live_shard3of4.json` | 238.0 KB | dict[23] | 08-10 00:19 |

### TinyRecursiveModels/results/ (figdata 제외) — 88개, 22.0 MB

| 경로 | 크기 | 행수/shape | mtime(UTC) |
|---|---|---|---|
| `a0_nq10_s0_final_ema_h0_halt32.json` | 946.9 KB | dict[11] | 08-05 08:51 |
| `a0_nq10_s0_final_ema_h0_halt64.json` | 946.9 KB | dict[11] | 08-05 11:56 |
| `eps_probe/csv/lambda_d_t.csv` | 62.9 KB | rows=1152 | 08-09 16:11 |
| `eps_probe/csv/sigma_n_t.csv` | 21.1 KB | rows=384 | 08-09 16:11 |
| `eps_probe/gc10p_lambda.json` | 12.7 KB | dict[18] | 08-09 16:10 |
| `eps_probe/gc10p_sigma.json` | 9.1 KB | dict[17] | 08-09 16:09 |
| `eps_probe/gc8_lambda.json` | 13.3 KB | dict[18] | 08-09 10:13 |
| `eps_probe/gc8_sigma.json` | 9.1 KB | dict[17] | 08-09 10:13 |
| `eps_probe/gc8p_lambda.json` | 12.6 KB | dict[18] | 08-09 11:34 |
| `eps_probe/gc8p_sigma.json` | 9.1 KB | dict[17] | 08-09 11:34 |
| `eps_probe/nq10_lambda.json` | 12.4 KB | dict[18] | 08-09 10:14 |
| `eps_probe/nq10_sigma.json` | 9.1 KB | dict[17] | 08-09 10:13 |
| `eps_probe/nq10_step138534_lambda.json` | 5.1 KB | dict[18] | 08-09 11:49 |
| `eps_probe/nq10_step41980_lambda.json` | 5.1 KB | dict[18] | 08-09 11:49 |
| `eps_probe/nq10_step4198_lambda.json` | 5.5 KB | dict[18] | 08-09 11:49 |
| `eps_probe/nq10_step92356_lambda.json` | 5.1 KB | dict[18] | 08-09 11:49 |
| `eps_probe/nq8_lambda.json` | 12.5 KB | dict[18] | 08-09 10:13 |
| `eps_probe/nq8_sigma.json` | 9.1 KB | dict[17] | 08-09 10:13 |
| `eps_probe/nq8fa_lambda.json` | 12.8 KB | dict[18] | 08-09 10:34 |
| `eps_probe/nq8fa_lambda_bigdelta_f32.json` | 12.7 KB | dict[18] | 08-09 10:51 |
| `eps_probe/nq8fa_lambda_f64.json` | 12.8 KB | dict[18] | 08-09 10:55 |
| `eps_probe/nq8fa_sigma.json` | 9.1 KB | dict[17] | 08-09 10:34 |
| `eps_probe/nq8fa_step18730_lambda.json` | 8.7 KB | dict[18] | 08-09 10:53 |
| `eps_probe/nq8fa_step28095_lambda.json` | 8.7 KB | dict[18] | 08-09 10:53 |
| `eps_probe/nq8fa_step46825_lambda.json` | 8.7 KB | dict[18] | 08-09 10:54 |
| `eps_probe/nq8fa_step9365_lambda.json` | 9.3 KB | dict[18] | 08-09 10:53 |
| `eps_probe/sudoku34_lambda.json` | 12.5 KB | dict[18] | 08-09 10:39 |
| `eps_probe/sudoku34_sigma.json` | 9.1 KB | dict[17] | 08-09 10:38 |
| `eps_probe/sudoku_lambda.json` | 12.5 KB | dict[18] | 08-09 10:14 |
| `eps_probe/sudoku_sigma.json` | 9.1 KB | dict[17] | 08-09 10:13 |
| `fig3/ema_s0_n20_h16_lim20000.json` | 2.1 KB | dict[22] | 08-08 13:19 |
| `fig3/ema_s0_n20_h16_lim20000.shard0.npz` | 683.6 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U396 | 08-08 13:17 |
| `fig3/ema_s0_n20_h16_lim20000.shard1.npz` | 725.5 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U400 | 08-08 13:18 |
| `fig3/ema_s0_n20_h16_lim20000.shard2.npz` | 686.2 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U399 | 08-08 13:18 |
| `fig3/ema_s0_n20_h16_lim20000.shard3.npz` | 628.7 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U401 | 08-08 13:19 |
| `fig3/ema_s0_n20_h16_lim20000.shard4.npz` | 703.5 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U401 | 08-08 13:17 |
| `fig3/ema_s0_n20_h16_lim20000.shard5.npz` | 651.0 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U401 | 08-08 13:19 |
| `fig3/ema_s0_n20_h16_lim20000.shard6.npz` | 721.0 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U400 | 08-08 13:18 |
| `fig3/ema_s0_n20_h16_lim20000.shard7.npz` | 684.6 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U401 | 08-08 13:17 |
| `fig3/ema_s1_n20_h16_lim20000.json` | 2.1 KB | dict[22] | 08-08 13:37 |
| `fig3/ema_s1_n20_h16_lim20000.shard0.npz` | 682.9 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U396 | 08-08 13:35 |
| `fig3/ema_s1_n20_h16_lim20000.shard1.npz` | 726.7 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U400 | 08-08 13:36 |
| `fig3/ema_s1_n20_h16_lim20000.shard2.npz` | 684.1 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U399 | 08-08 13:36 |
| `fig3/ema_s1_n20_h16_lim20000.shard3.npz` | 628.1 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U401 | 08-08 13:36 |
| `fig3/ema_s1_n20_h16_lim20000.shard4.npz` | 702.7 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U401 | 08-08 13:35 |
| `fig3/ema_s1_n20_h16_lim20000.shard5.npz` | 651.4 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U402 | 08-08 13:36 |
| `fig3/ema_s1_n20_h16_lim20000.shard6.npz` | 720.7 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U401 | 08-08 13:35 |
| `fig3/ema_s1_n20_h16_lim20000.shard7.npz` | 684.5 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U400 | 08-08 13:35 |
| `fig3/live_s0_n20_h16_lim20000.json` | 2.1 KB | dict[22] | 08-08 13:54 |
| `fig3/live_s0_n20_h16_lim20000.shard0.npz` | 747.2 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U400 | 08-08 13:53 |
| `fig3/live_s0_n20_h16_lim20000.shard1.npz` | 766.0 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U404 | 08-08 13:54 |
| `fig3/live_s0_n20_h16_lim20000.shard2.npz` | 742.8 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U404 | 08-08 13:53 |
| `fig3/live_s0_n20_h16_lim20000.shard3.npz` | 690.5 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U406 | 08-08 13:54 |
| `fig3/live_s0_n20_h16_lim20000.shard4.npz` | 741.3 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U406 | 08-08 13:53 |
| `fig3/live_s0_n20_h16_lim20000.shard5.npz` | 726.4 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U407 | 08-08 13:54 |
| `fig3/live_s0_n20_h16_lim20000.shard6.npz` | 763.3 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U406 | 08-08 13:53 |
| `fig3/live_s0_n20_h16_lim20000.shard7.npz` | 728.8 KB | cand(2500, 20, 81)|u1 v(2500, 20)<f4 inputs(2500, 81)|u1 labels(2500, 81)|u1 meta()<U406 | 08-08 13:53 |
| `fig3/xcheck_ema_s0_n20_h16_lim4000.json` | 2.0 KB | dict[22] | 08-08 13:01 |
| `fig3/xcheck_ema_s0_n20_h16_lim4000.shard0.npz` | 145.1 KB | cand(500, 20, 81)|u1 v(500, 20)<f4 inputs(500, 81)|u1 labels(500, 81)|u1 meta()<U395 | 08-08 13:01 |
| `fig3/xcheck_ema_s0_n20_h16_lim4000.shard1.npz` | 142.0 KB | cand(500, 20, 81)|u1 v(500, 20)<f4 inputs(500, 81)|u1 labels(500, 81)|u1 meta()<U398 | 08-08 13:01 |
| `fig3/xcheck_ema_s0_n20_h16_lim4000.shard2.npz` | 121.6 KB | cand(500, 20, 81)|u1 v(500, 20)<f4 inputs(500, 81)|u1 labels(500, 81)|u1 meta()<U398 | 08-08 13:01 |
| `fig3/xcheck_ema_s0_n20_h16_lim4000.shard3.npz` | 142.3 KB | cand(500, 20, 81)|u1 v(500, 20)<f4 inputs(500, 81)|u1 labels(500, 81)|u1 meta()<U399 | 08-08 13:01 |
| `fig3/xcheck_ema_s0_n20_h16_lim4000.shard4.npz` | 143.6 KB | cand(500, 20, 81)|u1 v(500, 20)<f4 inputs(500, 81)|u1 labels(500, 81)|u1 meta()<U399 | 08-08 13:01 |
| `fig3/xcheck_ema_s0_n20_h16_lim4000.shard5.npz` | 141.8 KB | cand(500, 20, 81)|u1 v(500, 20)<f4 inputs(500, 81)|u1 labels(500, 81)|u1 meta()<U397 | 08-08 13:01 |
| `fig3/xcheck_ema_s0_n20_h16_lim4000.shard6.npz` | 144.3 KB | cand(500, 20, 81)|u1 v(500, 20)<f4 inputs(500, 81)|u1 labels(500, 81)|u1 meta()<U399 | 08-08 13:01 |
| `fig3/xcheck_ema_s0_n20_h16_lim4000.shard7.npz` | 156.4 KB | cand(500, 20, 81)|u1 v(500, 20)<f4 inputs(500, 81)|u1 labels(500, 81)|u1 meta()<U399 | 08-08 13:01 |
| `halt8/SMOKE_nq8_s0_final_ema_h0_hms16.json` | 63.6 KB | dict[11] | 08-06 04:49 |
| `halt_sweep_20260806/nq10_repro_s0_final_ema_h0_hms16.json` | 947.0 KB | dict[11] | 08-06 07:52 |
| `halt_sweep_20260806/nq8_s0_final_ema_h0_hms16.json` | 63.6 KB | dict[11] | 08-06 05:00 |
| `halt_sweep_20260806/nq8_s0_final_ema_h0_hms32.json` | 63.6 KB | dict[11] | 08-06 05:15 |
| `halt_sweep_20260806/nq8_s0_final_ema_h0_hms64.json` | 63.6 KB | dict[11] | 08-06 05:47 |
| `halt_sweep_20260806/nq8_s0_final_ema_h0_hms8.json` | 63.6 KB | dict[11] | 08-06 04:52 |
| `halt_sweep_20260806/nq8_s0_final_live_h0_hms16.json` | 63.7 KB | dict[11] | 08-06 05:05 |
| `halt_sweep_20260806/nq8_s0_final_live_h0_hms32.json` | 63.7 KB | dict[11] | 08-06 05:26 |
| `halt_sweep_20260806/nq8_s0_final_live_h0_hms64.json` | 63.7 KB | dict[11] | 08-06 06:07 |
| `halt_sweep_20260806/nq8_s0_final_live_h0_hms8.json` | 63.6 KB | dict[11] | 08-06 04:55 |
| `halt_sweep_20260806/nq8_s0_peak_ema_h0_hms16.json` | 63.6 KB | dict[11] | 08-06 06:12 |
| `halt_sweep_20260806/nq8_s0_peak_live_h0_hms16.json` | 63.7 KB | dict[11] | 08-06 06:18 |
| `halt_sweep_20260806/nq8_s1_final_ema_h0_hms16.json` | 63.6 KB | dict[11] | 08-06 05:00 |
| `halt_sweep_20260806/nq8_s1_final_ema_h0_hms32.json` | 63.6 KB | dict[11] | 08-06 05:16 |
| `halt_sweep_20260806/nq8_s1_final_ema_h0_hms64.json` | 63.6 KB | dict[11] | 08-06 05:47 |
| `halt_sweep_20260806/nq8_s1_final_ema_h0_hms8.json` | 63.6 KB | dict[11] | 08-06 04:52 |
| `halt_sweep_20260806/nq8_s1_final_live_h0_hms16.json` | 63.6 KB | dict[11] | 08-06 05:05 |
| `halt_sweep_20260806/nq8_s1_final_live_h0_hms32.json` | 63.6 KB | dict[11] | 08-06 05:26 |
| `halt_sweep_20260806/nq8_s1_final_live_h0_hms64.json` | 63.6 KB | dict[11] | 08-06 06:07 |
| `halt_sweep_20260806/nq8_s1_final_live_h0_hms8.json` | 63.6 KB | dict[11] | 08-06 04:55 |
| `halt_sweep_20260806/nq8_s1_peak_ema_h0_hms16.json` | 63.6 KB | dict[11] | 08-06 06:13 |
| `halt_sweep_20260806/nq8_s1_peak_live_h0_hms16.json` | 63.7 KB | dict[11] | 08-06 06:18 |

### TinyRecursiveModels/experiments/results/ — 922개, 32.6 MB (전부 .json, 평면 디렉터리)

파일이 많아 접두어별로 묶었다. 전체 파일명은 `ls experiments/results/` 로 확인.

| 접두어 | 개수 | 합계 크기 | mtime 범위(UTC) |
|---|---|---|---|
| `live_*` | 177 | 186.1 KB | 07-28 13:40 ~ 07-28 13:40 |
| `band_*` | 80 | 376.1 KB | 07-29 05:09 ~ 07-30 05:17 |
| `offup_*` | 68 | 69.6 KB | 07-30 02:32 ~ 07-30 02:32 |
| `v5gc8_*` | 60 | 218.2 KB | 07-31 04:28 ~ 07-31 05:58 |
| `fin_*` | 48 | 7.6 MB | 08-02 16:05 ~ 08-04 12:42 |
| `sweep50_*` | 42 | 47.1 KB | 07-30 02:32 ~ 07-30 02:32 |
| `rp_*` | 38 | 20.3 MB | 08-04 03:13 ~ 08-06 05:48 |
| `v5h_*` | 32 | 150.2 KB | 07-31 08:21 ~ 07-31 08:21 |
| `finh_*` | 28 | 1.8 MB | 08-02 23:11 ~ 08-02 23:11 |
| `gc5_*` | 28 | 82.6 KB | 07-30 02:32 ~ 08-04 07:55 |
| `v4s_*` | 28 | 133.4 KB | 07-30 09:31 ~ 07-30 11:12 |
| `ema_*` | 26 | 28.5 KB | 07-27 13:19 ~ 07-28 13:40 |
| `v2_*` | 24 | 114.8 KB | 07-30 02:32 ~ 07-30 02:32 |
| `v5nq10_*` | 24 | 115.8 KB | 07-31 02:47 ~ 07-31 05:49 |
| `v6h_*` | 24 | 114.5 KB | 07-31 09:08 ~ 07-31 10:53 |
| `b500_*` | 16 | 75.0 KB | 07-30 02:32 ~ 07-30 02:32 |
| `cov50_*` | 16 | 76.8 KB | 07-30 02:32 ~ 07-30 02:32 |
| `nq10_*` | 16 | 75.6 KB | 07-30 04:47 ~ 07-30 04:47 |
| `peak50_*` | 16 | 18.2 KB | 07-30 02:32 ~ 07-30 02:32 |
| `onehot_*` | 12 | 12.3 KB | 07-27 16:02 ~ 07-27 16:13 |
| `sweep_*` | 12 | 13.0 KB | 07-27 13:19 ~ 07-28 13:40 |
| `guide_*` | 11 | 13.7 KB | 07-28 13:40 ~ 07-28 13:40 |
| `cov_*` | 8 | 37.7 KB | 07-28 13:40 ~ 07-28 13:40 |
| `ext_*` | 8 | 8.9 KB | 07-28 13:40 ~ 07-28 13:40 |
| `fine_*` | 8 | 8.7 KB | 07-28 13:40 ~ 07-28 13:40 |
| `gc5c_*` | 8 | 29.4 KB | 07-30 04:47 ~ 07-30 04:47 |
| `klgrid_*` | 8 | 16.3 KB | 07-27 13:19 ~ 07-27 13:19 |
| `off29_*` | 8 | 8.4 KB | 07-28 13:40 ~ 07-28 13:40 |
| `diag_*` | 6 | 3.0 KB | 07-23 17:42 ~ 07-24 06:35 |
| `nq8_*` | 6 | 8.6 KB | 07-21 22:53 ~ 07-24 05:39 |
| `covcurve8_*` | 5 | 74.0 KB | 07-30 05:17 ~ 07-30 05:17 |
| `covcurve_*` | 4 | 427.8 KB | 07-30 04:47 ~ 07-30 04:47 |
| `fp_*` | 4 | 253.8 KB | 08-04 13:10 ~ 08-04 13:10 |
| `qhalt_*` | 4 | 1.4 KB | 07-24 05:44 ~ 07-24 05:44 |
| `contradiction_*` | 3 | 45.1 KB | 07-27 13:19 ~ 07-27 13:19 |
| `eval_*` | 2 | 3.0 KB | 07-21 16:18 ~ 07-24 05:14 |
| `kl_*` | 2 | 373 B | 07-21 16:58 ~ 07-21 17:00 |
| `lmean_*` | 2 | 4.9 KB | 07-21 16:20 ~ 07-21 16:20 |
| `sigma_*` | 2 | 1.3 KB | 07-24 00:42 ~ 07-24 00:43 |
| `task2_*` | 2 | 2.4 KB | 07-24 05:41 ~ 07-24 05:41 |
| `yemb_*` | 2 | 30.2 KB | 07-27 13:19 ~ 07-27 13:19 |
| `anchor_*` | 1 | 1.5 KB | 07-24 06:59 ~ 07-24 06:59 |
| `audit_*` | 1 | 1.5 KB | 07-27 13:19 ~ 07-27 13:19 |
| `migration_*` | 1 | 1.5 KB | 07-27 13:19 ~ 07-27 13:19 |
| `nh512_*` | 1 | 970 B | 07-23 05:16 ~ 07-23 05:16 |

---

## ④ 진행 중 / 대기 중인 지시 (2026-08-11 06:39 UTC 실측)

### 진행 중

| 항목 | 상태 | 근거 |
|---|---|---|
| **[50] `scripts/run_depth_util_50.sh`** — Sudoku depth utilization (hms 128 / 320) | **실행 중.** 02:37 기동. hms=128 8샤드 완료 + merge 완료(04:54). hms=320 8샤드 04:54 시작, 06:39 현재 진행 중 (GPU 0–7 util 100%, 각 2.6 GB) | `ps` / `nvidia-smi` / `reports/sudoku50/` mtime |
| **[29] `/workspace/home/chaeun/tmp/watch_push_29.sh`** — fprm·gram-repro 원격 ref 5분 감시 | **감시 중** (2일 10시간째). 마지막 대조 06:24, 두 리포 모두 변화 없음 (`refs_*.prev` 가 08-08 19:29 이후 그대로) | 프로세스 + `refs_*.prev/.cur` diff |

hms=128 merge 결과 (`reports/sudoku50/n20000_hms128_N20_h0_merged8.json`, n=20,000):
prior exact 0.7363 / pass@20 0.88855 / majority 0.8544 / verifier-selected 0.8842.

hms=320 완주 후 스크립트가 `experiments/depth_util_summary.py` 를 돌려
`results/figdata/depth_util_sudoku.csv` (현재 2행 = hms16 seed 0/1) 를 갱신한다.
같은 csv 의 meta 에 적힌 대로 **hms 8 / 32 / 64 는 62번 격자에 있고 아직 합쳐지지 않았다.**

### 대기 중 (34번 몫 — 이 박스에서 직접 확인 불가)

`/workspace/home/chaeun/tmp/gram50/base_code/REPORT_INDEX.md` §6 (스냅샷 시점 2026-08-07):

| 항목 | 담당 | 상태 |
|---|---|---|
| nq10 A3′ μ/σ ablation (μ0σ1 / μ1σ0) | 34번 | 재실행 필요 (08-05 리부트로 11% 진행분 소실) |
| nq8 Stage A `ii` 3셀 (`a7525_lr3e4`, `a7525_w2000`, `fx_a7525_w2000`) | 34번 | 46%/46%/33% 에서 리부트 중단, SPEC §9 로 resume 불가 |
| nq10 10×10 결과 JSON (`nq10_on_b520_s0/s1/s2`, `nq10_on_b450_s0`) | 29번 | 수령 대기 |

### 미커밋 상태

- `TinyRecursiveModels`: M `experiments/coverage_curve.py`, M `reports/DATA_MD5.md`;
  untracked `experiments/{depth_util_summary,eps_propagation_probe,merge_sudoku_shards,residual_probe,solution_counts}.py`, `experiments/run_eps_probe_50.sh`
- `fprm`: M `config/cfg_pretrain_sudoku.yaml`; untracked `config/arch/fprm_nqueens10_{S0p_nonoise,S2_convnone}.yaml`,
  `config/cfg_pretrain_nqueens10_{S0p_nonoise,S2_convnone}.yaml`, `experiments/chain_gpu{03,47}_50.sh`, `experiments/fprm_harness.py`

## 디스크

| 경로 | 크기 |
|---|---|
| `TinyRecursiveModels/checkpoints` | 91 GB |
| `fprm/checkpoints` | 52 GB |
| `fprm/results/figdata` | 1.9 GB |

---
---

# 추가 조사 (2026-08-11 07:14 UTC) — checkpoint 대장 / EMA·live 메커니즘

## 0. 접근 범위

| 서버 | 이 박스에서 접근 | 근거 |
|---|---|---|
| 50 lecun | 가능 (이 박스) | — |
| 29 hinton / 34 bengio / 62 | **불가** | `~/.ssh/config` 없음, `known_hosts` 만 존재. git remote 는 `github.com/clhaeun/gram-repro`, `github.com/nilskiKonjIzDunava/fprm` 뿐 |

29 / 34 / 62 항목은 이 박스에서 값을 낼 수 없다 → 전부 **불명**.

## ① EMA / live 구분 메커니즘 — 확정

### 1-1. 저장 측 (pretrain.py)

| 줄 | 코드 | 산출 |
|---|---|---|
| `pretrain.py:736` | `save_train_state(config, train_state_eval)` | `step_N` = **EMA 사본** |
| `pretrain.py:737-739` | `if config.ema: save_train_state(config, train_state, suffix=".live")` | `step_N.live` = **raw(live) SGD 가중치** |
| `pretrain.py:683` | `train_state_eval.model = ema_helper.ema_copy(...)` | EMA 사본 생성 지점 |
| `pretrain.py:700-727` | `if config.ema:` raw eval 병행, `[EVAL RAW @ step N]` 로그 + wandb `*_raw` | **파일 없이도** 같은 step 의 raw 수치가 로그에 남는다 |

### 1-2. 읽기 측 — 두 리포가 서로 다른 메커니즘

**gram-repro (TRM) — `--weights` 플래그가 없다.**

| 줄 | 코드 |
|---|---|
| `experiments/gram_harness.py:40` | `def load_model(ckpt_dir, step):` |
| `experiments/gram_harness.py:47` | `cfg.load_checkpoint = os.path.join(ckpt_dir, step)` ← **축은 `--step` 문자열 하나뿐** |
| `experiments/gram_harness.py:48` | `cfg.ema = False   # checkpoint is already the EMA copy` |
| `experiments/gram_harness.py:64` | `net = getattr(ts.model, "_orig_mod", ts.model)` (torch.compile 언랩) |
| `pretrain.py:287` | `state_dict = torch.load(config.load_checkpoint, map_location="cuda")` |
| `pretrain.py:300` | `model.load_state_dict(state_dict, assign=True)` |

소비자 3종 전부 동일: `sweep_harness.py:116`, `gc_harness.py:88`, `sudoku_harness.py:186` → `load_model(args.ckpt_dir, args.step)`.
`--weights` 인자는 세 파일 어디에도 없다. EMA/live 는 `--step step_N` vs `--step step_N.live` 로만 갈린다.

**fprm — `--weights` 플래그가 있다.**

| 줄 | 코드 |
|---|---|
| `experiments/fprm_harness.py:275` | `ap.add_argument("--weights", default="ema", choices=["ema","live"])` |
| `experiments/fprm_harness.py:133` | `def load_model(ckpt_dir, step, weights="ema", device="cuda"):` |
| `experiments/fprm_harness.py:145-147` | `if weights == "ema":` → `step_N`, `torch.load(..., weights_only=True)` |
| `experiments/fprm_harness.py:148-150` | `elif weights == "live":` → `step_N_train_state.pt`, `["model"]` |
| `experiments/fprm_harness.py:301` | `load_model(args.ckpt_dir, args.step, args.weights)` |
| `experiments/fprm_harness.py:321` | 결과 JSON 에 `"weights": args.weights` 기록 |

### 1-3. state_dict 키 실측

`torch.load(..., map_location='meta')` 로 키만 읽었다.

| checkpoint | 타입 | 키 수 | ema/shadow 관련 키 | `_orig_mod.` 프리픽스 |
|---|---|---|---|---|
| `gc10_paper_34/step_87620` | OrderedDict | 27 | **없음** | 27 / 27 (전부) |
| `gc8_paper_34/step_45540` | OrderedDict | 27 | **없음** | 27 / 27 |
| `nq10_repro_s0/step_138534` | OrderedDict | 27 | **없음** | 27 / 27 |

키 목록 앞 4개(3종 동일): `_orig_mod.model.inner.prior_logvar_bias`, `_orig_mod.model.inner.post_logvar_bias`, `_orig_mod.model.inner.H_init`, `_orig_mod.model.inner.L_init`
뒤 3개: `_orig_mod.model.inner.post_logvar.down_proj.weight`, `_orig_mod.model.inner.lrpm_head.weight`, `_orig_mod.model.inner.lrpm_head.bias`

`pretrain.py:284` 는 `puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"` 로 이 프리픽스를 전제한다.

### 1-4. gc10_paper_34 의 EMA 0.691 / live 0.579 출처

두 수치가 나온 파일:

| 파일 | 1행 | 마지막 행 |
|---|---|---|
| `reports/gc/gc10_step87620_ema_h0.log` | `/home/chaeun/gram-repro/pretrain.py:287: FutureWarning...` | `.../gc10_paper_34/step_87620  conflict=0.268±0.589 cov@20=0.6906 maj_canon=0.9375 vsel=0.9219` |
| `reports/gc/gc10_step87620_live_h0.log` | `/home/chaeun/gram-repro/pretrain.py:287: ...` | `.../gc10_paper_34/step_87620.live  conflict=0.247±0.550 cov@20=0.5790 maj_canon=0.9010 vsel=0.8958` |

- 두 로그의 1행 경로가 **`/home/chaeun/gram-repro/`** = 29번 hinton 경로다 (50번은 `/workspace/TinyRecursiveModels`).
- live 로그는 `Loading checkpoint checkpoints/.../step_87620.live` 를 찍었다 → 측정 시점 그 박스에 `.live` 가 존재했다.
- 50번의 `checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/` 실측 (08-09_15:52): `all_config.yaml`, `gram.py`, `losses.py`, `step_87620` **4개뿐. `.live` 없음.**
- `reports/gc/` 파일 20건 mtime 전부 08-08_15:41 (일괄 복사).

### 1-5. gc8_paper_34 / nq10_repro_s0 디렉터리 대조

| run | 파일 | ema md5(12) | live md5(12) |
|---|---|---|---|
| `gc10_paper_34` | `all_config.yaml` `gram.py` `losses.py` `step_87620` | `de2e2d95934d` | **파일 없음** |
| `gc8_paper_34` | + `step_45540` `step_45540.live` | `99adc3f1644e` | `bedee38e310c` |
| `nq10_repro_s0` | `step_138534` `step_138534.live` 외 33 step | `f0dc3d5a58cd` | `acb6b9bb6e69` |

### 1-6. 부수 실측 — ema md5 == live md5 인 run 1건

`Nqueens-10-ACT-torch/nq10_A3p_ii_a7525_lr2e4` `step_96554` 와 `step_96554.live` 가 md5 `0e5f317ac094` 로 **바이트 동일**. 전 78 run 중 이 1건뿐이다.

## ② gc8_paper_34 coverage — 세 축 전부

`experiments/gc_harness.py:33-45` `coverage_at_N()` 이 저장하는 `coverage` 는 **행 단위 per-puzzle 비율의 평균**(`cov = found/totals` → `cov[have].mean()`) 하나뿐이다. micro 는 저장된 `found_mean`/`total_mean` 에서 재계산했다.

GC test 행 중복 실측: gc8 `rows=255, unique_rows=255` / gc10 `rows=192, unique_rows=192` → **row_macro == puzzle_macro** (pair 확장 없음).

| checkpoint | weight | path | N | hseed | micro | row_macro | puzzle_macro | maj_canon | vsel | conflict |
|---|---|---|---|---|---|---|---|---|---|---|
| `gc8_paper_34/step_45540` | EMA | harness (`gc_harness`) | 20 | 0 | **0.6296** | **0.8221** | **0.8221** | 0.8980 | 0.8000 | 0.310±0.619 |
| `gc8_paper_34/step_45540` | EMA | harness | 20 | 1 | 0.6512 | 0.8388 | 0.8388 | 0.9020 | 0.8118 | 0.3129 |
| `gc8_paper_34/step_45540.live` | live | harness | 20 | 0 | 0.6985 | 0.8879 | 0.8879 | 0.9804 | 0.9373 | 0.0743 |
| `gc8_paper_34/step_45540.live` | live | harness | 20 | 1 | 0.7135 | 0.8956 | 0.8956 | 0.9804 | 0.9294 | 0.0747 |
| `gc10_paper_34/step_87620` | EMA | harness | 20 | 0 | 0.4003 | **0.6906** | 0.6906 | 0.9375 | 0.9219 | 0.2680 |
| `gc10_paper_34/step_87620` | EMA | harness | 20 | 1 | 0.4070 | 0.6873 | 0.6873 | 0.8958 | 0.8542 | 0.2565 |
| `gc10_paper_34/step_87620.live` | live | harness | 20 | 0 | 0.3148 | **0.5790** | 0.5790 | 0.9010 | 0.8958 | 0.2469 |
| `gc10_paper_34/step_87620.live` | live | harness | 20 | 1 | 0.3210 | 0.5893 | 0.5893 | 0.8646 | 0.8542 | 0.2557 |

criterion: coverage 는 canonical unique 해 수 / sidecar 해 수. maj_canon = `majority_at_N_valid_canonical`, vsel = `verifier_selected_at_N_valid`.
n: gc8 255 / gc10 192. 결과 JSON: `reports/gc/gc8_step45540_{ema,live}_h{0,1}.json`, `reports/gc/gc10_step87620_{ema,live}_h{0,1}.json`.

gc8 EMA row_macro hseed 0/1 평균 = **0.8305**.

## ③ nq8 seed 1 checkpoint

`checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s1/step_67428` — 39,880,162 B, md5 `419f59a9aa2a`, mtime 08-02 13:17
`.live`: `step_67428.live` 39,880,162 B, md5 `47ef71fca559`

| checkpoint | weight | path | N | hseed | prior valid | prior strict | maj | vsel | cov micro | cov row_macro | cov puzzle_macro | JSON |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nq8_final_add_s0/step_67428` | EMA | harness (`sweep_harness` 구판 `fin_`) | 20 | 0 | 0.9973 | 0.9973 | 1.0000 | 0.9913 | 0.8695 | 0.9655 | 재집계 불가 | `experiments/results/fin_nq8_final_add_s0_step67428_ema_s0.json` |
| 〃 | EMA | 〃 | 20 | 1 | 0.9972 | 0.9972 | 1.0000 | 0.9921 | 0.8639 | 0.9633 | 재집계 불가 | `..._ema_s1.json` |
| `nq8_final_add_s0/step_67428.live` | live | 〃 | 20 | 0 | 0.9514 | 0.9514 | 0.9976 | 0.9543 | 0.8676 | 0.9632 | 재집계 불가 | `..._live_s0.json` |
| 〃 | live | 〃 | 20 | 1 | 0.9535 | 0.9533 | 0.9961 | 0.9661 | 0.8625 | 0.9621 | 재집계 불가 | `..._live_s1.json` |
| `nq8_final_add_s1/step_67428` | EMA | 〃 | 20 | 0 | 0.9899 | 0.9899 | 1.0000 | 0.9953 | 0.8681 | 0.9663 | 재집계 불가 | `fin_nq8_final_add_s1_step67428_ema_s0.json` |
| 〃 | EMA | 〃 | 20 | 1 | 0.9899 | 0.9899 | 1.0000 | 0.9961 | 0.8700 | 0.9658 | 재집계 불가 | `..._ema_s1.json` |
| `nq8_final_add_s1/step_67428.live` | live | 〃 | 20 | 0 | 0.9843 | 0.9843 | 0.9992 | 0.9913 | 0.8324 | 0.9487 | 재집계 불가 | `..._live_s0.json` |
| 〃 | live | 〃 | 20 | 1 | 0.9833 | 0.9833 | 0.9992 | 0.9945 | 0.8191 | 0.9403 | 재집계 불가 | `..._live_s1.json` |

n = 1270 행 (pair 확장 nq8 test). EMA, hseed 0/1 평균: s0 = 0.99725, s1 = 0.98990 → **2-seed 평균 0.9936, 개별 [99.72 / 98.99]** (`REPORT_final_pair_50.md §1` 의 99.4±0.5 [99.72/98.99] 와 일치).
puzzle_macro: `fin_` JSON 에 per-row 원자료가 없어 재집계 불가 (`REPORT_final_pair_50.md §1` 각주 ‡ 와 동일).

## ④ FPRM 셀 전수

git HEAD: fprm `7e65985318245abb997da476f045b4f9016cb600` / gram-repro `f93824ee94cca79464350d3a095810b8e4e6c5d5`

| 셀 | checkpoint 디렉터리 | step | ema 파일:크기:md5 | live(train_state):크기:md5 | 완주 |
|---|---|---|---|---|---|
| **A_sd1** | **없음** | — | — | — | — |
| **S1** | **없음** | — | — | — | — |
| S2_conv3 | `checkpoints/fprm-nqueens10/nq10_fprm_S2_conv3_s0/` | 138,534 | `step_138534`:130,082,322:`388d6a391baa` | `step_138534_train_state.pt`:325,235,986:`47ce7a6a7c62` | 완주 (계획 138,548) |
| S3_conv16 | `.../nq10_fprm_S3_conv16_s0/` | 138,534 | `step_138534`:130,082,322:`7c8533664704` | `..._train_state.pt`:325,235,986:`2798afb1310a` | 완주 |
| S0p_nonoise | `.../nq10_fprm_S0p_nonoise_s0/` | 138,534 | `step_138534`:54,577,792:`62c9deb4f425` | `..._train_state.pt`:136,472,628:`0463150983bb` | 완주 |
| S2_convnone | `.../nq10_fprm_S2_convnone_s0/` | 138,534 | `step_138534`:130,073,681:`1c2fdfae1e26` | `..._train_state.pt`:325,201,548:`341a691d9651` | 완주 |
| (참고) fixed | `.../nq10_fprm_fixed_s0/` | 83,960 | `step_83960`:54,573,666:`fc01fd88c327` | `..._train_state.pt`:136,453,690:`c43cc3876904` | 중단 60.6% |
| (참고) stock | `.../nq10_fprm_stock_s0/` | — | **step 파일 0** (`all_config.yaml` 1,927 B 만) | — | — |

A_sd1 / S1 은 **config 만 존재**하고 checkpoint 가 없다:
`config/arch/fprm_nqueens10_A_sd1.yaml`, `config/cfg_pretrain_nqueens10_A_sd1.yaml`, `config/arch/fprm_nqueens10_S1.yaml`, `config/cfg_pretrain_nqueens10_S1.yaml`.
커밋 `3e0c4e48` 은 이 리포에 없다 — `git cat-file -t 3e0c4e48` → `fatal: Not a valid object name 3e0c4e48`. 미커밋 작업트리 여부는 이 박스에서 확인 불가 → **불명**.

### FPRM harness 값 (4축 + coverage 3축)

path = harness (`experiments/fprm_harness.py`, 4샤드 merge) · N = 20 · hseed = 0 · data `data/nqueens-10/test` · n_puzzles 19,220 행 / 6,513 고유 입력

| 셀 | weight | prior valid | prior strict | maj valid | maj strict | cov micro | cov row_macro | cov puzzle_macro | eval_steps_mean | JSON |
|---|---|---|---|---|---|---|---|---|---|---|
| S0p_nonoise | EMA | 0.1726 | 0.1726 | 0.1738 | 0.1738 | 0.0202 | 0.1103 | 0.2310 | 517.34 | `results/fprm_harness/S0p_ema.json` |
| S0p_nonoise | live | 0.0161 | 0.0161 | 0.0164 | 0.0164 | 0.0021 | 0.0120 | 0.0266 | 187.54 | `S0p_live.json` |
| S2_conv3 | EMA | 0.4148 | 0.4147 | 0.4443 | 0.4443 | 0.1724 | 0.3787 | 0.5026 | 517.05 | `S2_ema.json` |
| S2_conv3 | live | 0.3498 | 0.3478 | 0.3946 | 0.3924 | 0.1063 | 0.2510 | 0.3434 | 968.31 | `S2_live.json` |
| S3_conv16 | EMA | 0.1846 | 0.1846 | 0.2469 | 0.2469 | 0.0517 | 0.1944 | 0.3270 | 1000.00 | `S3_ema.json` |
| S3_conv16 | live | 0.0025 | 0.0025 | 0.0031 | 0.0031 | 0.0008 | 0.0042 | 0.0084 | 1000.00 | `S3_live.json` |
| S2_convnone | EMA | 0.2088 | 0.2088 | 0.4650 | 0.4650 | 0.1589 | 0.4377 | 0.6026 | 257.94 | `CN_ema.json` |
| S2_convnone | live | 0.0693 | 0.0693 | 0.2137 | 0.2137 | 0.0693 | 0.1970 | 0.2797 | 390.61 | `CN_live.json` |

cov micro = `found_mean / total_mean` (재계산) · row_macro = JSON 의 `coverage` · puzzle_macro = JSON 의 `coverage_macro_by_input` (`n_inputs_scored` 6,513).
vsel: fprm harness 출력에 verifier-selected 필드 없음 → **미측정**.
A_sd1 / S1 harness 값: checkpoint 없음 → **불명**.

## (a) 최종 checkpoint md5 — 전수

`md5sum` 앞 12자. 스크립트 `/workspace/home/chaeun/tmp/md5_final.sh`, 원자료 `md5_final.tsv` (139행).
표는 본문 보고에 그대로 실었다.

## (b) 추가 — H_cycles

`H_cycles: 3`, `H_layers: 0` — **78 run 전부 동일**.

## (e) 학습 중 / 중단 run 상태 (07:14 UTC)

| 항목 | 값 |
|---|---|
| 학습 프로세스 (`pretrain.py`) | **0건** — 이 박스에서 학습 중인 run 없음 |
| 평가 프로세스 (`sudoku_harness.py`) | **8건 생존** — `run_depth_util_50.sh` hms=320 8샤드, 04:54 시작, GPU 0–7 util 100% |
| hms=320 출력 JSON | **0건** (아직 미완) |
| tmux 세션 | **없음** — `tmux ls` → `no server running on /tmp/tmux-0/default`, uid 1048 은 `/tmp/tmux-1048/default` 없음 |
| 중단 run 22건 | 최종 step·비율은 위 ① run 목록 표에 있음. 전부 프로세스 없음 |
| fprm 학습 로그 마지막 mtime | `S0p_nonoise` 08-09 17:14 · `S2_conv3` 08-09 08:33 · `S2_convnone` 08-10 04:44 · `S3_conv16` 08-09 09:19 · `fixed` 08-08 20:42 |

---
---

# 추가 조사 (2026-08-11) — GC conflict 축

## ① conflict 계산부 전문

### `evaluators/graph_coloring.py:48-78` — `verify_colorings(preds, inputs, n)`

```
48: def verify_colorings(preds: np.ndarray, inputs: np.ndarray, n: int) -> Dict[str, np.ndarray]:
49:     # preds/inputs: [B, n(n-1)/2] int. 채점결과: 각 [B]
50:     B = preds.shape[0]
51:     # 가능한 간선들의 idx (출발-도착)
52:     iu, iv = np.triu_indices(n, 1)
53:
54:     # 노드에 칠해진 색들 n개
55:     colors = preds[:, :n]                                   # [B, n]
56:     # 주어진 색으로 칠해진 게 맞는지
57:     colors_ok = ((colors >= COLOR_MIN) & (colors <= COLOR_MAX)).all(-1)
58:
59:     # 실제로 해당 문제에서 연결된 부분
60:     edge = inputs == EDGE                                   # [B, P]
61:     # 동일 간선에 연결된 노드 2개의 색이 같은지
62:     same = colors[:, iu] == colors[:, iv]                   # [B, P]
63:     # 연결됨 & 색 동일 -> conflict!임
64:     conflicts = (edge & same).sum(-1)
65:
66:     valid = colors_ok & (conflicts == 0)
```
상수: `evaluators/graph_coloring.py:35-36` `EDGE = 2`, `COLOR_MIN, COLOR_MAX = 3, 5`

- 세는 대상: **양 끝점이 같은 토큰인 간선의 개수** (`edge & same` 의 합). 위반 노드쌍도 토큰 불일치도 아니다.
- `same` (62행) 은 `colors_ok` 를 조건에 넣지 않는다 → 두 끝점이 같은 비색 토큰(예: 0)이어도 `conflicts` 에 계상된다. `colors_ok` 는 66행 `valid` 에만 쓰인다.
- 반환 `conflicts` 는 **[B] 벡터, 행(=샘플)당 정수**.

### `experiments/gc_harness.py:28-31` — `flat_stats`

```
28: def flat_stats(preds, inputs, n):
29:     """② Conflict: 전 샘플의 위반 edge 수 mean±std (논문 2.7±2.1 형식)."""
30:     r = verify_colorings(preds.reshape(-1, preds.shape[-1]), np.tile(inputs, (preds.shape[0], 1)), n)
31:     return dict(conflicts_mean=float(r["conflicts"].mean()), conflicts_std=float(r["conflicts"].std()))
```

### 호출 경로

| 줄 | 코드 |
|---|---|
| `gc_harness.py:93-94` | `pri = rollout(net, inputs, labels, K=args.N, sigma_scale_=args.sigma_scale, mode="prior", seed=args.seed, verbose=False)` |
| `gc_harness.py:95` | `cand, v = pri["preds"], pri["v"]` — `cand` shape `[N, B, L]` |
| `gc_harness.py:98` | `res["prior_all_samples"] = flat_stats(cand, inputs, args.n)` |
| `gc_harness.py:109` | majority 는 `verify_colorings(majc,...)["valid"].mean()` 만 쓴다 (conflict 미사용) |
| `gc_harness.py:110` | vsel 도 `["valid"].mean()` 만 쓴다 (conflict 미사용) |
| `gc_harness.py:114` | 출력 `conflict={p['conflicts_mean']:.3f}±{p['conflicts_std']:.3f}` |

### 답

| 질문 | 답 |
|---|---|
| 무엇을 세는가 | **양 끝점 동색(동일 토큰) 간선 수**. `evaluators/graph_coloring.py:64` |
| 집계 축 | **(d) 그 외** — `preds.reshape(-1, L)` 로 `[N,B]` 를 `N*B` 행으로 평탄화한 뒤 그 전체의 산술평균 (`gc_harness.py:30-31`). 표본 수 = gc8 20×255 = 5,100 / gc10 20×192 = 3,840. (a) 와는 N 이 인스턴스마다 동일해 수치가 같지만 코드상 축은 "전 샘플 평균"이다. (b) test 전체 합 아님, (c) N 합 후 인스턴스 평균 아님 |
| ± 의 축 | `np.ndarray.std()` = **ddof=0 모표준편차**, 위와 같은 `N*B` 평탄 배열에 대해. 인스턴스 간 변동과 샘플 간 변동이 섞여 있다. seed 축 아님 |
| N=20 중 어느 샘플 | **prior 20샘플 전부** (`gc_harness.py:98` 이 `cand` 전체를 넘긴다). majority 선택본·vsel 선택본은 conflict 를 재지 않는다 (`:109`, `:110`) |

## ② 세 축 재집계

`reports/gc/gc*_step*_h*.json` 의 `prior_all_samples` 에는 `conflicts_mean` / `conflicts_std` 두 스칼라만 있다 (배열형 필드 0개). per-sample `conflicts` 배열은 저장돼 있지 않다. N 이 전 인스턴스 20 으로 동일하므로 세 축은 저장된 평균의 정확한 선형변환이다.

- `conflict_per_instance_mean` = `conflicts_mean`
- `conflict_sum_over_N_mean` = `conflicts_mean × 20`
- `conflict_test_total` = `conflicts_mean × B` (인스턴스당 평균의 test 합) — 전 샘플 총합은 `× N × B` 로 병기
- 인스턴스별 분산은 pooled std 만 저장돼 분리 불가

검산: `reports/gc/units_gc8_ema.json` `model.c_violated_edges_sum_over_N.mean` = **6.2** = 0.31 × 20 ✓ / `units_gc10_ema.json` = **5.359375** = 0.26796875 × 20 ✓
(생성기 `experiments/gc_conflict_units.py:83-92`, `:88` `"c_violated_edges_sum_over_N": stat(viol.sum(0))`, 커밋 `55ebba5`)

## ③ 8셀 전부

`data/graph-coloring-8/test` B=255, 간선/그래프 mean 13.1176 (min 7 / max 18, 총합 3,345, n(n−1)/2 = 28)
`data/graph-coloring-10/test` B=192, 간선/그래프 mean 16.6927 (min 11 / max 27, 총합 3,205, n(n−1)/2 = 45)

| 셀 | weight | path | N | hseed | criterion | per_sample_mean (=per_instance_mean) | pooled_std | sum_over_N_mean | test_total (×B) | grand_total (×N×B) |
|---|---|---|---|---|---|---|---|---|---|---|
| gc8 h0 | EMA | harness | 20 | 0 | prior 전 샘플 위반간선 | 0.310000 | 0.619122 | 6.2000 | 79.05 | 1,581.0 |
| gc8 h1 | EMA | harness | 20 | 1 | 〃 | 0.312941 | 0.629744 | 6.2588 | 79.80 | 1,596.0 |
| gc8 h0 | live | harness | 20 | 0 | 〃 | 0.074314 | 0.281744 | 1.4863 | 18.95 | 379.0 |
| gc8 h1 | live | harness | 20 | 1 | 〃 | 0.074706 | 0.285100 | 1.4941 | 19.05 | 381.0 |
| gc10 h0 | EMA | harness | 20 | 0 | 〃 | 0.267969 | 0.588797 | 5.3594 | 51.45 | 1,029.0 |
| gc10 h1 | EMA | harness | 20 | 1 | 〃 | 0.256510 | 0.563734 | 5.1302 | 49.25 | 985.0 |
| gc10 h0 | live | harness | 20 | 0 | 〃 | 0.246875 | 0.549612 | 4.9375 | 47.40 | 948.0 |
| gc10 h1 | live | harness | 20 | 1 | 〃 | 0.255729 | 0.548405 | 5.1146 | 49.10 | 982.0 |

JSON: `reports/gc/gc8_step45540_{ema,live}_h{0,1}.json`, `reports/gc/gc10_step87620_{ema,live}_h{0,1}.json`
checkpoint 필드: gc8 = `.../gc8_paper_34/step_45540[.live]`, gc10 = `.../gc10_paper_34/step_87620[.live]`

## ④ gc10 `step_87620.live`

- 이 박스 `checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/` 파일 4개: `all_config.yaml`, `gram.py`, `losses.py`, `step_87620` (md5 `de2e2d95934d`). `.live` **없음**.
- 34번 접근 수단 없음 (`~/.ssh/config` 없음, git remote 는 GitHub 2곳뿐) → **받아올 수 없다.** md5 `e25d35ccb584` 대조 **불가**.
- 단, ③ 의 gc10 live 4셀은 이미 채워져 있다. 값의 출처는 29번에서 생성돼 08-08 15:41 에 일괄 복사된 `reports/gc/gc10_step87620_live_h{0,1}.json` 이다 (로그 1행 경로 `/home/chaeun/gram-repro/pretrain.py:287`).
