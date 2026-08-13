# [50] 퍼즐별 덤프 확장 — Fig 4 right / Fig 15

2026-08-11.  재학습 없음, 추론만.  coverage = sum(found)/sum(n_solutions).

## 조건

EMA (bare `step_N` 파일) · harness · sigma_scale 1.0 · mu_scale 1.0 · hms 16 · harness seed 0 · test 전수.
N ∈ {1, 5, 10, 20} 각각 독립 프로세스 (`experiments/perpuzzle_dump.py`, 실행 `experiments/run_perpuzzle_50.sh`).

| tag | checkpoint | step | 행수 | unique 입력 |
|---|---|---|---|---|
| `nq8` | `checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s0` | 67428 | 1,270 | 772 |
| `nq8_s1` | `checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s1` | 67428 | 1,270 | 772 |
| `nq10` | `checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0` | 138534 | 19,220 | 6,513 |

`n_rows_no_total` = 0 (세 대상 모두 sidecar 미매칭 행 없음).

## 산출 경로

```
results/figdata/nq8_perpuzzle_N{1,5,10,20}.csv        각 1,270 행 (+ 헤더)
results/figdata/nq8_s1_perpuzzle_N{1,5,10,20}.csv     각 1,270 행
results/figdata/nq10_perpuzzle_N{1,5,10,20}.csv       각 19,220 행
```

같은 이름의 `.meta.json` 동반.  `results/figdata` -> `/workspace/fprm/results/figdata` (symlink).

열: `run, step, weight, harness_seed, N, hms, sigma_scale, row, ihash, n_solutions,
n_valid_samples, n_valid_consistent, found, first_valid`.

- `n_solutions` : sidecar `inputs[sha1(x)[:16]]["solutions"]` 길이.  `sweep_harness.solution_counts:38-46` ·
  `experiments/solution_counts.py:36` 과 같은 정의.
- `n_valid_samples` : N 표본 중 valid (consistent 불요구).
- `n_valid_consistent` : N 표본 중 valid ∧ consistent.  ★스펙 외 추가 열★ — `n_valid_samples` 정의가
  두 갈래라 둘 다 남겼다.
- `found` : N 표본이 맞힌 서로 다른 valid ∧ consistent 보드 수.
- `first_valid` : 첫 표본(k=0) 1개가 valid (0/1).  Fig 4 right 축.

행 단위 (test 전수).  같은 ihash 가 여러 행에 나오므로 퍼즐 축 집계는 `ihash` 로 downstream 에서 한다.

## 값

coverage = sum(found)/sum(n_solutions)

| N | nq8 | nq8_s1 | nq10 |
|---|---|---|---|
| 1 | 0.336441 | 0.334576 | 0.097756 |
| 5 | 0.632658 | 0.637453 | 0.280008 |
| 10 | 0.763719 | 0.764518 | 0.393431 |
| 20 | 0.869473 | 0.868141 | 0.519632 |

sum_found / sum_n_solutions

| N | nq8 | nq8_s1 | nq10 |
|---|---|---|---|
| 1 | 1,263 / 3,754 | 1,256 / 3,754 | 18,469 / 188,930 |
| 5 | 2,375 / 3,754 | 2,393 / 3,754 | 52,902 / 188,930 |
| 10 | 2,867 / 3,754 | 2,870 / 3,754 | 74,331 / 188,930 |
| 20 | 3,264 / 3,754 | 3,259 / 3,754 | 98,174 / 188,930 |

`first_valid` (행 평균, N 전체 동일): nq8 0.994488 · nq8_s1 0.988976 · nq10 0.960926.

`n_valid_samples` 평균

| N | nq8 | nq8_s1 | nq10 |
|---|---|---|---|
| 1 | 0.9945 | 0.9890 | 0.9609 |
| 5 | 4.9858 | 4.9457 | 4.8106 |
| 10 | 9.9701 | 9.9039 | 9.6193 |
| 20 | 19.9457 | 19.7976 | 19.2411 |

`n_found_gt_total` = 0 (12 파일 전부).

## 정합성

`experiments/check_perpuzzle_50.py`.

① N=20 의 found 가 기존 산출물과 일치

| 대상 | 기준 | 결과 |
|---|---|---|
| nq8 | `experiments/results/fin_nq8_final_add_s0_step67428_ema_s0.json` | found_mean 2.5700787402 / 2.5700787402, total_mean 2.9559055118 / 2.9559055118 |
| nq8_s1 | `experiments/results/fin_nq8_final_add_s1_step67428_ema_s0.json` | found_mean 2.5661417323 / 2.5661417323 |
| nq10 | `experiments/results/rp_nq10_repro_s0_step138534_ema_s0.json` | `coverage_at_20.per_row.found` 19,220 행 전수 일치, 불일치 0 행 · totals 전수 일치 |
| nq10 | `results/figdata/nq10_perpuzzle.csv` (unique 6,513 행) | `found_n1` 6129/6129 · `found_n5` 12320/12320 · `found_n10` 14793/14793 · `found_n20` 16586/16586 불일치 0 행 · `valid_n1` 불일치 0 행 · `n_solutions` 전수 일치 |

② coverage 대조 — 기존 JSON 의 `coverage_at_20.coverage` 는 행 축 macro 이므로 micro 대조는
`found_mean / total_mean` 으로 한다.

| 대상 | 기존 (found_mean/total_mean) | 새 덤프 sum/sum |
|---|---|---|
| nq8 | 0.8694725626 | 0.8694725626 |
| nq8_s1 | 0.8681406500 | 0.8681406500 |
| nq10 | 0.5196316096 | 0.5196316096 |

③ `n_found_gt_total` = 0 (12 파일 전부).  기존 기준 JSON 도 0.

추가 확인: N 간 단조성 `found_N1 <= found_N5 <= found_N10 <= found_N20` 위반 0 행,
`found_N1 == n_valid_consistent_N1` True, `first_valid` 가 4 개 N 파일에서 동일.

## 독립 실행에 대한 사실

각 N 은 별도 프로세스에서 K=N 으로만 rollout 했다 (prefix 자르기 아님).
`experiments/gram_harness.py:152` 가 표본마다 `torch.random.manual_seed(seed * 100003 + k)` 로
재시드하므로, 같은 harness seed 에서 표본 k 는 N 값과 무관하게 동일하다.
위의 단조성 위반 0 행과 `first_valid` 동일이 그 결과다.

## 별건

`reports/RESULT_found_gt_total_50.md` 참조.
