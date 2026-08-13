# [50 별건] `n_found_gt_total` — coverage_at_N vs coverage_at_20

작성 2026-08-11.  값과 경로만 적는다.

## 1. 두 필드의 sidecar (줄번호)

| 필드 | 정의 파일:줄 | 값 주입 | sidecar 인자 default | 실행 시 실제 값 |
|---|---|---|---|---|
| `coverage_at_20` | `/workspace/TinyRecursiveModels/experiments/sweep_harness.py:52-92` | `:138-139` `coverage_at_N(cand, inputs, args.n, solution_counts(inputs, args.sidecar))` | `:106` `data/nqueens-8/sidecar_8.json` | nq10 실행은 `--sidecar data/nqueens-10/sidecar_10.json` |
| `coverage_at_N` | `/workspace/fprm/experiments/fprm_harness.py:89-131` | `:355-356` `coverage_at_N(cand, inputs, args.n, solution_counts(inputs, args.sidecar))` | `:277` `data/nqueens-10/sidecar_10.json` | JSON 기록 `"sidecar": "data/nqueens-10/sidecar_10.json"` |

`solution_counts` 본체: `sweep_harness.py:38-46` / `fprm_harness.py:75-87`.
둘 다 `json.load(open(sidecar))["inputs"][sha1(x)[:16]]` 의 `len(e["solutions"])` 이다.
`experiments/solution_counts.py:36` 의 `len(e[field])` 과 같은 정의.
`experiments/coverage_curve.py:12` 는 `sweep_harness.solution_counts` 를 그대로 import 한다.

파일 동일성:

```
/workspace/fprm/data/nqueens-10 -> /workspace/TinyRecursiveModels/data/nqueens-10   (symlink)
md5 c3ac619c874cabf2837cf7223303eccb  data/nqueens-10/sidecar_10.json
md5 def8aac8f11ac931e7004cac505d509d  data/nqueens-8/sidecar_8.json
```

→ 두 필드는 같은 파일을 읽는다.

## 2. 169 건은 저장된 산출물에 없다

`/workspace/**/*.json` 전수 스캔에서 `n_found_gt_total` 최대값:

| 값 | 파일 | 필드 |
|---|---|---|
| 50 | `TinyRecursiveModels/experiments/results/v4s_nq8_v4_add_b007_s2_step_68340_live_s1.json` | `coverage_at_20` |
| 43 | `fprm/results/fprm_harness/S3_ema.json` | `coverage_at_N` |
| 35 | `.../v4s_nq8_v4_add_b007_s2_step_68340_live_s0.json` | `coverage_at_20` |
| 30 | `.../fin_nq8_final_add_s1_step59936_live_s1.json` | `coverage_at_20` |
| 25 | `TinyRecursiveModels/reports/g3/nq10_repro_s0_mu0_sg1.json` | `coverage_at_20` |

`coverage_at_N` 을 가진 파일은 40 개, 그 중 0 이 아닌 것은 7 개 (43 / 15 / 12 / 9 / 7 / 1 / 1).
S3_ema.json = shard 4 개 합 (7+9+15+12 = 43).  `"169"` 문자열은 json/md/py 어디에도 없다.

## 3. 43 건의 내역 (`fprm/results/fprm_harness/S3_ema_shard*of4.json` 의 `coverage_at_N.per_row`)

checkpoint `checkpoints/fprm-nqueens10/nq10_fprm_S3_conv16_s0/step_138534`, ema, N=20, seed 0,
batch_size 192, max_iter_eval 1000, 19220 행.

(n_solutions, found) 분포: (1,2) 38 건 · (2,3) 2 건 · (1,3) 2 건 · (3,4) 1 건.  ihash 43 개 전부 서로 다름.

ihash 3 개:

| ihash | shard/row | clues | sidecar 해 개수 | 브루트포스 해 개수 | found |
|---|---|---|---|---|---|
| `14eff4632c56e44b` | 0 / 1740 | (0,6) (5,2) (8,8) | 1 | 1 | 2 |
| `1769c09b70d89deb` | 0 / 1879 | (0,9) (7,7) (9,4) | 1 | 1 | 2 |
| `1c0086ce7a1d507b` | 0 / 2206 | (0,1) (4,2) (9,4) | 1 | 1 | 2 |

브루트포스 = 10-queens 전체 해 724 개를 열거해 clue 로 거른 개수.
43 건 전부 sidecar 개수 == 브루트포스 개수 (불일치 0/43).  → sidecar 는 부족하지 않다.

## 4. 원인

`found` 는 ★토큰 벡터★ 를 유일화한다:

```
sweep_harness.py:60   len({tuple(cand[k, b]) for k in range(N) if ok[k, b]})
fprm_harness.py       (동일 식)
```

판정은 퀸 칸만 읽는다:

```
evaluators/nqueens.py:44-45   pq = preds.reshape(B,n,n) == QUEEN(2)
evaluators/nqueens.py:65      valid = count_ok & row_ok & col_ok & d1_ok & d2_ok
evaluators/nqueens.py:68      consistent = (~iq | pq).all(-1)
```

빈칸 토큰 값은 판정에 안 들어간다.  같은 해 보드에서 빈칸 하나만 `1 -> 0` 으로 바꿔도
valid·consistent 는 그대로 True 이고 토큰 벡터는 달라진다 (직접 확인: 세 벡터 모두
valid=True consistent=True, 서로 다른 토큰 벡터 3 · 서로 다른 퀸 집합 1).

실측 (`experiments/found_dedupe_probe.py`, `fprm/experiments/found_dedupe_probe_fprm.py`):

| 체크포인트 | 표본 | 예측 토큰 값 분포 (전체) | valid∧consistent 보드의 토큰 분포 | found(토큰) vs found(퀸집합) |
|---|---|---|---|---|
| `nq10_repro_s0/step_138534` ema | 앞 2048 행 × N=20 | `{1: 3,692,106, 2: 403,894}` | `{1: 3,530,610, 2: 392,290}` | 11681 vs 11681, 차이 행 0 |
| `nq10_fprm_S3_conv16_s0/step_138534` ema | 앞 192 행 × N=20 | `{0: 1425, 1: 358,341, 2: 24,234}` | `{0: 9, 1: 30,231, 2: 3,360}` | 61 vs 57, 차이 행 4 |

산출물: `reports/found_dedupe_nq10_repro_s0.json` · `/workspace/fprm/results/found_dedupe_S3_ema_first192.json`.

S3 는 빈칸에 PAD(0) 를 섞어 낸다.  퀸 배치가 같은 보드가 서로 다른 토큰 벡터로 두 번 세어지고,
n_solutions 가 1~3 으로 작은 퍼즐에서 그 초과분이 total 을 넘어 `n_found_gt_total` 로 잡힌다.
`nq10_repro_s0` 는 토큰이 {1,2} 뿐이라 같은 자리에서 0 건이다.
