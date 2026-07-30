# REPORT_contradiction.md — "KL≈0인데 posterior valid=100%" 모순 규명

- 일시: 2026-07-27
- 브랜치 `main`, HEAD `4c847f1`. 인터프리터 `/home/chaeun/venv-gram/bin/python`
- **GPU: 4번만 사용.** 0~3은 건드리지 않았다. 학습 launch **없음** (Task 3 미실행, 사유는 §3).
- 신규 파일: `experiments/diag_contradiction.py`
- 결과 json: `experiments/results/contradiction_gram_nq8_{fix_b,mean_nh512,mean1}.json`
- 학습 코드(`models/`, `pretrain.py`) **변경 없음**. 전부 런타임 monkeypatch/hook, 측정 후 원복(§5).

---

## 1. Task 1 — fix_b의 posterior valid

`experiments/diag_posterior.py`로 posterior/prior를 같은 스크립트에서 나란히 측정.
n=256 test puzzle, K=1, seed=0.

| run (step_17085) | 경로 | valid | consistent | strict | queen_count mean (min–max) |
|---|---|---|---|---|---|
| **gram_nq8_fix_b** | posterior | **1.0000** | 1.0000 | 1.0000 | 8.00 (8–8) |
| **gram_nq8_fix_b** | prior | **0.0000** | 0.0000 | 0.0000 | 10.57 (6–16) |
| gram_nq8_mean_nh512 | posterior | 1.0000 | 1.0000 | 1.0000 | 8.00 (8–8) |
| gram_nq8_mean_nh512 | prior | 0.0000 | 0.0000 | 0.0000 | 31.97 (26–39) |
| gram_nq8_mean1 | posterior | 1.0000 | 1.0000 | 1.0000 | 8.00 (8–8) |
| gram_nq8_mean1 | prior | 0.0000 | 0.0000 | 0.0000 | 1.57 (0–4) |

**판정: posterior 100%. 모순은 Task 1에서 해소되지 않는다.** 서로 다른 run의 관찰을
섞은 것이 아니었다 — `gram_nq8_fix_b` **단일 run**에서 posterior valid=100%와
`diag/kl_raw`≈0.0014가 동시에 성립한다. → Task 2로 진행.

---

## 2. Task 2 — ε 경로 검증

전부 `experiments/diag_contradiction.py`. posterior 경로, n=256(valid 측정) / B=64(KL·μ_q 측정),
sup_steps=16, seed=0.

### 2a. ε=0 ablation

`reparameterize`(`gram.py:247-250`)를 `lambda mu, logvar: mu`로 임시 치환.

| run | posterior baseline | posterior eps=0 | prior baseline | prior eps=0 |
|---|---|---|---|---|
| fix_b | 1.0000 | **1.0000** | 0.0000 | 0.0000 |
| mean_nh512 | 1.0000 | **1.0000** | 0.0000 | 0.0000 |
| mean1 | 1.0000 | **1.0000** | 0.0000 | 0.0000 |

세 run 모두 posterior valid가 그대로 1.0000. 원복 후 재측정도 1.0000.

**이 결과를 지시문의 판정 기준에 그대로 대입하면 안 된다.** 지시는
"valid 유지 → ε이 정보를 안 나른다"였는데, `gram.py:247-250`에서 `reparameterize`가
반환하는 값(코드상 변수명 `eps`)은 **노이즈가 아니라 잠재 표본 전체** `mu + noise·std`다:
```python
247      def reparameterize(self, mu, logvar):
248          std = torch.exp(0.5*logvar) * self.sigma_scale
249          eps = torch.randn_like(std)
250          return mu + eps * std
```
그리고 `gram.py:349,378`의 `z_H = z_H + eps`는 이 **표본 전체**를 더한다.
따라서 `reparameterize → mu`는 `mu_q`는 남기고 **노이즈 항만** 제거한다.
측정이 말하는 것은 "**노이즈 항은 정보를 안 나른다**"까지이고,
`mu_q`를 통한 y 전달 여부는 이 실험이 가르지 못한다. 그래서 2d를 추가했다.

### 2b. transition별 KL (`gram.py:343-345` no_grad / `gram.py:368-369` grad)

`H_cycles=3` → forward 1회당 transition 3개. **trans0/trans1은 `torch.no_grad()` 블록
(`gram.py:326-350`), trans2만 grad 경로**(`gram.py:353-369`)이고 **Eq 14의 KL 페널티는
trans2에만 걸린다**. 학습 로그의 `diag/kl_raw`도 trans2 하나다 —
`gram.py:382`가 grad 경로의 `kl`(366행)만 `_latent_diag`에 넘긴다:
```python
382          diag = self._latent_diag(mu_p, logvar_p, mu_q, logvar_q, kl) if use_posterior else None
```

sup_step × transition 격자 (KL, D축 합 / batch·L축 평균):

| run | sup0 trans0 | sup0 trans1 | sup0 trans2 | sup≥1 정상상태 (trans0/1/2) |
|---|---|---|---|---|
| **fix_b** | **13.8083** | 0.0025 | 0.0022 | ≈0.0023 / 0.0023 / 0.0023 |
| mean_nh512 | 40.4102 | 20.1510 | 3.0380 | ≈3.23 / 3.23 / 3.23 |
| mean1 | 31.5714 | 5.9252 | 4.9686 | ≈4.82 / 4.82 / 4.82 |

(전체 16×3 격자는 json의 `2b.grid`.)

- **fix_b**: KL이 0에 가까운 것은 사실이나, **sup_step 0의 transition 0에서만 13.8083**이다.
  나머지 47개 transition은 전부 0.0022–0.0025. 학습 로그가 보던 `diag/kl_raw`(=trans2)는
  0.0023이고, 이는 학습 최종 step의 0.0014와 같은 자릿수다.
- **mean_nh512 / mean1은 KL이 0이 아니다** (정상상태 3.23 / 4.82).
  "KL≈0" 관찰은 **fix_b 고유**이지 세 run 공통이 아니다.

### 2c. μ_q의 y 의존성

`post_mu`에 forward hook을 걸어, 같은 `z_H`(= `u_y − y_emb`로 복원)에 배치를 한 칸 굴린
다른 라벨 `y'`를 넣어 μ_q를 다시 계산했다.

- `rel_change_from_y = ‖μ_q(y) − μ_q(y′)‖ / ‖μ_q(y)‖`
- `rel_gap_q_vs_p = ‖μ_q − μ_p‖ / ‖μ_q‖`

| run | idx0 change_from_y | idx0 gap_q_vs_p | idx≥1 평균 change_from_y | idx≥1 평균 gap_q_vs_p |
|---|---|---|---|---|
| fix_b | 0.26969 | 0.46660 | **0.84559** | **0.00756** |
| mean_nh512 | 0.24352 | 1.16602 | **1.02758** | 0.73681 |
| mean1 | 0.28258 | 0.89585 | **0.40847** | 0.45604 |

**세 run 모두 μ_q는 y에 강하게 의존한다** (y를 바꾸면 μ_q가 자기 노름의 41–103%만큼 변한다).
`post_mu`가 y를 무시하는 것이 아니다.

fix_b에서 특징적인 것: idx≥1에서 `gap_q_vs_p = 0.0076`(q≈p, 그래서 KL≈0)인데
동시에 `change_from_y = 0.846`이다. 즉 **μ_q(y_true) ≈ μ_p 이면서, y를 바꾸면 μ_q가 크게 변한다.**
`μ_p = prior_mu(z_H)`이므로 이는 idx≥1 시점의 `z_H`가 이미 y와 정합한 상태라는 뜻이다.
(`‖z_H‖`는 전 구간 22.6285로 일정 — `mean_nh512`/`mean1`도 22.625–22.627.)

### 2d. transition 단위 y 게이팅 (내가 추가한 측정)

2a가 `mu_q` 경로를 가르지 못하므로, transition별로 q를 **p로 정확히 치환**하는 게이트를 만들었다.
`allow`에 없는 transition에서는
`μ_q := prior_mu(z_H)`, `logvar_q := prior_logvar(z_H) + prior_logvar_bias − post_logvar_bias`
로 바꾼다(`gram.py:339`가 호출 결과에 `post_logvar_bias`를 더하므로 미리 뺀다 → `gram.py:334`의
prior와 bit 수준으로 같은 값). 전역 transition 인덱스 0..47.

정확성 확인: `allow={}`(y 전면 차단)의 결과가 prior baseline과 일치해야 한다 →
fix_b queens 10.57 vs prior 10.57, mean1 1.57 vs 1.57. **일치.**

| allow | fix_b valid | mean_nh512 valid | mean1 valid |
|---|---|---|---|
| ALL (baseline) | 1.0000 | 1.0000 | 1.0000 |
| {} (y 전면 차단) | 0.0000 | 0.0000 | 0.0000 |
| **{0} (첫 transition만)** | **0.9961** | **0.0000** | **1.0000** |
| **ALL−{0} (첫것만 차단)** | **0.0039** | **1.0000** | **1.0000** |
| sup0 전체 {0,1,2} | 0.9961 | 0.9531 | 1.0000 |
| 원복 후 재측정 | 1.0000 | 1.0000 | 1.0000 |

- **fix_b**: transition 0 **하나만** y를 받으면 valid 0.9961, transition 0 **하나만** 막으면 0.0039.
  y 주입 지점이 transition 0에 **집중**돼 있다.
- **mean_nh512**: 정반대. transition 0만으로는 0.0000, transition 0을 빼도 1.0000.
- **mean1**: 둘 다 1.0000 (중복 주입).

---

## 3. Task 3 (β sweep) — 미실행

지시문의 발동 조건은 "Task 1에서 **posterior collapse로 판명되면**"이다.
Task 1 결과는 posterior valid = **1.0000**으로, 조건이 성립하지 않는다.
지시문 구조상 Task 2("Task 1이 posterior 100%일 때만")와 배타적 분기라
Task 2 쪽을 수행했다. GPU 학습은 돌리지 않았다. Task 4(재현성 교차 확인)도
"여유 시" 선택 항목이라 미실행 — 대신 mean1/nh512를 직접 측정해 §2 표에 병기했다.

---

## 4. 모순의 정체 (관측된 것만)

**모순은 fix_b에서 성립하며, 원인은 "KL이 0"과 "KL을 재는 지점"이 서로 다른 대상을 가리킨 것이다.**

측정된 사실만 나열하면:

1. `H_cycles=3`이므로 forward 1회에 latent transition이 3개 있다.
   그중 **KL 페널티를 받는 것은 trans2 하나**이고(`gram.py:353-369`, Eq 14),
   trans0·trans1은 `torch.no_grad()` 블록 안이라 **페널티도 gradient도 없다**(`gram.py:326-350`).
2. 학습 로그의 `diag/kl_raw`는 **trans2만** 본다(`gram.py:382`가 366행의 `kl`만 전달).
   fix_b의 trans2 KL = 0.0023 (학습 최종 step 로그 0.0014와 동일 자릿수).
3. 같은 fix_b에서 **sup_step 0 / trans0의 KL = 13.8083** — 계측되지 않던 지점이다.
4. y 게이팅(2d): fix_b는 **transition 0만 열어도 valid 0.9961**, **transition 0만 막으면 0.0039**.
5. μ_q는 y에 강하게 의존한다(2c, `rel_change_from_y` 0.85). `post_mu`가 y를 무시하는 것이 아니다.
6. transition 0 이후로는 `μ_q ≈ μ_p`(gap 0.0076)이면서도 μ_q는 여전히 y 의존적(0.85).

즉 fix_b에서 관측된 그림은: **y는 supervision step 0의 transition 0에서 z_H로 들어가고,
그 지점은 no_grad 블록이라 KL 페널티 대상이 아니며, 계측되던 "KL≈0"은 그 뒤의
페널티 대상 transition(trans2)의 값이다.** 두 관찰은 서로 다른 transition의 것이므로
애초에 모순이 아니다.

**단, 이 메커니즘은 세 run에 공통이 아니다.** mean_nh512는 정반대로 transition 0을
막아도 valid 1.0000이고 정상상태 KL이 3.23이다. mean1은 어느 쪽 하나만으로도 1.0000이다.
"y가 transition 0에 집중된다"는 fix_b에 대한 관측이지 일반 법칙으로 측정된 바 없다.

세 run에 공통인 것은: **posterior 100% / prior 0%**, **노이즈 항 제거해도 posterior 100%**,
**μ_q의 강한 y 의존성**, 그리고 **KL 페널티가 3개 transition 중 1개에만 걸린다는 코드 구조**.

---

## 5. 임의로 정한 것 / 원복한 임시 변경 전부

**임시 변경(전부 런타임, 파일 수정 없음, 컨텍스트 매니저로 원복):**

1. `inner.reparameterize` → `lambda mu, logvar: mu` (2a). `deterministic_latent()`에서 원복.
2. `inner.trace_kl = True` (2b). `finally`에서 `False`로 원복.
3. `inner.post_mu`에 forward hook (2c). `h.remove()`로 원복.
   hook 안에서 `mod(...)`를 다시 부르므로 재진입 플래그(`busy`)를 뒀다.
4. `inner._y_embeddings` / `inner.post_mu` / `inner.post_logvar` 치환 (2d).
   `post_mu`/`post_logvar`는 등록된 submodule이라 함수를 그대로 대입할 수 없어
   `nn.Module` 래퍼(`_Fn`)로 감싸 끼웠다. `y_gate()`의 `finally`에서 원래 모듈 복원.
5. 스크립트 끝에서 5개 전부 원상태인지 assert하고 통과를 출력한다
   (`[원복 확인] reparameterize / trace_kl / post_mu / post_logvar / _y_embeddings 전부 원상태.`).
   세 run 모두 통과. **디스크의 checkpoint/코드는 수정되지 않았다.**

**내가 정한 것:**

6. **2d를 추가했다.** 지시에 없는 측정이다. 2a가 `mu_q` 경로를 가르지 못한다는 것을
   §2a에 적은 이유로 판단해서 넣었다.
7. **2a의 판정 기준을 지시문 그대로 쓰지 않았다.** 지시는 "valid 유지 → ε이 정보를
   안 나른다 → y는 다른 경로로 흐른다(코드 재검토 필요)"였으나, `gram.py:250`의 반환값이
   노이즈가 아니라 `mu + noise·std`이므로 그 추론이 성립하지 않는다. 사실만 적었다.
8. **다른 y**는 배치를 1칸 roll한 것(`torch.roll(labels, 1, dims=0)`)으로 정했다 — 즉
   다른 퍼즐의 정답. 무작위 토큰이 아니라 실제 분포 안의 라벨이다.
9. **표본 수**: valid 측정 n=256, KL/μ_q 측정 B=64, sup_steps=16, seed=0.
   `diag_posterior.py`의 기존 기본값(256)에 맞췄다.
10. **비교 대상 run 2개(mean_nh512, mean1) 추가.** 지시는 fix_b만 요구했으나,
    "이전 run의 posterior 100%"가 같은 스크립트에서 재현되는지 확인하려고 함께 측정했다.
11. `_kl_trace`의 인덱스 0,1 = no_grad transition, 2 = grad transition이라는 대응은
    `gram.py`의 append 순서(345행 2회 → 369행 1회)에서 온 것이고, 별도 실측으로
    라벨을 확인하지는 않았다.

---

## 6. 다음 후보

측정이 가리키는 지점은 "β를 얼마로 하느냐"보다 "**KL을 어느 transition에 거느냐**"다.
fix_b에서 y가 들어간 transition 0은 β를 아무리 키워도 페널티를 받지 않는다
(`torch.no_grad()` 안이라 gradient 자체가 없다).

1. **KL을 전 transition에 거는 arm.** `gram.py`의 no_grad 루프를 grad 경로로 바꾸고
   3개 transition 전부에 KL을 합산. 논문 Eq 14 이탈이므로 별도 arm으로 표시 필요.
   비용 ~37분/run.
2. **β sweep (지시문 Task 3).** 발동 조건이 성립하지 않아 미실행. 돌린다면
   `RUN=gram_nq8_fix_c BETA=0.3 GPUS=4,5 PORT=29631 bash train_nqueens8_fix.sh`,
   `RUN=gram_nq8_fix_d BETA=1.0 GPUS=6,7 PORT=29641 bash train_nqueens8_fix.sh`.
   trans2에만 작용하므로 fix_b형 누출에는 직접 듣지 않을 수 있다 — mean_nh512형
   (누출이 trans≥1에 퍼진 경우)에는 작용할 여지가 있다.
3. **Task 4 재현성 교차 확인.** warmup 2000 + EMA on으로 mean1 설정을 재현.
   §2에서 mean1/nh512/fix_b의 누출 위치가 서로 달랐으므로, 이 메커니즘이
   run별로 흔들리는지 자체를 봐야 한다.

재측정 커맨드:
```bash
source ~/venv-gram/bin/activate
CUDA_VISIBLE_DEVICES=4 python experiments/diag_contradiction.py \
  --ckpt-dir checkpoints/Nqueens-8-ACT-torch/<RUN> --step step_17085 \
  --num 256 --kl-batch 64 --sup-steps 16 \
  --out experiments/results/contradiction_<RUN>.json
# --skip 2a,2c 등으로 부분 실행 가능
```
