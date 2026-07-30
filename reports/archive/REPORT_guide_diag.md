# REPORT_guide_diag.md — guide(prior_mu) 생사 진단 (29번 hinton, GPU 7)

- 일시: 2026-07-28
- 서버: **hinton**, repo `/home/chaeun/gram-repro`, conda env `gram`
- 사용 GPU: **7번 1장** (0~3 학습 4-run, 4~6 `.live` 스캔 — 전부 미접촉)
- 학습 없음. 기존 checkpoint inference 측정만.

**GPU 확보 경위**: 지시 시점에 유휴 GPU가 0장이었다. 0~3 은 학습이라 손댈 수 없고,
4~7 은 내가 건 `.live` 스캔(측정)이었으므로 **GPU 7 의 잔여 5 job 을 4/5/6 에 재분배**해
7번을 비웠다. 도는 학습에는 접근하지 않았다.

---

## 1. 사용 checkpoint

| 라인 | run | step | 가중치 | 선택 근거 |
|---|---|---|---|---|
| **OFF** | `nq8_off_b220_ema6k_s0` | **26197** | **`.live` (raw SGD)** | `.live` 곡선 30/30 스캔 완료, 그 최댓값 (78.48%) |
| **ON** | `nq8_on_b200_ema` | **15946** | **`.live` (raw SGD)** | `.live` 곡선 15/15 완료, 그 최댓값 (27.71%) |

**지시받은 ON 비교군 `nq8_on_b160_ema_s0` 은 이 서버에 없다.** 이 박스의 β=1.6 run 은
`nq8_on_b160`(EMA off) 하나뿐이라 `.live` 파일이 존재하지 않는다(EMA off 면 `.live` 를 저장하지 않음).
1.6 은 50번 담당분이다. 그래서 **이 서버에서 `.live` 곡선이 완비되고 peak 가 가장 높은 ON run**
(`nq8_on_b200_ema`)으로 대체했다. 임의 결정이며 5절에 명시한다.

OFF 쪽은 마침 진단 착수 시점에 `.live` 스캔이 30/30 으로 끝나 peak(26197)가 확정된 상태였다.

---

## 2. Task 1 — 3모드 harness (N=20, test 772 전수, harness seed 0/1)

모드 정의 (`gram.py:297-300` 의 `reparameterize`: `std = exp(0.5·logvar)·sigma_scale; return mu + eps·std`):

- **(a) 정상** — 기본값 (`mu_scale=1, sigma_scale=1`)
- **(b) μ=0** (stochasticity only) — `--mu-scale 0` → `eps = σ·noise`
- **(c) σ=0** (guide only) — `--sigma-scale 0` → `eps = μ`, 결정론적

### 2-1. OFF: `nq8_off_b220_ema6k_s0` / `step_26197.live`

| 모드 | prior valid (s0 / s1) | **평균** | pass@20 | distinct@20 | queens (min–max) | posterior† |
|---|---|---|---|---|---|---|
| (a) 정상 | 78.48 / 78.90% | **78.69%** | 100.00 / 100.00% | 4.75 | 7.93 (5–9) | 100.00% |
| **(b) μ=0** | 84.25 / 84.47% | **84.36%** | 100.00 / 99.87% | 4.11 | 7.98 (3–10) | 84.59 / 83.16% |
| (c) σ=0 | 82.90 / 82.90% | **82.90%** | 82.90 / 82.90% | **1.00** | 7.90 (6–8) | 100.00% |

### 2-2. ON: `nq8_on_b200_ema` / `step_15946.live`

| 모드 | prior valid (s0 / s1) | **평균** | pass@20 | distinct@20 | queens (min–max) | posterior† |
|---|---|---|---|---|---|---|
| (a) 정상 | 27.71 / 27.37% | **27.54%** | 97.41 / 98.06% | 12.84 | 8.66 (5–32) | 100.00% |
| **(b) μ=0** | 29.04 / 28.63% | **28.84%** | 97.67 / 98.58% | 12.37 | 8.53 (6–33) | 32.12 / 27.33% |
| (c) σ=0 | 23.58 / 23.58% | **23.58%** | 23.58 / 23.58% | **1.00** | 15.01 (7–33) | 100.00% |

† posterior 열의 (b) 값이 낮은 것은 **측정 아티팩트**다. `--mu-scale` 을 prior/posterior 두 rollout 에
같이 넘겨서 posterior 의 `mu_q` 까지 0 이 됐다. 지시대로 이 진단은 prior 경로만 보므로 판정에 쓰지 않는다.
(a)·(c) 의 posterior 는 정상값이다.

### 2-3. 논문 Table 3b 병기

| variant | 논문 N-Queens | **OFF b220 (우리)** | **ON b200 (우리)** |
|---|---|---|---|
| full GRAM | 99.69 | 78.69 | 27.54 |
| stochasticity only (μ=0) | **50.27** | **84.36** | **28.84** |
| guide only (σ=0) | **0.00** | **82.90** | **23.58** |

`distinct@20 = 1.00` 은 (c) 가 실제로 결정론이 됐음을 확인해 준다(20샘플이 전부 동일 보드).
그래서 (c) 의 pass@20 은 single-sample valid 와 같은 값이다.

---

## 3. Task 2 — μ·σ 크기 실측

`reparameterize` 호출을 가로채 기록. prior 경로, `no_grad`(inference_mode), 측정 후 원복.
`H_cycles = 3` 이므로 **transition 은 3개**다 (`gram.py:378` 무-grad 루프 2회 + 최종 1회).
지시의 "K=4" 와 다르며, 실측값 3 으로 보고한다.

- `‖μ‖` : per-token L2 norm (hidden 512차원 전체), 배치·토큰 평균
- `‖σ‖` : per-token L2 norm of std vector = `sqrt(Σ_i σ_i²)` — 노이즈 항 `σ⊙ε` 의 기대 노름 스케일.
  `‖μ‖` 와 같은 축이라 직접 비교 가능하다. (원소별 평균 `σ_elem` 은 축이 달라 참고용으로만 병기.)

### OFF `nq8_off_b220_ema6k_s0` / `step_26197.live` (호출 48회 = 16 ACT step × 3 transition)

| transition | `‖μ‖` | `‖σ‖` | **`‖μ‖/‖σ‖`** | σ_elem |
|---|---|---|---|---|
| 0 | 0.6459 | 19.3276 | **0.0334** | 0.85326 |
| 1 | 0.6560 | 19.4751 | **0.0337** | 0.85956 |
| 2 | 0.6621 | 19.4801 | **0.0340** | 0.85962 |

### ON `nq8_on_b200_ema` / `step_15946.live`

| transition | `‖μ‖` | `‖σ‖` | **`‖μ‖/‖σ‖`** | σ_elem |
|---|---|---|---|---|
| 0 | 0.1509 | 14.9182 | **0.0101** | 0.65871 |
| 1 | 0.1494 | 14.9831 | **0.0100** | 0.65877 |
| 2 | 0.1455 | 14.9174 | **0.0098** | 0.65877 |

**guide 의 크기는 노이즈의 1~3.4% 다.** transition 깊이에 따른 변화는 거의 없다
(OFF 0.0334→0.0340, ON 0.0101→0.0098).

---

## 4. Task 3 — solution-count 버킷 분해 ((a) 정상 모드, harness seed 0)

sidecar(`data/nqueens-8/sidecar_8.json`) 조회 성공률 **100.00%** (772/772).
키는 `sha1(flat token board uint8 bytes)[:16]`.

### OFF `nq8_off_b220_ema6k_s0` / `step_26197.live`

| 해 개수 버킷 | 문제 수 | valid rate | pass@20 |
|---|---|---|---|
| 1 | 494 | 77.28% | 100.00% |
| 2–5 | 267 | **80.75%** | 100.00% |
| 6–20 | 11 | 77.73% | 100.00% |
| 21+ | **0** | — | — |

### ON `nq8_on_b200_ema` / `step_15946.live`

| 해 개수 버킷 | 문제 수 | valid rate | pass@20 |
|---|---|---|---|
| 1 | 494 | 26.19% | 96.76% |
| 2–5 | 267 | **30.71%** | 98.88% |
| 6–20 | 11 | 22.73% | 90.91% |
| 21+ | **0** | — | — |

**다해 버킷 급락 없음.** 두 checkpoint 모두 2–5 버킷이 오히려 단해(1)보다 높다.
6–20 버킷은 문제 수가 11개뿐이라 표본이 작다. **21+ 버킷은 test 셋에 문제가 0개**라 측정 불가다.

---

## 5. 판정 기준 대비 (수치만)

지시가 미리 고정한 표에 대입한다.

| 기준 | 조건 | 측정값 | 해당 여부 |
|---|---|---|---|
| 가설 지지 | (a)≈(b) 차이 <5%p **그리고** ‖μ‖≪σ | OFF 차이 **+5.67%p**, ON 차이 **+1.30%p**; ‖μ‖/‖σ‖ = 0.033 / 0.010 | ON 은 조건 충족. OFF 는 5.67%p 로 5%p 를 0.67%p 넘음 |
| 가설 기각 | (a)≫(b) 차이 >15%p | OFF −5.67%p, ON −1.30%p (**(b)가 더 높음**) | 해당 없음 |
| 중간 | guide 부분 작동 | — | 아래 참조 |
| 논문 정합 | (c)≈0% | OFF **82.90%**, ON **23.58%** | **해당 없음 — 크게 벗어남** |
| 국소화 | 다해 버킷만 급락 | 1 / 2–5 / 6–20 = 77.3 / 80.8 / 77.7 (OFF), 26.2 / 30.7 / 22.7 (ON) | 해당 없음 |

**기준표에 없는 결과가 나왔다**: 두 checkpoint 모두 **(b) μ=0 이 (a) 정상보다 높다**
(OFF +5.67%p, ON +1.30%p). 기준표는 "(a)≈(b)" 또는 "(a)≫(b)" 만 상정했고
"(b)>(a)" 행이 없다. 차이의 크기는 OFF 가 5%p 문턱을 근소하게 넘지만 **부호가 반대**이므로,
"guide 가 기여한다"는 방향의 증거는 아니다.

수치를 그대로 적으면:
- guide 제거(μ=0)는 prior valid 를 **떨어뜨리지 않았다.** 두 라인 모두 소폭 상승했다.
- guide 크기는 노이즈의 **1.0~3.4%** 다.
- σ=0(guide only)은 논문의 0.00 과 달리 **82.90% / 23.58%** 다. 이 모델은 노이즈를 빼도
  단일 출력이 유효 보드를 낸다. 다만 `distinct@20 = 1.00` 이라 **다양성이 0** 이고,
  그래서 pass@20 이 single 값에 묶인다((a) 는 pass@20 100% / 97.4%).
- 즉 이 checkpoint 들에서 노이즈의 관측된 역할은 **단일 정확도 향상이 아니라 다양성 공급**이다
  (OFF: distinct 1.00 → 4.75, pass@20 82.90% → 100.00%).

**해석은 사용자 몫이다. 위 문턱 대입 이상으로 단정하지 않는다.**

### 확인 못 한 것
1. **논문 ablation 과의 조건 차이** — 논문의 50.27 / 0.00 은 μ=0 / σ=0 으로 **학습까지 한** 모델이고,
   우리는 μ 를 학습한 모델을 **inference 시점에만** 0 으로 만든 것이다. 같은 축의 비교가 아니다.
2. **다른 checkpoint 로의 일반화** — 2개만 쟀다. β·라인·step 이 다르면 달라질 수 있다.
3. **21+ 해 버킷** — test 셋에 해당 문제가 0개라 Fig 7 의 극단 구간은 측정 불가.
4. **(b) 에서 posterior 가 낮은 것** — 측정 아티팩트(2절 †). 별도로 재보지 않았다.

---

## 6. 코드 변경 diff + 원복 여부

**`models/recursive_reasoning/gram.py` 는 건드리지 않았다.** 진단 시점에 GPU 0~3 에서 학습 4-run 이
이 모듈을 쓰고 있었기 때문이다. 변경은 전부 `experiments/` 안이며 **기본값에서 기존 동작 불변**이다.
**원복하지 않고 남겼다** (기본값 경로가 bit-identical 이므로).

### `experiments/gram_harness.py` — `mu_scale` 컨텍스트매니저 추가

```diff
+@contextmanager
+def mu_scale(net, scale):
+    """Scale the guide (mu) before sampling.  scale=0 -> "stochasticity only"
+    (eps = sigma * noise, guide removed).  scale=1 (default) is a no-op wrapper
+    that returns exactly what reparameterize returned before, so the normal path
+    stays bit-identical.
+    """
+    if float(scale) == 1.0:                 # no-op: do not even wrap
+        yield
+        return
+    inner = net.model.inner
+    orig = inner.reparameterize
+    def patched(mu, logvar):
+        return orig(mu * float(scale), logvar)
+    inner.reparameterize = patched
+    try:
+        yield
+    finally:
+        inner.reparameterize = orig
```
```diff
 def rollout(net, inputs, labels, K, sigma_scale_=1.0, mode="prior",
-            batch_size=512, seed=0, verbose=True):
+            batch_size=512, seed=0, verbose=True, mu_scale_=1.0):
...
-    with sigma_scale(net, sigma_scale_):
+    with sigma_scale(net, sigma_scale_), mu_scale(net, mu_scale_):
```

`scale == 1.0` 이면 래핑조차 하지 않으므로 **기존 호출은 monkeypatch 를 거치지 않는다.**

### `experiments/sweep_harness.py` — `--mu-scale` 인자

```diff
+    # Table 3b ablation: --mu-scale 0 = "stochasticity only", --sigma-scale 0 = "guide only".
+    ap.add_argument("--mu-scale", type=float, default=1.0)
...
-                  mode="prior", seed=args.seed, verbose=False)
+                  mode="prior", seed=args.seed, verbose=False, mu_scale_=args.mu_scale)
-                   mode="posterior", seed=args.seed, verbose=False)
+                   mode="posterior", seed=args.seed, verbose=False, mu_scale_=args.mu_scale)
-           "seed": args.seed, "sigma_scale": args.sigma_scale, "tag": args.tag}
+           "seed": args.seed, "sigma_scale": args.sigma_scale, "mu_scale": args.mu_scale,
+           "tag": args.tag}
```

### 모드 (c) σ=0 — **신규 코드 없음**
`gram.py:231` 의 `self.sigma_scale` 과 `gram_harness.py:119-128` 의 기존 `sigma_scale()`
컨텍스트매니저, `sweep_harness.py --sigma-scale` 인자를 그대로 썼다. `--sigma-scale 0`.

### `experiments/diag_guide.py` — 신규 파일 (Task 2·3 전용)
`reparameterize` 를 임시 패치해 μ/σ 노름을 기록하고 `finally` 에서 원복한다(3절).
버킷 분해는 sidecar 해시 조회 + `verify_boards`. 모델·학습 코드는 건드리지 않는다.

### 결과 파일
`experiments/results/guide_<tag>_<mode>_seed<S>.json` (9개),
`experiments/results/guide_diag_<tag>.json` (2개).
(a) 정상 seed 0 은 기존 `live_*.json` 을 재사용했다 — 기본값이 bit-identical 이므로 동일 측정이다.
