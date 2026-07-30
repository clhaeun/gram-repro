# REPORT_nograd.md — `torch.no_grad()`의 출처: TRM 상속인가 우리 추가인가

- 일시: 2026-07-27
- 브랜치 `main`, HEAD `4c847f1`
- **코드 수정 없음. 학습 없음. GPU 미사용.** 전부 읽기/grep/diff/git.
- 대조 기준: `/tmp/trm_repo` (`git clone --depth 1 https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git`)
  → clone된 HEAD가 `c011037 Clarify update message in README`인데,
  **이것이 이 저장소의 부모 커밋과 동일 커밋이다**(§3). 즉 이 repo는 TRM의 fork이고,
  upstream 원본을 추측 없이 정확히 집어낼 수 있다.

---

## 1. Task 1 — `gram.py`의 `no_grad` 전수 식별

```
$ grep -n "no_grad\|inference_mode" models/recursive_reasoning/gram.py
178:        with torch.no_grad():
253:    @torch.no_grad()
326:        with torch.no_grad():
444:        with torch.no_grad():
```

**4개다** (지시문은 3종을 예상했다). 분류:

### ① `gram.py:178-180` — Q-head 초기화. **분류: 위 3종 어디도 아님 (init 전용).**
```python
176        # Q head special init
177        # Init Q to (almost) zero for faster learning during bootstrapping
178        with torch.no_grad():
179            self.q_head.weight.zero_()
180            self.q_head.bias.fill_(-5)  # type: ignore
```
`__init__` 안. 학습 gradient 경로와 무관. TRM `trm.py:156-160`과 **주석까지 동일**.

### ② `gram.py:253-254` — `_latent_diag` 데코레이터. **분류: 2번(계측 전용, 무관).**
```python
253    @torch.no_grad()
254    def _latent_diag(self, mu_p, logvar_p, mu_q, logvar_q, kl):
```
로깅 전용. `gram.py:382`에서만 호출되고 반환값은 metrics로만 나간다. TRM에 대응물 없음(우리 추가).

### ③ `gram.py:326-350` — **forward의 recursive 루프. 분류: 1번. 이것이 문제의 그 no_grad다.**
```python
325        # H_cycles-1 without grad
326        with torch.no_grad():
327            for _H_step in range(self.config.H_cycles-1):
328                for _L_step in range(self.config.L_cycles):
329                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
330
331                z_H = self.L_level(z_H, z_L, **seq_info)
332
333                mu_p = self.prior_mu(z_H)  # 차원 크기 맞춰주기
334                logvar_p = self.prior_logvar(z_H) + self.prior_logvar_bias
335
336                if use_posterior:
337                    u_y = z_H + y_emb  # concat하기
338                    mu_q = self.post_mu(u_y)
339                    logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
340                    eps = self.reparameterize(mu_q, logvar_q)
341
342                    # 계측 전용: 이 transition은 학습 시 KL 페널티를 받지 않는다 (Eq 14)
343                    if trace_kl:
344                        kl_t = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
345                        self._kl_trace.append(kl_t.sum(-1).mean().item())
346                else:
347                    eps = self.reparameterize(mu_p, logvar_p)
348
349                z_H = z_H + eps
350                lrpm_z.append(z_H[:, 0])
```
대응하는 grad 경로 (`gram.py:352-372`):
```python
352        # 1 with grad
353        for _L_step in range(self.config.L_cycles):
354            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
355
356        z_H = self.L_level(z_H, z_L, **seq_info)
357
358        mu_p = self.prior_mu(z_H)
359        logvar_p = self.prior_logvar(z_H) + self.prior_logvar_bias
360
361        if use_posterior:
362            u_y = z_H + y_emb
363            mu_q = self.post_mu(u_y)
364            logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
365            eps = self.reparameterize(mu_q, logvar_q)
366            kl = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
```
→ **`kl`이 손실로 나가는 곳은 366행 하나뿐**이다. 326 블록 안의 `kl_t`(344행)는
`trace_kl`이 켜졌을 때만 계산되고 `.item()`으로 리스트에 append될 뿐 그래프에 안 걸린다.

### ④ `gram.py:444` — ACT halting. **분류: 위 3종 어디도 아님 (ACT wrapper).**
```python
444        with torch.no_grad():
445            # Step
446            new_steps = new_steps + 1
447            is_last_step = new_steps >= self.config.halt_max_steps
```
`TinyRecursiveReasoningModel_ACTV1.forward`(래퍼) 안. TRM `trm.py:267-271`과 동일.

**지시문 3분류에 대한 정정 2건**
- 지시문의 "3. trace_kl 계측 부분"은 **독립된 `no_grad`가 아니다.** trace_kl(343-345)은
  ③의 상속된 no_grad 블록 *안에* 얹힌 코드다.
- 지시문에 없던 `no_grad` 2개(① q-head init, ④ ACT halting)가 실재하며 둘 다 TRM 것이다.

---

## 2. Task 2 — TRM 원본과 대조 (결정적 증거)

```
$ grep -n "no_grad" /tmp/trm_repo/models/recursive_reasoning/trm.py
158:        with torch.no_grad():     <- q-head init      (GRAM 178)
208:        with torch.no_grad():     <- recursive 루프    (GRAM 326)
267:        with torch.no_grad():     <- ACT halting      (GRAM 444)
```
TRM 3개 ↔ GRAM 4개. **차이 1개 = `_latent_diag`의 `@torch.no_grad()`(GRAM 253, 계측 전용).**

### TRM `trm.py:204-216` (원본 그대로)
```python
204        # Forward iterations
205        it = 0
206        z_H, z_L = carry.z_H, carry.z_L
207        # H_cycles-1 without grad
208        with torch.no_grad():
209            for _H_step in range(self.config.H_cycles-1):
210                for _L_step in range(self.config.L_cycles):
211                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
212                z_H = self.L_level(z_H, z_L, **seq_info)
213        # 1 with grad
214        for _L_step in range(self.config.L_cycles):
215            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
216        z_H = self.L_level(z_H, z_L, **seq_info)
```

### `diff -u` (TRM 204-216 → GRAM 320-356)
```diff
         # Forward iterations
         it = 0
         z_H, z_L = carry.z_H, carry.z_L
+        lrpm_z = []
+
         # H_cycles-1 without grad
         with torch.no_grad():
             for _H_step in range(self.config.H_cycles-1):
                 for _L_step in range(self.config.L_cycles):
                     z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
+
                 z_H = self.L_level(z_H, z_L, **seq_info)
+
+                mu_p = self.prior_mu(z_H)  # 차원 크기 맞춰주기
+                logvar_p = self.prior_logvar(z_H) + self.prior_logvar_bias
+
+                if use_posterior:
+                    u_y = z_H + y_emb  # concat하기
+                    mu_q = self.post_mu(u_y)
+                    logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
+                    eps = self.reparameterize(mu_q, logvar_q)
+
+                    # 계측 전용: 이 transition은 학습 시 KL 페널티를 받지 않는다 (Eq 14)
+                    if trace_kl:
+                        kl_t = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
+                        self._kl_trace.append(kl_t.sum(-1).mean().item())
+                else:
+                    eps = self.reparameterize(mu_p, logvar_p)
+
+                z_H = z_H + eps
+                lrpm_z.append(z_H[:, 0])
+
         # 1 with grad
         for _L_step in range(self.config.L_cycles):
             z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
+
         z_H = self.L_level(z_H, z_L, **seq_info)
```

**`-` 줄이 하나도 없다.** TRM의 루프 골격은 한 글자도 삭제·수정되지 않았고,
GRAM은 그 안에 줄을 **삽입만** 했다. 구체적으로:
- `with torch.no_grad():`, `for _H_step in range(H_cycles-1)`, `# H_cycles-1 without grad`,
  `# 1 with grad` — **주석 문자열까지 TRM과 동일**.
- 같은 관용구가 TRM의 다른 변형에도 있다:
  `trm_singlez.py:203-209`, `trm_hier6.py:229-235` (둘 다 `# H_cycles-1 without grad` + `with torch.no_grad():`).
  → TRM 계열의 표준 구조지 우리가 만든 것이 아니다.

**판정: `gram.py:326`의 `no_grad`(및 H_cycles-1 / 1-with-grad 분할)는 TRM 상속이다.**

**단, 같은 무게로 기록해야 할 구분이 있다.**
TRM에는 잠재변수가 아예 없다 — TRM의 no_grad 블록 안에는 `L_level` 호출만 있다.
**prior/posterior 샘플링과 `z_H = z_H + eps`를 그 상속된 no_grad 블록 *안에* 넣은 것은
GRAM 쪽 코드의 배치 결정**이고, TRM에는 대응물이 없다(diff의 `+` 줄 전부).
즉 직전 진단에서 확인된 "y가 무페널티 trans0으로 흐른다"는 경로는
**상속된 no_grad**와 **우리 저장소가 그 안에 넣은 잠재 주입**의 조합에서 나온다.
이 배치가 GRAM 논문 저자 구현과 같은지는 **확인 불가**(§5).

---

## 3. Task 3 — git 이력 교차 확인

```
$ git log --oneline -- models/recursive_reasoning/gram.py
4c847f1 GRAM reproduction

$ git log --oneline -S "no_grad" -- models/recursive_reasoning/gram.py
4c847f1 GRAM reproduction

$ git show --stat --oneline 4c847f1 -- models/recursive_reasoning/
4c847f1 GRAM reproduction
 models/recursive_reasoning/gram.py      | 474 ++++++++++++++++++++++++++++++++
 models/recursive_reasoning/gram_0722.py | 455 ++++++++++++++++++++++++++++++
 2 files changed, 929 insertions(+)

$ git cat-file -e c011037:models/recursive_reasoning/gram.py
fatal: path 'models/recursive_reasoning/gram.py' exists on disk, but not in 'c011037'

$ git ls-tree --name-only c011037 models/recursive_reasoning/
models/recursive_reasoning/hrm.py
models/recursive_reasoning/transformers_baseline.py
models/recursive_reasoning/trm.py
models/recursive_reasoning/trm_hier6.py
models/recursive_reasoning/trm_singlez.py
```

- `gram.py`는 **커밋 1개(`4c847f1`)에서 474줄 전체가 한 번에 추가**됐다. 삭제 0줄.
  → **git 이력만으로는 "최초부터 있었는가"를 가릴 수 없다.** GRAM 작업 전체가 단일 커밋이라
  중간 과정이 남아 있지 않다. 지시문이 기대한 "중간에 추가/이동했는지"는 **git으로는 확인 불가**.
  대신 §2의 upstream 구조 diff가 그 역할을 한다(그리고 결론은 동일하다).
- `c011037`(= clone한 TRM HEAD와 같은 커밋) 시점에 `gram.py`는 **없었다**.
  이 저장소가 TRM fork이고 `4c847f1`이 우리 작업 커밋임이 확정된다.

**저장소 안에 남은 GRAM 초기 산출물로 보강** — 셋 다 잠재 주입이 이미 no_grad 안에 있다:

| 파일 | 라인 | 내용 |
|---|---|---|
| `models/recursive_reasoning/gram_0722.py` | 312-332 | `# H_cycles-1 without grad` + `with torch.no_grad():` 안에 `prior_mu`/`post_mu`/`z_H = z_H + eps` |
| `exp/legacy/archive/gram_halt_only_as_run.py` | 265-285 | 동일 (분기가 `if self.training:`인 더 옛 형태) |
| `exp/legacy/archive/gram_halt_only_tts.py` | 274-294 | 동일 |

→ 저장소에 남아 있는 **가장 이른 GRAM 산출물부터 이미 이 배치**다. 중간에 옮긴 흔적은 없다.

---

## 4. Task 4 — 논문 Eq-14와 대조

**코드가 하는 일(측정·인용으로 확정된 것):**

| transition | grad | KL 손실 | 근거 |
|---|---|---|---|
| trans0, trans1 (H_cycles-1개) | **없음** | **없음** | `gram.py:326` no_grad 블록. 344행 `kl_t`는 `trace_kl` 시에만, `.item()`이라 그래프 밖 |
| trans2 (마지막 1개) | 있음 | **있음** | `gram.py:352-366`, `kl = self.kl_loss(...)` 366행 |

즉 코드는 **supervision step당 마지막 transition에만 gradient + KL**이다.

**논문 Eq-13/Eq-14 텍스트와의 대조는 이 세션에서 확인 불가.**
저장소에 GRAM 논문 PDF·본문이 없다:
```
$ find . \( -iname "*.pdf" -o -iname "*.tex" \) -not -path "./.git/*"   # 결과 없음
```
`README.md:8`의 `[Paper](https://arxiv.org/abs/2510.04871)`은 **TRM 논문**이고 GRAM 논문이 아니다.

따라서 "코드가 Eq-14와 정확히 일치하는가"에 대해 내가 확인할 수 있는 것은
**코드가 '마지막 transition만 grad+KL'을 한다는 사실**까지다.
그것이 논문 Eq-14의 진술과 일치하는지는 **논문 원문을 대조하지 못했으므로 확인 불가**.

중요: 저장소 곳곳의 "Eq 14" 라벨은 **전부 우리가 단 주석/문서**이지 논문 대조 결과가 아니다 —
`gram.py:342`, `experiments/trace_kl.py:3`, `REPORT_audit.md:86,90`, `REPORT_diag.md:36`,
`REPORT_contradiction.md:66,155,219`, `experiments/diag_contradiction.py:92`.
이 라벨들을 근거로 삼으면 순환 논증이 된다.

**조건부로 말할 수 있는 것**: 지시문이 요약한 Eq-14 정의(“supervision step당 마지막
transition만 gradient + KL, 앞 transition은 gradient truncation”)가 논문 그대로라면,
코드는 그 정의와 일치하고, y가 trans0으로 흐르는 것은 버그가 아니라 truncation의
구조적 귀결이다. **그 전제(논문 문구)를 내가 검증하지 못했다.**

---

## 5. Task 5 — 저자 구현 단서 (논문 A.3 / Fig 8)

**확인 불가.** 저장소에 GRAM 논문 파일이 없고(§4), `A.3`/`Fig 8`/`ELBO`를 저장소 전체에서
grep해도 GRAM 논문 관련 문자열은 나오지 않는다(히트는 전부 TRM 코드의 PQN 링크
`https://arxiv.org/abs/2407.04811`, 그리고 우리 자신의 REPORT_*.md 문장들이다).

저자 공개 구현체도 이 박스에 없다. 지시가 읽기 전용이라 웹에서 논문/구현을 받아오지 않았다.
→ "저자도 truncation을 썼는가"는 **이 세션에서 판단 근거 없음**.

---

## 6. 결론 (한 줄)

**`gram.py:326`의 `no_grad`는 TRM 상속이고**(upstream `trm.py:208`과 주석까지 동일, diff에 삭제 0줄,
`trm_singlez.py`/`trm_hier6.py`에도 같은 관용구), **코드는 "supervision step당 마지막 transition만
grad+KL"을 구현한다 — 이것이 논문 Eq-14인지는 저장소에 논문이 없어 확인 불가.**

덧붙일 단서 하나: **잠재 주입(`prior_mu`/`post_mu`/`z_H = z_H + eps`)을 그 상속된 no_grad
블록 안에 배치한 것은 TRM에 대응물이 없는 GRAM 측 코드**이며(§2 diff의 `+` 줄 전부),
저장소에 남은 가장 이른 GRAM 산출물부터 이미 그렇다(§3). 이 배치가 저자 구현과 같은지는 확인 불가.

### 다음 방향을 가르는 미해결 질문 (측정으로 답 못 한 것)
1. GRAM 논문 Eq-13/Eq-14의 실제 문구 — 논문 확보 필요. 이것이 "사수 질문 vs 코드 수정"을 가른다.
2. 저자 구현에서 잠재 주입이 no_grad 안인지 밖인지 — 공개 구현 확보 필요.
