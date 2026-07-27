# REPORT_yemb.md — posterior의 y 통과 진단: y_emb 스케일 + Sudoku 대조

- 일시: 2026-07-27
- 브랜치 `main`, HEAD `4c847f1`
- **학습 launch 없음. 아키텍처 수정 없음.** 전부 기존 checkpoint 측정.
- **GPU 4,5,6 만 사용. 0~3은 건드리지 않았다** (측정 내내 타 작업 447–462MiB 점유 중이었고 무관).
- 신규 파일: `experiments/diag_yemb.py` (런타임 hook, 측정 후 원복 + assert)
- 결과 json: `experiments/results/yemb_nq8_nh512.json`,
  `experiments/results/yemb_sudoku_halt_only.json`,
  `experiments/results/klgrid_sudoku_halt_only.json`

---

## 1. Task 1 — N-Queens에서 y_emb vs z_H 크기

### 코드: 두 임베딩은 **같은 스케일**을 쓴다

`gram.py:146` — 스케일 정의:
```python
146        self.embed_scale = math.sqrt(self.config.hidden_size)     # = sqrt(512) = 22.6274
147        embed_init_std = 1.0 / self.embed_scale
```
`gram.py:202-222` `_input_embeddings` 끝:
```python
222        return self.embed_scale * embedding
```
`gram.py:225-232` `_y_embeddings` 전문:
```python
225    def _y_embeddings(self, labels):
226        labels = torch.where(labels==IGNORE_LABEL_ID, 0, labels)
227        embedding = self.embed_tokens(labels.to(torch.int32))
228
229        if self.puzzle_emb_len > 0:
230            embedding = F.pad(embedding, (0, 0, self.puzzle_emb_len, 0))
231
232        return self.embed_scale * embedding
```

**둘 다 `self.embed_scale`(= √512)을 곱하고, 같은 `self.embed_tokens`를 공유한다.**
y만 따로 키우는 계수는 없다. 차이는 `_y_embeddings`가 앞쪽 `puzzle_emb_len=16` 위치를
0으로 pad 하는 것뿐(229-230행)이고, `_input_embeddings`는 그 자리에 puzzle embedding을
넣는다(`gram.py:214`).

### 측정: `u_y = z_H + y_emb` 지점의 노름

`gram_nq8_mean_nh512/step_17085`, test 128개, posterior 경로, 48개 transition 전부.
`post_mu`에 forward hook을 걸어 입력 `u_y`를 받고 `z_H = u_y − y_emb`로 분리했다.

| idx | sup | trans | ‖z_H‖ | ‖y_emb‖ | ratio | ‖z_H‖(non-pad) | ‖y_emb‖(non-pad) | **ratio(non-pad)** |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 22.6275 | 10.3783 | 0.4587 | 22.6279 | 12.9729 | **0.5733** |
| 1 | 0 | 1 | 22.6283 | 10.3783 | 0.4586 | 22.6285 | 12.9729 | 0.5733 |
| 2 | 0 | 2 | 22.6273 | 10.3783 | 0.4587 | 22.6273 | 12.9729 | 0.5733 |
| 3 | 1 | 0 | 22.6272 | 10.3783 | 0.4587 | 22.6272 | 12.9729 | 0.5733 |
| … | | | | | | | | |
| 47 | 15 | 2 | 22.6276 | 10.3783 | 0.4587 | 22.6276 | 12.9729 | 0.5733 |

- pad 위치 16/80. 전 토큰 평균(0.4587)은 그 0들 때문에 낮게 나오므로
  **non-pad 기준 0.5733**이 실제 비율이다.
- **48개 transition 전부 동일**하다(소수 4자리까지). trans0가 특별히 크지 않다.
- ‖z_H‖ = 22.627 = √512로 고정 — `L_level` 출력이 RMSNorm을 거치기 때문이다
  (`models/layers.py:163 rms_norm`). y_emb는 매 transition 같은 텐서(`gram.py:311`에서
  forward당 1회 계산)라 값이 안 변한다.

### 판정: **y_emb ≫ z_H 가설은 성립하지 않는다**

지시문 기준은 "y_emb ≫ z_H (예: 5× 이상) → 크기 문제".
실측은 **0.573×** — y_emb가 z_H보다 오히려 **작다**. 5×는커녕 1×도 안 된다.
→ 기준대로면 "크기 아님. 구조 문제" 쪽이다. 다만 Task 2가 이 결론을 다시 바꾼다(아래).

---

## 2. Task 2 — Sudoku 대조 (성공 모델)

`gram_sudoku_halt_only/step_65100` (single 70.60% 재현 확인된 그 checkpoint), test 128개.

### 2a. posterior 입력 불변성 — **Sudoku는 입력 의존적이다**

| 조건 | Sudoku (halt_only) | N-Queens (nh512) |
|---|---|---|
| posterior / 정상 입력: label 완전일치 | **82.03%** | 100.00% |
| posterior / 정상 입력: token 일치 | 92.98% | 100.00% |
| posterior / **입력 shuffle**: label 완전일치 | **0.00%** | 100.00% |
| posterior / 입력 shuffle: token 일치 | 11.11% | 100.00% |
| **입력 shuffle 시 원본 출력과 동일한 보드** | **0.00%** | **100.00%** |
| **틀린 라벨 주입 → 그 틀린 라벨 재현** | **0.00%** | **100.00%** |
| (참고) prior / 정상 입력: label 완전일치 | 69.53% | 0.00% |

**Sudoku posterior는 y를 복사하지 않는다.** 틀린 라벨을 줘도 그 라벨을 0.00% 재현하고,
입력을 바꾸면 출력이 100% 바뀐다. posterior(82.03%)와 prior(69.53%)의 격차도 12.5%p로 작다.

N-Queens는 정반대다 — 위 표 오른쪽 열 전부.

### 2b. y_emb 스케일 — **두 과제가 사실상 같다**

| | ‖z_H‖(non-pad) | ‖y_emb‖(non-pad) | **ratio(non-pad)** | pad/전체 위치 |
|---|---|---|---|---|
| Sudoku halt_only | 22.6268 | 11.2439 | **0.4969** | 16 / 97 |
| N-Queens nh512 | 22.6279 | 12.9729 | **0.5733** | 16 / 80 |

**0.497 vs 0.573.** 같은 자릿수, 차이 15%. Sudoku는 이 비율에서 정상 동작한다.

### 판정: **y_emb 스케일 조정은 처방이 아니다**

지시문의 두 갈래에 그대로 대입하면:
- "Sudoku도 입력 불변 + y_emb ≫ z_H → 손대면 Sudoku 깨짐" — **아니다.** Sudoku는 입력 의존적이다.
- "Sudoku는 입력 의존적 + 다른 스케일 → N-Queens 고유 문제, y_emb 조정 안전" — **절반만 맞다.**
  Sudoku가 입력 의존적인 건 맞지만 **스케일은 다르지 않다**(0.497 vs 0.573).

따라서 세 번째 결론이 된다: **y 복사는 N-Queens 고유 현상이지만, y_emb 스케일로 설명되지 않는다.**
같은 스케일에서 Sudoku는 안 복사하고 N-Queens는 복사한다. 스케일을 낮추는 처방은
원인을 겨냥하지 못하고, Sudoku가 이미 잘 도는 값을 벗어나게 만든다.

### 2c. 스케일이 아니라면 무엇이 다른가 — 측정된 차이 하나

두 run의 **유효 KL 가중치**가 크게 다르다.

`gram_sudoku_halt_only`의 kl_reduction은 config 키에 **없다**(그 필드 도입 이전 학습).
대신 그 run이 저장한 자기 코드 스냅샷
`checkpoints/.../gram_sudoku_halt_only/losses.py:94`가 실제로 무엇을 돌렸는지 남긴다:
```python
94            kl_loss = outputs["kl"].sum(-1).sum(-1).sum()  # D는 합, L은 평균(x) L을 합으로 (o), batch는 합
95            # kl_loss = outputs["kl"].sum(-1).mean(-1).sum()
```
→ **L축 sum**. 현재 `models/losses.py:103-105`의 `sum` 분기와 같다.

`kl_coef=0.1`(all_config.yaml:12), L = 81+16 = **97** positions.
L-sum은 L-mean의 L배이므로 **L-mean 환산 유효 β = 0.1 × 97 = 9.7**.

N-Queens는 L = 64+16 = 80, L-mean이므로 β가 곧 유효값이다.

| run | kl_coef | reduction | L | **L-mean 환산 유효 β** |
|---|---|---|---|---|
| gram_sudoku_halt_only (71.85%) | 0.1 | sum | 97 | **9.7** |
| N-Queens 이번 sweep 최고점 (β=1.5) | 1.5 | mean | 80 | **1.5** |
| N-Queens 최대 시도 (β=3.0) | 3.0 | mean | 80 | **3.0** |
| N-Queens 원래 설정 (β=0.07) | 0.07 | mean | 80 | **0.07** |

**성공한 Sudoku의 유효 β(9.7)는 이번 N-Queens sweep 최고점(1.5)의 6.5배,
탐색 최대치(3.0)의 3.2배다.** N-Queens는 아직 Sudoku가 있는 영역을 안 밟았다.

그리고 trans0 KL도 같은 방향을 가리킨다
(`experiments/results/klgrid_sudoku_halt_only.json`, posterior, B=64):

| 모델 | sup0 trans0 KL | 정상상태 trans0 |
|---|---|---|
| **Sudoku halt_only (성공)** | **0.0133** | 0.0090 |
| N-Queens β=0.07 | 13.8083 | 0.0023 |
| N-Queens β=1.5 (최고점) | 0.5053 | 0.0003 |
| N-Queens β=3.0 | 0.1859 | 0.0006 |

Sudoku는 trans0에서도 KL이 0.0133으로 **평평하다**(전 transition 0.008–0.013).
N-Queens β=0.07의 13.81 같은 첫 transition 스파이크가 **없다**.
N-Queens는 β를 3.0까지 올려도 trans0가 0.186으로 Sudoku(0.0133)보다 14배 높다.

**주의(교란 요인)**: 두 checkpoint는 β 말고도 다르다 — Sudoku는 `L_cycles=6`(N-Queens 4),
`mlp_t=true`(false), `pos_encodings=none`(rope), `noise_head_expansion=4.0`(1.5),
데이터·step 수(65100 vs 17085)도 다르다(`all_config.yaml` 양쪽).
**유효 β 차이가 y-복사 차이의 원인이라는 것은 이 측정으로 증명되지 않았다** — 상관까지다.

---

## 3. Task 3 — trans0에 KL을 걸려면 (조사만, 구현 안 함)

### 현재 배선

| 위치 | 코드 | 역할 |
|---|---|---|
| `gram.py:326` | `with torch.no_grad():` | H_cycles-1개 transition 전체를 감쌈 |
| `gram.py:343-345` | `if trace_kl: kl_t = ...; self._kl_trace.append(kl_t...item())` | trans0/1 KL은 **계측만**, `.item()`이라 그래프 밖 |
| `gram.py:366` | `kl = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)` | 손실로 나가는 **유일한** KL |
| `gram.py:389` | `return new_carry, output, (...), kl, lrpm_logits, diag` | kl 텐서 1개 반환 |
| `gram.py:425` | `... , kl, lrpm_logits, diag = self.inner(...)` | 래퍼가 언팩 |
| `gram.py:436` | `outputs["kl"] = kl` | 손실 헤드로 전달 |
| `losses.py:103-105` | `kl_per_pos = outputs["kl"].sum(-1)` → L reduction → batch 합 | `[B, L, D]` 가정 |

### 바꿔야 할 곳 (최소 범위)

1. **`gram.py:326`의 no_grad 범위 축소.** 지금은 루프 전체를 감싼다. trans0 KL에
   gradient를 주려면 latent 헤드(333-345행) 호출이 no_grad 밖이어야 한다.
   `L_level` 호출(329, 331행)만 no_grad로 남기고 그 뒤를 밖으로 빼는 형태
   (루프 안에 `with torch.no_grad():` 블록을 두는 구조로 재배치).
   - 이때 `z_H`는 detach 상태로 헤드에 들어가므로, gradient는
     `prior_mu/prior_logvar/post_mu/post_logvar`와 두 logvar bias **가중치로만** 흐르고
     recursive core(`L_level`)로는 안 흐른다. **Eq-14의 core truncation은 유지된다.**
   - 단 `gram.py:349`의 `z_H = z_H + eps`에서 `eps`가 grad를 가지면 이후 `L_level`
     호출로 그래프가 이어진다. 끊으려면 그 지점에서 `z_H`를 다시 detach 해야 한다
     (안 하면 core truncation이 깨진다).
2. **KL 누적.** 지금은 `kl` 스칼라 텐서 1개. transition별 KL을 더해 같은 `[B, L, D]`
   모양으로 합산하거나 리스트로 반환. 손대는 곳: `gram.py:389`(반환), `:425`(언팩), `:436`(전달).
   같은 shape로 더하면 `losses.py`는 **무수정 가능**.
3. **`losses.py`**: 위처럼 shape를 유지하면 수정 불필요. transition 수로 나눌지
   (평균) 합칠지(합)는 새 config 키가 필요하고, 그건 `losses.py:95-105` 분기 확장.
4. **on/off 스위치**: 기존 run 재현이 깨지지 않게 arch config 키(예: `kl_all_transitions: false`)로
   기본 off. `gram.py`의 pydantic 필드 추가(`gram.py:71-81` 부근 관례대로).

### Eq-14 이탈 정도

- 지시문 요약 기준 Eq-14 = "supervision step당 마지막 transition만 gradient + KL",
  Eq-13 = "전 transition KL(full ELBO)".
- 위 1~3은 **KL 항만** 전 transition으로 확장하고 **core gradient truncation은 유지**한다.
  → Eq-13도 Eq-14도 아닌 **중간 형태**다. Eq-13은 (내가 아는 요약대로라면) core에도
  gradient가 흐르는 full ELBO일 텐데, 여기서는 core가 여전히 끊겨 있다.
- **주의**: 직전 `REPORT_nograd.md`에서 확정했듯 **Eq-13/Eq-14의 실제 논문 문구는
  이 저장소에 논문이 없어 확인 불가**다. 위 "이탈 정도" 평가는 지시문이 준 요약을
  전제로 한 것이고, 논문 원문 대조가 아니다.
- 메모리/속도: trans0/1의 latent 헤드가 그래프에 올라오므로 activation이 늘어난다.
  `L_level`은 여전히 no_grad라 증가폭은 헤드(SwiGLU 4개)분으로 제한된다. 실측 안 함.

---

## 4. 결론

**처방은 [y_emb 스케일 조정]이 아니다. 측정이 지지하는 것은 [KL 가중치를 더 올리는 것]이고,
[trans0 KL]은 아직 근거가 간접적이다.**

근거 수치:

1. **y_emb는 z_H를 압도하지 않는다.** non-pad 비율 **0.5733**(N-Queens 48개 transition 전부 동일).
   지시문 기준(5× 이상)의 1/9 수준이고, 1보다도 작다.
2. **두 임베딩은 같은 `embed_scale`(√512)을 공유한다**(`gram.py:222` vs `:232`, 같은 `embed_tokens`).
   y만 편향되게 키우는 계수는 코드에 없다.
3. **성공한 Sudoku도 거의 같은 비율(0.4969)에서 돈다.** 스케일이 원인이면 Sudoku도 복사해야
   하는데, Sudoku는 틀린 라벨 재현 **0.00%**, 입력 shuffle 시 출력 변화 **100%**로 복사하지 않는다.
   → 스케일을 낮추는 처방은 원인을 못 겨냥하면서 Sudoku가 이미 잘 도는 값을 벗어나게 만든다.
4. **측정된 큰 차이는 유효 KL 가중치다**: Sudoku 9.7(L-sum 0.1×97) vs N-Queens 최고점 1.5.
   그리고 β를 0.07→1.5로 올렸을 때 N-Queens prior valid가 0%→30.29%로 올랐다(직전 sweep).
   Sudoku가 있는 9.7 영역은 **아직 안 밟았다**.
5. **trans0 KL**: Sudoku 0.0133(평평) vs N-Queens β=3.0의 0.1859(14배), β=0.07의 13.81(1000배).
   β를 올리면 trans0가 간접적으로 눌린다(13.81→0.1859). 직접 개입(Task 3)은
   코드 변경이 필요하고 Eq-14 이탈이며, **β를 더 올리는 것으로 같은 방향을 갈 수 있는지
   아직 안 해봤다** — 그쪽이 코드 변경 0이다.

**확인 못 한 것 (중요)**
- 유효 β 차이가 y-복사 차이의 **원인**인지: 상관만 봤다. Sudoku/N-Queens는 β 외에도
  `L_cycles`(6 vs 4), `mlp_t`, `pos_encodings`, `noise_head_expansion`, 데이터, step 수가 다르다.
- β를 9.7 근처까지 올리면 N-Queens가 어떻게 되는지: **미측정.** 이번 sweep 최대가 3.0이었고,
  3.0은 17085 step에서 아직 상승 중이었다(수렴 안 함).
- N-Queens가 다중해(H(y|x)>0), Sudoku가 유일해(H(y|x)=0)라는 차이가 y-복사를 만드는지:
  **측정으로 가르지 않았다.** 그럴듯한 후보지만 이 리포트에 근거 없음.
- Eq-13/Eq-14 논문 문구: 저장소에 논문 없음(`REPORT_nograd.md` §4-5와 동일). **확인 불가.**

---

## 5. 임의로 정한 것

1. **N-Queens 대상 checkpoint를 `gram_nq8_mean_nh512/step_17085`로 골랐다.**
   지시가 그것 또는 "가장 최근 정상 N-Queens ckpt"를 허용했다. nh512는 §2-4 원래 관찰의
   대상이고 noise_head가 논문 크기(9,966,085)라 골랐다.
2. **pad 위치를 분리해 두 비율을 보고했다.** `_y_embeddings`가 앞 16칸을 0으로 pad 하므로
   (`gram.py:229-230`) 전 토큰 평균은 아래로 끌린다. non-pad 기준을 정본으로 삼았다.
   지시문 스니펫은 전 토큰 평균만 찍는데, 그 값(0.4587)도 표에 병기했다.
3. **z_H 복원 방식**: `post_mu` hook의 입력 `u_y`에서 `y_emb`를 빼는 방식.
   `gram.py:337`/`:363`에 print를 넣는 대신 이 방법을 쓴 것은 파일 수정을 피하기 위해서다.
   측정 후 `_y_embeddings`/`post_mu` 원복을 assert로 확인했다(두 run 모두 통과).
4. **n=128, sup_steps=16, seed=0.** §2-4의 기존 측정과 맞췄다.
5. **Sudoku "posterior valid"를 label 완전일치(exact board)로 정의**했다. Sudoku는 유일해라
   제약 검증기 대신 라벨 일치가 정답 기준이고, 이는 `eval_harness.py`의
   `unique_solution=True` 처리와 같은 관례다(`eval_harness.py:56-60`).
6. **입력 shuffle = 배열 역순**(`inputs[::-1]`), **틀린 라벨 = 배치 1칸 roll**
   (`np.roll(labels,1,axis=0)`). §2-4와 동일한 방식.
7. **유효 β 환산(9.7)은 내 계산**이다: L-sum = L-mean × L, L = seq_len + puzzle_emb_len = 97.
   reduction이 sum이라는 근거는 config 키가 아니라 그 run의 코드 스냅샷
   `checkpoints/.../gram_sudoku_halt_only/losses.py:94`다(config에는 `kl_reduction` 키가 아예 없다).
8. **Task 3의 "최소 범위"는 내 판단**이다. 실제로 짜보지 않았으므로 누락이 있을 수 있다.
   특히 `gram.py:349` 이후 재-detach 필요성은 코드를 읽고 추론한 것이지 실행으로 확인하지 않았다.
9. Sudoku KL 그리드 첫 시도는 `diag_contradiction.py`의 `--data` 기본값이 N-Queens 경로라
   shape 오류(97 vs 80)로 실패했다. `--data`를 sudoku로 주고 재실행했다. 코드는 안 고쳤다.
