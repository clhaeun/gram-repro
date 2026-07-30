# REPORT_prep.md — GPU 없이 가능한 준비 작업 (one-hot y_emb + 데이터/evaluator 검증)

- 일시: 2026-07-27
- 브랜치 `main`, HEAD `4c847f1` (커밋 안 함)
- **GPU 미사용. 학습 launch 없음.** 전부 `CUDA_VISIBLE_DEVICES=""` CPU 또는 정적 검사.
  (작업 도중 GPU 0~7 전체가 타 사용자 작업 1265–1690MiB로 채워졌다. 어차피 안 썼다.)
- 수정한 파일: `models/recursive_reasoning/gram.py`, `train_nqueens8_fix.sh`
- 신규 파일: `experiments/verify_yonehot.py`

---

## 1. Task 1 — one-hot y_emb 구현 (토글)

### 1-1. 현재 코드 (수정 전)

`gram.py:225-232` 원본:
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
`embed_tokens`(`gram.py:149`)는 학습되는 `CastedEmbedding(vocab_size, hidden_size)`이고,
`embed_scale = math.sqrt(hidden_size)`(`gram.py:146`).

### 1-2. **projection 설계: 근거와 선택**

> **핵심: `nn.Embedding` 조회는 `one_hot @ W` 와 수학적으로 완전히 같다.**
> 따라서 "one-hot + 학습가능 projection"은 **지금 코드와 동일한 함수**이고 효과가 0이다.
> 사수 지시의 "통로 좁히기"가 성립하려면 projection이 **없거나 고정**이어야 한다.

세 후보와 트레이드오프:

| 안 | 내용 | 학습 파라미터 | y 통로 용량 | 문제 |
|---|---|---|---|---|
| A. 학습가능 projection (3→512) | `one_hot @ W_trainable` | 1,536 | **현재와 동일** | **현재 경로와 수학적으로 같음 → 무의미.** 채택 안 함 |
| B. **zeropad** | one-hot [V]을 512차원 앞쪽에, 나머지 0 | **0** | 위치당 log2(V)=1.58 bit | 512차원 중 3차원만 씀. 뒤 509차원은 z_H 전용이 됨 |
| C. **frozen_random** | 고정 랜덤 [V,512] 행렬로 사영(buffer) | **0** | 위치당 log2(V)=1.58 bit | 512차원 전체에 퍼지지만 방향이 임의 고정 |

**B와 C 둘 다 구현하고 `y_onehot_proj`로 고르게 했다. 기본은 B(zeropad).**
근거: B가 파라미터도 buffer도 만들지 않아 state_dict가 기존과 **완전히 동일**하다(§1-4 표).
C는 buffer `y_onehot_mat` 1개가 늘어 state_dict 키가 바뀐다.
정보 용량은 B·C가 같다(둘 다 위치당 V개 값 중 하나). **어느 쪽이 학습에 유리한지는
돌려봐야 안다 — 임의로 하나만 넣지 않고 둘 다 남겼다.**

**스케일**: 두 경로 모두 사영 전 위치당 L2 노름 = 1로 맞췄고(C는 행별 단위노름 정규화),
기존 두 임베딩 경로(`gram.py:222`, `:232`)와 같이 `embed_scale`을 곱한다.
`y_onehot_scale`(기본 1.0)을 추가로 곱해 스케일만 따로 조정할 수 있다. 실측 비교는 §1-5.

### 1-3. 구현 diff

`git diff --stat models/recursive_reasoning/gram.py` → **53 insertions(+), 2 deletions(-)**.
삭제된 2줄은 `_y_embeddings` 본문을 `else:` 분기로 옮긴 것뿐이다:
```
-        embedding = self.embed_tokens(labels.to(torch.int32))
-        return self.embed_scale * embedding
```

config 필드 (`gram.py:81-95`):
```python
    y_onehot: bool = False
    #   "zeropad"       : one-hot [V] 을 hidden_size 앞쪽에 두고 나머지 0. 파라미터 0개.
    #   "frozen_random" : 고정(비학습) 랜덤 행렬 [V, hidden_size] 로 사영. 파라미터 0개(buffer).
    y_onehot_proj: str = "zeropad"
    #   embed_scale(√hidden) 곱 위에 추가로 곱하는 계수. 1.0 이면 위치당 노름 = √hidden.
    y_onehot_scale: float = 1.0
```

`__init__`의 조건부 생성 (`gram.py:166-182`) — **False면 아무것도 만들지 않는다**:
```python
        if self.config.y_onehot:
            if self.config.y_onehot_proj == "frozen_random":
                g = torch.Generator(device="cpu").manual_seed(0)
                m = torch.randn(self.config.vocab_size, self.config.hidden_size, generator=g)
                m = m / m.norm(dim=-1, keepdim=True)          # 행별 단위노름 -> 출력 노름 1
                self.y_onehot_mat = nn.Buffer(m.to(self.forward_dtype), persistent=True)
```
전역 RNG 대신 **로컬 `torch.Generator`**를 쓴다. 전역을 소비하면 뒤따르는 파라미터의
init 스트림이 밀려 기존 run 재현(`experiments/make_raw_ema.py`의 w0 복원 포함)이 깨진다.

`_y_embeddings` (`gram.py:255-280`):
```python
    def _y_onehot_embeddings(self, labels):
        oh = F.one_hot(labels.to(torch.int64), num_classes=self.config.vocab_size)
        oh = oh.to(self.forward_dtype)
        if self.config.y_onehot_proj == "zeropad":
            return F.pad(oh, (0, self.config.hidden_size - self.config.vocab_size))
        return oh @ self.y_onehot_mat.to(oh.dtype)

    def _y_embeddings(self, labels):
        labels = torch.where(labels==IGNORE_LABEL_ID, 0, labels)
        if self.config.y_onehot:
            embedding = self._y_onehot_embeddings(labels)
        else:
            embedding = self.embed_tokens(labels.to(torch.int32))

        if self.puzzle_emb_len > 0:
            embedding = F.pad(embedding, (0, 0, self.puzzle_emb_len, 0))

        if self.config.y_onehot:
            return self.embed_scale * self.config.y_onehot_scale * embedding
        return self.embed_scale * embedding
```
`puzzle_emb_len=16` zero-pad 규약(229-230행 원본)은 **두 경로 공통**으로 유지된다.

### 1-4. 정적 검증 결과 (CPU, `experiments/verify_yonehot.py`)

```
$ CUDA_VISIBLE_DEVICES="" python experiments/verify_yonehot.py
```

**(1) `ast.parse` 통과.**

**(2) num_params (N-Queens nh512 config, vocab=3, seq=64)**

| 설정 | params | buffers | 판정 |
|---|---|---|---|
| `y_onehot=False` (기존) | **9,966,085** | 15,880 | ✅ 논문 대응값과 일치 |
| `y_onehot=True, zeropad` | **9,966,085** | 15,880 | 증가분 0 |
| `y_onehot=True, frozen_random` | **9,966,085** | 17,416 | 파라미터 증가분 0, buffer `y_onehot_mat`(3×512=1,536) 추가 |

**세 경우 모두 학습 파라미터 수가 같다.** one-hot이 표현력을 추가하지 않음을 수치로 확인.

**(3) state_dict 키 집합 — 실제 checkpoint `gram_nq8_mean_nh512/step_17085`와 대조**

| 설정 | 누락 키 | 추가 키 |
|---|---|---|
| `y_onehot=False` | **없음** | **없음** |
| `y_onehot=True, zeropad` | **없음** | **없음** |
| `y_onehot=True, frozen_random` | 없음 | `['y_onehot_mat']` |

→ `pretrain.py:266`이 `load_state_dict(..., assign=True)`(strict)이므로,
**False와 zeropad는 기존 checkpoint를 그대로 로드할 수 있다.**
frozen_random은 키가 하나 늘어 기존 checkpoint 로드 시 strict 오류가 난다
(새 학습 전용. 기존 ckpt 재개용이 아니다).

**(4) 기존 all_config.yaml 호환 (y_onehot 키 없음 → 기본 False)**

| checkpoint | yaml에 `y_onehot` 키 | 해석된 config |
|---|---|---|
| `gram_nq8_mean_nh512` | 없음 | `y_onehot=False, proj=zeropad, scale=1.0` |
| `gram_sudoku_halt_only` | 없음 | `y_onehot=False, proj=zeropad, scale=1.0` |

**(5) CPU forward 1회 (B=8, L=64, float32)**

| 설정 | logits | kl | y_emb shape | pad 위치 |
|---|---|---|---|---|
| False | (8, 64, 3) | (8, 80, 512) | (8, 80, 512) | 16/80 |
| True zeropad | (8, 64, 3) | (8, 80, 512) | (8, 80, 512) | 16/80 |
| True frozen_random | (8, 64, 3) | (8, 80, 512) | (8, 80, 512) | 16/80 |

**shape 에러 없음. 세 경로 전부 동일 shape.**

### 1-5. y_emb 스케일 비교 (√512 곱 유지 여부 판단 재료)

| 경로 | ‖y_emb‖ (non-pad) | ‖z_H‖ 대비 비율 |
|---|---|---|
| 학습 embedding, **랜덤 init** (CPU 측정) | 23.5597 | 1.0412 |
| 학습 embedding, **학습 후 실측** (`experiments/results/yemb_nq8_nh512.json`) | **12.9729** | **0.5733** |
| one-hot zeropad, `scale=1.0` | 22.6250 | 0.9999 |
| one-hot frozen_random, `scale=1.0` | 22.6223 | 0.9998 |

(‖z_H‖은 RMSNorm 출력이라 √512 = 22.627 고정.)

**관측**: 학습된 embedding은 init(23.56)에서 12.97로 **줄어들었다**.
one-hot에 `scale=1.0`을 쓰면 22.63이 되어 **학습 후 실측값의 1.74배**다.
one-hot은 학습으로 크기를 못 줄이므로(고정), 이 값이 그대로 간다.
`y_onehot_scale=0.573`으로 두면 학습된 경로의 실측 노름과 맞는다.
**어느 쪽이 맞는지는 돌려봐야 안다 — 기본 1.0으로 두고 스윕 대상으로 남겼다(§6).**

### 1-6. Sudoku 안전성 (코드 레벨)

- `config/arch/gram.yaml`(Sudoku), `gram_sudoku_halt_only.sh`, `gram_sudoku_lmean_v2.sh`에
  `y_onehot` 문자열 **0건**(`grep -c` 확인).
- 따라서 Sudoku 경로는 `y_onehot=False` → `_y_embeddings`가 원본과 **같은 두 줄**을 실행한다
  (§1-3의 삭제 2줄이 `else:` 분기로 그대로 이동).
- state_dict 키 변화 없음(§1-4 표), 파라미터 수 변화 없음, RNG 소비 변화 없음
  (False면 `torch.Generator`조차 만들지 않음).
- **판정: y_onehot을 지정하지 않는 한 Sudoku 경로는 bit-identical.**
  단 이것은 **코드·정적 검증 수준**이다. Sudoku 재학습으로 71.85%를 다시 확인하지는 않았다.

### 1-7. `train_nqueens8_fix.sh` 환경변수 추가

```bash
YONEHOT="${YONEHOT:-False}"
YOH_PROJ="${YOH_PROJ:-zeropad}"      # zeropad | frozen_random
YOH_SCALE="${YOH_SCALE:-1.0}"
...
  arch.y_onehot=${YONEHOT} arch.y_onehot_proj=${YOH_PROJ} arch.y_onehot_scale=${YOH_SCALE} \
```
`bash -n` 통과. dry-run 조립 확인:
```
기본            : arch.y_onehot=False arch.y_onehot_proj=zeropad arch.y_onehot_scale=1.0
YONEHOT=True    : arch.y_onehot=True  arch.y_onehot_proj=zeropad arch.y_onehot_scale=1.0
frozen_random   : arch.y_onehot=True  arch.y_onehot_proj=frozen_random arch.y_onehot_scale=0.573
```
부수 수정: `DRYRUN=1`일 때 GPU 점유 pre-flight를 건너뛰게 했다.
조립만 확인하려는데 남의 작업이 GPU를 잡고 있으면 막히기 때문이다.
**실제 launch 경로의 점유 검사는 그대로다**(실측: 타 작업 1675MiB 상태에서
`FATAL: GPU 4 사용 중 (1675MiB)`로 정상 차단됨).

---

## 2. Task 2 — 데이터 split 재확인

`data/nqueens-8/{train,test}/all__inputs.npy` 직접 비교 (byte 단위 `tobytes()`).

| 항목 | train | test |
|---|---|---|
| rows | 7,194 | 772 |
| unique inputs | **4,376** | **772** |
| input당 label 개수 | 1~18 (분포 아래) | 전부 **1** |

- **train ∩ test unique-input 교집합 = 0** ✅
- **(x, y) pair 교집합 = 0** ✅
- 4,376 + 772 = **5,148** = sidecar `meta.num_unique_inputs`(5,148) → **전체 unique input을
  빠짐없이 두 쪽으로 분할**. 겹침도 누락도 없다.

train의 input당 label 개수 분포:
`{1:2878, 2:964, 3:315, 4:93, 5:25, 6:36, 7:8, 8:26, 12:7, 14:8, 16:9, 18:7}`

### x-split 무결성 (한 x의 solution들이 한쪽에만 있는가)

unique-input 교집합이 0이므로 **정의상 어떤 x도 양쪽에 걸치지 않는다.** 추가로 sidecar
(`data/nqueens-8/sidecar_8.json`, sha1 해시 키)와 대조:

| split | unique | sidecar에 존재 | labels == solutions | labels < solutions | labels > solutions |
|---|---|---|---|---|---|
| train | 4,376 | 4,376 | **4,376** | 0 | 0 |
| test | 772 | 772 | 494 | **278** | 0 |

- **train은 각 x의 해집합을 전부 저장**한다(4,376/4,376 완전 일치).
- **test는 278개 x에서 해집합 일부만 저장**한다(x당 label 1개 규약 때문).
  이것은 누출이 아니라 "test는 witness 1개만 보관"이라는 설계다.
  다만 **test 라벨과의 일치로 채점하면 278개(36%)는 다른 정답을 내도 오답 처리**된다 —
  Task 3의 제약 기반 evaluator가 필요한 이유와 같은 사안이다.

**판정: 누출 0. 데이터 재생성 불필요.**

---

## 3. Task 3 — evaluator 실제 적용 확인

### 3-1. `nqueens/accuracy` = valid_rate (라벨 무관)

- `verify_boards(preds, inputs, n)` — **labels 인자가 없다**(`evaluators/nqueens.py:38`).
- `NQueens.required_outputs = {"inputs", "preds"}` (`evaluators/nqueens.py:79`) — **labels 미요청**.
- `evaluators/nqueens.py:129`:
```python
129            "nqueens/accuracy": float(acc[1] / total),
```
  `acc[1]`은 `valid.sum()`. `valid = count_ok & row_ok & col_ok & d1_ok & d2_ok`
  (`evaluators/nqueens.py:61`) — 순수 제약 만족.

**드라이런 실측**: `required_outputs={'preds','inputs'}`, `'labels' in required_outputs = False`.

### 3-2. 실제로 로드되는 형식 — **확정**

`create_evaluators` 경로에서 두 형식을 직접 파싱:

| 커맨드 형식 | 결과 |
|---|---|
| `evaluators="[nqueens@NQueens]"` | **FAIL** — `TypeError: BaseModel.__init__() takes 1 positional argument but 2 were given` |
| `'evaluators=[{name: nqueens@NQueens}]'` | **OK** — `['nqueens@NQueens']` |

`create_evaluators` 드라이런(실제 로드):
```
로드된 evaluator: ['NQueens']  n=8
required_outputs={'preds', 'inputs'}  (labels 포함? False)
```
→ **동작하는 형식은 `'evaluators=[{name: nqueens@NQueens}]'` 하나다.**
`pretrain.py:53`이 `evaluators: List[EvaluatorConfig]`이고 `EvaluatorConfig`(`pretrain.py:41-43`)가
`name` 필드를 가진 매핑을 요구하기 때문. 지시문·`evaluators/nqueens.py:29` docstring의
`evaluators="[nqueens@NQueens]"`는 **동작하지 않는다**(docstring은 지시대로 수정하지 않았다).

### 3-3. `train_nqueens8_fix.sh` 확인 — **수정 불필요**

`train_nqueens8_fix.sh:54`:
```bash
  'evaluators=[{name: nqueens@NQueens}]' \
```
이미 동작하는 형식이다. 앞선 β sweep 4개 run에서 `nqueens/accuracy`가 실제로 찍힌 것과 일치.

### 3-4. 정답 라벨 772개 → valid=1.0 재확인

```
valid=1.0000 consistent=1.0000 strict=1.0000 queen_count mean=8.0000 n=772
```
✅ 검증기 정확.

---

## 4. Task 4 — one-hot 메커니즘과 남은 위험

### 4-1. 현재 학습 embedding이 y 통과를 가능케 하는 구조

- `embed_tokens`는 `CastedEmbedding(vocab_size=3, hidden_size=512)`(`gram.py:149`),
  **학습되는 3×512 = 1,536 파라미터**.
- 위치당 정보량 자체는 log2(3)=1.58 bit로 one-hot과 같지만, **512차원 임의 방향을
  학습으로 고를 수 있다** — 즉 `post_mu`가 가장 읽기 쉬운 방향으로 정렬될 수 있다.
- 실측(`REPORT_yemb.md` §1): 학습 후 ‖y_emb‖이 init 23.56 → 12.97로 조정됐다.
  크기까지 학습으로 맞춘 흔적이다.
- 결과(`REPORT_klsweep.md` §2-4): posterior가 y를 100% 복사, 틀린 라벨도 99~100% 재현.

### 4-2. one-hot이 통로를 좁히는 방식

- zeropad: y는 **512차원 중 앞 3차원**에만 나타나고 값은 {0, embed_scale}뿐.
  방향·크기 모두 **학습 불가**. `post_mu`가 y에 맞춰 정렬할 대상이 없다.
- frozen_random: 방향은 512차원에 퍼지지만 **고정**이라 마찬가지로 학습 불가.
- 두 경우 모두 **학습 파라미터 0개 추가**(§1-4 실측)로, y 인코딩이 최적화 대상에서 빠진다.

### 4-3. 남은 위험 — **코드로 확정 못 하는 것**

1. **`post_mu`는 그대로다.** `post_mu`는 학습되는 SwiGLU(`gram.py:188-191`)이고
   입력이 `u_y = z_H + y_emb`다. y_emb가 고정이어도 **`post_mu`가 그 고정 패턴을 읽어
   latent에 실어나르도록 학습될 수 있다.** one-hot은 y_emb 쪽 자유도만 없앨 뿐
   post_mu 쪽 자유도는 그대로다. → **막힌다는 보장 없음. 학습해봐야 안다.**
2. **`z_H + y_emb` 덧셈 결합은 유지된다.** zeropad는 앞 3차원에서 z_H와 y가 겹쳐
   서로를 오염시킨다(z_H의 그 3차원 성분과 구분 불가). frozen_random은 512차원 전체에
   퍼져 겹침이 넓다. 어느 쪽이 나은지 **미확인**.
3. **스케일 선택의 영향 미확인.** `scale=1.0`은 학습된 값(12.97)의 1.74배(§1-5).
   과대하면 오히려 y가 더 잘 통과할 수 있다. **미측정.**
4. **Sudoku(vocab=11)에서의 거동 미확인.** zeropad면 512 중 11차원. 이번엔 안 켰다.
5. **정보이론적으로 one-hot이 y 복사를 막지 못할 수 있다.** 위치당 1.58 bit면
   8×8 보드 전체로 64×1.58 ≈ 101 bit이고, latent(512차원 연속값)는 그보다 훨씬 크다.
   **통로가 좁아진 게 아니라 인코딩만 고정된 것**일 수 있다. 이 구분은 학습 결과로만 갈린다.

**요약: one-hot은 "y_emb의 학습 자유도"를 0으로 만든다(정적 검증 완료). 그것이
"posterior의 y 복사"를 실제로 막는지는 이 리포트의 어떤 측정으로도 확정할 수 없다.**

---

## 5. 임의로 정한 것 전부

1. **projection 후보를 2개(zeropad / frozen_random) 다 구현**하고 기본을 zeropad로 했다.
   지시가 "임의로 하나 고르지 말라"고 해서 고르지 않고 토글로 남겼다.
   기본을 zeropad로 한 이유는 state_dict가 기존과 완전히 같아서다(§1-4).
   학습가능 projection(안 A)은 **구현하지 않았다** — 현재 경로와 수학적으로 동일해서다.
2. **`y_onehot_scale` 필드를 추가했다.** 지시에 없던 손잡이다. §1-5에서 one-hot 노름이
   학습된 값의 1.74배로 나와, 스케일을 코드 수정 없이 스윕할 수 있게 했다. 기본 1.0.
3. **`embed_scale`(√512) 곱을 one-hot 경로에도 유지**했다. 기존 두 경로
   (`gram.py:222`, `:232`)와 구조를 맞추는 쪽이 덜 놀랍다고 판단. `y_onehot_scale`로 조정 가능.
4. **frozen_random의 시드를 0으로 고정**하고 **로컬 `torch.Generator`**를 썼다.
   전역 RNG를 소비하면 다른 파라미터 init 스트림이 밀린다.
5. **frozen_random 행렬을 행별 단위노름으로 정규화**해 zeropad와 출력 노름을 맞췄다.
   안 하면 두 안의 스케일이 달라져 비교가 안 된다.
6. **`train_nqueens8_fix.sh`에 `DRYRUN=1`이면 GPU 점유 검사를 건너뛰게** 했다.
   지시에 없던 수정이다. 실제 launch 경로의 검사는 안 건드렸다.
7. **CPU 검증의 B=8**은 `config.batch_size`와 같아야 한다(`CastedSparseEmbedding`이
   고정 버퍼). 처음 B=4로 짜서 shape 오류가 났고 8로 맞췄다.
8. **CPU 검증은 float32**로 돌렸다(`.to(torch.float32)`). 학습은 bfloat16이다.
   `_y_embeddings` 출력 dtype은 bfloat16임을 따로 확인했다.
9. Sudoku 안전성은 **코드/정적 수준만** 확인했다. 재학습 검증 안 함(GPU 금지).

---

## 6. GPU 생기면 바로 돌릴 실험 목록

전제: β=1.5가 이번 sweep 최고점(prior valid 30.29%, pass@20 94.17%),
성공한 Sudoku의 L-mean 환산 유효 β는 9.7로 **아직 미탐색 대역**(`REPORT_yemb.md` §2c).

**우선순위 1 — one-hot on/off × β (핵심 처방 검증, 각 ~1h40m/1GPU)**
```bash
cd /home/chaeun/gram-repro
# 대조군: β=1.5 는 이미 있음 (nq8_b15, one-hot off)
RUN=nq8_oh_b15  BETA=1.5 YONEHOT=True GPUS=4 PORT=29651 bash train_nqueens8_fix.sh
RUN=nq8_oh_b07  BETA=0.7 YONEHOT=True GPUS=5 PORT=29652 bash train_nqueens8_fix.sh
RUN=nq8_oh_b30  BETA=3.0 YONEHOT=True GPUS=6 PORT=29653 bash train_nqueens8_fix.sh
RUN=nq8_oh_b007 BETA=0.07 YONEHOT=True GPUS=7 PORT=29654 bash train_nqueens8_fix.sh
```
`nq8_oh_b007`이 특히 중요하다 — **β를 원래 값으로 두고 one-hot만 켠** 순수 대조군이라
one-hot 단독 효과를 β와 분리한다.

**우선순위 2 — 미탐색 β 대역 (Sudoku 유효 β 9.7 방향)**
```bash
RUN=nq8_b50  BETA=5.0  GPUS=4 PORT=29661 bash train_nqueens8_fix.sh
RUN=nq8_b100 BETA=10.0 GPUS=5 PORT=29662 bash train_nqueens8_fix.sh
```
β=3.0은 step 17085에서 아직 상승 중이었다(수렴 안 함) — epoch 증량도 후보.

**우선순위 3 — 부수 축**
```bash
# one-hot projection 비교
RUN=nq8_ohfr_b15 BETA=1.5 YONEHOT=True YOH_PROJ=frozen_random GPUS=6 PORT=29663 bash train_nqueens8_fix.sh
# one-hot 스케일을 학습된 값(0.573)에 맞춤
RUN=nq8_ohs_b15 BETA=1.5 YONEHOT=True YOH_SCALE=0.573 GPUS=7 PORT=29664 bash train_nqueens8_fix.sh
```

**각 run 완주 후 측정 (기존 스크립트 그대로)**
```bash
# prior N=20 + posterior, valid 기준
python experiments/sweep_harness.py --ckpt-dir checkpoints/Nqueens-8-ACT-torch/<RUN> \
  --step <peak 또는 step_17085> --N 20 --seed 0 --out experiments/results/sweep_<RUN>_<step>.json
# y 복사 여부 (핵심 판정): 틀린 라벨 재현율 / 입력 shuffle 불변성
python experiments/diag_yemb.py --ckpt-dir checkpoints/Nqueens-8-ACT-torch/<RUN> \
  --step <step> --data data/nqueens-8/test --task nqueens --out experiments/results/yemb_<RUN>.json
# transition별 KL
python experiments/diag_contradiction.py --ckpt-dir checkpoints/Nqueens-8-ACT-torch/<RUN> \
  --step <step> --skip 2a,2c,2d --out experiments/results/klgrid_<RUN>.json
```
**one-hot 성공 판정 기준**: `diag_yemb`의
`post_wrong_label_reproduces_it`이 100% → 크게 하락, `post_shuffled_same_as_real`이 100% → 하락.
(Sudoku 성공 모델의 값은 둘 다 **0.00%**다 — `experiments/results/yemb_sudoku_halt_only.json`.)

**Sudoku 회귀 (y_onehot 기본 False가 안 깨졌는지, GPU 1장 ~5분)**
```bash
python experiments/eval_harness.py --ckpt-dir checkpoints/Sudoku-extreme-1k-aug-1000-test10k-ACT-torch/gram_sudoku_halt_only \
  --step step_65100 --task sudoku --num-puzzles 1000 --k 20 --seed 0 \
  --out experiments/results/sudoku_regression_after_yonehot.json
# 기대: single ~70.60% / majority@20 ~88.40% / pass@20 ~93.10% (±1%p)
```
