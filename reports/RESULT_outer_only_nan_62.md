# 결과: outer_only(Eq3 단독) 구성은 학습 불가 — α₁ 이 유일한 수축 인자

실행: 2026-08-06, 62번(jurgen), nqueens-10 Stage A
상태: **확정 결과**. 실패한 실험이 아니라 **관측된 성질**이다.

## 한 줄

> **outer_only(Eq3 단독) 구성은 `pre-norm` + `norm_placement=none` 하에서 step 50 이내 forward NaN.
> α₁ 이 유일한 수축 인자이며 Eq2/Eq3 는 분리 불가.**

## 로그 경로

| 항목 | 경로 |
|---|---|
| 본 런 로그 | `logs/nq10_A3p_oo_lr3e4_s0.log` |
| 최초 기동 실패분(wandb 자격증명, 본건 무관) | `logs/nq10_A3p_oo_lr3e4_s0.wandbfail.log` |
| 런처 wrapper | scratchpad 전용, repo 미포함 (`arch.outer_only=true` 1줄 외 런처와 조립 동일, DRYRUN 대조 확인) |

RUN=`nq10_A3p_oo_lr3e4_s0` / GPU 3,4 / PORT 29972 / SEED 0
설정: `residual_scale=input-independent`, `alpha_1_init=0.75`, `alpha_2_init=0.25`,
`outer_only=true`, `lr=3e-4`, `nodecay_marked=True`, `EPOCHS=990 EVALINT=30`

## 플래그가 실제로 먹었다는 증거

```
[nodecay] weight_decay=0 대상 1개: ['_orig_mod.model.inner.L_level.alpha_2_param']
```
`outer_only=false` 면 2개(`alpha_2_param`, `alpha_1_param`)다.
1개라는 것은 `fp_transformer.py:169-177` 의 `if not self.outer_only` 분기가 타서
`alpha_1_param` 이 **생성되지 않았다**는 뜻 — 런처 헤더가 예고한 시그니처 그대로다.

## 관측

```
[step  50] kl_loss=nan lm_loss=nan lprm_loss=nan q_halt_loss=nan lr=3.00e-05
[step 100] kl_loss=nan lm_loss=nan lprm_loss=nan q_halt_loss=nan lr=6.00e-05
[step 150] ... nan ...   lr=9.00e-05
[step 250] ... nan ...   lr=1.50e-04
(step 314 에서 kill. 회복 구간 없음)
```

세 가지가 동시에 성립한다:

1. **점진적 발산이 아니다.** 최초 기록 지점(step 50), warmup 중 `lr=3e-5` 에서 이미 NaN.
2. **전 손실이 동시에** NaN. 특정 loss 의 gradient 폭주가 아니라 **forward 측 오염**이다.
3. **동시 실행된 대조군은 정상.** 같은 시각 셀 P(`α₂=0.5`, outer_only=false)는
   step 800 에서 `train/accuracy=0.9659`, `kl_loss=2.11` 로 정상 하강.
   → 데이터·런처·환경·wandb 문제가 아니라 **`outer_only` 분기 고유 현상**이다.

## 메커니즘

`config/arch/gram_nqueens10_fp.yaml`:
- `norm_type: pre-norm` (51행)
- `norm_placement: none` (57행)

pre-norm 이라 블록 내 `out` 은 `rms_norm(hidden)` 에서 계산돼 유계다.
그러나 **residual stream `hidden` 자체는 어디서도 정규화되지 않는다** —
`norm_placement: none` 이라 `FixedPointTransformer` 입·출구 정규화도 없다
(`fp_transformer.py:247-260`).

`outer_only=true` 는 `fp_transformer.py:236-239` 에서
```python
if self.outer_only:
    one = hidden_states.new_ones(1, 1, 1)
    beta_2 = 1 - alpha_2
    alpha_1, beta_1 = one, one
```
로 두므로, 블록 갱신 `hidden = alpha_1*hidden + beta_1*out` (`fp_transformer.py:88`)이
`hidden = hidden + out` 인 **감쇠 없는 순수 누적**이 된다.

재귀 깊이: `L_layers=2 × L_cycles=4 × H_cycles=3 × halt_max_steps≤16` → 최대 ~384 회 누적.
유계 증분이 384 회 무감쇠로 쌓이면 bfloat16(`forward_dtype: bfloat16`) overflow → inf → NaN.

바깥 Eq3 `α₂·hidden + β₂·input` 은 α₂=0.25 로 수축이지만 **L_level 호출당 1회**뿐이라
블록 내 누적을 상쇄하지 못한다.

→ **`outer_only` 가 제거하는 α₁ 이 이 아키텍처의 유일한 수축 인자였다.**

## FPRM Theorem 1 과의 관계

정리의 커플링은
```
β₂ = 1 − α₂ · α₁^(2L)
β₁ = β₂ (1 − α₁) / (1 − α₁^(2L) + 1e-5)
```
이다 (`fp_transformer.py:241-242`).

`α₁ = 1` 을 넣으면
- `α₁^(2L) = 1` → `β₂ = 1 − α₂` (α₁ 의존성 소멸)
- `β₁` 의 분자 `(1 − α₁) = 0`, 분모 `(1 − α₁^(2L)) = 0` → **0/0 형태**.
  코드에서는 `+1e-5` 정칙화 때문에 `β₁ → 0` 으로 떨어지지만, `outer_only` 분기는
  이 식을 아예 우회하고 `β₁ = 1` 로 강제한다.

즉 **α₁=1 에서는 Theorem 1 의 커플링 자체가 무의미해진다.**
본 관측은 **정리의 유계성이 α₁ < 1 에 본질적으로 의존함을 실증**한 것이다.
α₁ 은 "튜닝 가능한 하이퍼파라미터" 가 아니라 **유계성의 전제**다.

★ **FPRM 논문에 이 ablation 이 없다. 우리 고유 관측이다.**

## 함의 — 원래 의도했던 축 분리는 성립하지 않는다

셀 Q 의 목적은 "A3′ 붕괴 원인이 블록 내 감쇠(Eq2)인지 입력 혼합(Eq3)인지" 가르는 것이었다.
그러나 α₁ 을 1 로 두면 학습 신호가 아예 생기지 않으므로 **이 방식으로는 가를 수 없다.**
Eq2 와 Eq3 는 독립 축이 아니다 — Eq2 의 α₁ 이 전체 유계성을 떠받치고 있다.

부수적으로: lr 을 낮춰 재시도할 축이 아니다. warmup `lr=3e-5` 에서 이미 NaN 이고
forward 측 오염이므로 optimizer 설정과 무관하다.

## 사상 최초 실행이었다

보관된 config 20건이 **전부 `outer_only: false`** 다:
```
$ grep -h "outer_only" reports/configs/*/*/all_config.yaml | sort | uniq -c
     20 outer_only: false
```
`outer_only=true` 는 이 프로젝트에서 한 번도 검증된 적 없는 경로였다.

## 이 축을 살리려면 (★ 62번 승인 필요, 미착수)

residual stream 을 묶을 장치가 같이 필요하다:

1. `arch.norm_placement=output` — L_level 출구에서 rms_norm. **config 만으로 가능, 코드 수정 불요.**
2. `arch.norm_type=post-norm` — 블록마다 정규화. 역시 config 만으로 가능.
3. α₁ 을 1 이 아니라 1 근방 고정값(예: 0.99)으로 — **코드 수정 필요.**

★ 어느 쪽도 A3′ 정준 설정을 벗어나므로 셀 P/R 과 **직접 비교 불가**다.
   또한 1·2 는 "α₁=1 정확히" 가 아니게 되므로 의도한 깨끗한 축 분리도 아니다.
   축 분리를 정말 원하면 설계 재검토가 필요하다.
