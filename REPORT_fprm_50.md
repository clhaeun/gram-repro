# REPORT_fprm_50 — FPRM 기여1(pre-norm + residual scaling) 이식 (2026-08-02)

코드 작업만 수행. 학습 launch 없음 (pair final run 8건이 GPU 전점유 중).

## 1. 변경 요약

| 파일 | 변경 |
|---|---|
| `models/fp_transformers.py` → `models/fp_transformer.py` | **git mv** (사수 스펙 파일명은 단수형; 업로드본이 복수형이었다). 내용 변경 2건: 주석 처리돼 있던 `from .config import ReasoningModelConfig` 줄 삭제, `FixedPointTransformerBlock.__init__(self, config: ReasoningModelConfig)` → `(self, config)` (어노테이션이 def 실행 시점에 평가되어 import 자체가 NameError 로 죽는 상태였다). fp_layers import 는 업로드본에 이미 반영돼 있었다. |
| `models/layers.py` | **불변** (fp_layers 의 CastedLinear 가 persistent nn.Buffer(mask) 를 등록 → 기존 checkpoint strict load 즉사 방지. 별도 파일 공존). |
| `models/recursive_reasoning/gram.py` | [2-1] Config 15개 필드 — 사용자 초안이 이미 반영돼 있어 스펙 대조만 수행(전부 일치). [2-2] L_level 분기 활성화 (초안은 docstring 주석 상태였다). import 는 분기 안. 호출부 4곳 불변. [5] `_latent_diag` 에 `diag/z_h_norm`, `diag/z_l_norm` 추가 (최종 transition, detach, norm(dim=-1).mean()). 호출부에서 z_H/z_L 전달. |
| `pretrain.py` | [3-1] `nodecay_marked: bool = False`. [3-2] `_adamw_param_arg` 에 `_no_weight_decay` 속성 기반 분류(by_attr) + 켰는데 마킹 파라미터가 없으면 raise. 세 플래그 전부 False 면 기존과 글자 그대로 동일 경로. |
| `config/arch/gram_nqueens8_fp.yaml`, `gram_sudoku_fp.yaml` | 신규. 기존 대응 yaml(gram_nqueens8.yaml / gram.yaml) 복사 + FPRM 14개 필드 명시. 기존 yaml 불변. |

### yaml 특이사항 2건
- **causal 은 yaml 에 넣지 않았다**: `pretrain.py:189` 의 create_model 이 `causal=False` 를
  직접 주입하므로 yaml 에도 있으면 `dict()` 중복 키워드로 즉사한다 (seq_len/vocab_size 와
  같은 부류). 두 yaml 에 주석으로 박제.
- gram_nqueens8_fp 의 `kl_reduction: sum` — 원본 gram_nqueens8.yaml 은 mean 이지만
  사수 지시("기존 정준 설정: kl_reduction sum")에 따라 sum + kl_coef 0.07(sum 눈금).

## 2. [6] 검증 전문 (CPU, GPU 불사용)

```
[6-1] import fp_transformer OK
[6-2] use_fp=False 모델 키 27개, checkpoint 키 27개
      모델-ckpt: 없음
      ckpt-모델: 없음
[6-3] strict load OK (nq8_final_add_s0/step_67428)
[6-3b] 구세대(v7, FPRM 필드 이전) checkpoint strict load OK
[6-4] print(model)에 FixedPointTransformer(Block) 포함: True
      params: False=9,966,085  True=9,966,087  차이=+2
      alpha 파라미터: ['model.inner.L_level.alpha_2_param', 'model.inner.L_level.alpha_1_param']
      fp 관련 buffer 예: ['model.inner.L_level.layers.0.self_attn.qkv_proj.mask', ...]
[6-5] 반환 타입: generator (generator 그대로 = 현행 bit-identical: True)
[추가] nodecay_marked=True + residual_scale=fixed -> 의도대로 raise:
       "nodecay_marked 를 켰는데 _no_weight_decay 파라미터가 없다"
[추가] input-independent + nodecay_marked=True -> group 2개, nodecay 2개 (wd=0.0)
```

읽는 법:
- (2) OFF 모델의 state_dict 키 집합이 checkpoint(변경 전 코드로 저장)와 양방향 차집합
  모두 공집합 = 완전 동일.
- (3) 사수 예시의 nq8_final_add_s2 는 29번 소유라, 같은 조건의 50번
  **nq8_final_add_s0/step_67428** 로 strict load 성공 + FPRM 필드가 아예 없던 코드로
  저장된 **nq10_v7_catnd_e2010_s3/step_96547** 도 strict load 성공.
- (4) fixed 모드 파라미터 증가는 α₁/α₂ 스칼라 2개뿐(requires_grad=False).
  fp_layers CastedLinear 의 mask buffer 는 ON 모델에만 존재 — layers.py 분리가 맞았다.
- (5) 세 플래그 False → `model.parameters()` generator 그대로 = AdamW 인자 현행 불변.

[9] yaml 재검: 두 파일 첫 바이트 `23 20`("# "), zero-width(e2 80 8b) 0건
(gram_nqueens8_fp 작성 직후 첫머리에 하나 섞였던 것을 xxd 로 잡아 정정).
hydra compose 실로드: 두 arch 모두 use_fp_transformer=True, norm_type=pre-norm,
residual_scale=fixed, α=0.75/0.25, norm_placement=none, normalize_input_injection=False.

## 3. [8] 미확정 2건 (사수 확인 대기)

- `norm_placement: none`, `normalize_input_injection: false` 는 **FPRM 논문 default
  미확인** 상태다. "추가 개입 없음"이 보수적이라는 판단이지 논문 확인값이 아니다.
- 웹/공개 repo 참조는 사수 지시로 하지 않는다 (추정 참조는 오염). 논문 PDF 는 사수가
  보유 — 확인값 회신 시 yaml 두 곳 + 이 절을 갱신한다.

## 4. [10] watcher 오판 정정

watch_final_50.sh 가 최종 checkpoint 를 total_steps(67,443/58,778)로 기다렸으나
실제 저장은 eval 경계(36×1,873=**67,428** / 14×4,198=**58,772**)다.
- 오판·건너뜀 발생: **nq8_final_add_s0, nq8_final_add_s1** 2건 (실제로는 정상 완주,
  wandb Synced 확인). → step 정정 후 watcher 재시작, 최종 step harness 자동 재개.
- nq10 6런: 오판 시점에 전부 학습 중이어서 **건너뛴 run 없음**. 정정된 58,772 로 감시 중.
- 정점 harness: EMA 정점은 두 런 모두 최종 step 과 동일(67,428)이라 별도 측정 불요.
  raw 정점은 s0=step 29,968(0.9890), s1=step 59,936(0.9890) → {EMA,.live}×hs{0,1}
  8건을 최종 step harness 종료 후 자동 실행하도록 큐(queue_nq8_peaks.sh) 가동.

## 5. push 범위

05ba06c(pair 확장 라운드) + 본 라운드 코드 커밋. fin_* harness 결과는 생성 중이라
다음 라운드에 커밋한다. 학습 launch 없음.
