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

## 3. [8] 논문 대조 결과 (사수 PDF 확인 회신 반영, 2026-08-02)

- `norm_placement: none` — **FPRM Appendix Table 6 확인값** (FPRM/Looped Transformer
  모두 none). 미확인 → 확인으로 승격, yaml 주석 갱신 완료.
- `normalize_input_injection: false` — **코드 전용 knob, 논문 전문에 서술 없음.
  false 유지.** yaml 주석 갱신 완료.
- 웹/공개 repo 참조는 하지 않았고 하지 않는다 (추정 참조는 오염).

### 3-2. [8-2] lr/wd 근거 정정

"pre-norm 이라서 wd 를 내려야 한다"는 **틀렸다**. FPRM 표 전개:

| | lr | wd | α init | optimizer |
|---|---|---|---|---|
| Table 4 Sudoku | 1e-3 | 1e-3 | 0.75, 0.25 | AdamW |
| Table 5 Maze | 1e-4 | 1.0 | 0.75, 0.25 | Adam-Atan2 |
| Table 6 state-track | 1e-3 | 1e-2 | 0.5, 0.5 | Adam-Atan2 |
| Table 7 ARC | 1e-3 | 1e-2 | 0.75, 0.25 | Adam-Atan2 |

Maze 가 TRM 과 동일한 lr 1e-4 / wd 1.0 에서 pre-norm + residual scaling 을 쓴다 —
wd 인하는 pre-norm 의 요구사항이 아니라 task 별 튜닝이다. Table 4 값을 쓰는 arm 의
근거는 "우리 task 가 Sudoku 이므로 Table 4 를 따름"으로 정정. α init 도 task 의존
(state-tracking 만 0.5/0.5) — 우리는 0.75/0.25 유지. 두 yaml 주석에 표와 함께 기록.

### 3-3. [8-3] A3′ arm — run config 두 벌 (`train_nqueens8_fp.sh`)

새 arch yaml 없음 (arch 는 A2 와 동일한 gram_nqueens8_fp). 런처 하나에 ARM 필수
인자로 두 run config 를 담았다 (ARM 미지정 시 즉시 중단 — 잘못된 arm 무음 launch 방지):

| arm | lr | wd | 근거 |
|---|---|---|---|
| `ARM=A2` | 1e-3 | 1e-3 | FPRM Table 4 (우리 task 가 Sudoku 계열) |
| `ARM=A3p` (A3′) | 1e-4 | 1.0 | Maze 선례 + 기존 확립 사실(전부 wd=1.0 전제)과의 비교 축. A0(기존 L_level, 동일 lr/wd)와 L_level 하나만 달라 pre-norm 단독 효과 직접 측정 |

DRYRUN 조립 확인: A2 → lr=1e-3 weight_decay=1e-3 (puzzle_emb 동일), A3p → lr=1e-4
weight_decay=1.0. puzzle_emb_lr/wd 는 repo 관례대로 본체 추종. **학습 launch 안 함.**

## 4. [10] watcher 오판 정정 + step 계산식

### step 번호의 출처와 off-by-N 의 원인 (개별 정정이 아니라 계산식 확정)

- 내가 watcher 에 넣은 67,443/58,778 은 **하드코딩**이었고, 출처는 pretrain.py:279
  `total_steps = int(epochs × total_groups / global_batch_size)` 다.
- 실제 최종 step 이 그보다 작은 이유: `puzzle_dataset.py:226-227` `_iter_train` 이
  iter(=eval_interval epochs 단위)마다 batch 를 뽑다가 **마지막 미달 배치를 버린다**
  ("Drop last batch"). 따라서

  **final = (epochs // eval_interval) × floor(eval_interval × total_groups / global_batch_size)**

- 4개 anchor 전수 검산 (전부 일치):
  | run | 식 | 결과 | 실측 |
  |---|---|---|---|
  | nq8_final (7200ep/200) | 36 × floor(200×7,194/768=1873.4) | 67,428 | 67,428 ✓ |
  | nq10_final (420ep/30) | 14 × floor(30×107,480/768=4198.4) | 58,772 | 58,772 ✓ |
  | nq10_v7 (2010ep/30, 구데이터) | 67 × floor(30×36,907/768=1441.7) | 96,547 | 96,547 ✓ |
  | nq8_v5/v6 (12000ep/200, 구데이터) | 60 × floor(200×4,376/768=1139.6) | 68,340 | 68,340 ✓ |

- 재발 방지: `/workspace/home/chaeun/calc_final_step.sh` — run 의 all_config.yaml +
  data 의 train/dataset.json 에서 위 식으로 계산 (하드코딩 금지). ★단 data_paths 의
  "현재" dataset.json 을 읽으므로 run 이후 데이터가 재빌드된 과거 run 에는 못 쓴다
  (v7 을 지금 돌리면 pair 데이터 groups 로 281,266 이 나온다 — launch 직후 걸 때 쓰는 용도).

### 오판으로 harness 를 건너뛴 run 전수 목록

| run | 오판 시각 | 실제 상태 | 조치 |
|---|---|---|---|
| nq8_final_add_s0 | 08-02 13:20 | step_67428 정상 완주 (wandb Synced) | 정정 후 최종 step harness 4건 완료 |
| nq8_final_add_s1 | 08-02 13:20 | step_67428 정상 완주 (wandb Synced) | 동일 |

이상 **2건이 전부**다. nq10 6런은 오판 시점에 전부 학습 중이었고, 정정된 58,772 기준
watcher 가 add_s0/s1·catnd_s0/s1 완주분 harness 를 진행 중 (add_s2/catnd_s2 는 학습 중).

- 정점 harness: EMA 정점은 두 nq8 런 모두 최종 step 과 동일(67,428)이라 별도 측정 불요.
  raw 정점은 s0=step 29,968(0.9890), s1=step 59,936(0.9890) → {EMA,.live}×hs{0,1}
  8건을 최종 step harness 종료 후 자동 실행하도록 큐(queue_nq8_peaks.sh) 가동.

## 5. push 범위

05ba06c(pair 확장 라운드) + 본 라운드 코드 커밋. fin_* harness 결과는 생성 중이라
다음 라운드에 커밋한다. 학습 launch 없음.
