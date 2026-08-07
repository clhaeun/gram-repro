# TODO — 감사 보고 12건 (62번 접수, 논문 마감 후 처리)

접수일: 2026-08-06 (62번 세션)

★ **이 문서는 TODO 기록 전용이다. 아래 항목 때문에 코드를 고치지 마라.**
진행 중인 run 의 재현성과 checkpoint 로딩이 걸려 있다.

## 분류

### 즉시 처리 불필요 — 실행 경로 밖 (1, 2, 3번)
현재 전 arm 이 `stablemax` + `weight_dropout=0.0` 이라 지적된 코드가 실행되지 않는다.

- **2번 (mask 버퍼가 state_dict 에 포함)** 은 **버그가 아니라 의도된 설계**다.
  `models/fp_layers.py` 를 `models/layers.py` 와 분리한 이유가 정확히 **기존 checkpoint 보호**다.
  "고치면" 기존 체크포인트가 로드 불가가 된다. ★건드리지 마라.

### 처리 금지 — 재현성/체크포인트 파괴 (4, 5, 9번)
손대면 진행 중인 run 의 재현성이 깨지고 checkpoint 로딩이 실패한다.
논문 마감 후에도, 고칠 거면 **새 arch 이름 + 새 파일**로 분기해야 한다
(`fp_layers.py` 분리와 같은 패턴).

### 스타일 — 논문 마감 후 (나머지)
기능 영향 없음. 마감 후 일괄 처리.

---

## 별건 — 62번 박스(jurgen) 로컬 환경 이슈 2건

`scripts/train_nqueens10_stageA.sh` 에 **커밋하지 않은 로컬 수정 2줄**이 있다.
실험 의미론(semantics)은 하나도 바꾸지 않는다. 다른 박스의 기본 동작도 그대로다.

1. **python 경로**: 이 박스에 `~/venv-gram` 이 없다. conda `gram` 이 정답 경로.
   `export PATH="$HOME/miniconda3/envs/gram/bin:$PATH"` 를 추가해 명시 고정.
   (스크립트 헤더가 이미 경고하는 그 함정 — `source ... || true` 라 조용히 실패한다.)

2. **wandb**: 이 박스엔 wandb 자격증명이 **전무**하다
   (`~/.netrc` 없음, `WANDB_API_KEY` 미설정, `~/.config/wandb/settings` 없음).
   런처가 `export WANDB_MODE=online` 을 **하드코딩**해서 첫 기동 3셀이 전부
   `wandb.errors.UsageError: No API key configured` 로 즉사했다.
   → `export WANDB_MODE="${WANDB_MODE:-online}"` 로 바꿔 호출자가 override 만 할 수 있게 열었다.
     **기본값은 online 그대로**라 34/50/29번 동작은 불변.
   → 62번 3셀은 전부 `WANDB_MODE=offline` 로 기동. 지표는 `wandb/offline-run-*` 에
     전부 남으므로, 키가 생기면 `wandb sync wandb/offline-run-*` 로 나중에 올릴 수 있다.

★ **상류(upstream) 결정 필요**: 위 2번을 런처에 정식 반영할지.
   반영하면 자격증명 없는 박스에서 "조용히 죽는" 대신 굴러간다.
   기본값을 안 바꾸므로 기존 박스에 영향 없다. — 62번 판단 대기.

## 셀 Q (outer_only) — 확정 결과로 승격됨

step 50 이내 forward NaN 으로 kill. **실패가 아니라 결과다.**
전문은 → **`reports/RESULT_outer_only_nan_62.md`**
(관측·메커니즘·FPRM Theorem 1 과의 관계·로그 경로 전부 그쪽에 있다. 여기 중복 기술 안 한다.)

62번 회신: 진단 수용, kill 판단 옳음. "outer_only 지시는 내가 틀렸다" 확인받음.

남는 TODO 는 런처 손잡이뿐:
- 런처에 `OUTER_ONLY` 손잡이가 없어 scratchpad wrapper 로 우회했다.
  정식화할 거면 `OUTER_ONLY` 를 런처에 추가하는 게 맞다 — **마감 후**.
  (단 이 축은 현 arch 에서 학습 불가로 판명됐으므로 우선순위 낮다.)

## 데이터 md5 기준값 부재 — 별도 파일로 분리
→ **`reports/DATA_MD5.md`**. 50번 대조가 아직 비어 있다.
