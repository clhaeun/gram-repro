# archive/bak_originals/ — 수정 이전 스냅샷 11개 (**유일본, 지우지 말 것**)

2026-07-30 정리에서 `*.bak*` 을 삭제 후보로 올렸다가 **보존으로 뒤집은 파일들**이다.
근거: blob 해시로 검사했더니 **11개 중 10개의 내용이 git 히스토리 어디에도 없다.**
(`git hash-object` → 해당 경로의 전 커밋 blob 과 대조, 그리고 `git cat-file -e` 로 객체 존재 확인.)
`ema.py.bak_0724_0645` 만 히스토리에 있으나 같이 둔다.

**즉 이것들은 백업이 아니라 유일본이다.** 수정 이전 코드가 어떻게 돌았는지를 증명하는 물증이고,
`reports/archive/` 의 결론 중 몇 개는 이 파일이 없으면 재확인이 불가능하다.

원위치는 파일명에서 알 수 있다(`losses.py.bak_*` → `models/losses.py`). 아래 "무엇의 증거인가"는
현재 파일과 `diff` 를 떠서 확인한 내용이다.

## 2026-07-21 — KL reduction 사건 (`REPORT_night2.md`)

| 파일 | 스냅샷 시점 | 무엇의 증거인가 |
|---|---|---|
| `losses.py.bak_0721_1603` | 수정 직전 | **★ as-run L-mean 의 물증.** `kl_loss = outputs["kl"].sum(-1).mean(-1).sum()` 이 살아 있고 sum 버전이 주석 처리돼 있다. "reduction 을 주석 토글로 골랐고 그래서 L-mean 으로 돌았다"는 [REPORT_night2.md](../../reports/archive/REPORT_night2.md) 결론의 **유일한 코드 증거**다 |
| `losses.py.bak_0721_1659` | 1차 수정 후 | 같은 라인이 아직 남아 있는 중간 상태 (변경 16줄) |
| `gram.py.bak_0721_1603` | `kl_reduction` 키 도입 전 | dataclass 에 `kl_reduction` 필드가 **없던** 상태. 키가 나중에 추가됐음을 보여준다 (변경 222줄) |
| `gram.py.bak_0721_1659` | 〃 중간 | 같은 계열 중간 상태 (변경 174줄) |
| `gram.yaml.bak_0721_1659` | 〃 | `kl_reduction` 키가 없던 arch 기본값 |
| `gram_nqueens8.yaml.bak_0721_1659` | 〃 | **★ β 눈금의 물증.** 당시 주석이 `kl_reduction: mean  # 이 실험군의 실효 눈금이다(β 값이 전부 mean 기준)` 이다. `REPORT_FINAL.md` §2.1 의 "29번/초기 run 은 mean 눈금" 판정을 코드 쪽에서 뒷받침한다 |
| `pretrain.py.bak_0721_1603` | 〃 | `dual_eval` / `nodecay_*` 손잡이가 **없던** 상태 (변경 152줄) |

## 2026-07-24 — SEQ_LEN / evaluator / EMA (`REPORT_diag2.md`, `REPORT_ema_test.md`)

| 파일 | 스냅샷 시점 | 무엇의 증거인가 |
|---|---|---|
| `gram_harness.py.bak_0724_0251` | SEQ_LEN 수정 직전 | **`SEQ_LEN = 81` 하드코딩과 `np.zeros((batch_size, SEQ_LEN))` 이 그대로 있다.** N-Queens(64)를 sudoku 길이로 깎던 버그의 물증 — [REPORT_diag2.md](../../reports/archive/REPORT_diag2.md) Task 1 |
| `nqueens.py.bak_0724_0522` | 실패 사유 분해 추가 전 | evaluator 가 `count_ok/row_ok/col_ok/d1_ok/d2_ok` 를 아직 내보내지 않던 상태. "실패 사유는 queen 개수"라는 분해가 이 시점 이후에 가능해졌다 |
| `pretrain.py.bak_0724_0645` | `dual_eval`/`nodecay_*` 추가 전 | 같은 손잡이가 없던 상태 (변경 125줄). 0721 판과 함께 두면 두 단계 변화가 보인다 |
| `ema.py.bak_0724_0645` | bias-correction 주석 추가 전 | EMA shadow 가 init 가중치 `w0` 로 시작해 `mu^t` 만큼 남는다는 설명이 **없던** 상태. 그 주석(N-Queens 17k step 에서 18%)이 [REPORT_ema_test.md](../../reports/archive/REPORT_ema_test.md) Task 1 의 근거다. **이 파일만 git 히스토리에 존재한다** |

## 왜 `git mv` 가 아니라 `mv` 였는가

`.bak*` 은 애초에 git 에 추적되지 않은 파일이라 `git mv` 대상이 아니다(그래서 히스토리에도
없었다). `mv` 후 `git add` 로 추적을 시작했다 — 이번 커밋부터 git 이 이 파일들을 보존한다.
