# reports/configs — 학습 설정 스냅샷 (전 박스 공용)

`checkpoints/<project>/<run>/` 안의 **바이너리가 아닌 것**을 그대로 복사한 것이다.
`checkpoints/` 는 `.gitignore` 대상이라 checkpoint 가 사라지면 설정도 함께 사라진다.
**재실행 시 설정 복원의 유일한 근거가 이 디렉토리다.**

- 경로 규칙: `reports/configs/<project>/<run>/`
- 박스 구분이 경로에 없다. 같은 run 이름을 두 박스가 학습했다면 **내용이 같은지 확인하고** 올릴 것.
  (2026-08-05 50번 190건 통합 시점 기준, 29번↔50번 겹치는 run 은 2건이고 둘 다 바이트 동일이었다.)
- 어느 박스에서 돌았는지는 `reports/snapshot_<box>_<시각>.md` 의 run 목록으로 판별한다.

## 무엇을 넣나

| 파일 | 왜 남기나 |
|---|---|
| `all_config.yaml` | hydra 가 실제로 해석한 최종 설정. 재실행 복원의 근거. |
| `gram.py` | 그 run 이 실제로 쓴 아키텍처 코드. 현재 HEAD 와 다를 수 있다. |
| `losses.py` | 그 run 이 실제로 쓴 loss 코드. 동상. |

`step_*` / `*.live` (nq8 ~67 MB, nq10 ~128 MB) 는 **넣지 않는다.** 그건 rsync 로 옮긴다.

★ `gram.py` / `losses.py` 는 **run 시점 스냅샷**이다. 현재 repo 의
`models/recursive_reasoning/gram.py` · `models/losses.py` 와 다르면
**어느 쪽이 틀린 게 아니라 그 run 이 옛 코드로 돈 것**이다. 결과를 비교하기 전에 확인한다.
(34번 43 run 기준 코드 스냅샷이 20 종으로 갈린다.)

## 이름 충돌 — 지우지 말고 박스 접미사로 남긴다

박스가 다르면 **같은 run 이름이 다른 설정일 수 있다.** 덮어쓰면 한쪽 근거가 사라진다.

- 규칙: 먼저 등록된 쪽이 `all_config.yaml` 을 유지하고,
  다른 쪽은 같은 디렉토리에 `all_config.<box>.yaml` 로 남긴다.
- 실제 사례 (2026-08-05): `Nqueens-8-ACT-torch/gram_nq8_fix_noema` 가
  `global_batch_size` 하나만 다르다 — 기존본 **256**, 34번(bengio) **768**.
  → 34번 것을 `all_config.bengio.yaml` 로 병기했다. 둘 다 실재한 run 이다.

## 갱신

새 run 을 띄우면 (첫 checkpoint 가 떨어진 뒤) 아래를 돌리고 **바로** 커밋한다.

```bash
find checkpoints \( -name all_config.yaml -o -name gram.py -o -name losses.py \) |
while read f; do d="reports/configs/$(dirname "${f#checkpoints/}")"; mkdir -p "$d"; cp -p "$f" "$d/"; done
git status --short reports/configs      # 기존 파일이 M 으로 뜨면 이름 충돌이다. 위 규칙대로.
```

## 이력

- 29번(hinton): `f825a1b` 로 81건 등록, 이 규칙을 만들었다.
- 50번(lecun): 2026-08-05 에 109건 추가(총 190건). 처음엔 `run_configs/lecun/` 에 올렸다가
  규칙이 갈리는 것을 막으려고 이쪽으로 통합했다.
- 34번(bengio): 2026-08-05 에 43 run 추가. `gram.py` / `losses.py` 스냅샷과
  이름 충돌 규칙을 여기서 추가했다.
- `run_configs/hinton/` 은 이 규칙 이전의 29번 구본이다. 새 설정은 여기 말고 `reports/configs/` 에 넣을 것.
