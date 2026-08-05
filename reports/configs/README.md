# reports/configs — 학습 설정 스냅샷 (전 박스 공용)

`checkpoints/<project>/<run>/all_config.yaml` 을 그대로 복사한 것이다.
`checkpoints/` 는 `.gitignore` 대상이라 checkpoint 가 사라지면 설정도 함께 사라진다.
**재실행 시 설정 복원의 유일한 근거가 이 디렉토리다.**

- 경로 규칙: `reports/configs/<project>/<run>/all_config.yaml`
- 박스 구분이 경로에 없다. 같은 run 이름을 두 박스가 학습했다면 **내용이 같은지 확인하고** 올릴 것.
  (2026-08-05 50번 190건 통합 시점 기준, 29번↔50번 겹치는 run 은 2건이고 둘 다 바이트 동일이었다.)
- 어느 박스에서 돌았는지는 `reports/snapshot_<box>_<시각>.md` 의 run 목록으로 판별한다.

## 이력

- 29번(hinton): `f825a1b` 로 81건 등록, 이 규칙을 만들었다.
- 50번(lecun): 2026-08-05 에 109건 추가(총 190건). 처음엔 `run_configs/lecun/` 에 올렸다가
  규칙이 갈리는 것을 막으려고 이쪽으로 통합했다.
- `run_configs/hinton/` 은 이 규칙 이전의 29번 구본이다. 새 설정은 여기 말고 `reports/configs/` 에 넣을 것.

새 run 을 띄우면 이 디렉토리에도 설정을 복사해 함께 커밋한다.
