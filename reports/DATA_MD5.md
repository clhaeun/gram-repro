# 데이터셋 md5 기준값 대장

★ **이 파일의 존재 이유**: 62번에서 nqueens-10 데이터를 검산하려 했으나
**기준값이 repo 어디에도 기록돼 있지 않아 대조가 불가능했다.**
같은 사고를 막기 위해 여기에 서버별 실측값을 모은다.
새 서버에서 데이터를 쓰기 전 이 표와 대조하라.

## 검산 명령 (모든 서버 공통)

```bash
cd ~/gram-repro && python3 -c "
import numpy as np, hashlib, json
for sp in ['train','test']:
    a=np.load(f'data/nqueens-10/{sp}/all__inputs.npy')
    m=json.load(open(f'data/nqueens-10/{sp}/dataset.json'))
    print(sp, a.shape, m['total_groups'], hashlib.md5(a.tobytes()).hexdigest()[:12])
"
```

md5 는 `all__inputs.npy` 의 `tobytes()` 앞 12자리다.

## nqueens-10

| 서버 | split | shape | total_groups | md5(12) | 확인일 |
|---|---|---|---|---|---|
| **62번** (jurgen) | train | (107480, 100) | 107480 | `703026643e60` | 2026-08-06 |
| **62번** (jurgen) | test  | (19220, 100)  | 19220  | `9b6a03fe8c8a` | 2026-08-06 |
| **50번** | train | ❓ 미기록 | ❓ | ❓ **대조 필요** | — |
| **50번** | test  | ❓ 미기록 | ❓ | ❓ **대조 필요** | — |

62번 데이터 파일 타임스탬프: `2026-08-02 13:12` (이후 변경 없음, 재빌드 안 함).

### 상태
- shape 와 `total_groups` 는 **기대값과 정확히 일치** → 62번은 이 값으로 진행 승인됨(62번 지시).
- ★ **50번 md5 미대조.** 위 명령을 50번에서 실행해 이 표를 채울 것.
  일치하면 그대로 확정, 불일치하면 즉시 보고 — 두 서버 결과의 직접 비교가 무효가 된다.

## nqueens-8

| 서버 | split | shape | total_groups | md5(12) | 확인일 |
|---|---|---|---|---|---|
| — | — | 미기록 | — | — | — |

★ nq8 도 같은 문제가 있다. 기준값이 없다. 여유 생기면 채울 것.
