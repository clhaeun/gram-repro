# [34 → 50] `gc10_paper_34` checkpoint 전송 요청

## 상태 — 아직 도착하지 않았다

2026-08-09 16:05 확인 시점에 50번에는
`checkpoints/Graph-coloring-10-ACT-torch/` 디렉토리 자체가 없었다.
(gc8 때와 같은 실패 모드를 막기 위해) 목적지 디렉토리는 미리 만들어 두었다:

```
/workspace/TinyRecursiveModels/checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34   (컨테이너 경로)
/home/chaeun/trm_chaeun/TinyRecursiveModels/checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34   (호스트 경로)
```

데이터는 이미 있다 — `data/graph-coloring-10` 실측:

| split | rows | seq_len | vocab | total_groups | total_puzzles | md5(inputs)[:12] | puzzle_indices=arange(N+1) | group_indices=arange(N+1) |
|---|---|---|---|---|---|---|---|---|
| train | 13,461 | 45 | 6 | 1,088 | 13,461 | `085538a60f49` | True | False |
| test | **192** | 45 | 6 | **192** | 192 | `79fe0dad7d33` | True | True |

test 192행 / total_groups 192 — 지시하신 값과 일치한다.

## 무엇을

대상 경로는 50번 repo 의 harness 결과 파일에 그대로 적혀 있다:
`reports/gc/gc10_step87620_ema_h0.json` → `"checkpoint": "checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/step_87620"`

```
checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/step_87620
checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/step_87620.live
checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/all_config.yaml
checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/gram.py
checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/losses.py
```

gc8 기준 step 파일 1개가 약 39MB → **총 약 80MB** 예상.
`evaluator_*` 디렉토리와 나머지 step 은 필요 없다.

## 어떻게

★ 경로 주의 — 50번은 컨테이너 경로와 호스트 경로가 다르다.
`/proc/self/mountinfo` 실측: 호스트 `/home/chaeun/trm_chaeun` = 컨테이너 `/workspace`.
ssh/scp 는 호스트에 붙으므로 **`~/trm_chaeun/...`** 를 써야 한다
(`/workspace/...` 를 쓰면 `scp: remote mkdir ...: No such file or directory`).

### 34번(bengio, `chaeun@143.248.135.34`, repo 루트 `~/gram-repro`) 에서 실행

```bash
cd ~/gram-repro
scp -p \
  checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/step_87620 \
  checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/step_87620.live \
  checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/all_config.yaml \
  checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/gram.py \
  checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/losses.py \
  chaeun@143.248.135.50:~/trm_chaeun/TinyRecursiveModels/checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/
```

`~` 가 전개되지 않으면 절대경로:
`chaeun@143.248.135.50:/home/chaeun/trm_chaeun/TinyRecursiveModels/checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34/`

### 34번에서 함께 보내 주실 md5 (도착본과 대조용)

```bash
cd ~/gram-repro && python3 -c "
import hashlib,os
D='checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34'
for f in ['step_87620','step_87620.live']:
    p=os.path.join(D,f)
    print(f, os.path.getsize(p), hashlib.md5(open(p,'rb').read()).hexdigest()[:12])
"
```

test 데이터 md5 도 대조 가능하면 함께:
`python3 -c "import numpy as np,hashlib;print(hashlib.md5(np.load('data/graph-coloring-10/test/all__inputs.npy').tobytes()).hexdigest()[:12])"`
→ 50번 값 `79fe0dad7d33`

## 도착 후 50번이 돌릴 것

`experiments/run_eps_probe_50.sh:30` 에 `gc10p` 분기를 추가해 두었다.

```bash
GPU=<free> TASK=gc10p N=192 bash experiments/run_eps_probe_50.sh
```

test 전체가 192행이므로 N=192 (batch 64 × 3) 로 전량을 쓴다.
산출: `results/eps_probe/gc10p_{sigma,lambda}.json`, 로그 `logs/eps_probe_gc10p.log`
