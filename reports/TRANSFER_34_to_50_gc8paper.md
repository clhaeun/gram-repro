# [34 → 50] `gc8_paper_34` checkpoint 전송 요청

## 왜

50번의 ε 전파 probe (`reports/RESULT_eps_probe_50.md` ④) 가 gc8 을
`checkpoints/Graph-coloring-8-ACT-torch/gc8_repro_s0 / step_9400` 으로 쟀는데,
이 셀은 붕괴 셀이다:

- `reports/snapshot_lecun_20260805_0755.md` §gc8_repro_s0 — eval 20지점 전부
  `EMA exact_acc 0.0000`, `EMA kl_loss 0.0000`.
- 50번 probe 실측 — prior σ 가 48 transition 전 구간 0.997309~0.997317
  (변동폭 0.0009%). σ head 가 상수를 뱉는 상태.

이 상태에서 나온 λ(초반) 0.080~0.112 는 정상 모델의 수축인지 붕괴 모델의
산물인지 구분되지 않는다. coverage 0.82~0.90 인 셀로 다시 재야 한다.

## 무엇을

box 34 (bengio) 의

```
checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/step_45540
checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/step_45540.live
checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/all_config.yaml
checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/gram.py
checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/losses.py
```

`step_*` 파일은 gc8 기준 1개당 약 39MB → **총 약 80MB**.
`evaluator_*` 디렉토리와 나머지 step 은 필요 없다.

근거 (이 경로는 50번 repo 의 harness 결과 파일에 그대로 적혀 있다):
`reports/gc/gc8_step45540_ema_h0.json` → `"checkpoint": "checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/step_45540"`

| 파일 | coverage@20 (N=20, sigma_scale 1.0, n=255) |
|---|---|
| `step_45540` (EMA) | 0.8221 (h0) / 0.8388 (h1) |
| `step_45540.live` | 0.8879 (h0) / 0.8956 (h1) |

## 어떻게

50번 컨테이너에는 `rsync` 가 없고 `scp` / `ssh` 만 있다.

★★ 경로 주의 — 50번은 컨테이너 경로와 호스트 경로가 다르다.
`/proc/self/mountinfo` 실측: 호스트 `/home/chaeun/trm_chaeun` 가 컨테이너
`/workspace` 로 bind mount 돼 있다 (ext4, /dev/nvme1n1p2).

| 보는 위치 | repo 루트 |
|---|---|
| 50번 컨테이너 안 (probe 를 돌리는 곳) | `/workspace/TinyRecursiveModels` |
| **50번 호스트 (scp/ssh 로 붙는 곳)** | **`/home/chaeun/trm_chaeun/TinyRecursiveModels`** (= `~/trm_chaeun/...`) |

scp 목적지에 `/workspace/...` 를 쓰면
`scp: remote mkdir ...: No such file or directory` 가 난다 — ssh 는 호스트에
붙기 때문이다. 아래 명령은 전부 호스트 경로를 쓴다.
34번(bengio) 호스트는 **`chaeun@143.248.135.34`**, repo 루트 `~/gram-repro`
(62번은 `chaeun@143.248.135.62`, `reports/TRANSFER_50_to_62.md:4`).

### 0. 50번에 목적지 디렉토리 생성 (50번에서)

컨테이너 안에서 만들어도 호스트에 그대로 보인다 (같은 파일시스템).

```bash
# 50번 컨테이너 안에서
cd /workspace/TinyRecursiveModels
mkdir -p checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34
# 또는 50번 호스트에서
mkdir -p ~/trm_chaeun/TinyRecursiveModels/checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34
```

★ scp 는 목적지 디렉토리를 스스로 만들지 않는다. 이 단계를 건너뛰면 실패한다.

### 1-A. 34번에서 밀어 넣기 (34번에서 실행)

```bash
cd ~/gram-repro
scp -p \
  checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/step_45540 \
  checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/step_45540.live \
  checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/all_config.yaml \
  checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/gram.py \
  checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/losses.py \
  chaeun@143.248.135.50:~/trm_chaeun/TinyRecursiveModels/checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34/
```

### 1-B. 50번에서 끌어오기 (50번에서 실행, 34번 접속이 가능한 경우)

```bash
cd /workspace/TinyRecursiveModels
D=checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34
mkdir -p $D
scp -p chaeun@143.248.135.34:~/gram-repro/$D/{step_45540,step_45540.live,all_config.yaml,gram.py,losses.py} $D/
```

### 2. 도착 검산 (50번에서)

```bash
cd /workspace/TinyRecursiveModels && python3 -c "
import torch,hashlib,os,math
D='checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34'
for f in ['step_45540','step_45540.live']:
    p=os.path.join(D,f)
    print(f, os.path.getsize(p), hashlib.md5(open(p,'rb').read()).hexdigest()[:12])
    sd=torch.load(p,map_location='cpu',weights_only=True)
    print('   n_keys',len(sd),
          'prior_logvar_bias',float([v for k,v in sd.items() if k.endswith('prior_logvar_bias')][0]))
"
```

34번에서 같은 명령을 돌려 md5(12) 2건을 함께 보내 주면 대조한다.

## 도착 후 50번이 돌릴 것

`experiments/run_eps_probe_50.sh` 에 `gc8p` 분기를 추가해 두었다.

```bash
GPU=6 TASK=gc8p N=192 bash experiments/run_eps_probe_50.sh
```

산출: `results/eps_probe/gc8p_{sigma,lambda}.json`
