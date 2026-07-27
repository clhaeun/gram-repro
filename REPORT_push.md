# REPORT_push.md — one-hot y_emb push + 50번 실행 커맨드

- 일시: 2026-07-27
- **GPU 미사용. 학습 없음.** 정적 검증 + git 작업만.
- 커밋: **`55ef8c2`**, 원격 `https://github.com/clhaeun/gram-repro.git` `main` 에 push 완료.

---

## 1. Task 1 — sh 환경변수 + dry-run 조립

### diff (`train_nqueens8_fix.sh`)

지시대로 변수명을 `YOPROJ` / `YOSCALE` 로 맞추고 기본 scale 을 0.573 으로 바꿨다:
```bash
# 사수 처방 4: posterior 의 y 를 one-hot 으로. 기본 False = 기존 동작.
#   YOSCALE 기본 0.573 = 학습된 embed_tokens 의 실측 노름(12.97)/‖z_H‖(22.627).
#   one-hot 은 노름이 고정이라 학습으로 못 줄이므로 처음부터 맞춰 둔다.
#   (YONEHOT=False 면 arch.y_onehot_scale 은 코드상 쓰이지 않는다 -- gram.py 의
#    scale 곱은 y_onehot 분기 안에만 있다. 그래서 기본 경로는 여전히 bit-identical.)
YONEHOT="${YONEHOT:-False}"
YOPROJ="${YOPROJ:-zeropad}"          # zeropad | frozen_random
YOSCALE="${YOSCALE:-0.573}"
```
torchrun 주입:
```bash
  arch.y_onehot=${YONEHOT} arch.y_onehot_proj=${YOPROJ} arch.y_onehot_scale=${YOSCALE} \
```

`bash -n` 통과.

### dry-run 조립 결과 (DRYRUN=1, 실행 안 함)

**기본 (YONEHOT 미지정):**
```
arch.kl_coef=0.07
arch.y_onehot=False            <- 기존 동작
arch.y_onehot_proj=zeropad
arch.y_onehot_scale=0.573      <- y_onehot=False 라 코드상 미사용
ema=False
+run_name=gram_nq8_fix_b
evaluators=[{name: nqueens@NQueens}]
```

**`YONEHOT=True YOSCALE=0.573 BETA=0.07`:**
```
arch.kl_coef=0.07
arch.y_onehot=True
arch.y_onehot_proj=zeropad
arch.y_onehot_scale=0.573
ema=False
+run_name=nq8_oh_b007
```

`y_onehot_scale` 이 항상 주입되지만 `gram.py` 의 scale 곱이 `y_onehot` 분기 **안에만**
있으므로(`_y_embeddings` 의 `if self.config.y_onehot:` 반환문), False 경로는 영향 없다.

---

## 2. Task 2 — 정적 재검증 (CPU, `experiments/verify_yonehot.py`)

`ast.parse` 통과 (`gram.py`, `verify_yonehot.py`).

### num_params + state_dict 키 (실제 `gram_nq8_mean_nh512/step_17085` 대조)

| 설정 | params | buffers | 누락 키 | 추가 키 |
|---|---|---|---|---|
| `y_onehot=False` | **9,966,085** | 15,880 | **0** | **0** |
| `True, zeropad, s=1.0` | 9,966,085 | 15,880 | 0 | 0 |
| `True, frozen_random, s=1.0` | 9,966,085 | 17,416 | 0 | `y_onehot_mat` |
| **`True, zeropad, s=0.573`** | **9,966,085** | 15,880 | **0** | **0** |

네 경우 모두 **학습 파라미터 수 동일**. `False` 와 `zeropad`(scale 무관)는 기존 checkpoint 를
strict load 로 그대로 읽을 수 있다.

### y_emb 노름 (CPU forward, non-pad 위치 평균)

| 경로 | ‖y_emb‖ | ‖z_H‖(=√512=22.627) 대비 |
|---|---|---|
| 학습 embedding, 랜덤 init | 23.5597 | 1.0412 |
| **학습 embedding, 학습 후 실측** (`experiments/results/yemb_nq8_nh512.json`) | **12.9729** | **0.5733** |
| one-hot zeropad, s=1.0 | 22.6250 | 0.9999 |
| one-hot frozen_random, s=1.0 | 22.6223 | 0.9998 |
| **one-hot zeropad, s=0.573** | **12.9375** | **0.5718** |

**s=0.573 이 학습된 embedding 실측값(12.9729)과 일치한다** (차이 0.035, bf16 반올림 범위).

### 기존 config 호환 / forward

- `gram_nq8_mean_nh512`, `gram_sudoku_halt_only` 두 `all_config.yaml` 모두 `y_onehot` 키 없음
  → `y_onehot=False` 로 해석.
- CPU forward: 네 경로 전부 `logits=(8,64,3)`, `kl=(8,80,512)` 동일. shape 에러 없음.

---

## 3. Task 3 — push 결과

### 사전 검사

| 항목 | 결과 |
|---|---|
| 커밋 대상 파일 수 | **50** |
| 총 용량 | **420 KB** |
| 1MB 초과 파일 | **없음** |
| `checkpoints/` `data/` `wandb/` `logs/` `*.npy` `*.pt` 혼입 | **없음** (`.gitignore` 로 제외) |

### 결과

```
커밋   : 55ef8c2  "one-hot y_emb 토글(zeropad/frozen_random) + y_onehot_scale, ..."
변경   : 50 files changed, 7767 insertions(+), 12 deletions(-)
push   : 4c847f1..55ef8c2  main -> main   (exit 0)
확인   : git rev-list --left-right --count origin/main...HEAD  ->  0  0
         git ls-remote origin main -> 55ef8c2c27a930bdee7a797576cae16b8c254ce9
```

주요 변경:
- `models/recursive_reasoning/gram.py` (+55/−2) — one-hot 토글
- `train_nqueens8_fix.sh` (+79, 신규)
- 진단 스크립트 5종: `diag_contradiction.py`, `diag_yemb.py`, `make_raw_ema.py`,
  `sweep_harness.py`, `verify_yonehot.py`
- REPORT 7종, `experiments/results/*.json` 측정 결과

### ⚠️ 인증에 관한 사실 두 가지 (50번에 영향)

1. **이 서버의 push 는 VS Code Server 의 `GIT_ASKPASS` 로 통과했다.**
   ```
   GIT_ASKPASS=/home/chaeun/.vscode-server/cli/servers/.../extensions/git/dist/askpass.sh
   ```
   `credential.helper` 없음, `gh auth` 로그인 없음, `~/.ssh/id_*` 없음.
   즉 **이 인증은 VS Code 세션에 묶여 있고 다른 서버로 따라가지 않는다.**

2. **저장소가 private 이다.** 인증 없이 접근 시:
   ```
   $ GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/true git ls-remote https://github.com/clhaeun/gram-repro.git main
   remote: Repository not found.
   fatal: Authentication failed
   ```
   → **50번에서 `git pull` 하려면 그쪽에 인증을 따로 세팅해야 한다.** §4-0 참조.

---

## 4. Task 4 — 50번(lecun, docker, GPU 0~3)에서 돌릴 커맨드

### 4-0. 먼저 인증 (private repo라 필수, 셋 중 하나)

```bash
# (A) gh CLI — 가장 간단
gh auth login          # 브라우저/디바이스 코드
gh auth setup-git

# (B) PAT 저장 — docker 컨테이너에 적합
git config --global credential.helper store
git pull               # 최초 1회 username + PAT(repo 권한) 입력, 이후 저장됨

# (C) SSH 키
ssh-keygen -t ed25519 -C "gram-50" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub          # -> GitHub Settings > SSH and GPG keys 에 등록
git remote set-url origin git@github.com:clhaeun/gram-repro.git
```

### 4-1. pull

```bash
cd <50번의 gram 폴더>          # docker 안 경로에 맞춰 조정
git fetch origin && git status  # 로컬 수정이 있으면 먼저 정리(stash/commit)
git pull origin main            # 55ef8c2 를 받는다
git log --oneline -1            # 55ef8c2 확인
```

**주의**: `checkpoints/` `data/` `wandb/` `logs/` 는 `.gitignore` 라 **git 으로 안 간다.**
50번에 `data/nqueens-8/` 이 없으면 학습이 안 된다 — rsync 로 따로 옮기거나
`dataset/build_nqueens_dataset.py` 로 재생성해야 한다.

### 4-2. 실행 전 확인 (GPU 안 씀, 각 1초)

```bash
# 조립 확인 (실행 안 함) — y_onehot 이 붙는지, evaluators 형식이 맞는지
DRYRUN=1 RUN=nq8_oh_b007 YONEHOT=True YOSCALE=0.573 BETA=0.07 bash train_nqueens8_fix.sh

# venv 확인: 스크립트가 `source ~/venv-gram/bin/activate 2>/dev/null || true` 라
# 50번에 그 venv 가 없으면 조용히 현재 python 으로 진행한다. torch 가 맞는지 먼저 볼 것.
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

### 4-3. 3-run (GPU 0,1,2 / 각 1장 / 1장당 ~1h40m, 병렬)

```bash
cd <50번의 gram 폴더>

# 1. one-hot 단독 효과 (β 원래값, one-hot 만) — 가장 중요한 대조군
RUN=nq8_oh_b007  YONEHOT=True  YOSCALE=0.573 BETA=0.07 EMA=False GPUS=0 PORT=29651 bash train_nqueens8_fix.sh

# 2. one-hot + 좋은 β
RUN=nq8_oh_b15   YONEHOT=True  YOSCALE=0.573 BETA=1.5  EMA=False GPUS=1 PORT=29652 bash train_nqueens8_fix.sh

# 3. 대조군 (one-hot off, β=1.5 — bengio 에서 prior valid 30.29% 나온 값의 재현)
RUN=nq8_ctrl_b15 YONEHOT=False               BETA=1.5  EMA=False GPUS=2 PORT=29653 bash train_nqueens8_fix.sh
```

각 run 은 launch 직전 GPU 점유(>1000MiB)를 검사해 걸리면 `FATAL` 로 멈춘다.
백그라운드로 돌리려면 각 줄 앞에 `nohup`, 뒤에 `> logs/<RUN>.nohup 2>&1 &`.

**launch 직후 확인할 것**
```bash
grep -c "SWITCH TO EMA" logs/<RUN>.log        # 0 이어야 함 (EMA off)
grep -E "y_onehot|kl_coef|lr_warmup" checkpoints/Nqueens-8-ACT-torch/<RUN>/all_config.yaml
grep -oE "\[EVAL @ step [0-9]+\].*nqueens/accuracy=[0-9.]+" logs/<RUN>.log | head -2
#   -> nqueens/accuracy 가 찍히면 evaluator 가 실제로 켜진 것
```

### 4-4. 완주 후 측정 (bengio 와 동일 스크립트)

```bash
# peak step 찾기
grep -oE "\[EVAL @ step [0-9]+\].*nqueens/accuracy=[0-9.]+" logs/<RUN>.log

# prior N=20 + posterior, valid 기준
python experiments/sweep_harness.py --ckpt-dir checkpoints/Nqueens-8-ACT-torch/<RUN> \
  --step <peak 또는 step_17085> --N 20 --seed 0 --out experiments/results/sweep_<RUN>.json

# ★ one-hot 성공 판정: y 복사가 줄었는가
python experiments/diag_yemb.py --ckpt-dir checkpoints/Nqueens-8-ACT-torch/<RUN> \
  --step <step> --data data/nqueens-8/test --task nqueens --out experiments/results/yemb_<RUN>.json
```

**판정 기준** (`diag_yemb.py` 출력):

| 지표 | 현재 N-Queens (실패) | 성공한 Sudoku | one-hot 성공이면 |
|---|---|---|---|
| `post_wrong_label_reproduces_it` | 100% | **0.00%** | 크게 하락 |
| `post_shuffled_same_as_real` | 100% | **0.00%** | 크게 하락 |
| prior valid | 0~30% | (69.53% exact) | 상승 |

(Sudoku 값 출처: `experiments/results/yemb_sudoku_halt_only.json`)

---

## 5. 임의로 정한 것

1. **git identity 를 `clhaeun` / `catgptshared@gmail.com` 로 `--local` 설정했다.**
   커밋에 반드시 필요한데 로컬·전역 모두 미설정이었고, 사용자에게 물어본 질문이
   중단돼 답을 못 받았다. 근거: 원격 URL 소유자명(`clhaeun`), 세션 등록 이메일.
   **틀렸으면 지금 고칠 수 있다** (이미 push 됐지만 커밋 1개뿐):
   ```bash
   git config --local user.email "<맞는 이메일>"
   git commit --amend --reset-author --no-edit
   git push --force-with-lease origin main
   ```
2. **`YOSCALE` 기본을 0.573 으로** 했다(지시대로). config 필드
   `y_onehot_scale` 의 **코드 기본값은 1.0 그대로** 두었다 — 기존 checkpoint 해석을
   바꾸지 않기 위해서다. 즉 sh 를 통하면 0.573, 코드를 직접 쓰면 1.0.
3. **변수명을 `YOH_PROJ`/`YOH_SCALE` → `YOPROJ`/`YOSCALE` 로 바꿨다** (지시문 표기에 맞춤).
   이전 리포트(`REPORT_prep.md` §1-7)에 적힌 옛 이름은 더 이상 동작하지 않는다.
4. **커밋을 1개로 묶었다.** 지시의 `git add -A` 를 그대로 따랐다. one-hot 구현,
   진단 스크립트, REPORT 7종, 측정 json 이 한 커밋에 들어간다.
5. **`DRYRUN=1` 일 때 GPU 점유검사 생략**은 앞선 작업(`REPORT_prep.md` §5-6)에서 넣은 것으로,
   이번 커밋에 함께 들어갔다.
6. **50번의 GPU 번호를 0,1,2 로 적었다** (지시문 "GPU 0~3" 기준). docker 안에서 GPU 번호가
   다르게 보일 수 있으니 `nvidia-smi` 로 확인 후 조정할 것. **미확인.**
7. **50번 경로·venv·데이터 존재 여부는 확인 못 했다** (그 서버에 접속하지 않았다).
   §4-1, §4-2 의 경고는 코드에서 추론한 것이다.
