# in-loop eval 스냅샷 — hinton @ 2026-08-05 16:50:04

checkpoint 소실 대비 텍스트 보존본. `logs/<run>.log` 의
`[EVAL @ step]`(EMA) / `[EVAL RAW @ step]`(raw) 을 그대로 옮긴 것이다.

## nq10_A3p_ii_a7525_w2000

- 진행: **step 35780 / 138548** (25.8%) · **5.12 it/s** · elapsed 2:09:15 · **ETA 5:34:16**
- 로그: `logs/nq10_A3p_ii_a7525_w2000.log` (mtime 2026-08-05 16:50:04)
- eval 지점: 8개
- 최근 train @ step 35750: `train/accuracy=0.9742 train/exact_accuracy=0.4455 train/kl_loss=0.6067 train/lm_loss=0.1196 train/lprm_loss=0.0090 train/q_halt_accuracy=0.4273 train/q_halt_loss=0.4577 lr=0.0001`

설정 (실행 커맨드라인):

```
/home/chaeun/miniconda3/envs/gram/bin/python3.11 -u pretrain.py arch=gram_nqueens10_fp data_paths=[data/nqueens-10] evaluators=[{name: nqueens@NQueens}] epochs=990 eval_interval=30 seed=0 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 lr_min_ratio=1.0 nodecay_marked=True arch.mlp_t=False arch.pos_encodings=rope arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 global_batch_size=768 arch.kl_coef=0.045 arch.noise_head_expansion=1.5 arch.residual_scale=input-independent arch.alpha_1_init=0.75 arch.alpha_2_init=0.25 ema=True ema_rate=0.9999 lr_warmup_steps=2000 +run_name=nq10_A3p_ii_a7525_w2000
```

| step | EMA strict_acc | raw strict_acc | EMA queen_cnt | raw queen_cnt | EMA kl_loss | raw kl_loss | EMA lm_loss | raw lm_loss | EMA exact_acc | raw exact_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| 4198 | 0.0000 | 0.0002 | 8.8778 | 9.6370 | 0.0000 | 0.0000 | 0.4521 | 0.3759 | 0.0000 | 0.0000 |
| 8396 | 0.0000 | 0.0006 | 3.3368 | 7.5273 | 0.0000 | 0.0000 | 0.3157 | 0.3331 | 0.0000 | 0.0002 |
| 12594 | 0.0000 | 0.0048 | 3.2688 | 9.3470 | 0.0000 | 0.0000 | 0.2930 | 0.4105 | 0.0000 | 0.0015 |
| 16792 | 0.0000 | 0.0019 | 4.6698 | 9.4786 | 0.0000 | 0.0000 | 0.2913 | 0.5049 | 0.0000 | 0.0006 |
| 20990 | 0.0000 | 0.0029 | 7.0927 | 9.0845 | 0.0000 | 0.0000 | 0.2874 | 0.4465 | 0.0000 | 0.0009 |
| 25188 | 0.0000 | 0.0029 | 7.3446 | 8.5860 | 0.0000 | 0.0000 | 0.2753 | 0.3385 | 0.0000 | 0.0009 |
| 29386 | 0.0008 | 0.0087 | 9.5124 | 9.1431 | 0.0000 | 0.0000 | 0.3212 | 0.3748 | 0.0002 | 0.0027 |
| 33584 | 0.0031 | 0.0104 | 10.2096 | 9.8112 | 0.0000 | 0.0000 | 0.3686 | 0.4372 | 0.0009 | 0.0025 |

## nq10_A3p_ii_a7525_lr3e4

- 진행: **step 34346 / 138548** (24.8%) · **4.97 it/s** · elapsed 2:09:12 · **ETA 5:49:42**
- 로그: `logs/nq10_A3p_ii_a7525_lr3e4.log` (mtime 2026-08-05 16:50:04)
- eval 지점: 8개
- 최근 train @ step 34300: `train/accuracy=0.9820 train/exact_accuracy=0.6753 train/kl_loss=0.7220 train/lm_loss=0.0962 train/lprm_loss=0.0110 train/q_halt_accuracy=0.6753 train/q_halt_loss=0.4789 lr=0.0003`

설정 (실행 커맨드라인):

```
/home/chaeun/miniconda3/envs/gram/bin/python3.11 -u pretrain.py arch=gram_nqueens10_fp data_paths=[data/nqueens-10] evaluators=[{name: nqueens@NQueens}] epochs=990 eval_interval=30 seed=0 lr=3e-4 puzzle_emb_lr=3e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 lr_min_ratio=1.0 nodecay_marked=True arch.mlp_t=False arch.pos_encodings=rope arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 global_batch_size=768 arch.kl_coef=0.045 arch.noise_head_expansion=1.5 arch.residual_scale=input-independent arch.alpha_1_init=0.75 arch.alpha_2_init=0.25 ema=True ema_rate=0.9999 lr_warmup_steps=500 +run_name=nq10_A3p_ii_a7525_lr3e4
```

| step | EMA strict_acc | raw strict_acc | EMA queen_cnt | raw queen_cnt | EMA kl_loss | raw kl_loss | EMA lm_loss | raw lm_loss | EMA exact_acc | raw exact_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| 4198 | 0.0000 | 0.0217 | 7.4214 | 8.6502 | 0.0000 | 0.0000 | 0.4267 | 0.3253 | 0.0000 | 0.0064 |
| 8396 | 0.0000 | 0.4071 | 5.7723 | 9.6370 | 0.0000 | 0.0000 | 0.3492 | 0.3416 | 0.0000 | 0.1169 |
| 12594 | 0.0000 | 0.3552 | 12.8154 | 9.8721 | 0.0000 | 0.0000 | 0.4063 | 0.4967 | 0.0000 | 0.0961 |
| 16792 | 0.0000 | 0.2774 | 15.8093 | 9.8494 | 0.0000 | 0.0000 | 0.3886 | 0.5696 | 0.0000 | 0.0774 |
| 20990 | 0.0000 | 0.6345 | 8.9658 | 9.5114 | 0.0000 | 0.0000 | 0.2798 | 0.3586 | 0.0000 | 0.1892 |
| 25188 | 0.0001 | 0.6504 | 7.1257 | 9.2780 | 0.0000 | 0.0000 | 0.2457 | 0.3410 | 0.0000 | 0.1974 |
| 29386 | 0.0037 | 0.5986 | 6.3898 | 9.3469 | 0.0000 | 0.0000 | 0.2284 | 0.3391 | 0.0015 | 0.1765 |
| 33584 | 0.0786 | 0.6736 | 6.9421 | 9.1110 | 0.0000 | 0.0000 | 0.2155 | 0.3278 | 0.0327 | 0.2074 |

