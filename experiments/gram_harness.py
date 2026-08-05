"""Shared plumbing for the current N-Queens / GC harnesses.

GRAM's latent transition is stochastic, so K independent forward passes on the
*same* input already give K distinct answers -- unlike TRM, which is deterministic
and needs test-time augmentation for sample diversity.  Every rollout here is
therefore plain repetition: no symmetry transform, no inverse mapping.

Provides:
  load_model()      checkpoint -> eager ACTLossHead (all_config.yaml + ema=False)
  rollout()         K forward passes -> preds / v_logits / token accuracy
  sigma_scale(), mu_scale()   rollout 이 쓰는 추론시 스케일링 컨텍스트

소비자는 세 곳뿐이고 전부 load_model + rollout 만 쓴다:
  experiments/sweep_harness.py  experiments/gc_harness.py  experiments/coverage_curve.py

★ 2026-08-05 축소.  참조 0 으로 확인된 17종을 제거했다:
    dist_setup dist_gather dist_teardown shard sharded_rollout load_puzzles
    capture_kl capture_sigma pass_at_j majority_vote metrics auroc
    print_hits_histogram save_json add_common_args out_dir_for SEQ_LEN
  전부 experiments/legacy/gram_harness_legacy.py 에 그대로 남아 있고,
  e1~e4 · sigma_probe · eval_harness 는 그쪽을 import 한다.
  capture_* 프로브의 유래(diag_sigma.py pattern)와 kl_loss 호출 횟수 주석도
  사본 쪽에 있다.  선례는 exp/diag_sigma.py.
"""
import os
from contextlib import contextmanager

import numpy as np
import torch
from omegaconf import OmegaConf

import pretrain as P

PAD = 0                 # dataset.json: pad_id = ignore_label_id = 0


# --------------------------------------------------------------------------- #
# checkpoint / data loading
# --------------------------------------------------------------------------- #
def load_model(ckpt_dir, step):
    """eval_checkpoint.py pattern: all_config.yaml + load_checkpoint, ema=False.

    Returns the EAGER ACTLossHead (unwrapping torch.compile via ._orig_mod) so the
    monkeypatches and the sigma_scale attribute take effect without retracing.
    """
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "all_config.yaml"))
    cfg.load_checkpoint = os.path.join(ckpt_dir, step)
    cfg.ema = False                  # checkpoint is already the EMA copy
    cfg.eval_save_outputs = []
    cfg.evaluators = []
    # harness 는 추론만 한다 -- optimizer param_group 은 의미가 없다.  끄지 않으면
    # pretrain.py 의 load_checkpoint 가 `load_state_dict(..., assign=True)` 로
    # Parameter 객체를 통째로 교체하면서 fp_transformer 가 붙인 `_no_weight_decay`
    # 속성이 사라지고, _adamw_param_arg 가 "마킹 0개" 로 보고 ValueError 로 즉사한다.
    # (학습은 load_checkpoint=None 이라 이 경로를 안 타서 마킹 2개가 살아 있다.)
    cfg.nodecay_marked = False
    config = P.PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))

    _, train_metadata = P.create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)
    ts = P.init_train_state(config, train_metadata, rank=0, world_size=1)

    net = getattr(ts.model, "_orig_mod", ts.model)   # eager ACTLossHead
    net.eval()
    return net, config


def _batch(inputs, labels, lo, hi, batch_size):
    """Fixed-size padded batch on GPU (tail batch padded so shapes stay static)."""
    seq_len = inputs.shape[-1]          # was hardcoded SEQ_LEN=81 (sudoku); breaks N-Queens (64)
    inp = np.zeros((batch_size, seq_len), dtype=np.int32)
    lab = np.zeros((batch_size, seq_len), dtype=np.int32)
    inp[: hi - lo] = inputs[lo:hi]
    lab[: hi - lo] = labels[lo:hi]
    return {
        "inputs": torch.from_numpy(inp).cuda(),
        "labels": torch.from_numpy(lab).cuda(),
        # test set has a single blank identifier (all__puzzle_identifiers is all-zero)
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32).cuda(),
    }


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
@contextmanager
def sigma_scale(net, scale):
    inner = net.model.inner
    prev = inner.sigma_scale
    inner.sigma_scale = float(scale)
    try:
        yield
    finally:
        inner.sigma_scale = prev


@contextmanager
def mu_scale(net, scale):
    """Scale the guide (mu) before sampling.  scale=0 -> "stochasticity only"
    (eps = sigma * noise, guide removed).  scale=1 (default) is a no-op wrapper
    that returns exactly what reparameterize returned before, so the normal path
    stays bit-identical.

    ★ 이것은 추론 시 스케일링이다.  논문 Table 3b 는 학습 시 ablation 이므로
      직접 대응하지 않는다.
    """
    if float(scale) == 1.0:                 # no-op: do not even wrap
        yield
        return
    inner = net.model.inner
    orig = inner.reparameterize

    def patched(mu, logvar):
        return orig(mu * float(scale), logvar)

    inner.reparameterize = patched
    try:
        yield
    finally:
        inner.reparameterize = orig


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def rollout(net, inputs, labels, K, sigma_scale_=1.0, mode="prior",
            batch_size=512, seed=0, verbose=True, mu_scale_=1.0):
    """K independent forward passes over the same puzzles.

    mode="prior"     : eps ~ p(.|u)     (no labels used)
    mode="posterior" : eps ~ q(.|u,y)   (y injected, sees labels)

    Selected via the explicit use_posterior kwarg on gram.py's forward, so the
    module stays in eval mode throughout: ACT halting remains deterministic
    (full halt_max_steps) and only the latent path differs between the two.

    Returns dict with
      preds     [K, N, 81] int64 argmax tokens
      v         [K, N]     float32, LPRM head at the final transition
      tok_acc   [K, N]     float32 per-sample token accuracy
    """
    assert mode in ("prior", "posterior")
    n = len(inputs)
    seq_len = inputs.shape[-1]          # was hardcoded SEQ_LEN=81 (sudoku); breaks N-Queens (64)
    preds = np.zeros((K, n, seq_len), dtype=np.int64)
    v = np.zeros((K, n), dtype=np.float32)

    with sigma_scale(net, sigma_scale_), mu_scale(net, mu_scale_):
        for k in range(K):
            torch.random.manual_seed(seed * 100003 + k)
            for lo in range(0, n, batch_size):
                hi = min(lo + batch_size, n)
                batch = _batch(inputs, labels, lo, hi, batch_size)
                with torch.device("cuda"):
                    carry = net.initial_carry(batch)

                p_final = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
                v_final = torch.zeros(batch_size, dtype=torch.float32, device="cuda")
                done = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
                while True:
                    carry, _, _, out, all_finish = net(
                        carry=carry, batch=batch, return_keys=["preds", "lrpm_logits"],
                        use_posterior=(mode == "posterior"))
                    newly = carry.halted & (~done)
                    if newly.any():
                        p_final[newly] = out["preds"][newly].long()
                        v_final[newly] = out["lrpm_logits"][newly][:, -1].float()
                    done |= carry.halted
                    if all_finish:
                        break

                preds[k, lo:hi] = p_final[: hi - lo].cpu().numpy()
                v[k, lo:hi] = v_final[: hi - lo].cpu().numpy()
            if verbose:
                print(f"  rollout {mode} sigma={sigma_scale_} {k + 1}/{K}", flush=True)

    mask = labels != PAD
    tok_acc = ((preds == labels[None]) & mask[None]).sum(-1) / np.maximum(mask.sum(-1), 1)[None]
    return dict(preds=preds, v=v, tok_acc=tok_acc.astype(np.float32))
