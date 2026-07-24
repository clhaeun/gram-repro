"""Diagnose the sigma_p-inflation hypothesis on GRAM checkpoints.

Runs ONE training-mode forward per checkpoint, captures the exact
(mu_q, logvar_q, mu_p, logvar_p) tensors handed to kl_loss at the final
transition, and prints variance statistics + KL decomposition.

Usage (single GPU, no torchrun needed):
  python diag_sigma.py <checkpoint_dir> [step_file ...]
If no step_file given, scans all step_* files in the dir.

Hypothesis: prior inflates sigma_p to keep KL low while the posterior
smuggles y through mu_q.  Signature:  logvar_p >> logvar_q,  and the KL
term (mu_q-mu_p)^2 / var_p is suppressed despite a large raw mean gap.
"""
import os, sys, glob
# keep compile ON so checkpoint keys ("_orig_mod.*") load; we run the forward
# eagerly through ._orig_mod so the monkeypatch fires without tracing.

import torch
from omegaconf import OmegaConf
import pretrain as P


IGNORE = -100


def stats(name, t):
    t = t.float().flatten()
    tq = t if t.numel() <= 1_000_000 else t[torch.randperm(t.numel(), device=t.device)[:1_000_000]]
    q = torch.quantile(tq, torch.tensor([0.01, 0.5, 0.99], device=t.device))
    print(f"  {name:16s} mean={t.mean():+.4f}  std={t.std():.4f}  "
          f"[p1={q[0]:+.4f}  med={q[1]:+.4f}  p99={q[2]:+.4f}]  "
          f"min={t.min():+.4f} max={t.max():+.4f}")


def accuracy(logits, labels):
    """token acc over valid positions + exact (full-sequence) acc."""
    preds = logits.argmax(-1)
    mask = labels != IGNORE
    counts = mask.sum(-1).clamp_min(1)
    is_correct = mask & (preds == labels)
    tok_acc = (is_correct.sum(-1).float() / counts).mean().item()
    exact = ((is_correct.sum(-1) == mask.sum(-1)).float()).mean().item()
    return tok_acc, exact


def rollout(inner, batch, B, training):
    """one inner transition from a fresh reset carry; posterior if training else prior."""
    inner.train(training)
    with torch.device("cuda"):
        carry = inner.reset_carry(
            torch.ones(B, dtype=torch.bool, device="cuda"), inner.empty_carry(B))
    with torch.no_grad():
        _, logits, _, kl, _ = inner(carry, batch)
    return logits, kl


def run_one(ckpt_dir, ckpt_file):
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "all_config.yaml"))
    cfg.load_checkpoint = ckpt_path
    cfg.ema = False
    cfg.eval_save_outputs = []
    cfg.evaluators = []
    config = P.PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))
    torch.random.manual_seed(config.seed)

    train_loader, train_metadata = P.create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)

    ts = P.init_train_state(config, train_metadata, rank=0, world_size=1)

    # ts.model is the compiled OptimizedModule (checkpoint loaded into it).
    # eager_net is the underlying ACTLossHead; call it directly to stay eager.
    eager_net = getattr(ts.model, "_orig_mod", ts.model)
    inner = eager_net.model.inner

    cap = {}
    orig_kl = inner.kl_loss
    def patched(mu_q, logvar_q, mu_p, logvar_p):
        cap['mu_q'] = mu_q.detach(); cap['logvar_q'] = logvar_q.detach()
        cap['mu_p'] = mu_p.detach(); cap['logvar_p'] = logvar_p.detach()
        return orig_kl(mu_q, logvar_q, mu_p, logvar_p)
    inner.kl_loss = patched

    batch = None
    for _, b, _ in train_loader:
        batch = {k: v.cuda() for k, v in b.items()}
        break
    B = batch["inputs"].shape[0]

    # posterior rollout (train mode: eps ~ q(.|u,y), y injected) -> also fills cap{}
    post_logits, _ = rollout(inner, batch, B, training=True)
    post_tok, post_exact = accuracy(post_logits, batch["labels"])
    # prior rollout (eval mode: eps ~ p(.|u), no y) -- SAME batch
    prior_logits, _ = rollout(inner, batch, B, training=False)
    prior_tok, prior_exact = accuracy(prior_logits, batch["labels"])

    # Probe A: shuffle y fed to posterior (y_emb), score vs TRUE labels
    perm = torch.randperm(B, device="cuda")
    batch_shuf = dict(batch); batch_shuf["labels"] = batch["labels"][perm]
    shufY_logits, _ = rollout(inner, batch_shuf, B, training=True)
    shufY_tok, shufY_exact = accuracy(shufY_logits, batch["labels"])

    # Probe B: deterministic latent eps=mu (no noise), posterior mu_q vs prior mu_p
    orig_rep = inner.reparameterize
    inner.reparameterize = lambda mu, logvar: mu
    detq_logits, _ = rollout(inner, batch, B, training=True)
    detq_tok, detq_exact = accuracy(detq_logits, batch["labels"])
    detp_logits, _ = rollout(inner, batch, B, training=False)
    detp_tok, detp_exact = accuracy(detp_logits, batch["labels"])
    inner.reparameterize = orig_rep

    lp = cap['logvar_p'].float(); lq = cap['logvar_q'].float()
    mp = cap['mu_p'].float();     mq = cap['mu_q'].float()
    var_p = lp.exp(); var_q = lq.exp()
    gap2 = (mq - mp).pow(2)

    print("=" * 78)
    print(f"CHECKPOINT  {ckpt_file}")
    print("=" * 78)
    if hasattr(inner, "prior_logvar_bias"):
        print(f"  prior_logvar_bias = {inner.prior_logvar_bias.item():+.4f}   "
              f"post_logvar_bias = {inner.post_logvar_bias.item():+.4f}   "
              f"(unbounded std = exp(0.5*logvar))")
    else:
        print(f"  sigma_max = {inner.sigma_max:.3f}   "
              f"(bounded std = sigma_max * sigmoid(raw); no logvar_bias)")
    print("  -- log-variances (effective, incl. bias) --")
    stats("logvar_p", lp)
    stats("logvar_q", lq)
    stats("sigma_p", (0.5 * lp).exp())
    stats("sigma_q", (0.5 * lq).exp())
    print("  -- mean gap & KL decomposition --")
    stats("(mu_q-mu_p)^2", gap2)
    stats("gap2 / var_p", gap2 / var_p)      # <-- the smuggling term; small => explained-as-noise
    stats("var_q / var_p", var_q / var_p)
    kl_raw = 0.5 * (lp - lq - 1 + (var_q + gap2) / var_p)
    print(f"  raw per-elem KL mean = {kl_raw.mean().item():+.5f}")
    print(f"  logvar_p - logvar_q  mean = {(lp - lq).mean().item():+.4f}   "
          f"(>0 => prior wider than posterior)")
    print("  -- rollout accuracy on same batch (1 transition, fresh carry) --")
    print(f"  posterior (train, eps~q, y injected):  tok={post_tok:6.2%}  exact={post_exact:6.2%}")
    print(f"  prior     (eval,  eps~p, no y):        tok={prior_tok:6.2%}  exact={prior_exact:6.2%}")
    print(f"  ACC GAP posterior-prior:  tok={post_tok - prior_tok:+.2%}  exact={post_exact - prior_exact:+.2%}")
    print("  -- mechanism probes --")
    print(f"  A: posterior w/ SHUFFLED y (score vs true):  tok={shufY_tok:6.2%}  exact={shufY_exact:6.2%}"
          f"   (~prior => y is the source; ~100% => y not needed)")
    print(f"  B: deterministic eps=mu_q (no noise):        tok={detq_tok:6.2%}  exact={detq_exact:6.2%}"
          f"   (high => y-signal lives in the MEAN)")
    print(f"  B: deterministic eps=mu_p (no noise):        tok={detp_tok:6.2%}  exact={detp_exact:6.2%}")
    print(f"  |mu_q| rms={mq.pow(2).mean().sqrt():.4f}  |mu_p| rms={mp.pow(2).mean().sqrt():.4f}  "
          f"|mu_q-mu_p| rms={gap2.mean().sqrt():.4f}")
    # verdict: the real test is NOT sigma_p vs sigma_q (they track each other),
    # but whether a LARGE raw mean gap is made KL-cheap by a LARGE absolute sigma_p.
    gap_rms = gap2.mean().sqrt().item()
    kl_gap_term = (0.5 * gap2 / var_p).mean().item()
    print("  ---------------------------------------------------------------")
    print(f"  >>> raw mean gap |mu_q-mu_p| rms = {gap_rms:.3f}  (LARGE => posterior smuggles answer in mean)")
    print(f"  >>> but its KL cost 0.5*gap^2/var_p = {kl_gap_term:.4f}  (tiny => sigma_p absorbs it)")
    if gap_rms > 1.0 and kl_gap_term < 0.1:
        print(f"  >>> SMUGGLING CONFIRMED: big mean gap, ~free in KL because sigma_p is large")
    print()


def main():
    ckpt_dir = sys.argv[1]
    files = sys.argv[2:] or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(ckpt_dir, "step_*")))
    print(f"checkpoints to probe: {files}\n")
    for f in files:
        run_one(ckpt_dir, f)


if __name__ == "__main__":
    main()
