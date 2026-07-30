"""Task 1.7: measure init-weight contamination in an EMA checkpoint.

Standard EMA with shadow initialized to the init weights w0 (models/ema.py:14)
and no bias correction leaves, after t steps, a mu^t fraction of w0 in the
shadow:  shadow_t = (1-mu) sum mu^{t-i} w_i  +  mu^t w0.

We reconstruct w0 by re-instantiating the model with the same seed, VALIDATE the
reconstruction against the checkpoint's H_init/L_init (fixed buffers = exact
init, never SGD-updated), then report for the noise-head weights:
  cos(W_ema, w0),  ||w0|| / ||W_ema||,  and the predicted mu^t fraction.

Also emits the *debiased* weights  (W_ema - mu^t w0) / (1 - mu^t)  as a new
state_dict so Task 3 can load and re-evaluate it.
"""
import argparse, json, os, sys
import torch
from omegaconf import OmegaConf
# experiments/diag/ 로 한 단계 내려왔으므로 repo root 는 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.functions import load_model_class


def build_init(all_config_yaml, meta, seed, device="cuda"):
    """Reproduce create_model's eager init under the same seed.

    all_config_yaml is the full PretrainConfig dump; arch lives under `arch`.
    """
    full = OmegaConf.to_container(OmegaConf.load(all_config_yaml), resolve=True)
    arch = dict(full["arch"])
    name = arch.pop("name"); arch.pop("loss")
    cfg = dict(**arch, batch_size=full["global_batch_size"], vocab_size=meta["vocab_size"],
               seq_len=meta["seq_len"], num_puzzle_identifiers=meta["num_puzzle_identifiers"],
               causal=False)
    torch.random.manual_seed(seed)
    with torch.device(device):
        model = load_model_class(name)(cfg)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch-yaml", required=True)
    ap.add_argument("--data", default="data/nqueens-8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, required=True, help="training step t for mu^t")
    ap.add_argument("--mu", type=float, default=0.9999)
    ap.add_argument("--emit-debiased", default=None, help="path to write debiased state_dict")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data, "train", "dataset.json")))
    model = build_init(args.arch_yaml, meta, args.seed)
    init_sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}

    sd = torch.load(args.ckpt, map_location="cpu")
    # ckpt keys: _orig_mod.model.inner.*  ->  inner.*  (matches eager ACTV1 state_dict)
    strip = lambda k: k.replace("_orig_mod.model.", "")
    ckpt = {strip(k): v.float() for k, v in sd.items()}

    mu_t = args.mu ** args.step
    print(f"mu={args.mu} t={args.step}  mu^t = {mu_t:.4f}  (init fraction if uncorrected)")

    # --- validation: H_init/L_init are fixed buffers = exact init ---
    print("\n[validation vs fixed buffers]")
    for b in ("inner.H_init", "inner.L_init"):
        if b in ckpt and b in init_sd:
            a, c = init_sd[b], ckpt[b]
            same = torch.allclose(a, c, atol=1e-3)
            cos = torch.nn.functional.cosine_similarity(a.flatten(), c.flatten(), dim=0).item()
            print(f"  {b}: allclose={same}  cos={cos:.6f}  ||init||={a.norm():.3f} ||ckpt||={c.norm():.3f}")

    # --- contamination on noise-head weights ---
    print("\n[noise-head contamination]")
    heads = ["inner.prior_mu", "inner.prior_logvar", "inner.post_mu", "inner.post_logvar"]
    debiased = {k: v.clone() for k, v in sd.items()}
    for h in heads:
        for sub in ("gate_up_proj.weight", "down_proj.weight"):
            key = f"{h}.{sub}"
            if key not in ckpt or key not in init_sd:
                continue
            w0, we = init_sd[key], ckpt[key]
            cos = torch.nn.functional.cosine_similarity(w0.flatten(), we.flatten(), dim=0).item()
            # debiased = (we - mu^t w0) / (1 - mu^t)
            deb = (we - mu_t * w0) / (1 - mu_t)
            # how much does removing mu^t*w0 change the weight (relative)
            rel = (mu_t * w0).norm() / we.norm()
            print(f"  {key:34} cos(w0,ema)={cos:+.4f} ||w0||/||ema||={w0.norm()/we.norm():.3f} "
                  f"mu^t*||w0||/||ema||={rel:.3f} ||deb-ema||/||ema||={(deb-we).norm()/we.norm():.3f}")
            # write debiased back with the original (prefixed) key
            for orig in sd:
                if strip(orig) == key:
                    debiased[orig] = deb.to(sd[orig].dtype)

    if args.emit_debiased:
        # Debias EVERY EMA-shadowed param (all requires_grad params), not just the
        # noise heads: shadow_t = (1-mu^t) W_avg + mu^t w0 for all of them.
        # Buffers (H_init, L_init, puzzle_emb.weights) are NOT shadowed -> leave as-is.
        BUFFERS = {"inner.H_init", "inner.L_init", "inner.puzzle_emb.weights"}
        n_deb = 0
        for orig in sd:
            k = strip(orig)
            if k in BUFFERS or k not in init_sd:
                continue
            deb = (ckpt[k] - mu_t * init_sd[k]) / (1 - mu_t)
            debiased[orig] = deb.to(sd[orig].dtype)
            n_deb += 1
        torch.save(debiased, args.emit_debiased)
        print(f"\ndebiased {n_deb} shadowed params (buffers left as init) -> {args.emit_debiased}")


if __name__ == "__main__":
    main()
