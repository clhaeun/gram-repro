"""y_onehot 토글 정적 검증 (CPU 전용, GPU 미사용).

확인 항목
  1. y_onehot=False 로 인스턴스화 -> num_params 가 기존 값과 동일한가
  2. y_onehot=True(zeropad / frozen_random) -> 빌드 성공, 새 파라미터 0개
  3. CPU forward 1회 -> y_emb shape/norm 을 z_H 와 비교
  4. y_onehot 키가 없는 기존 all_config.yaml -> 기본 False 로 해석되는가
  5. y_onehot=False 일 때 state_dict 키 집합이 기존 checkpoint 와 정확히 일치하는가
"""
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# experiments/legacy/diag/ 로 내려왔으므로 repo root 는 부모의 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.recursive_reasoning.gram import (            # noqa: E402
    TinyRecursiveReasoningModel_ACTV1_Inner as Inner,
    TinyRecursiveReasoningModel_ACTV1Config as Cfg,
)

NQ_CKPT = "checkpoints/Nqueens-8-ACT-torch/gram_nq8_mean_nh512"
SUDOKU_CKPT = "checkpoints/Sudoku-extreme-1k-aug-1000-test10k-ACT-torch/gram_sudoku_halt_only"


def cfg_from_ckpt(ckpt_dir, vocab_size, seq_len, npi=1, batch_size=8, **over):
    """checkpoint 의 all_config.yaml arch 섹션 + 데이터 차원으로 config 조립."""
    y = OmegaConf.load(os.path.join(ckpt_dir, "all_config.yaml"))
    arch = OmegaConf.to_container(y.arch, resolve=True)
    for k in ("name", "loss"):
        arch.pop(k, None)
    arch.update(batch_size=batch_size, vocab_size=vocab_size, seq_len=seq_len,
                num_puzzle_identifiers=npi, causal=False)
    arch.update(over)
    return Cfg(**arch)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def build(tag, **over):
    torch.random.manual_seed(0)
    c = cfg_from_ckpt(NQ_CKPT, vocab_size=3, seq_len=64, **over)
    m = Inner(c).to(torch.float32)
    n = nparams(m)
    nbuf = sum(b.numel() for b in m.buffers())
    print(f"  [{tag:<34}] params={n:>10,}  buffers={nbuf:>8,}  "
          f"y_onehot={c.y_onehot} proj={c.y_onehot_proj} scale={c.y_onehot_scale}")
    return m, c, n


def main():
    print("== 1/2. 인스턴스화 + num_params (N-Queens nh512 config, vocab=3, seq=64) ==")
    m_off, c_off, n_off = build("y_onehot=False (기존)")
    m_zp, c_zp, n_zp = build("y_onehot=True zeropad", y_onehot=True)
    m_fr, c_fr, n_fr = build("y_onehot=True frozen_random", y_onehot=True,
                             y_onehot_proj="frozen_random")
    m_s, c_s, n_s = build("y_onehot=True zeropad s=0.573", y_onehot=True,
                          y_onehot_scale=0.573)
    print(f"  기대(논문 대응) = 9,966,085 | False 일치: {n_off == 9966085}")
    print(f"  True(zeropad) 파라미터 증가분      = {n_zp - n_off}")
    print(f"  True(frozen_random) 파라미터 증가분 = {n_fr - n_off}  "
          f"(buffer 로만 추가, requires_grad 아님)")
    fr_extra = [k for k, _ in m_fr.named_buffers() if k not in dict(m_off.named_buffers())]
    print(f"  frozen_random 이 추가한 buffer: {fr_extra}")

    print("\n== 5. state_dict 키 집합 (기존 checkpoint 와 비교) ==")
    real = torch.load(os.path.join(NQ_CKPT, "step_17085"), map_location="cpu", weights_only=False)
    real_keys = {k.replace("_orig_mod.model.inner.", "") for k in real}
    for tag, m in (("False", m_off), ("True/zeropad", m_zp), ("True/frozen_random", m_fr),
                   ("True/zeropad s=0.573", m_s)):
        mk = set(m.state_dict().keys())
        print(f"  [{tag:<18}] 누락={sorted(real_keys - mk)}  추가={sorted(mk - real_keys)}")

    print("\n== 4. y_onehot 키 없는 기존 all_config.yaml -> 기본값 해석 ==")
    for name, d, v, s in (("nqueens nh512", NQ_CKPT, 3, 64), ("sudoku halt_only", SUDOKU_CKPT, 11, 81)):
        raw = OmegaConf.load(os.path.join(d, "all_config.yaml"))
        has = "y_onehot" in OmegaConf.to_container(raw.arch)
        c = cfg_from_ckpt(d, vocab_size=v, seq_len=s)
        print(f"  [{name:<17}] yaml 에 y_onehot 키 존재={has} -> config.y_onehot={c.y_onehot} "
              f"(proj={c.y_onehot_proj}, scale={c.y_onehot_scale})")

    print("\n== 3. CPU forward 1회 + y_emb / z_H 노름 비교 ==")
    B, L = 8, 64          # B 는 config.batch_size 와 같아야 한다 (CastedSparseEmbedding 이 고정 버퍼)
    labels = torch.randint(1, 3, (B, L))
    for tag, m, c in (("False (학습 embedding)", m_off, c_off),
                      ("True zeropad", m_zp, c_zp),
                      ("True frozen_random", m_fr, c_fr)):
        with torch.no_grad():
            y = m._y_embeddings(labels).float()
        n = y.norm(dim=-1)
        nz = n > 0
        print(f"  [{tag:<24}] y_emb shape={tuple(y.shape)} dtype={m._y_embeddings(labels).dtype} "
              f"| ‖y‖ non-pad mean={n[nz].mean():.4f}  pad위치={int((~nz).sum(1).float().mean())}/{y.shape[1]}")

    # 전체 forward 로 shape 에러 없는지 (CPU, float32)
    print("\n  -- 전체 forward (CPU) --")
    for tag, m, c in (("False", m_off, c_off), ("True zeropad", m_zp, c_zp),
                      ("True frozen_random", m_fr, c_fr)):
        batch = {"inputs": torch.randint(1, 3, (B, L), dtype=torch.int32),
                 "labels": torch.randint(1, 3, (B, L), dtype=torch.int32),
                 "puzzle_identifiers": torch.zeros(B, dtype=torch.int32)}
        carry = m.empty_carry(B)
        carry = m.reset_carry(torch.ones(B, dtype=torch.bool), carry)
        with torch.no_grad():
            out = m(carry, batch, use_posterior=True)
        newc, logits, qs, kl, lrpm, diag = out
        zH = newc.z_H.float()
        print(f"  [{tag:<18}] logits={tuple(logits.shape)} kl={tuple(kl.shape)} "
              f"‖z_H‖/token={zH.norm(dim=-1).mean():.4f}")

    print("\n  -- ‖y_emb‖ / ‖z_H‖ 비율 (z_H 는 RMSNorm 출력이라 √512=22.627 고정) --")
    zref = 22.627
    for tag, m in (("False", m_off), ("True zeropad s=1.0", m_zp),
                   ("True frozen_random", m_fr), ("True zeropad s=0.573", m_s)):
        with torch.no_grad():
            y = m._y_embeddings(labels).float()
        n = y.norm(dim=-1); nz = n > 0
        print(f"  [{tag:<24}] ‖y‖={n[nz].mean():7.4f}  ratio={n[nz].mean()/zref:.4f}")
    print(f"  (측정된 학습 embedding 실제값: ‖y‖=12.9729 ratio=0.5733 "
          f"-- experiments/results/yemb_nq8_nh512.json)")


if __name__ == "__main__":
    main()
