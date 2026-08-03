from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        
        # KL reduction 은 sum 고정: D축 합 -> L축 합 -> batch 합.
        # 'mean' 은 run 을 붕괴시킨다 (sum = gram_sudoku_halt_only 71.85%,
        #  mean = gram_sudoku_lmean_v2 1.8%).  분기 자체를 없애 재발을 막는다.
        kl_loss = torch.zeros((), device=lm_loss.device)   # 추론(prior) 시 kl 없음
        if outputs.get("kl") is not None:
            kl_loss = outputs["kl"].sum(-1).sum(-1).sum()

        lprm_loss = torch.zeros((), device=lm_loss.device)
        if "lrpm_logits" in outputs:
            # LPRM (Appendix A.2 Eq.16): r = 최종 출력 정확도, L = Σ_t (v_ψ(z_t) - r)²
            r = (is_correct.to(torch.float32) / loss_divisor).sum(-1).detach()
            lprm_loss = ((outputs["lrpm_logits"] - r.unsqueeze(-1)) ** 2).sum(-1).sum()

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "lprm_loss": lprm_loss.detach(),
        })

        # FPRM boundedness diagnostics from gram.py (diag/z_h_norm, diag/z_l_norm) --
        # already-detached scalars, batch means; `diag/` namespace 는 pretrain.py 가
        # world_size 로만 정규화한다.
        if outputs.get("diag") is not None:
            metrics.update(outputs["diag"])

        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}


        kl_coef   = getattr(self.model.config, "kl_coef", 0.0)
        lprm_coef = getattr(self.model.config, "lprm_coef", 1.0)

        # Total loss (기존 TRM loss + kl_loss + lprm_loss)
        return new_carry, lm_loss + kl_coef * kl_loss + 0.5 * (q_halt_loss + q_continue_loss) + lprm_coef * lprm_loss, metrics, detached_outputs, new_carry.halted.all()
