from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.
    

    use_fp_transformer: bool = False
    norm_type: str = "pre-norm"               # 'pre-norm'|'post-norm'|'peri-norm'
    residual_scale: Optional[str] = "fixed"   # None|'fixed'|'input-independent'|'input-dependent'
    alpha_1_init: float = 0.75
    alpha_2_init: float = 0.25
    outer_only: bool = False
    norm_placement: str = "none"              # 'input'|'output'|'none'
    normalize_input_injection: bool = False
    conv_type: str = "none"
    conv_kernel_size: int = 3
    conv_bias: bool = False
    causal: bool = False
    softmax_temp: float = 1.0
    use_spec_norm_linear: bool = False
    weight_dropout: float = 0.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense


    kl_coef: float = 0.1
    kl_balance: float = 0.8
    lprm_coef: float = 1.0

    eps_scale: float = 1.0
    damping_eta: float = 1.0

    noise_head_expansion: float = 4.0

    diag_kl_per_transition: bool = False

    ss_prob: float = 0.0
    ss_seed: int = 0  # rank 에 의존하면 안 된다 -- _ss_gen 주석 참조

    logvar_bias_init: float = 0.0


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        if self.config.use_fp_transformer:
            from models.fp_transformer import FixedPointTransformer
            self.L_level = FixedPointTransformer(self.config, self.config.L_layers)
        else:
            self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

        D = self.config.hidden_size
        nh_exp = self.config.noise_head_expansion

        self.prior_mu = SwiGLU(hidden_size=D, expansion=nh_exp)
        self.prior_logvar = SwiGLU(hidden_size=D, expansion=nh_exp)
        self.post_mu = SwiGLU(hidden_size=D, expansion=nh_exp)
        self.post_logvar = SwiGLU(hidden_size=D, expansion=nh_exp)
        self.lrpm_head = CastedLinear(D, 1, bias=True)

        _lv_init = float(self.config.logvar_bias_init)
        self.prior_logvar_bias = nn.Parameter(torch.tensor(_lv_init))
        self.post_logvar_bias = nn.Parameter(torch.tensor(_lv_init))

        self.sigma_scale = 1.0

        self.diag_kl_sum: Dict[str, float] = {}
        self.diag_kl_n: int = 0

        # rank 무관한 시드를 쓴다.  전역 RNG 는 rank 마다 시드가 달라(pretrain.py)
        # 분기가 갈리고, all_reduce 호출 집합이 어긋나 NCCL 이 교착한다.
        self._ss_gen = torch.Generator()
        self._ss_gen.manual_seed(self.config.ss_seed * 1000003 + 777)


    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding


    def _y_embeddings(self, labels):
        # 0.573 = embed_tokens 노름(12.97) / ‖z_H‖(22.63) 실측 비율.
        labels = torch.where(labels==IGNORE_LABEL_ID, 0, labels)
        oh = F.one_hot(labels.to(torch.int64), num_classes=self.config.vocab_size)
        embedding = F.pad(oh.to(self.forward_dtype),
                          (0, self.config.hidden_size - self.config.vocab_size))
        if self.puzzle_emb_len > 0:
            embedding = F.pad(embedding, (0, 0, self.puzzle_emb_len, 0))
        return self.embed_scale * 0.573 * embedding


    def kl_loss(self, mu_q, logvar_q, mu_p, logvar_p):
        mu_q, logvar_q, mu_p, logvar_p = (t.to(torch.float32) for t in (mu_q, logvar_q, mu_p, logvar_p))

        def _kl(mu_q, logvar_q, mu_p, logvar_p):
            var_q = torch.exp(logvar_q)
            var_p = torch.exp(logvar_p)
            return 0.5 * (logvar_p - logvar_q - 1 + (var_q + (mu_q - mu_p)**2 )/ var_p)
            
        b = self.config.kl_balance
        return b * _kl(mu_q.detach(), logvar_q.detach(), mu_p, logvar_p) + (1 - b) * _kl(mu_q, logvar_q, mu_p.detach(), logvar_p.detach())


    def _use_posterior_this_transition(self, use_posterior: bool) -> bool:
        # ss_prob=0.0 에서는 torch.rand 를 부르지 않는다 -- 부르면 RNG 소비가 달라져
        # 기존 run 과 비트 동일성이 깨진다.
        if not use_posterior:
            return False
        if self.config.ss_prob <= 0.0:
            return True
        return bool(torch.rand((), generator=self._ss_gen) >= self.config.ss_prob)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) * self.sigma_scale
        eps = torch.randn_like(std)
        return mu + eps * std


    @torch.no_grad()
    def _latent_diag(self, z_H, z_L, mu, logvar):
        return {
            "diag/z_h_norm": z_H.detach().float().norm(dim=-1).mean(),
            "diag/z_l_norm": z_L.detach().float().norm(dim=-1).mean(),
            "diag/eps_mu_norm": mu.detach().float().norm(dim=-1).mean(),
            "diag/eps_std_norm": (torch.exp(0.5 * logvar.detach().float()) * self.sigma_scale).norm(dim=-1).mean(),
        }


    @torch.no_grad()
    def _record_train_diag(self, t: int, mu_q, logvar_q, mu_p, logvar_p, dst: dict, kl=None):
        std_q = torch.exp(0.5 * logvar_q.detach().float())
        if kl is None:
            kl = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
        dst[f"diag/train_kl_t{t}"] = kl.detach().sum(-1).sum(-1).mean()
        dst[f"diag/train_std_mean_t{t}"] = std_q.mean()
        dst[f"diag/train_std_dimstd_t{t}"] = std_q.std(dim=-1).mean()
        dst[f"diag/train_mu_absmean_t{t}"] = mu_q.detach().float().abs().mean()


    def diag_kl_reset(self):
        self.diag_kl_sum = {}
        self.diag_kl_n = 0


    def diag_kl_result(self):
        if self.diag_kl_n == 0:
            return None
        return {k: v / self.diag_kl_n for k, v in sorted(self.diag_kl_sum.items())}


    @torch.no_grad()
    def _diag_kl_accumulate(self, dst: dict):
        for k, v in dst.items():
            self.diag_kl_sum[k] = self.diag_kl_sum.get(k, 0.0) + float(v)
        self.diag_kl_n += 1


    @torch.no_grad()
    def _record_diag_kl(self, t: int, z_H, y_emb, mu_p, logvar_p, dst: dict):
        # 측정 전용.  여기서 만든 mu_q 로 eps 를 뽑거나 z_H 를 갱신하면 궤적이 바뀐다.
        u_y = z_H + y_emb
        mu_q = self.post_mu(u_y)
        logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
        kl = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
        std_q = torch.exp(0.5 * logvar_q.float())
        dst[f"diag/kl_t{t}"] = kl.sum(-1).sum(-1).mean()
        # *_norm 은 per-position L2(D축) 뒤 나머지 축 평균, *_absmean/*_mean 은
        # (B,L,D) 전체 element 평균 -- 두 축은 sqrt(D) 만큼 눈금이 다르다.
        dst[f"diag/eps_mu_norm_t{t}"] = mu_q.float().norm(dim=-1).mean()
        dst[f"diag/eps_std_norm_t{t}"] = std_q.norm(dim=-1).mean()
        dst[f"diag/eps_mu_absmean_t{t}"] = mu_q.float().abs().mean()
        dst[f"diag/eps_std_mean_t{t}"] = std_q.mean()
        dst[f"diag/eps_std_dimstd_t{t}"] = std_q.std(dim=-1).mean()



    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], use_posterior: Optional[bool] = None) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if use_posterior is None:
            use_posterior = self.training

        y_emb = self._y_embeddings(batch["labels"]) if use_posterior else None

        train_diag = {} if use_posterior else None

        diag_kl_dst = None
        diag_kl_y = None
        if self.config.diag_kl_per_transition:
            diag_kl_dst = {}
            diag_kl_y = y_emb if y_emb is not None else self._y_embeddings(batch["labels"])

        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        lrpm_z = []

        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                t = _H_step + 1
                for _L_step in range(self.config.L_cycles):
                    z_tilde = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    if self.config.damping_eta != 1.0:
                        z_L = self.config.damping_eta * z_tilde \
                              + (1.0 - self.config.damping_eta) * z_L
                    else:
                        z_L = z_tilde

                z_H = self.L_level(z_H, z_L, **seq_info)

                mu_p = self.prior_mu(z_H)
                logvar_p = self.prior_logvar(z_H) + self.prior_logvar_bias

                if self._use_posterior_this_transition(use_posterior):
                    u_y = z_H + y_emb
                    mu_q = self.post_mu(u_y)
                    logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
                    eps = self.reparameterize(mu_q, logvar_q)
                    if train_diag is not None:
                        self._record_train_diag(t, mu_q, logvar_q, mu_p, logvar_p, train_diag)
                else:
                    eps = self.reparameterize(mu_p, logvar_p)

                if diag_kl_dst is not None:
                    self._record_diag_kl(t, z_H, diag_kl_y, mu_p, logvar_p, diag_kl_dst)

                z_H = z_H + self.config.eps_scale * eps
                lrpm_z.append(z_H[:, 0])

        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_tilde = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            if self.config.damping_eta != 1.0:
                z_L = self.config.damping_eta * z_tilde \
                      + (1.0 - self.config.damping_eta) * z_L
            else:
                z_L = z_tilde

        z_H = self.L_level(z_H, z_L, **seq_info)

        mu_p = self.prior_mu(z_H)
        logvar_p = self.prior_logvar(z_H) + self.prior_logvar_bias

        # prior 로 뽑히면 q=p 라 KL 이 0 이므로 항 자체를 뺀다 (kl=None).
        use_post_last = self._use_posterior_this_transition(use_posterior)
        if use_post_last:
            u_y = z_H + y_emb
            mu_q = self.post_mu(u_y)
            logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
            eps = self.reparameterize(mu_q, logvar_q)
            kl = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
            if train_diag is not None:
                self._record_train_diag(self.config.H_cycles, mu_q, logvar_q,
                                        mu_p, logvar_p, train_diag, kl=kl)
        else:
            eps = self.reparameterize(mu_p, logvar_p)
            kl = None

        # Logging-only; posterior path (= training) only, so eval stays untouched.
        diag = self._latent_diag(z_H, z_L, mu_q, logvar_q) if use_post_last else None

        if train_diag:
            diag = {**diag, **train_diag} if diag is not None else dict(train_diag)

        if diag_kl_dst is not None:
            self._record_diag_kl(self.config.H_cycles, z_H, diag_kl_y, mu_p, logvar_p, diag_kl_dst)
            self._diag_kl_accumulate(diag_kl_dst)
            diag = {**diag, **diag_kl_dst} if diag is not None else diag_kl_dst

        z_H = z_H + self.config.eps_scale * eps
        lrpm_z.append(z_H[:, 0])


        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        q_logits = self.q_head(z_H[:, 0].detach()).to(torch.float32)

        lrpm_logits = self.lrpm_head(torch.stack(lrpm_z, dim=1)).to(torch.float32).squeeze(-1)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), kl, lrpm_logits, diag

class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor], use_posterior: Optional[bool] = None) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), kl, lrpm_logits, diag = self.inner(new_inner_carry, new_current_data, use_posterior=use_posterior)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "lrpm_logits": lrpm_logits
        }

        if kl is not None:
            outputs["kl"] = kl

        # metrics-only side channel (dict of scalars); never a loss term
        if diag is not None:
            outputs["diag"] = diag

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _, _ = self.inner(new_inner_carry, new_current_data, use_posterior=use_posterior)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
