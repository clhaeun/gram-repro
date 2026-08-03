from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
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
    

    # --- FPRM 기여1: pre-norm + residual scaling (fp_transformer) ----------
    # False = 기존 ReasoningModule 경로 그대로. 기존 yaml/checkpoint 무영향.
    use_fp_transformer: bool = False
    norm_type: str = "pre-norm"               # 'pre-norm'|'post-norm'|'peri-norm'
    residual_scale: Optional[str] = "fixed"   # None|'fixed'|'input-independent'|'input-dependent'
    alpha_1_init: float = 0.75                # FPRM Table 4 (Sudoku)
    alpha_2_init: float = 0.25                # FPRM Table 4 (Sudoku)
    outer_only: bool = False                  # True = Eq3 만 (α1=β1=1)
    norm_placement: str = "none"              # 'input'|'output'|'none' / FPRM default 미확인
    normalize_input_injection: bool = False   # FPRM default 미확인
    conv_type: str = "none"                   # 기여1만 이식 → conv off
    conv_kernel_size: int = 3                 # off 여도 __init__ 이 읽음
    conv_bias: bool = False
    causal: bool = False                      # pretrain.py 가 이미 주입 중, 필드 선언만 필요
    softmax_temp: float = 1.0
    use_spec_norm_linear: bool = False
    weight_dropout: float = 0.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense


    kl_coef: float = 0.1
    kl_balance: float = 0.8
    lprm_coef: float = 1.0

    # KL reduction over the sequence axis.  This used to be selected by
    # commenting/uncommenting a line in losses.py, which is how gram_nq8_* got
    # launched on "mean": gram_sudoku_halt_only (71.85%) ran "sum",
    # gram_sudoku_lmean_v2 (1.8%) ran "mean".  Set it explicitly in the arch
    # yaml; the default here is the value the working run used.
    kl_reduction: str = "sum"   # "sum" | "mean"


    noise_head_expansion: float = 4.0

    # prior/posterior logvar 헤드에 더하는 학습가능 스칼라 bias 의 초기값.
    # -4.0 이면 초기 sigma = exp(-2) = 0.135 로 노이즈를 작게 시작한다(기존 하드코딩값).
    # 0.0 이면 초기 sigma = 1.0 이라 KL 이 초반에 훨씬 크게 잡힌다.
    logvar_bias_init: float = -4.0

    # --- posterior 의 y 인코딩 (사수 처방 4: one-hot) -----------------------
    # 기본 False = 기존 동작(학습된 embed_tokens 조회). 기존 checkpoint 호환을 위해
    # False 일 때는 어떤 모듈/버퍼도 새로 만들지 않는다(pretrain.py:266 이 strict load).
    #
    # 주의: nn.Embedding 조회는 수학적으로 one_hot @ W 와 **완전히 같다**.
    # 따라서 "one-hot + 학습가능 projection" 은 현재 경로와 동일해져 아무 효과가 없다.
    # 통로를 실제로 좁히려면 projection 이 없거나(zeropad) 고정(frozen_random)이어야 한다.
    y_onehot: bool = False
    #   "zeropad"       : one-hot [V] 을 hidden_size 앞쪽에 두고 나머지 0. 파라미터 0개.
    #   "frozen_random" : 고정(비학습) 랜덤 행렬 [V, hidden_size] 로 사영. 파라미터 0개(buffer).
    y_onehot_proj: str = "zeropad"
    #   embed_scale(√hidden) 곱 위에 추가로 곱하는 계수. 1.0 이면 위치당 노름 = √hidden.
    y_onehot_scale: float = 1.0

    # --- posterior 의 y 주입 방식 -------------------------------------------
    #   "add"    : u_y = z_H + y_emb                       (기존. 파라미터 0개)
    #   "concat" : u_y = W([z_H ; y_emb])   W: 2D -> D     (파라미터 2D*D)
    # prior 경로는 어느 쪽이든 손대지 않는다: mu_p/logvar_p 는 계속 z_H(D차원)만 받고,
    # KL 은 여전히 D 차원 대각 가우시안 두 개 사이의 것이라 구조가 바뀌지 않는다.
    # W 는 [I ; I] 로 초기화되므로 step 0 에서 concat 은 add 와 수치적으로 동일하다 --
    # 두 arm 의 차이는 학습이 W 를 항등에서 밀어낸 결과로만 남는다.
    y_inject: str = "add"   # "add" | "concat"


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
        # B, L, D = hidden_states.shape
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


# transition loop
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

        # one-hot y 경로. y_onehot=False 면 **아무것도 만들지 않는다** -- 기존
        # checkpoint 가 strict load(pretrain.py:266) 로 그대로 들어와야 하기 때문.
        if self.config.y_onehot:
            if self.config.y_onehot_proj == "frozen_random":
                # 전역 RNG 를 건드리지 않으려고 로컬 generator 를 쓴다. 전역을 소비하면
                # 다른 파라미터의 init 스트림이 밀려 기존 run 재현이 깨진다.
                g = torch.Generator(device="cpu").manual_seed(0)
                m = torch.randn(self.config.vocab_size, self.config.hidden_size, generator=g)
                m = m / m.norm(dim=-1, keepdim=True)          # 행별 단위노름 -> 출력 노름 1
                self.y_onehot_mat = nn.Buffer(m.to(self.forward_dtype), persistent=True)
            elif self.config.y_onehot_proj != "zeropad":
                raise ValueError(
                    f"y_onehot_proj must be 'zeropad' or 'frozen_random', got {self.config.y_onehot_proj!r}")
            if self.config.y_onehot_proj == "zeropad" and self.config.vocab_size > self.config.hidden_size:
                raise ValueError(
                    f"y_onehot_proj='zeropad' needs vocab_size({self.config.vocab_size}) "
                    f"<= hidden_size({self.config.hidden_size})")

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
        # FPRM 기여1: use_fp_transformer=True 면 pre-norm + residual scaling 블록으로 교체.
        # import 는 분기 안에 둔다 -- False 인 기존 run 이 fp_* 파일 상태와 무관하게 돌아야 한다.
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

        # concat 일 때만 사영을 만든다. "add" 면 모듈을 아예 만들지 않으므로 기존
        # checkpoint 의 strict load(pretrain.py:266)가 그대로 통과한다.
        if self.config.y_inject == "concat":
            self.y_concat_proj = CastedLinear(2 * D, D, bias=False)
            # [I ; I] 초기화: step 0 에서 W([z_H ; y_emb]) == z_H + y_emb 라
            # "add" arm 과 정확히 같은 지점에서 출발한다.
            with torch.no_grad():
                self.y_concat_proj.weight.copy_(
                    torch.cat([torch.eye(D), torch.eye(D)], dim=-1))
        elif self.config.y_inject != "add":
            raise ValueError(
                f"y_inject must be 'add' or 'concat', got {self.config.y_inject!r}")

        # 노이즈를 작게 시작하기 (기본 -4.0 = 기존 하드코딩값, arch.logvar_bias_init 로 덮어씀)
        self.prior_logvar_bias = nn.Parameter(torch.tensor(self.config.logvar_bias_init))
        self.post_logvar_bias = nn.Parameter(torch.tensor(self.config.logvar_bias_init))

        # Test-time에 sampled latent std 조정 실험용
        self.sigma_scale = 1.0


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


    def _y_onehot_embeddings(self, labels):
        """one-hot 기반 y 인코딩. 학습 파라미터 0개.

        nn.Embedding 조회 == one_hot @ W 이므로, projection 이 학습 가능하면
        기존 경로와 수학적으로 동일해진다. 여기서는 projection 이 없거나(zeropad)
        고정 buffer(frozen_random) 라서 y -> latent 통로가 실제로 좁아진다.
        두 경로 모두 위치당 L2 노름 1 로 나오게 맞췄다(아래 embed_scale 곱 전 기준).
        """
        oh = F.one_hot(labels.to(torch.int64), num_classes=self.config.vocab_size)
        oh = oh.to(self.forward_dtype)
        if self.config.y_onehot_proj == "zeropad":
            return F.pad(oh, (0, self.config.hidden_size - self.config.vocab_size))
        return oh @ self.y_onehot_mat.to(oh.dtype)

    def _y_embeddings(self, labels):
        labels = torch.where(labels==IGNORE_LABEL_ID, 0, labels)
        if self.config.y_onehot:
            embedding = self._y_onehot_embeddings(labels)
        else:
            embedding = self.embed_tokens(labels.to(torch.int32))

        if self.puzzle_emb_len > 0:
            embedding = F.pad(embedding, (0, 0, self.puzzle_emb_len, 0))

        if self.config.y_onehot:
            return self.embed_scale * self.config.y_onehot_scale * embedding
        return self.embed_scale * embedding


    def kl_loss(self, mu_q, logvar_q, mu_p, logvar_p):
        mu_q, logvar_q, mu_p, logvar_p = (t.to(torch.float32) for t in (mu_q, logvar_q, mu_p, logvar_p))

        def _kl(mu_q, logvar_q, mu_p, logvar_p):
            var_q = torch.exp(logvar_q)
            var_p = torch.exp(logvar_p)
            return 0.5 * (logvar_p - logvar_q - 1 + (var_q + (mu_q - mu_p)**2 )/ var_p)
            
        b = self.config.kl_balance
        return b * _kl(mu_q.detach(), logvar_q.detach(), mu_p, logvar_p) + (1 - b) * _kl(mu_q, logvar_q, mu_p.detach(), logvar_p.detach())


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) * self.sigma_scale
        eps = torch.randn_like(std)
        return mu + eps * std


    def _inject_y(self, z_H, y_emb):
        """posterior 입력 u_y 를 만든다.  prior 는 이 함수를 쓰지 않는다.

        grad 구간에서도 호출되므로 no_grad 를 걸면 안 된다 -- 걸면 y_concat_proj 가
        학습되지 않고 posterior 로 가는 gradient 도 끊긴다.
        """
        if self.config.y_inject == "concat":
            return self.y_concat_proj(torch.cat([z_H, y_emb], dim=-1))
        return z_H + y_emb


    @torch.no_grad()
    def _y_proj_deviation(self):
        """‖W - [I;I]‖_F / ‖[I;I]‖_F.  0 이면 아직 add 와 같은 함수다.

        주의 -- 이 스칼라만으로 "concat 이 add 와 다른 함수를 배웠다"고 판정하면 안 된다.
        AdamW 가 model.parameters() 전체에 weight_decay 를 걸기 때문에(pretrain.py:185,
        예외 group 없음) W 는 gradient 와 무관하게 매 step (1 - lr*wd) 배로 줄어든다.
        W = a*[I;I] 인 순수 축소만 있어도 이 값은 |1 - a| 가 되고, lr=1e-4/wd=1.0/40k step
        이면 a -> 0.019 라 값이 0.98 까지 올라간다.  실측(step 1139): dev=0.0882 중
        축소분 |1-a|=0.0866, 방향 잔차는 0.0170 뿐이었다.
        판정에 쓸 값은 checkpoint 의 W 에서 사후 분해한
            a = <W, [I;I]> / 2D,   잔차 = ‖W - a*[I;I]‖_F / ‖[I;I]‖_F
        중 잔차 쪽이다.  W 는 state_dict 에 그대로 남으므로 재학습 없이 계산할 수 있다.
        """
        D = self.config.hidden_size
        dev = self.y_concat_proj.weight.detach().float().clone()   # [D, 2D]
        dev[:, :D].diagonal().sub_(1.0)
        dev[:, D:].diagonal().sub_(1.0)
        return dev.norm() / math.sqrt(2.0 * D)


    @torch.no_grad()
    def _latent_diag(self, mu_p, logvar_p, mu_q, logvar_q, kl, z_H=None, z_L=None):
        """Scalar diagnostics for the final (grad-carrying) transition.

        Everything is detached and derived from tensors the forward pass has
        already produced -- no extra model evaluation.  Quantiles run on a
        strided subsample because torch.quantile caps out near 2**24 elements.
        """
        out = {}

        def _std_stats(logvar, tag):
            std = (0.5 * logvar.detach().float()).exp() * self.sigma_scale
            flat = std.reshape(-1)
            sub = flat[:: max(1, flat.numel() // 1000000)]
            q = torch.quantile(sub, torch.tensor([0.5, 0.95], device=sub.device))
            out[f"diag/sigma_{tag}_std_mean"] = flat.mean()
            out[f"diag/sigma_{tag}_std_p50"] = q[0]
            out[f"diag/sigma_{tag}_std_p95"] = q[1]

        _std_stats(logvar_p, "prior")
        out["diag/mu_p_norm"] = mu_p.detach().float().norm(dim=-1).mean()

        # FPRM Figure 2b 재현용: 최종 transition 의 ‖z_H‖ (배치·시퀀스 평균).
        # boundedness 가 1차 판정 지표 -- accuracy 보다 먼저 본다.
        if z_H is not None:
            out["diag/z_h_norm"] = z_H.detach().float().norm(dim=-1).mean()
        if z_L is not None:
            out["diag/z_l_norm"] = z_L.detach().float().norm(dim=-1).mean()

        if logvar_q is not None:
            _std_stats(logvar_q, "post")
            out["diag/mu_q_norm"] = mu_q.detach().float().norm(dim=-1).mean()

        if kl is not None:
            # per-position KL: summed over D, averaged over batch and sequence.
            # kl_balance is already folded into `kl`; kl_coef is not.
            out["diag/kl_raw"] = kl.detach().float().sum(-1).mean()

        # concat 사영이 항등에서 얼마나 벗어났는지. 0 이면 add 와 같은 함수다.
        if self.config.y_inject == "concat":
            out["diag/y_proj_dev"] = self._y_proj_deviation()

        return out



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

        # transition별 KL 계측용 (inner.trace_kl = True 일 때만 동작)
        trace_kl = getattr(self, "trace_kl", False)
        if trace_kl:
            self._kl_trace = []

        y_emb = self._y_embeddings(batch["labels"]) if use_posterior else None

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
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                z_H = self.L_level(z_H, z_L, **seq_info)

                mu_p = self.prior_mu(z_H)
                logvar_p = self.prior_logvar(z_H) + self.prior_logvar_bias

                if use_posterior:
                    u_y = self._inject_y(z_H, y_emb)
                    mu_q = self.post_mu(u_y)
                    logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
                    eps = self.reparameterize(mu_q, logvar_q)

                    # 계측 전용: 이 transition은 학습 시 KL 페널티를 받지 않는다 (Eq 14)
                    if trace_kl:
                        kl_t = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
                        self._kl_trace.append(kl_t.sum(-1).mean().item())
                else:
                    eps = self.reparameterize(mu_p, logvar_p)

                z_H = z_H + eps
                lrpm_z.append(z_H[:, 0])

        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

        z_H = self.L_level(z_H, z_L, **seq_info)

        mu_p = self.prior_mu(z_H)
        logvar_p = self.prior_logvar(z_H) + self.prior_logvar_bias

        if use_posterior:
            u_y = self._inject_y(z_H, y_emb)
            mu_q = self.post_mu(u_y)
            logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
            eps = self.reparameterize(mu_q, logvar_q)
            kl = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)

            if trace_kl:
                self._kl_trace.append(kl.detach().sum(-1).mean().item())
        else:
            mu_q = logvar_q = None
            eps = self.reparameterize(mu_p, logvar_p)
            kl = None

        # Logging-only; posterior path (= training) only, so eval stays untouched.
        diag = self._latent_diag(mu_p, logvar_p, mu_q, logvar_q, kl, z_H=z_H, z_L=z_L) if use_posterior else None

        z_H = z_H + eps
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
