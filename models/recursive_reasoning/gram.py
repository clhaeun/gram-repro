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

    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense


    kl_coef: float = 0.1
    kl_balance: float = 0.8
    lprm_coef: float = 1.0

    # ε 주입 계수. FPRM Theorem 1 의 유계성은 "모든 덧셈에 계수" 를 전제하는데
    # GRAM 의 z_H = z_H + eps 만 계수가 없어 그 자리에서 보장이 깨진다.
    # 1.0 = 현행과 비트 단위 동일.
    eps_scale: float = 1.0

    # inner z_L loop damped update. 1.0 = 현행과 비트 단위 동일.
    # η 고정. patience/adaptive 는 넣지 않는다 -- L_cycles(4) < FPRM patience(10) 이라
    # 발동 기회 자체가 없다.
    damping_eta: float = 1.0

    noise_head_expansion: float = 4.0

    # 평가 시 posterior 를 ★측정 전용★ 으로 계산해 transition 별 KL 을 기록한다.
    # GRAM Eq 14 는 KL 을 마지막 transition 에만 걸므로(논문이 biased surrogate 로
    # 명시) 앞쪽 transition 의 prior 가 미학습 상태로 남는지 실측하기 위한 것이다.
    # False = 현행과 비트 단위 동일.
    diag_kl_per_transition: bool = False

    # 학습 중 각 transition 에서 확률 ss_prob 로 posterior 대신 prior 를 쓴다.
    # 학습은 posterior(정답 봄) / 추론은 prior(정답 없음) 라는 exposure bias 완화용.
    # 0.0 = 현행과 비트 단위 동일 (torch.rand 호출 자체를 건너뛴다).
    ss_prob: float = 0.0


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

        # prior/posterior logvar 헤드에 더하는 학습가능 스칼라 bias.  최종 설정 init 0.0
        # (sigma=1.0 시작).  파라미터 자체는 state_dict 키이자 GRAM 노이즈 head 의 일부다.
        self.prior_logvar_bias = nn.Parameter(torch.tensor(0.0))
        self.post_logvar_bias = nn.Parameter(torch.tensor(0.0))

        # Test-time sampled latent std 조정 (논문 Table 3 ablation; harness 의
        # sigma_scale()/mu_scale() 컨텍스트가 이 속성을 바꾼다)
        self.sigma_scale = 1.0

        # diag_kl_per_transition 누적기 (측정 전용).  forward 궤적과 무관하며
        # 플래그가 꺼져 있으면 건드리지 않는다.  harness 가 reset -> rollout ->
        # result 순으로 쓴다.
        self.diag_kl_sum: Dict[str, float] = {}
        self.diag_kl_n: int = 0


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
        # 사수 처방 4 (posterior 의 y-copy 통로 차단): y 는 학습된 embedding 조회가 아니라
        # one-hot 을 zeropad 로 hidden_size 에 얹는다.  파라미터 0개, state_dict 무영향.
        # 0.573 = 학습된 embed_tokens 노름(12.97) / ‖z_H‖(22.63) 실측 비율.
        # one-hot 은 노름이 고정이라 학습으로 못 줄이므로 처음부터 맞춘 값.
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


    def _use_posterior_this_transition(self, use_posterior: bool, device) -> bool:
        """[B] scheduled sampling: transition 마다 ★독립★ 으로 prior/posterior 를 뽑는다
        (sequence 당 1회가 아니다).

        ★ ss_prob=0.0 에서는 torch.rand 를 아예 부르지 않는다 -- 부르면 RNG 소비가
          달라져 기존 run 과의 비트 동일성이 깨진다.  분기로 처리해야 한다.
        """
        if not use_posterior:
            return False
        if self.config.ss_prob <= 0.0:
            return True
        return bool(torch.rand((), device=device) >= self.config.ss_prob)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) * self.sigma_scale
        eps = torch.randn_like(std)
        return mu + eps * std


    @torch.no_grad()
    def _latent_diag(self, z_H, z_L, mu, logvar):
        """FPRM Figure 2b 재현용: 최종 transition 의 ‖z_H‖/‖z_L‖ (배치·시퀀스 평균).
        boundedness 가 FPRM 1차 판정 지표 -- accuracy 보다 먼저 본다.

        eps_mu/eps_std 는 eps = mu + std*noise 의 두 항을 분리 기록한다.
        29번 실측에서 σ 는 붕괴 구간 0.09% 변동뿐이었으므로, 폭주 주범이
        mu 인지 z_H 자체인지 갈라야 eps_scale 셀 결과를 해석할 수 있다.
        """
        return {
            "diag/z_h_norm": z_H.detach().float().norm(dim=-1).mean(),
            "diag/z_l_norm": z_L.detach().float().norm(dim=-1).mean(),
            "diag/eps_mu_norm": mu.detach().float().norm(dim=-1).mean(),
            "diag/eps_std_norm": (torch.exp(0.5 * logvar.detach().float()) * self.sigma_scale).norm(dim=-1).mean(),
        }


    def diag_kl_reset(self):
        """diag_kl_per_transition 누적기 초기화 (harness 가 rollout 전에 부른다)."""
        self.diag_kl_sum = {}
        self.diag_kl_n = 0


    def diag_kl_result(self):
        """누적된 transition 별 KL/μ/σ 의 forward 평균.  누적이 없으면 None."""
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
        """transition t 의 posterior 를 ★측정 전용★ 으로 계산해 KL/μ/σ 를 적는다.

        ★★ 여기서 만든 mu_q 로 ε 을 뽑거나 z_H 를 갱신하면 절대 안 된다.  궤적이
           바뀌면 다른 것을 재게 된다.  post_mu/post_logvar/_y_embeddings 는 전부
           RNG 를 소비하지 않으므로 이 블록은 prior 경로의 난수 스트림도 건드리지
           않는다 (그래서 플래그 on/off 가 비트 단위로 같다).

        kl_t 는 losses.py 의 kl_loss 와 같은 눈금이다: D축 합 -> L축 합.  다만
        batch 는 합이 아니라 평균이다 -- diag/ 접두사는 pretrain.py 가 world_size
        로만 나누는 per-rank batch mean 규약이기 때문이다.
        """
        u_y = z_H + y_emb
        mu_q = self.post_mu(u_y)
        logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
        kl = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
        std_q = torch.exp(0.5 * logvar_q.float())
        dst[f"diag/kl_t{t}"] = kl.sum(-1).sum(-1).mean()
        # 두 축을 병기한다.  *_norm 은 per-position L2(D축) 뒤 나머지 축 평균,
        # *_absmean/*_mean 은 (B,L,D) 전체 element 평균이다.  두 축은 sqrt(D) 만큼
        # 눈금이 다르므로 인용 시 어느 축인지 밝힐 것.
        dst[f"diag/eps_mu_norm_t{t}"] = mu_q.float().norm(dim=-1).mean()
        dst[f"diag/eps_std_norm_t{t}"] = std_q.norm(dim=-1).mean()
        dst[f"diag/eps_mu_absmean_t{t}"] = mu_q.float().abs().mean()
        dst[f"diag/eps_std_mean_t{t}"] = std_q.mean()
        # 차원(D) 간 표준편차.  역산이 아니라 직접 계산한다 -- sigma 가 D 축에서
        # 균일한지(head 가 죽어 상수를 뱉는지) 를 mean 과 분리해 보기 위한 축.
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

        # 진단 B+ (측정 전용).  평가 시엔 use_posterior=False 라 y_emb 가 None 이므로
        # 여기서 label 을 따로 읽는다.  ★ 궤적에는 절대 쓰지 않는다.
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
                t = _H_step + 1                     # 1-based transition 번호
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

                if self._use_posterior_this_transition(use_posterior, z_H.device):
                    # KL 페널티는 마지막(grad) transition 에만 걸린다 (Eq 14) --
                    # 여기 no-grad transition 들은 샘플만 뽑는다.
                    u_y = z_H + y_emb
                    mu_q = self.post_mu(u_y)
                    logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
                    eps = self.reparameterize(mu_q, logvar_q)
                else:
                    eps = self.reparameterize(mu_p, logvar_p)

                # ★ eps 를 뽑은 뒤에 잰다 -- 난수 스트림에 끼어들지 않음을 눈으로
                #   확인할 수 있게. (_record_diag_kl 자체는 RNG 를 안 쓴다)
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

        # ★ [B] 마지막 transition 도 scheduled sampling 대상이다.  prior 로 뽑히면
        #   q=p 라 KL 이 0 이므로 항 자체를 뺀다 (kl=None -> losses.py 가 0 처리).
        use_post_last = self._use_posterior_this_transition(use_posterior, z_H.device)
        if use_post_last:
            u_y = z_H + y_emb
            mu_q = self.post_mu(u_y)
            logvar_q = self.post_logvar(u_y) + self.post_logvar_bias
            eps = self.reparameterize(mu_q, logvar_q)
            kl = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
        else:
            eps = self.reparameterize(mu_p, logvar_p)
            kl = None

        # Logging-only; posterior path (= training) only, so eval stays untouched.
        # ss_prob=0.0 이면 use_post_last == use_posterior 라 기존과 동일하다.
        diag = self._latent_diag(z_H, z_L, mu_q, logvar_q) if use_post_last else None

        # 마지막(grad) transition 도 같이 잰다 -- 이 자리만 KL 을 학습받으므로
        # 앞쪽 t 의 미학습 여부를 판정하려면 기준점이 필요하다.
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
