"""GRAM 수식 <-> 코드 대응 발췌 + 핵심 config  (reference excerpt / 수정판).

★ import 해서 쓰는 모듈이 아니다.  논문 수식에 대응하는 코드 조각만 원본 파일들
  (gram.py / losses.py / layers.py)에서 오려 모은 것이고, 각 조각 위에 어느 식의
  어느 부분인지를 적었다.

원본 위치:
  models/recursive_reasoning/gram.py
  models/losses.py
  models/layers.py

------------------------------------------------------------------------------
gram_min.py 대비 변경 2건 (서로 독립. arm 을 섞지 말 것)
------------------------------------------------------------------------------
(A) core FFN 1536 -> 512      : config 의 expansion 4 -> 1.5.  코드 수정 없음.
      round(1.5 * 512 * 2/3) = 512 -> _find_multiple(512, 256) = 512
      noise_head_expansion 은 별개 필드이므로 건드리지 않는다 (1.5 유지).

(B) f_H / f_L 분리            : H_level 모듈 신설 + forward 호출 2곳 교체.
      - 모듈 생성   : self.H_level 추가 (H_layers 개 Block)
      - forward     : no_grad 구간과 grad 구간 '둘 다' z_H = self.H_level(...)
                      ★ 한 곳만 고치면 두 구간이 다른 네트워크를 쓰게 되어 조용히 망가진다
      - config      : H_layers 0 -> 2 ('# ignored' 주석 삭제 -- 이제 실제로 쓰인다)

  ※ 원본 gram.py:206-210 은 use_fp_transformer if/else 두 갈래다.  이 발췌는 else
     갈래만 담았으므로, 실제 gram.py 를 고칠 때는 '양쪽 갈래 모두'에 H_level 을 추가할 것.
  ※ 기존 체크포인트는 못 읽는다 -- H_level.* 키가 없어 load_state_dict 가 실패한다.

  실측 파라미터 수 (hidden 512 / L_layers 2 / noise_head_expansion 1.5):
      L_level 3,670,016 + H_level 3,670,016 + noise heads 3,145,728 + 기타
      = 10,491,397  (논문 보고치 10M 과 일치.  '기타'는 vocab_size /
        num_puzzle_identifiers 에 따라 수천 단위로 달라진다)

------------------------------------------------------------------------------
config/arch/gram.yaml 기본값 기준 '값-보존' 제거 (gram_min.py 와 동일)
------------------------------------------------------------------------------
  - 기본값 no-op 계수 : damping_eta=1.0 / eps_scale=1.0 / sigma_scale=1.0 /
                        logvar_bias_init=0.0
  - 죽은 분기        : mlp_t=False, pos_encodings=="learned",
                       scheduled sampling(ss_prob=0.0), ACT continue(no_ACT_continue=True)
  - 계측 전용        : diag 반환 경로 일체
  - dead code / 방어 : it = 0, getattr(config, ..., 기본값), lrpm_logits 존재 가드

이에 따라 수식이 함께 바뀐 곳 2군데:
  [샘플링]  h_t <- h_t + lambda_eps * u_t   =>   h_t <- h_t + u_t
  I         0.5 * (L_Qhalt + L_Qcont)       =>   0.5 * L_Qhalt
"""


# =============================================================================
# [CONFIG]  핵심 설정값
# =============================================================================

# --- config/arch/gram_nqueens8.yaml (발췌, 변경 반영) --------------------------
# name: recursive_reasoning.gram@TinyRecursiveReasoningModel_ACTV1
# loss:
#   name: losses@ACTLossHead
#   loss_type: stablemax_cross_entropy
#
# # A: T = 바깥 transition 수, K = 안쪽 refinement 수
# H_cycles: 3
# L_cycles: 4
#
# H_layers: 2            # (B) 0 -> 2.  f_H 전용 Block 수
# L_layers: 2            #               f_L 전용 Block 수
#
# hidden_size: 512
# num_heads: 8
# expansion: 1.5         # (A) 4 -> 1.5.  core FFN inter = 512
# noise_head_expansion: 1.5   # B 의 prior/posterior 헤드용. core 와 별개 필드다.
#
# puzzle_emb_ndim: ${.hidden_size}
# puzzle_emb_len: 16
# pos_encodings: rope
# forward_dtype: bfloat16
#
# # G: ACT (halting)
# halt_exploration_prob: 0.1
# halt_max_steps: 16
#
# # C, I: KL / LPRM
# kl_coef: 0.07          # I 의 lambda_KL  (sudoku 0.1 / nq8 0.07 / nq10 0.045)
# kl_balance: 0.8        # C 의 b.  '가중치' 가 아니라 그래디언트 배분이다
# lprm_coef: 1.0         # I 의 lambda_LPRM
#
# # ※ config/arch/gram.yaml(sudoku)에는 noise_head_expansion 이 없어 기본값 4.0 이
# #   먹는다.  이번 변경과 무관한 별건이므로 따로 정리할 것.


# --- gram.py:31-98 (발췌) -----------------------------------------------------
# 위 yaml 을 받는 pydantic 스키마.  이 발췌가 실제로 참조하는 필드만 남겼다.
class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # A: T = H_cycles (바깥 transition), K = L_cycles (안쪽 refinement)
    H_cycles: int
    L_cycles: int

    # (B) H_layers 는 더 이상 ignored 가 아니다 -- H_level 의 Block 수다.
    H_layers: int
    L_layers: int

    # F: core Transformer.  expansion 은 SwiGLU 중간차원을 정한다.
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.

    # G: ACT (halting)
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"
    puzzle_emb_len: int = 16

    # B: prior/posterior 노이즈 헤드(SwiGLU) 의 expansion.  core 와 독립이다.
    noise_head_expansion: float = 4.0

    # C: KL balancing 비율 b
    kl_balance: float = 0.8

    # I: 총 손실 가중치 (losses.py 가 읽는다)
    kl_coef: float = 0.1
    lprm_coef: float = 1.0


# --- layers.py:20-21, 154 -----------------------------------------------------
# expansion 이 실제 FFN 폭이 되는 경로.  256 배수로 올림한다.
#   expansion=4   -> round(4   * 512 * 2/3) = 1365 -> 1536
#   expansion=1.5 -> round(1.5 * 512 * 2/3) =  512 ->  512   (= gate_up_proj [1024, 512])
def _find_multiple(a, b):
    return (-(a // -b)) * b

inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)


# =============================================================================
# [초기화]  인코더 및 초기 상태
#
#   e_x = sqrt(D) * [ PuzzleEmb(p) || Emb(x) ],   h_0 = H_init,  l_0 = L_init
# =============================================================================

# --- gram.py:260-280 -----------------------------------------------------
# e_x 를 만드는 곳.  토큰 임베딩 앞에 퍼즐 임베딩을 prefix 로 붙이고 sqrt(D) 를 곱한다.
# 제거: pos_encodings == "learned" 분기 (yaml 은 rope)
def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
    embedding = self.embed_tokens(input.to(torch.int32))

    if self.config.puzzle_emb_ndim > 0:
        puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

        pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
        if pad_count > 0:
            puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

        embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

    return self.embed_scale * embedding


# --- gram.py:180-181 -----------------------------------------------------
# 위 식의 sqrt(D) 스케일과, 그에 맞춰 잡은 임베딩 초기화 표준편차
self.embed_scale = math.sqrt(self.config.hidden_size)
embed_init_std = 1.0 / self.embed_scale


# --- gram.py:213-214 -----------------------------------------------------
# h_0, l_0.  학습 파라미터가 아니라 trunc-normal(std=1) 로 채운 버퍼다.
self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)


# --- gram.py:430-434 -----------------------------------------------------
# halt 된 시퀀스만 h_0 / l_0 으로 되돌린다 (ACT 스텝 경계).
def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
    return TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
        z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
    )


# --- gram.py:459 ---------------------------------------------------------
# forward 안에서 e_x 를 한 번만 만들어 두고 모든 transition 이 재사용한다.
input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])


# =============================================================================
# A.  계층적 잠재 전이   ★(B) f_L / f_H 분리 반영
#
#   f(z, u) = Block_n o ... o Block_1 (z + u)
#   l_t^(k) = f_L( l_t^(k-1), h_{t-1} + e_x ),    h_t = f_H( h_{t-1}, l_t^(K) )
#
#   -> 이제 f_L 과 f_H 는 서로 다른 파라미터를 갖는 별개 모듈이다
#      (기존: 둘 다 self.L_level 하나를 공유)
# =============================================================================

# --- gram.py:504-512 -----------------------------------------------------
# 전이 본체.  하위 레벨을 K(=L_cycles) 회 돌린 뒤 상위 레벨을 1 회 갱신한다.
# 하위 레벨의 주입이 e_x 가 아니라 (z_H + e_x) 라는 점이 핵심.
# 변경(B): 상위 갱신을 L_level -> H_level 로.  ★ D 섹션의 no_grad 구간도 같이 고칠 것
# 제거: damping_eta 분기 (+ 임시변수 z_tilde)
for _L_step in range(self.config.L_cycles):
    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

z_H = self.H_level(z_H, z_L, **seq_info)


# --- gram.py:164-168 -----------------------------------------------------
# f_L / f_H 의 공통 실체.  주입(input_injection)은 concat 이 아니라 '덧셈' 이다.
def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
    hidden_states = hidden_states + input_injection
    for layer in self.layers:
        hidden_states = layer(hidden_states=hidden_states, **kwargs)
    return hidden_states


# --- gram.py:210 ---------------------------------------------------------
# 변경(B): 하위/상위 전이를 별개 모듈로 만든다.  Block 이 config 를 그대로 받으므로
# H_level 도 L_level 과 동일한 구성(norm/attention/FFN)이 된다 -- 폭만 expansion 이 정한다.
# ★ 원본 gram.py 는 use_fp_transformer if/else 두 갈래이므로 양쪽 모두에 추가할 것.
self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
self.H_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])


# =============================================================================
# F.  내부 정규화 (post-norm)
#
#   z <- RMS( z + Attn(z) ),    z <- RMS( z + MLP(z) )
# =============================================================================

# --- gram.py:144-157 -----------------------------------------------------
# 정규화가 residual 덧셈 '뒤' 에 온다 -> post-norm.  L_level / H_level 이 공유하는 Block.
# 제거: mlp_t 분기 (nqueens/GC 는 mlp_t: False.  sudoku 는 True 라 이 분기가 살아 있다)
def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
    out = self.mlp(hidden_states)
    hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
    return hidden_states


# --- layers.py:163-169 ---------------------------------------------------
# 위에서 쓰는 RMS.  learnable gain 이 없는 순수 정규화 z / sqrt(E[z^2] + eps) 다.
def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


# =============================================================================
# B.  분포 추정 (Prior / Posterior)
#
#   mu_p, log sigma_p^2 = g_p( h_t )
#   mu_q, log sigma_q^2 = g_q( h_t + e_y )
# =============================================================================

# --- gram.py:514-524 -----------------------------------------------------
# prior 는 h_t 만, posterior 는 h_t + e_y (정답 주입) 를 받는다.
# 헤드가 내놓는 것은 sigma 가 아니라 logvar 다.
# 제거: logvar_bias 항, scheduled sampling(_use_posterior_this_transition -> use_posterior)
mu_p = self.prior_mu(z_H)
logvar_p = self.prior_logvar(z_H)

if use_posterior:
    u_y = z_H + y_emb
    mu_q = self.post_mu(u_y)
    logvar_q = self.post_logvar(u_y)
    eps = self.reparameterize(mu_q, logvar_q)


# --- gram.py:222-228 -----------------------------------------------------
# g_p / g_q 의 실체.  Linear 가 아니라 SwiGLU 이고, mu 와 logvar 가 별도 헤드다.
# 폭은 core 의 expansion 이 아니라 noise_head_expansion(1.5) 이 정한다 -> inter 512.
# 제거: prior_logvar_bias / post_logvar_bias 선언 (gram.py:234-236)
D = self.config.hidden_size
nh_exp = self.config.noise_head_expansion

self.prior_mu = SwiGLU(hidden_size=D, expansion=nh_exp)
self.prior_logvar = SwiGLU(hidden_size=D, expansion=nh_exp)
self.post_mu = SwiGLU(hidden_size=D, expansion=nh_exp)
self.post_logvar = SwiGLU(hidden_size=D, expansion=nh_exp)


# --- gram.py:441 ---------------------------------------------------------
# e_y 는 학습(posterior) 일 때만 만든다.  평가 때는 None 이라 posterior 경로가 막힌다.
y_emb = self._y_embeddings(batch["labels"]) if use_posterior else None


# --- gram.py:283-294 -----------------------------------------------------
# e_y 의 실체.  학습된 embedding 조회가 아니라 one-hot 을 zero-pad 로 hidden_size 에
# 얹는다 (파라미터 0개).  0.573 = 학습된 embed_tokens 노름 / ||z_H|| 실측 비율.
def _y_embeddings(self, labels):
    labels = torch.where(labels==IGNORE_LABEL_ID, 0, labels)
    oh = F.one_hot(labels.to(torch.int64), num_classes=self.config.vocab_size)
    embedding = F.pad(oh.to(self.forward_dtype), (0, self.config.hidden_size - self.config.vocab_size))
    if self.puzzle_emb_len > 0:
        embedding = F.pad(embedding, (0, 0, self.puzzle_emb_len, 0))
    return self.embed_scale * 0.573 * embedding


# =============================================================================
# [샘플링]  재매개변수화
#
#   u_t = mu + sigma * xi,   xi ~ N(0, I),   sigma = exp(log sigma^2 / 2)
#   h_t <- h_t + u_t
# =============================================================================

# --- gram.py:327-330 -----------------------------------------------------
# torch.distributions 의 rsample 이 아니라 손으로 짠 재매개변수화다.
# ★ 반환값이 논문의 u_t 이고, 함수 안의 지역변수 eps 가 노이즈 xi 다 (이름 충돌 주의).
# 제거: sigma_scale (test-time ablation 노브)
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# --- gram.py:529-531 -----------------------------------------------------
# 평가(prior) 경로.  q = p 이므로 KL 항 자체가 없다.
else:
    eps = self.reparameterize(mu_p, logvar_p)
    kl = None


# --- gram.py:550 ---------------------------------------------------------
# 샘플을 상태에 주입:  h_t <- h_t + u_t   (z_H 가 h_t, eps 가 u_t)
# 제거: eps_scale 계수
z_H = z_H + eps


# =============================================================================
# D.  그래디언트 절단 (Truncated Gradient)
#
#   dL/dtheta = (dL/dh_T)(dh_T/dtheta)  with  h_{T-1} = const
# =============================================================================

# --- gram.py:467-472 -----------------------------------------------------
# 앞의 T-1 개 transition 을 통째로 no_grad 로 돌린다 -> 마지막 하나만 역전파된다.
# 변경(B): 상위 갱신을 L_level -> H_level 로.
#   ★ A 섹션의 grad 구간과 '반드시 같은' 모듈을 써야 한다.  한 곳만 고치면 두 구간이
#     서로 다른 네트워크를 타면서 에러 없이 조용히 망가진다.
# 제거: 계측용 t = _H_step + 1, damping_eta 분기
with torch.no_grad():
    for _H_step in range(self.config.H_cycles-1):
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

        z_H = self.H_level(z_H, z_L, **seq_info)


# --- gram.py:555, 558 ----------------------------------------------------
# 같은 계열의 절단 두 곳: ACT 스텝 사이(캐리)와 Q 헤드 입력.
new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
q_logits = self.q_head(z_H[:, 0].detach()).to(torch.float32)


# =============================================================================
# E.  디코더 복원 손실
#
#   L_recon = sum_b (1/n_b) sum_i  -log p(y_bi | h_T)
# =============================================================================

# --- gram.py:556 ---------------------------------------------------------
# 디코더는 Linear 헤드 하나.  앞쪽 puzzle prefix 자리는 잘라낸다.
output = self.lm_head(z_H)[:, self.puzzle_emb_len:]


# --- losses.py:66-68, 87 -------------------------------------------------
# 시퀀스 안에서는 유효 토큰 평균(n_b 로 나눔), 배치는 합이다.
mask = (labels != IGNORE_LABEL_ID)
loss_counts = mask.sum(-1)
loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()


# --- losses.py:11-32 -----------------------------------------------------
# 위 loss_fn 의 실체 (yaml 의 loss_type: stablemax_cross_entropy).
# softmax 가 아니라 s(x) = x+1 (x>=0) / 1/(1-x) (x<0) 를 지수 자리에 쓰는 변형이다.
def s(x, epsilon=1e-30):
    return torch.where(x<0, 1/(1-x+ epsilon), x + 1)

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


# =============================================================================
# [종료]  최종 예측
#
#   y_hat = argmax_v [ W_dec h_T ]_v        (prior 경로)
# =============================================================================

# --- gram.py:438-441 -----------------------------------------------------
# eval() 이면 use_posterior=False -> y_emb=None -> 정답을 못 본다.
if use_posterior is None:
    use_posterior = self.training

y_emb = self._y_embeddings(batch["labels"]) if use_posterior else None


# --- losses.py:61-63 -----------------------------------------------------
# 최종 예측은 lm_head 출력의 argmax 하나뿐이다.
with torch.no_grad():
    outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)


# =============================================================================
# C.  KL Divergence
#
#   L_KL = b * D_KL( sg[q] || p ) + (1-b) * D_KL( q || sg[p] )
# =============================================================================

# --- gram.py:297-306 -----------------------------------------------------
# _kl 은 대각 가우시안의 닫힌형 KL.  바깥의 b(kl_balance)는 '가중치' 가 아니라
# 그래디언트 배분이다 -- detach 는 값을 안 바꾸므로 forward 값은 정확히 D_KL(q||p) 이고,
# b 만큼 prior 를, (1-b) 만큼 posterior 를 민다.  실제 손실 가중치는 I 의 kl_coef.
def kl_loss(self, mu_q, logvar_q, mu_p, logvar_p):
    mu_q, logvar_q, mu_p, logvar_p = (t.to(torch.float32) for t in (mu_q, logvar_q, mu_p, logvar_p))

    def _kl(mu_q, logvar_q, mu_p, logvar_p):
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        return 0.5 * (logvar_p - logvar_q - 1 + (var_q + (mu_q - mu_p)**2 )/ var_p)

    b = self.config.kl_balance
    return b * _kl(mu_q.detach(), logvar_q.detach(), mu_p, logvar_p) + (1 - b) * _kl(mu_q, logvar_q, mu_p.detach(), logvar_p.detach())


# --- gram.py:525 ---------------------------------------------------------
# 호출은 마지막(grad) transition 한 곳뿐이다.  앞쪽 transition 은 샘플만 뽑는다.
kl = self.kl_loss(mu_q, logvar_q, mu_p, logvar_p)


# --- losses.py:93-95 -----------------------------------------------------
# reduction 은 sum 고정: D축 합 -> L축 합 -> batch 합.
# 평가 경로에서는 "kl" 키가 없어 0 으로 남는다.
kl_loss = torch.zeros((), device=lm_loss.device)
if outputs.get("kl") is not None:
    kl_loss = outputs["kl"].sum(-1).sum(-1).sum()


# =============================================================================
# G.  Q Head
#
#   L_Q = sum_b BCE( Q_halt( sg[h_T^(0)] ),  1[ y_hat_b = y_b ] )
# =============================================================================

# --- gram.py:185, 218-220 ------------------------------------------------
# 출력 2개(halt, continue).  Q 를 (거의) 0 에서 시작시켜 bootstrapping 을 돕는다.
self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

with torch.no_grad():
    self.q_head.weight.zero_()
    self.q_head.bias.fill_(-5)


# --- gram.py:558, 561 ----------------------------------------------------
# detach 때문에 Q 손실의 그래디언트는 q_head 로만 가고 backbone 으로 안 흐른다.
# z_H[:, 0] 은 puzzle embedding 자리인 0 번 토큰이다.
# 제거: 반환 튜플의 계측용 diag
q_logits = self.q_head(z_H[:, 0].detach()).to(torch.float32)
return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), kl, lrpm_logits


# --- losses.py:88 --------------------------------------------------------
# MSE 가 아니라 BCE.  타깃은 '이 시퀀스를 전부 맞혔는가' 0/1 이다.
# 제거: q_continue 항 (no_ACT_continue: True 라 target_q_continue 가 생기지 않는다)
q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")


# =============================================================================
# H.  V Head (LPRM)
#
#   v_t = w^T h_t^(0) + b,     L_V = sum_b sum_t ( v_t - r_b )^2
# =============================================================================

# --- gram.py:229 ---------------------------------------------------------
# 잠재 -> 스칼라 가치.  Linear 하나뿐이다.
self.lrpm_head = CastedLinear(D, 1, bias=True)


# --- gram.py:464, 501, 551 -----------------------------------------------
# 모든 transition 의 잠재를 모은다: no-grad 구간(501)과 마지막 transition(551).
lrpm_z = []

lrpm_z.append(z_H[:, 0])      # (501) no-grad transition
lrpm_z.append(z_H[:, 0])      # (551) 마지막 transition


# --- gram.py:560 ---------------------------------------------------------
# (B, T, D) 로 쌓아 한 번에 사상 -> (B, T)
lrpm_logits = self.lrpm_head(torch.stack(lrpm_z, dim=1)).to(torch.float32).squeeze(-1)


# --- losses.py:97-101 ----------------------------------------------------
# 타깃 r 은 '최종 출력의 토큰 정확도' 이고 detach 되어 있으며 t 에 무관하게 같다.
# (bootstrap 도 discount 도 없는 순수 회귀)
# 제거: if "lrpm_logits" in outputs 가드 (항상 채워진다)
r = (is_correct.to(torch.float32) / loss_divisor).sum(-1).detach()
lprm_loss = ((outputs["lrpm_logits"] - r.unsqueeze(-1)) ** 2).sum(-1).sum()


# =============================================================================
# I.  총 손실
#
#   L = L_recon + kl_coef * L_KL + 0.5 * L_Qhalt + lprm_coef * L_LPRM
# =============================================================================

# --- losses.py:125-129 ---------------------------------------------------
# 네 항을 한 줄로 합친다.  Q 계수 0.5 만 하드코딩이고 나머지는 arch config 값이다.
# 제거: getattr(..., 기본값) 호환 방어, L_Qcont 항
kl_coef   = self.model.config.kl_coef
lprm_coef = self.model.config.lprm_coef

return new_carry, lm_loss + kl_coef * kl_loss + 0.5 * q_halt_loss + lprm_coef * lprm_loss, metrics, detached_outputs, new_carry.halted.all()


# =============================================================================
# [검증]  변경 후 확인 항목
# =============================================================================

# assert model.inner.L_level is not model.inner.H_level          # (B) 분리됐는가
# model.inner.L_level.layers[0].mlp.gate_up_proj.weight.shape    # (A) [1024, 512] 기대
#                                                               #     (gate+up 합쳐 2배)
# sum(p.numel() for p in model.parameters())                     # ~10.49M 기대
#
# 실측 (hidden 512 / L_layers=H_layers=2 / noise_head_expansion 1.5):
#   expansion=4.0 : core gate_up_proj (3072, 512), L_level 6,815,744
#   expansion=1.5 : core gate_up_proj (1024, 512), L_level 3,670,016
#                   + H_level 3,670,016 + noise heads 3,145,728 + 기타
#                   = 10,491,397
