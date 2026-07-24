import copy
import torch.nn as nn

class EMAHelper(object):
    """Exponential moving average of model parameters.

    IMPORTANT bias-correction note.  The shadow is initialised to the model's
    *init* weights w0 (register()), so after t updates

        shadow_t = (1 - mu^t) * W_avg  +  mu^t * w0

    i.e. a mu^t fraction of the random init survives.  At mu=0.9999 that is
    0.15% after 65k steps (Sudoku) but 18% after 17k steps (N-Queens).  The
    standard Adam-style correction `shadow / (1 - mu^t)` assumes shadow_0 = 0
    and is WRONG here -- it would *add* init instead of removing it.  The
    correct debiasing subtracts the surviving init first:

        W_avg = (shadow_t - mu^t * w0) / (1 - mu^t)

    which needs w0; we snapshot it at register() time.  The training-time
    shadow buffer itself is never modified by the correction.
    """

    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}
        self.w0 = {}            # init snapshot, needed for correct bias correction
        self.num_updates = 0

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                self.w0[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        self.num_updates += 1
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    # --- raw (uncorrected) copy: shadow as-is. Kept for exact old behaviour. ---
    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def _debiased(self, name, step):
        mt = self.mu ** step
        denom = 1.0 - mt
        if denom < 1e-8:                       # step==0 or numerically ~0: no correction
            return self.shadow[name].data
        return (self.shadow[name].data - mt * self.w0[name].data) / denom

    def ema_bias_corrected(self, module, step):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self._debiased(name, step).to(param.dtype))

    def ema_copy(self, module, step=None):
        """step=None -> raw EMA (legacy). step=int -> bias-corrected EMA."""
        module_copy = copy.deepcopy(module)
        if step is None:
            self.ema(module_copy)
        else:
            self.ema_bias_corrected(module_copy, step)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
