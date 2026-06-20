import math
from dataclasses import dataclass

import torch
from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, Muon
from torch.optim import AdamW


@dataclass(frozen=True)
class AdjustLrRmsNorm:
    """Pickleable Muon LR adjuster for constant element-wise RMS norm."""

    adam_beta1: float
    adam_beta2: float

    def __call__(self, lr, param_shape):
        fan_out, fan_in = param_shape[-2:]
        rms_adamw = math.sqrt((1 - self.adam_beta1) / (1 + self.adam_beta1))
        adjusted_ratio = rms_adamw * math.sqrt(max(fan_out, fan_in))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr


def expert_weights_split_fn(param: torch.Tensor):
    """
    Split (num_experts, fan_in, fan_out) → a list of num_experts tensors of shape (fan_in, fan_out).
    """
    return list(param.unbind(dim=0))


def expert_weights_recombine_fn(splits):
    return torch.cat(splits, dim=0)


def build_adjust_lr_rms_norm_func(adam_beta1, adam_beta2):
    """
    Adjust learning rate for constant element-wise RMS norm.
    https://arxiv.org/abs/2502.16982
    """
    # Muon stores adjust_lr inside optimizer param_groups, which are included in
    # optimizer.state_dict(). A nested closure cannot be pickled by torch.save.
    # Use a top-level callable object so Muon checkpoints can be saved/resumed.
    # beta2 is kept in the signature for optimizer-builder compatibility; this
    # RMS correction only depends on the Adam first-moment beta.
    return AdjustLrRmsNorm(adam_beta1=adam_beta1, adam_beta2=adam_beta2)


def extract_split_weights(model):
    no_wd_types = {"layernorm", "norm", "ln"}  # module name heuristic

    muon_params_2d = []  # 2-D+ weights → Muon
    muon_params_3d = []  # 3-D weights → Muon
    adamw_params = []  # scalars / output → AdamW  (with WD)
    no_wd_params = []  # layernorm / bias / embed → AdamW  (WD = 0)

    # Identify the output projection (last Linear before logits)
    output_modules = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" in name:
            output_modules.add(name)

    for mod_name, mod in model.named_modules():
        for param_name, param in mod.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            mod_lower = mod_name.lower()

            is_embedding = isinstance(mod, torch.nn.Embedding)
            is_output = mod_name in output_modules
            is_norm = any(k in mod_lower for k in no_wd_types)
            is_bias = param_name == "bias"
            is_scalar = param.ndim < 2
            # ── No-WD group (layernorm weights & biases, all biases, embeddings)
            if is_norm or is_bias or is_embedding:
                no_wd_params.append(param)

            # ── AdamW group (output head, scalars like MoE gate logits)
            elif is_output or is_scalar:
                adamw_params.append(param)

            # ── Muon group (all remaining 2-D+ matrices: attn, mlp, expert weights)
            else:
                if param.ndim == 2:
                    muon_params_2d.append(param)
                elif param.ndim == 3:
                    muon_params_3d.append(param)
    return muon_params_2d, muon_params_3d, adamw_params, no_wd_params


def build_optimizer_muon(
    model,
    lr_adamw: float = 3e-4,
    lr_layernorm: float = 3e-4,
    lr_muon: float = 2e-2,
    wd: float = 0.1,
    betas_adamw: tuple = (0.9, 0.9),
    momentum_muon: float = 0.95,
):
    """
    Optimizer groups for MoE models:
      - Muon   : all 2-D+ non-embedding, non-output weight matrices
      - AdamW  : scalars, output projection (with weight decay)
      - AdamW  : layernorm/bias/embedding params that must have WD = 0
    """
    muon_params_2d, muon_params_3d, adamw_params, no_wd_params = extract_split_weights(model)
    optimizers = []

    if muon_params_2d or muon_params_3d:
        muon_params_groups = []
        if muon_params_2d:
            muon_params_groups.append({"params": muon_params_2d, "split_fn": None, "recombine_fn": None})
        if muon_params_3d:
            muon_params_groups.append(
                {
                    "params": muon_params_3d,
                    # Use default 3d spliting
                    # "param_split_fn": expert_weights_split_fn,
                    # "param_recombine_fn": expert_weights_recombine_fn
                }
            )
        optimizers.append(
            Muon(
                muon_params_groups,
                lr=lr_muon,
                momentum=momentum_muon,
                weight_decay=wd,
                adjust_lr=build_adjust_lr_rms_norm_func(*betas_adamw),
                ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
                ns_algorithm="standard_newton_schulz",
                ns_use_kernels=False,
            )
        )

    if adamw_params:
        optimizers.append(AdamW(adamw_params, lr=lr_adamw, betas=betas_adamw, weight_decay=wd))

    if no_wd_params:
        optimizers.append(AdamW(no_wd_params, lr=lr_layernorm, betas=betas_adamw, weight_decay=0.0))

    return optimizers


def build_optimizer_adamw(
    model,
    lr_adamw: float = 3e-4,
    lr_layernorm: float = 3e-4,
    lr_muon: float = 2e-2,
    wd: float = 0.1,
    betas_adamw: tuple = (0.9, 0.9),
    momentum_muon: float = 0.95,
):
    """
    Optimizer groups for MoE models:
      - AdamW  : all 2-D+ non-embedding, non-output weight matrices,  scalars, output projection (with weight decay)
      - AdamW  : layernorm/bias/embedding params that must have WD = 0
    """
    muon_params_2d, muon_params_3d, adamw_params, no_wd_params = extract_split_weights(model)
    optimizers = []
    if adamw_params:
        optimizers.append(
            AdamW(muon_params_2d + muon_params_3d + adamw_params, lr=lr_adamw, betas=betas_adamw, weight_decay=wd)
        )

    if no_wd_params:
        optimizers.append(AdamW(no_wd_params, lr=lr_layernorm, betas=betas_adamw, weight_decay=0.0))

    return optimizers
