from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    # Config
    "MoEConfig": ("nanomoe.model.config", "MoEConfig"),
    # Model
    "MoETransformer": ("nanomoe.model.model", "MoETransformer"),
    "TransformerBlock": ("nanomoe.model.model", "TransformerBlock"),
    "create_model": ("nanomoe.model.model", "create_model"),
    "ModelOutput": ("nanomoe.model.output", "ModelOutput"),
    # MoE
    "MoELayer": ("nanomoe.model.moe", "MoELayer"),
    "Expert": ("nanomoe.model.moe", "Expert"),
    "TopKRouter": ("nanomoe.model.moe", "TopKRouter"),
    "BaseRouter": ("nanomoe.model.moe_router", "BaseRouter"),
    "LinearRouter": ("nanomoe.model.moe_router", "LinearRouter"),
    "NaiveTopKRouter": ("nanomoe.model.moe_router", "NaiveTopKRouter"),
    "SwitchTop1Router": ("nanomoe.model.moe_router", "SwitchTop1Router"),
    "SwitchRouter": ("nanomoe.model.moe_router", "SwitchRouter"),
    "StraightThroughTopKRouter": ("nanomoe.model.moe_router", "StraightThroughTopKRouter"),
    "StraightThroughRouter": ("nanomoe.model.moe_router", "StraightThroughRouter"),
    "GumbelStraightThroughTopKRouter": ("nanomoe.model.moe_router", "GumbelStraightThroughTopKRouter"),
    "GumbelSoftmaxStraightThroughRouter": ("nanomoe.model.moe_router", "GumbelSoftmaxStraightThroughRouter"),
    "PolicyGradientRouter": ("nanomoe.model.moe_router", "PolicyGradientRouter"),
    "DenseFFN": ("nanomoe.model.moe", "DenseFFN"),
    "SwiGLU": ("nanomoe.model.moe", "SwiGLU"),
    "softmax_normalize": ("nanomoe.model.moe_router", "softmax_normalize"),
    "sigmoid_normalize": ("nanomoe.model.moe_router", "sigmoid_normalize"),
    # Attention
    "Attention": ("nanomoe.model.attention", "Attention"),
    "RoPE": ("nanomoe.model.attention", "RoPE"),
    "apply_rope": ("nanomoe.model.attention", "apply_rope"),
    # Normalization
    "RMSNorm": ("nanomoe.model.model", "RMSNorm"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
