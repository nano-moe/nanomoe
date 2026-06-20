from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from nanomoe.model.model import MoETransformer


@dataclass
class RouterIORecord:
    """Captured input/output for a router forward pass."""

    inputs: Tensor
    router_logits: Tensor
    expert_indices: Tensor
    expert_weights: Tensor


@dataclass
class RouterIOCapture:
    """Router IO records for a single transformer layer."""

    layer_idx: int
    records: list[RouterIORecord]

    def clear(self) -> None:
        self.records.clear()


def register_router_io_hooks(
    model: MoETransformer,
    *,
    detach: bool = True,
    to_cpu: bool = False,
) -> tuple[list[RouterIOCapture], list[torch.utils.hooks.RemovableHandle]]:
    """Register forward hooks on each router to capture inputs and outputs.

    Args:
        model: A ``MoETransformer`` instance.
        detach: Whether to detach captured tensors from the autograd graph.
        to_cpu: Whether to move captured tensors to CPU memory.

    Returns:
        A tuple of (captures, handles). Call ``handle.remove()`` on each handle
        when done to unregister hooks.
    """
    captures: list[RouterIOCapture] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, layer in enumerate(model.layers):
        router = getattr(getattr(layer, "mlp", None), "router", None)
        if router is None:
            continue
        capture = RouterIOCapture(layer_idx=layer_idx, records=[])
        hook = _make_router_io_hook(capture.records, detach=detach, to_cpu=to_cpu)
        handles.append(router.register_forward_hook(hook))
        captures.append(capture)

    return captures, handles


def _make_router_io_hook(
    records: list[RouterIORecord],
    *,
    detach: bool,
    to_cpu: bool,
):
    @torch.compiler.disable
    def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if not inputs:
            raise RuntimeError("Router hook expected at least one input tensor.")
        hidden_states = inputs[0]
        if not isinstance(hidden_states, Tensor):
            raise TypeError(
                f"Router hook expected Tensor input, got {type(hidden_states)!r}"
            )

        if not isinstance(output, tuple) or len(output) != 3:
            raise TypeError(
                "Router hook expected (router_logits, expert_indices, expert_weights) output tuple."
            )
        router_logits, expert_indices, expert_weights = output
        if not isinstance(router_logits, Tensor):
            raise TypeError(f"Router logits must be Tensor, got {type(router_logits)!r}")
        if not isinstance(expert_indices, Tensor):
            raise TypeError(f"Expert indices must be Tensor, got {type(expert_indices)!r}")
        if not isinstance(expert_weights, Tensor):
            raise TypeError(f"Expert weights must be Tensor, got {type(expert_weights)!r}")

        def _capture(tensor: Tensor) -> Tensor:
            if detach:
                tensor = tensor.detach()
            if to_cpu:
                tensor = tensor.cpu()
            return tensor

        records.append(
            RouterIORecord(
                inputs=_capture(hidden_states),
                router_logits=_capture(router_logits),
                expert_indices=_capture(expert_indices),
                expert_weights=_capture(expert_weights),
            )
        )

    return hook
