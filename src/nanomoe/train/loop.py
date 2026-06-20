from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer

from nanomoe.data.types import PackedBatch
from nanomoe.train.checkpoint import Checkpointer
from nanomoe.train.logging import Logger
from nanomoe.train.prefetch import PrefetchConfig, maybe_prefetch


class TrainLoopConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    max_steps: int = 1000
    max_tokens: int | None = None
    gradient_accumulation: int = 1
    log_every: int = 10
    checkpoint_every: int = 500
    max_grad_norm: float = 1.0


class CudaProfilerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    enabled: bool = False
    start_step: int = 1
    profile_steps: int = 5


class TrainState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    step: int = 0
    tokens_seen: int = 0


def _default_token_count(batch: PackedBatch) -> int:
    return int(batch.token_weights.ne(0).sum().item())


def _unwrap_model(module: Module) -> Module:
    target = module
    if hasattr(target, "module"):
        target = target.module
    if hasattr(target, "_orig_mod"):
        target = target._orig_mod
    return target


def _maybe_set_router_aux_loss_accumulation(module: Module, enabled: bool) -> None:
    target = _unwrap_model(module)
    setter = getattr(target, "set_router_aux_loss_accumulation", None)
    if callable(setter):
        setter(enabled)


def _maybe_reset_router_aux_stats(module: Module) -> None:
    target = _unwrap_model(module)
    reset = getattr(target, "reset_router_aux_stats", None)
    if callable(reset):
        reset()


def train_loop(
    *,
    model: Module,
    data_iter: Iterator[PackedBatch],
    step_fn: Callable[[PackedBatch], tuple[Tensor, dict[str, float]]],
    optimizer: Optimizer,
    cfg: TrainLoopConfig,
    scheduler: Any | None = None,
    logger: Logger | None = None,
    checkpointer: Checkpointer | None = None,
    state: TrainState | None = None,
    data_iter_factory: Callable[[], Iterator[PackedBatch]] | None = None,
    grad_scaler: GradScaler | None = None,
    autocast_dtype: torch.dtype | None = None,
    token_count_fn: Callable[[PackedBatch], int] = _default_token_count,
    prefetch_config: PrefetchConfig | None = None,
    profile_config: CudaProfilerConfig | None = None,
) -> TrainState:
    if state is None:
        state = TrainState()

    device = next(model.parameters()).device
    if prefetch_config is not None:
        data_iter = maybe_prefetch(data_iter, device, prefetch_config)
        if data_iter_factory is not None:
            factory = data_iter_factory

            def wrapped_factory() -> Iterator[PackedBatch]:
                return maybe_prefetch(factory(), device, prefetch_config)

            data_iter_factory = wrapped_factory
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None and device.type in {"cuda", "cpu"}
        else nullcontext()
    )

    _maybe_set_router_aux_loss_accumulation(model, cfg.gradient_accumulation > 1)

    profile_enabled = (
        profile_config is not None
        and profile_config.enabled
        and profile_config.profile_steps > 0
        and device.type == "cuda"
        and torch.cuda.is_available()
    )
    profile_active = False
    profile_start = max(1, profile_config.start_step) if profile_enabled else 0
    profile_end = profile_start + profile_config.profile_steps - 1 if profile_enabled else 0

    accumulated_metrics: dict[str, float] = {}
    accumulated_tokens = 0
    start_time = time.time()

    try:
        while state.step < cfg.max_steps:
            if cfg.max_tokens is not None and state.tokens_seen >= cfg.max_tokens:
                break

            current_step = state.step + 1
            profile_window = profile_enabled and profile_start <= current_step <= profile_end

            if profile_enabled and not profile_active and current_step == profile_start:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                profile_active = True

            if profile_window:
                torch.cuda.nvtx.range_push("train_step")
            try:
                optimizer.zero_grad()
                _maybe_reset_router_aux_stats(model)
                step_tokens = 0
                step_metrics: dict[str, float] = {}

                for _ in range(cfg.gradient_accumulation):
                    if profile_window:
                        torch.cuda.nvtx.range_push("data_fetch")
                    try:
                        try:
                            batch = next(data_iter)
                        except StopIteration:
                            if data_iter_factory is None:
                                raise
                            data_iter = data_iter_factory()
                            batch = next(data_iter)
                    finally:
                        if profile_window:
                            torch.cuda.nvtx.range_pop()

                    step_tokens += token_count_fn(batch)

                    if profile_window:
                        torch.cuda.nvtx.range_push("forward_backward")
                    try:
                        with autocast_ctx:
                            loss, metrics = step_fn(batch)

                        scaled_loss = loss / cfg.gradient_accumulation
                        if grad_scaler is not None:
                            grad_scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                    finally:
                        if profile_window:
                            torch.cuda.nvtx.range_pop()

                    for k, v in metrics.items():
                        if isinstance(v, int | float):
                            scale = 1.0 if k.startswith("num_") else 1.0 / cfg.gradient_accumulation
                            step_metrics[k] = step_metrics.get(k, 0.0) + float(v) * scale

                if profile_window:
                    torch.cuda.nvtx.range_push("optimizer_step")
                try:
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)
                        if cfg.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        if cfg.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                        optimizer.step()
                finally:
                    if profile_window:
                        torch.cuda.nvtx.range_pop()

                state.tokens_seen += step_tokens
                lr = None
                if scheduler is not None:
                    try:
                        lr = scheduler.step(step=state.step, tokens_seen=state.tokens_seen)
                    except TypeError:
                        lr = scheduler.step()

                state.step += 1

                for k, v in step_metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
                accumulated_tokens += step_tokens

                if logger is not None and state.step % cfg.log_every == 0:
                    elapsed = time.time() - start_time
                    log_metrics = {f"train/{k}": v / cfg.log_every for k, v in accumulated_metrics.items()}
                    if lr is not None:
                        log_metrics["train/lr"] = lr
                    log_metrics["train/step"] = state.step
                    log_metrics["train/tokens_seen"] = state.tokens_seen
                    log_metrics["perf/tokens_per_sec"] = accumulated_tokens / max(elapsed, 1e-6)
                    log_metrics["perf/elapsed_time"] = elapsed
                    logger.log_metrics(log_metrics, step=state.step)

                    accumulated_metrics.clear()
                    accumulated_tokens = 0
                    start_time = time.time()

                if checkpointer is not None and state.step % cfg.checkpoint_every == 0:
                    checkpointer.save(
                        step=state.step,
                        model=model,
                        optimizer=optimizer,
                        tokens_seen=state.tokens_seen,
                        scheduler=scheduler,
                    )
            finally:
                if profile_window:
                    torch.cuda.nvtx.range_pop()

            if profile_active and current_step >= profile_end:
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                profile_active = False
    finally:
        if profile_active:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()

    return state
