"""Device prefetcher to overlap H2D copy with compute."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class PrefetchConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    enabled: bool = True
    pin_memory: bool = True
    non_blocking: bool = True


def _pin_tensor(tensor: Tensor, pin_memory: bool) -> Tensor:
    if pin_memory and tensor.device.type == "cpu":
        return tensor.pin_memory()
    return tensor


def _move_to_device[T](
    obj: T,
    device: torch.device,
    *,
    pin_memory: bool,
    non_blocking: bool,
) -> T:
    if obj is None:
        return cast(T, obj)
    if isinstance(obj, Tensor):
        tensor = _pin_tensor(obj, pin_memory)
        if tensor.device == device:
            return cast(T, tensor)
        return cast(T, tensor.to(device, non_blocking=non_blocking))
    if isinstance(obj, dict):
        return cast(
            T,
            {k: _move_to_device(v, device, pin_memory=pin_memory, non_blocking=non_blocking) for k, v in obj.items()},
        )
    if isinstance(obj, tuple):
        values = [_move_to_device(v, device, pin_memory=pin_memory, non_blocking=non_blocking) for v in obj]
        if hasattr(obj, "_fields"):
            return cast(T, type(obj)(*values))
        return cast(T, tuple(values))
    if isinstance(obj, list):
        return cast(
            T,
            [_move_to_device(v, device, pin_memory=pin_memory, non_blocking=non_blocking) for v in obj],
        )
    if hasattr(obj, "to") and callable(obj.to):
        try:
            return cast(T, obj.to(device, non_blocking=non_blocking, pin_memory=pin_memory))  # type: ignore[attr-defined]
        except TypeError:
            try:
                return cast(T, obj.to(device, non_blocking=non_blocking))  # type: ignore[attr-defined]
            except TypeError:
                return cast(T, obj.to(device))  # type: ignore[attr-defined]
    if is_dataclass(obj):
        values = {}
        for field in fields(obj):
            if not field.init:
                continue
            values[field.name] = _move_to_device(
                getattr(obj, field.name),
                device,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
            )
        dataclass_obj = cast(Any, obj)
        return cast(T, replace(dataclass_obj, **values))
    return cast(T, obj)


def _record_stream(obj: Any, stream: torch.cuda.Stream) -> None:
    if obj is None:
        return
    if isinstance(obj, Tensor):
        if obj.is_cuda:
            obj.record_stream(stream)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _record_stream(value, stream)
        return
    if isinstance(obj, tuple | list):
        for value in obj:
            _record_stream(value, stream)
        return
    if hasattr(obj, "record_stream") and callable(obj.record_stream):
        obj.record_stream(stream)
        return
    if is_dataclass(obj):
        for field in fields(obj):
            _record_stream(getattr(obj, field.name), stream)


class DevicePrefetcher[T](Iterator[T]):
    """Prefetch next batch to device using a dedicated CUDA stream."""

    def __init__(
        self,
        data_iter: Iterator[T],
        device: torch.device,
        *,
        pin_memory: bool = True,
        non_blocking: bool = True,
    ):
        self._iter = iter(data_iter)
        self._device = device
        self._pin_memory = pin_memory
        self._non_blocking = non_blocking
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" and torch.cuda.is_available() else None
        self._next: T | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return

        if self._stream is None:
            self._next = _move_to_device(
                batch,
                self._device,
                pin_memory=False,
                non_blocking=False,
            )
        else:
            with torch.cuda.stream(self._stream):
                self._next = _move_to_device(
                    batch,
                    self._device,
                    pin_memory=self._pin_memory,
                    non_blocking=self._non_blocking,
                )

    def __iter__(self) -> DevicePrefetcher[T]:
        return self

    def __next__(self) -> T:
        if self._next is None:
            raise StopIteration

        if self._stream is not None:
            current_stream = torch.cuda.current_stream(self._device)
            current_stream.wait_stream(self._stream)
            batch = self._next
            _record_stream(batch, current_stream)
        else:
            batch = self._next

        self._preload()
        return batch


def maybe_prefetch[T](
    data_iter: Iterator[T],
    device: torch.device,
    config: PrefetchConfig | None,
) -> Iterator[T]:
    if config is None or not config.enabled or device.type != "cuda":
        return data_iter
    return DevicePrefetcher(
        data_iter,
        device=device,
        pin_memory=config.pin_memory,
        non_blocking=config.non_blocking,
    )
