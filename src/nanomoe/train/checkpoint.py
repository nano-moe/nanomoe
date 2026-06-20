"""Checkpointing utilities for GRPO training.

Simplified from nmoe/checkpoint.py - supports:
- Async checkpointing with background thread
- Tracker file for latest checkpoint
- Keep-last rotation (auto-delete old checkpoints)
- State dict save/load for model, optimizer, scheduler, data
"""

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer

TRACKER = "latest_checkpoint.txt"
LATEST_DIR = "latest"


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _iteration_dir(base: str | Path, step: int) -> Path:
    return Path(base) / f"step_{step:07d}"


def _latest_dir(base: str | Path) -> Path:
    return Path(base) / LATEST_DIR


def _tracker_path(base: str | Path) -> Path:
    return Path(base) / TRACKER


def _fsync_file(path: str | Path) -> None:
    """Fsync a file to ensure durability."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str | Path) -> None:
    """Fsync a directory to ensure durability."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_torch_save(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    os.makedirs(path.parent, exist_ok=True)

    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
        _fsync_file(path)
        _fsync_dir(path.parent)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def write_tracker(base: str | Path, step: int) -> None:
    """Write the latest checkpoint step to tracker file (rank 0 only)."""
    if _rank() != 0:
        return

    os.makedirs(base, exist_ok=True)
    out = _tracker_path(base)
    tmp = Path(str(out) + f".tmp.{os.getpid()}")

    with open(tmp, "w") as f:
        f.write(str(int(step)))
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, out)
    _fsync_file(out)
    _fsync_dir(base)


def read_tracker(base: str | Path) -> int:
    """Read the latest checkpoint step from tracker file."""
    try:
        with open(_tracker_path(base)) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return -1


@dataclass
class CheckpointState:
    """State to checkpoint."""

    step: int
    tokens_seen: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None = None
    data_state: dict[str, Any] | None = None
    rng_state: dict[str, Any] | None = None
    config: dict[str, Any] | None = None


def _materialize_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    """Recursively move all tensors to CPU."""

    def _copy(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.cpu() if obj.is_cuda else obj
        if isinstance(obj, dict):
            return {k: _copy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_copy(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_copy(v) for v in obj)
        return obj

    return _copy(state)


def _coerce_rng_state(state: Any) -> torch.Tensor | None:
    """Best-effort conversion for legacy or serialized RNG states."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.to(dtype=torch.uint8, device="cpu")
    if isinstance(state, bytes | bytearray):
        return torch.tensor(list(state), dtype=torch.uint8)
    if isinstance(state, list | tuple):
        try:
            return torch.tensor(state, dtype=torch.uint8)
        except (TypeError, ValueError):
            return None
    return None


def _coerce_cuda_rng_state(state: Any) -> list[torch.Tensor] | None:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        tensor = _coerce_rng_state(state)
        return [tensor] if tensor is not None else None
    if isinstance(state, list | tuple):
        out: list[torch.Tensor] = []
        for item in state:
            tensor = _coerce_rng_state(item)
            if tensor is None:
                return None
            out.append(tensor)
        return out
    return None


def _optimizer_state_dict(optimizer: Optimizer | list[Optimizer]) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(optimizer, list):
        return [item.state_dict() for item in optimizer]
    return optimizer.state_dict()


def _load_optimizer_state_dict(optimizer: Optimizer | list[Optimizer], state: Any) -> None:
    if isinstance(optimizer, list):
        if not isinstance(state, list):
            raise TypeError("Expected a list of optimizer states for optimizer list checkpoint.")
        if len(optimizer) != len(state):
            raise ValueError(f"Optimizer state count mismatch: expected {len(optimizer)}, got {len(state)}.")
        for item, item_state in zip(optimizer, state, strict=True):
            item.load_state_dict(item_state)
        return
    optimizer.load_state_dict(state)


class _AsyncSaver(threading.Thread):
    """Background thread for async checkpoint saving."""

    def __init__(self, on_saved: Any = None) -> None:
        super().__init__(daemon=True)
        self.queue: queue.Queue[tuple[str, dict] | None] = queue.Queue(maxsize=2)
        self._on_saved = on_saved
        self._error: Exception | None = None
        self.start()

    def run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return

            path, state = item
            try:
                self._atomic_save(path, state)
                if self._on_saved:
                    self._on_saved(path)
            except Exception as e:
                self._error = e
            finally:
                self.queue.task_done()

    def submit(self, path: str, state: dict[str, Any]) -> None:
        if self._error is not None:
            raise RuntimeError(f"Async saver failed: {self._error}") from self._error

        cpu_state = _materialize_to_cpu(state)
        self.queue.put((path, cpu_state), block=True, timeout=120.0)

    def _atomic_save(self, path: str, state: dict[str, Any]) -> None:
        _atomic_torch_save(path, state)

    def wait(self) -> None:
        self.queue.join()
        if self._error is not None:
            raise RuntimeError(f"Async saver failed: {self._error}") from self._error

    def close(self) -> None:
        self.queue.put(None)
        self.join(timeout=60.0)


class _Purger(threading.Thread):
    """Background thread for deleting old checkpoints."""

    def __init__(self, base: str, keep_last: int) -> None:
        super().__init__(daemon=True)
        self.base = Path(base)
        self.keep_last = keep_last
        self._queue: queue.Queue[None] = queue.Queue()
        self.start()

    def run(self) -> None:
        while True:
            self._queue.get()
            self._purge_once()
            self._queue.task_done()

    def trigger(self) -> None:
        if self._queue.empty():
            self._queue.put(None)

    def _purge_once(self) -> None:
        if self.keep_last <= 0 or not self.base.exists():
            return

        # Find all step directories
        step_dirs = sorted(
            [p for p in self.base.iterdir() if p.is_dir() and p.name.startswith("step_")],
            key=lambda p: int(p.name.split("_")[1]),
        )

        if len(step_dirs) <= self.keep_last:
            return

        # Delete old checkpoints
        for p in step_dirs[: -self.keep_last]:
            try:
                for f in p.glob("*"):
                    f.unlink()
                p.rmdir()
            except Exception:
                pass  # Best effort


class Checkpointer:
    """Checkpoint manager with async IO and keep-last rotation.

    Set overwrite_latest=True to keep writing the newest checkpoint into
    <base>/latest while preserving the true step in the tracker and state.

    Usage:
        ckpt = Checkpointer("checkpoints/", keep_last=3, async_io=True)

        # Save
        ckpt.save(step, model, optimizer, ...)

        # Load
        step, state = ckpt.find_latest()
        if state:
            model.load_state_dict(state["model"])
    """

    def __init__(
        self,
        base: str,
        keep_last: int = 3,
        async_io: bool = True,
        overwrite_latest: bool = False,
    ) -> None:
        self.base = Path(base).absolute()
        self.keep_last = keep_last
        self.async_io = async_io
        self.overwrite_latest = overwrite_latest

        self._purger = _Purger(str(self.base), keep_last) if keep_last > 0 else None
        self._saver = _AsyncSaver(on_saved=self._on_saved) if async_io else None
        self._last_step = -1

    def _checkpoint_dir(self, step: int) -> Path:
        if self.overwrite_latest:
            return _latest_dir(self.base)
        return _iteration_dir(self.base, step)

    def _on_saved(self, path: str) -> None:
        if self._purger:
            self._purger.trigger()

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optimizer | list[Optimizer],
        tokens_seen: int = 0,
        scheduler: Any = None,
        data_state: dict | None = None,
        config: dict | None = None,
    ) -> str:
        """Save checkpoint for this rank."""
        self._last_step = step
        it_dir = self._checkpoint_dir(step)
        os.makedirs(it_dir, exist_ok=True)

        # Build state
        state = {
            "step": step,
            "tokens_seen": tokens_seen,
            "model": model.state_dict(),
            "optimizer": _optimizer_state_dict(optimizer),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if data_state is not None:
            state["data"] = data_state
        if config is not None:
            state["config"] = config

        # For distributed: each rank saves its own shard
        if _is_dist():
            path = str(it_dir / f"rank_{_rank():03d}.pt")
        else:
            path = str(it_dir / "checkpoint.pt")

        if self._saver is not None:
            self._saver.submit(path, state)
        else:
            cpu_state = _materialize_to_cpu(state)
            _atomic_torch_save(path, cpu_state)

        # Rank 0 writes tracker and manifest
        if _rank() == 0:
            manifest = {
                "step": step,
                "tokens_seen": tokens_seen,
                "world_size": _world_size(),
                "timestamp": time.time(),
            }
            manifest_path = it_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            write_tracker(self.base, step)

            if self._purger:
                self._purger.trigger()

        return path

    def find_latest(self) -> tuple[int, str | None]:
        """Find the latest checkpoint for this rank."""
        step = read_tracker(self.base)
        if step < 0:
            return -1, None

        it_dir = self._checkpoint_dir(step)
        if self.overwrite_latest and not it_dir.exists():
            it_dir = _iteration_dir(self.base, step)
        if not it_dir.exists():
            return -1, None

        if _is_dist():
            path = it_dir / f"rank_{_rank():03d}.pt"
        else:
            path = it_dir / "checkpoint.pt"

        if path.exists():
            return step, str(path)

        return -1, None

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer | list[Optimizer],
        scheduler: Any = None,
    ) -> tuple[int, int]:
        """Load latest checkpoint. Returns (step, tokens_seen)."""
        step, path = self.find_latest()
        if path is None:
            return 0, 0

        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(path, map_location=device, weights_only=False)

        model.load_state_dict(state["model"])
        _load_optimizer_state_dict(optimizer, state["optimizer"])

        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])

        # Restore RNG state
        if "rng" in state:
            torch_state = _coerce_rng_state(state["rng"].get("torch"))
            if torch_state is not None:
                torch.set_rng_state(torch_state)
            cuda_state = _coerce_cuda_rng_state(state["rng"].get("cuda"))
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)

        return state.get("step", 0), state.get("tokens_seen", 0)

    def wait(self) -> None:
        """Wait for async saves to complete."""
        if self._saver:
            self._saver.wait()

    def close(self) -> None:
        """Clean shutdown."""
        if self._saver:
            self._saver.wait()
            self._saver.close()
