from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import chz
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from nanomoe.lra import (
    TASK_SPECS,
    TransformerClassifierConfig,
    build_transformer_classifier,
    classification_collate_fn,
    load_lra_datasets,
)
from nanomoe.train import Checkpointer, CosineScheduler, cleanup_distributed, init_distributed, setup_logging


@dataclass(slots=True)
class EvalMetrics:
    loss: float
    accuracy: float


@chz.chz
class TrainConfig:
    task: str = "listops"
    data_root: str | None = None
    max_length: int | None = None
    imdb_val_split: float = 0.0
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    seed: int = 42

    batch_size: int = 32
    eval_batch_size: int = 64
    max_steps: int = 10_000
    eval_every: int = 500
    log_every: int = 50
    num_workers: int = 4

    d_model: int = 128
    num_layers: int = 8
    num_heads: int = 8
    ffn_hidden_size: int = 512
    dropout: float = 0.1
    pooling: str = "last"
    attention_backend: str = "sdpa"
    hull_top_k: int = 8

    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0

    dtype: str = "bfloat16"
    compile_model: bool = False
    distributed: bool = False

    log_dir: str = "checkpoints/lra"
    checkpoint_every: int = 1000
    wandb_project: str | None = "nanomoe-lra"
    wandb_name: str | None = None
    wandb_mode: str = "online"


def _config_to_dict(config: object) -> dict[str, object]:
    return dict(getattr(config, "__dict__", {}))


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype_map[dtype_name]


def _build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(batch.inputs, batch.attention_mask)
            loss = F.cross_entropy(logits, batch.labels)
        total_loss += float(loss.item()) * batch.labels.shape[0]
        total_correct += int((logits.argmax(dim=-1) == batch.labels).sum().item())
        total_examples += int(batch.labels.shape[0])

    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor(
            [total_loss, float(total_correct), float(total_examples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss = float(totals[0].item())
        total_correct = int(totals[1].item())
        total_examples = int(totals[2].item())

    model.train()
    return EvalMetrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
    )


def main(cfg: TrainConfig) -> None:
    distributed_enabled = cfg.distributed
    if distributed_enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed=True requires CUDA for single-node DDP")
        init_distributed()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if cfg.task not in TASK_SPECS:
        raise ValueError(f"Unsupported task: {cfg.task}. Available tasks: {sorted(TASK_SPECS)}")

    torch.manual_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    amp_dtype = None if cfg.dtype == "float32" else _resolve_dtype(cfg.dtype)
    use_grad_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = GradScaler(device="cuda", enabled=use_grad_scaler)

    datasets = load_lra_datasets(
        cfg.task,
        data_root=cfg.data_root,
        max_length=cfg.max_length,
        imdb_val_split=cfg.imdb_val_split,
        max_train_examples=cfg.max_train_examples,
        max_eval_examples=cfg.max_eval_examples,
        seed=cfg.seed,
    )
    max_length = cfg.max_length or datasets.spec.default_max_length
    model_config = TransformerClassifierConfig(
        vocab_size=len(datasets.vocab) if datasets.vocab is not None else None,
        num_classes=datasets.spec.num_classes,
        max_seq_len=max_length,
        input_mode=datasets.spec.input_mode,
        input_dim=datasets.spec.input_dim,
        pad_token_id=datasets.vocab.pad_id if datasets.vocab is not None else 0,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ffn_hidden_size=cfg.ffn_hidden_size,
        dropout=cfg.dropout,
        attention_backend=cfg.attention_backend,
        hull_top_k=cfg.hull_top_k,
        pooling=cfg.pooling,
        use_cls_token=cfg.pooling == "cls",
    )

    model = build_transformer_classifier(model_config).to(device)
    if amp_dtype is not None:
        model = model.to(dtype=amp_dtype)
    model_for_checkpoint = model
    if cfg.compile_model:
        model = torch.compile(model)
    if dist.is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
        )

    train_sampler = None
    if dist.is_initialized():
        train_sampler = DistributedSampler(
            datasets.train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.seed,
        )

    train_loader = DataLoader(
        datasets.train,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=classification_collate_fn(datasets.pad_value),
    )
    eval_loader = DataLoader(
        datasets.val,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=classification_collate_fn(datasets.pad_value),
    )
    test_loader = DataLoader(
        datasets.test,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=classification_collate_fn(datasets.pad_value),
    )

    optimizer = _build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineScheduler(
        optimizer,
        peak_lr=cfg.lr,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.max_steps,
    )

    log_dir = Path(cfg.log_dir) / cfg.task
    config_payload = {"train": _config_to_dict(cfg), "model": _config_to_dict(model_config)}
    logger = setup_logging(
        log_dir=log_dir,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        wandb_mode=cfg.wandb_mode,
        config=config_payload,
        console=True,
        console_every=cfg.log_every,
        rank=rank,
    )
    checkpointer = Checkpointer(str(log_dir / "checkpoints"), keep_last=3, async_io=True)

    train_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(train_epoch)
    train_iter = iter(train_loader)
    best_val_accuracy = -math.inf
    best_test_accuracy = -math.inf
    tokens_seen = 0

    try:
        for step in range(cfg.max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(train_epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(batch.inputs, batch.attention_mask)
                loss = F.cross_entropy(logits, batch.labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

            lr = scheduler.step(step=step)
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == batch.labels).float().mean().item()
            grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            batch_tokens = float(batch.num_tokens)
            if dist.is_available() and dist.is_initialized():
                train_totals = torch.tensor(
                    [float(loss.item()), accuracy, grad_norm_value, batch_tokens],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
                world_size_float = float(world_size)
                loss_value = float(train_totals[0].item() / world_size_float)
                accuracy_value = float(train_totals[1].item() / world_size_float)
                grad_norm_value = float(train_totals[2].item() / world_size_float)
                batch_tokens = float(train_totals[3].item())
            else:
                loss_value = float(loss.item())
                accuracy_value = accuracy

            tokens_seen += int(batch_tokens)
            logger.log_metrics(
                {
                    "train/loss": loss_value,
                    "train/accuracy": accuracy_value,
                    "train/lr": float(lr),
                    "train/grad_norm": grad_norm_value,
                    "train/num_tokens": int(batch_tokens),
                },
                step=step + 1,
            )

            if (step + 1) % cfg.eval_every == 0 or step == cfg.max_steps - 1:
                val_metrics = evaluate(model, eval_loader, device=device, amp_dtype=amp_dtype)
                test_metrics = evaluate(model, test_loader, device=device, amp_dtype=amp_dtype)
                logger.log_metrics(
                    {
                        "val/loss": val_metrics.loss,
                        "val/accuracy": val_metrics.accuracy,
                        "test/loss": test_metrics.loss,
                        "test/accuracy": test_metrics.accuracy,
                    },
                    step=step + 1,
                )

                if val_metrics.accuracy >= best_val_accuracy:
                    best_val_accuracy = val_metrics.accuracy
                    best_test_accuracy = test_metrics.accuracy
                    checkpointer.save(
                        step=step + 1,
                        model=model_for_checkpoint,
                        optimizer=optimizer,
                        tokens_seen=tokens_seen,
                        scheduler=scheduler,
                        config=config_payload,
                    )

            if (step + 1) % cfg.checkpoint_every == 0:
                checkpointer.save(
                    step=step + 1,
                    model=model_for_checkpoint,
                    optimizer=optimizer,
                    tokens_seen=tokens_seen,
                    scheduler=scheduler,
                    config=config_payload,
                )

        checkpointer.wait()
        logger.log_metrics(
            {
                "summary/best_val_accuracy": best_val_accuracy,
                "summary/best_test_accuracy": best_test_accuracy,
                "summary/num_parameters": model_for_checkpoint.num_parameters(),
            },
            step=cfg.max_steps,
        )
        logger.close()
        checkpointer.close()
    finally:
        if distributed_enabled:
            cleanup_distributed()


if __name__ == "__main__":
    config = chz.entrypoint(TrainConfig, allow_hyphens=True)
    main(config)
