"""Minimal SFT training script using packed batches."""

from dataclasses import dataclass
from typing import Any

import datasets
import torch
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer

from nanomoe.data import PackedBatch, PackedSFTDataset, SFTDatasetConfig, create_document_mask
from nanomoe.model import MoEConfig, create_model
from nanomoe.train import (
    Checkpointer,
    TrainLoopConfig,
    TrainState,
    WSDConfig,
    WSDScheduler,
    setup_logging,
    train_loop,
    unified_loss,
)


@dataclass
class TrainConfig:
    # Model
    model_preset: str = "small"
    num_experts: int | None = None
    num_experts_per_tok: int | None = None

    # Data
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    input_key: str = "messages"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    packed_seq_len: int = 2048  # Target tokens per packed sequence; max_seq_len controls per-document truncation
    max_seq_len: int | None = None

    # Training
    gradient_accumulation: int = 8
    max_steps: int = 1000
    max_tokens: int | None = None

    # Optimizer
    peak_lr: float = 1e-4
    floor_lr: float = 1e-6
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Logging & checkpointing
    log_every: int = 10
    checkpoint_every: int = 500
    checkpoint_dir: str = "checkpoints/sft"
    wandb_project: str | None = "nanomoe"
    wandb_name: str | None = None

    # System
    seed: int = 42
    dtype: str = "bfloat16"


def compute_sft_loss(
    model: Any,
    batch: PackedBatch,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    batch = batch.to(device)
    attention_mask = create_document_mask(batch.cu_seqlens, dtype=dtype)

    input_ids = batch.tokens.unsqueeze(0)
    position_ids = batch.position_ids.unsqueeze(0)

    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits[0]
    aux_loss = outputs.aux_loss

    if batch.labels is None:
        raise ValueError("PackedBatch.labels is required for SFT")

    token_loss = unified_loss(logits, batch.labels, batch.token_weights)
    total_loss = token_loss + aux_loss

    metrics = {
        "loss": token_loss.item(),
        "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
        "num_tokens": batch.token_weights[:-1].abs().sum().item(),
    }

    return total_loss, metrics


def main() -> None:
    cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[cfg.dtype]

    model_config = getattr(MoEConfig, cfg.model_preset)()
    if cfg.num_experts is not None:
        model_config.num_experts = cfg.num_experts
    if cfg.num_experts_per_tok is not None:
        model_config.num_experts_per_tok = cfg.num_experts_per_tok

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    model_config.vocab_size = len(tokenizer)

    model = create_model(model_config).to(device=device, dtype=dtype)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    lr_config = WSDConfig(
        peak_lr=cfg.peak_lr,
        floor_lr=cfg.floor_lr,
        warmup_steps=cfg.warmup_steps,
        sustain_tokens=0,
        decay_tokens=cfg.max_tokens or cfg.max_steps * cfg.packed_seq_len * cfg.gradient_accumulation,
    )
    scheduler = WSDScheduler(optimizer, lr_config)

    checkpointer = Checkpointer(cfg.checkpoint_dir, keep_last=3)
    logger = setup_logging(
        log_dir=cfg.checkpoint_dir,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        console=True,
        console_every=cfg.log_every,
        config=cfg,
    )

    hf_dataset = datasets.load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)
    sft_config = SFTDatasetConfig(
        packed_seq_len=cfg.packed_seq_len,
        max_seq_len=cfg.max_seq_len,
        input_key=cfg.input_key,
        seed=cfg.seed,
    )
    dataset = PackedSFTDataset(hf_dataset=hf_dataset, tokenizer=tokenizer, config=sft_config)

    resume_step, resume_tokens = checkpointer.load(model, optimizer, scheduler)
    state = TrainState(step=resume_step, tokens_seen=resume_tokens)

    loop_cfg = TrainLoopConfig(
        max_steps=cfg.max_steps,
        max_tokens=cfg.max_tokens,
        gradient_accumulation=cfg.gradient_accumulation,
        log_every=cfg.log_every,
        checkpoint_every=cfg.checkpoint_every,
        max_grad_norm=cfg.max_grad_norm,
    )

    scaler = GradScaler() if cfg.dtype == "float16" else None

    def data_iter_factory():
        return iter(dataset)

    data_iter = data_iter_factory()

    state = train_loop(
        model=model,
        data_iter=data_iter,
        data_iter_factory=data_iter_factory,
        step_fn=lambda b: compute_sft_loss(model, b, device, dtype),
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        checkpointer=checkpointer,
        cfg=loop_cfg,
        state=state,
        grad_scaler=scaler,
        autocast_dtype=None if cfg.dtype == "float32" else dtype,
    )

    checkpointer.save(
        step=state.step,
        model=model,
        optimizer=optimizer,
        tokens_seen=state.tokens_seen,
        scheduler=scheduler,
    )
    checkpointer.wait()
    logger.close()

    print(f"SFT complete. Final step: {state.step}, tokens_seen: {state.tokens_seen}")


if __name__ == "__main__":
    main()
