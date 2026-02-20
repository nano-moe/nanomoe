"""Small-scale pretraining script with sequence packing and document masking.

Usage:
    uv run python -m nanomoe.experiments.pretrain --max_steps=2

Features:
- Streams from HuggingFace datasets
- Packs sequences to 8k tokens with document masking (cu_seqlens)
- Uses custom MoE model or HuggingFace model
- Supports gradient accumulation, checkpointing, wandb logging
"""

from typing import Any

import chz
import datasets
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer

from nanomoe.data import PackedBatch, PackedPretrainDataset, create_document_mask
from nanomoe.model import MoEConfig, create_model
from nanomoe.train import (
    Checkpointer,
    CudaProfilerConfig,
    PrefetchConfig,
    TrainLoopConfig,
    TrainState,
    WSDConfig,
    WSDScheduler,
    cleanup_distributed,
    init_distributed,
    setup_logging,
    train_loop,
    unified_loss,
)


@chz.chz
class TrainConfig:
    # Model
    model_preset: str = "small"  # "tiny", "small", "medium", "large", or "custom"
    num_experts: int | None = None  # Override num_experts if set
    num_experts_per_tok: int | None = None  # Override if set
    attention_type: str | None = None  # Override attention backend if set
    moe_kernel: str | None = None  # Override MoE kernel if set

    # Data
    dataset_name: str = "nvidia/Nemotron-CC-Math-v1"
    dataset_config: str = "4plus"
    tokenizer_name: str = "Qwen/Qwen3-0.6B"
    pack_size: int = 8192
    max_seq_len: int | None = None
    text_key: str = "text"
    min_doc_len: int = 64
    prefetch_batches: int = 4
    shuffle_buffer: int = 10_000
    add_special_tokens: bool = False
    max_examples: int | None = None  # Limit streaming dataset (useful for overfit)

    # Training
    batch_size: int = 1  # Micro batch size (packed sequences)
    gradient_accumulation: int = 8
    max_steps: int = 1000
    max_tokens: int | None = None  # Stop after this many tokens (overrides max_steps)

    # Optimizer
    peak_lr: float = 1e-4
    floor_lr: float = 1e-6
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Logging & checkpointing
    log_every: int = 10
    checkpoint_every: int = 500
    checkpoint_dir: str = "checkpoints/pretrain"
    wandb_project: str | None = "nanomoe"
    wandb_name: str | None = None

    # System
    seed: int = 42
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    compile_model: bool = False  # Use torch.compile
    distributed: bool = False  # Use torch.distributed + DDP

    # Device prefetch
    device_prefetch: bool = True
    prefetch_pin_memory: bool = True
    prefetch_non_blocking: bool = True

    # Profiling (Nsight Systems via torch.cuda.profiler)
    profile_cuda: bool = False
    profile_start_step: int = 1
    profile_steps: int = 5


def compute_loss(
    model: Any,  # Can be MoETransformer or torch.compile'd version
    batch: PackedBatch,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    """Compute cross-entropy loss with document masking."""
    # Move batch to device if not already prefetched.
    if batch.tokens.device != device:
        batch = batch.to(device, non_blocking=True)

    # Create document-aware attention mask
    attention_mask = create_document_mask(batch.cu_seqlens, dtype=dtype)

    # Forward pass
    # Reshape for batch dim
    input_ids = batch.tokens.unsqueeze(0)  # [1, seq_len]
    position_ids = batch.position_ids.unsqueeze(0)  # [1, seq_len]

    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits[0]  # [seq_len, vocab]
    aux_loss = outputs.aux_loss

    # Shift for next-token prediction
    if batch.labels is None:
        raise ValueError("PackedBatch.labels is required for pretraining")
    token_loss = unified_loss(logits, batch.labels, batch.token_weights)

    # Add auxiliary loss
    total_loss = token_loss + aux_loss

    num_tokens = batch.token_weights[:-1].abs().sum().item()
    metrics = {
        "loss": token_loss.item(),
        "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
        "num_tokens": num_tokens,
        "num_docs": len(batch.cu_seqlens) - 1,
    }

    return total_loss, metrics


def main(cfg: TrainConfig) -> None:
    distributed_enabled = cfg.distributed
    if distributed_enabled:
        init_distributed()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_primary = rank == 0

    def log(msg: str) -> None:
        if is_primary:
            print(msg)

    # Set seed
    torch.manual_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + rank)

    # Device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[cfg.dtype]

    log(f"Device: {device}, dtype: {dtype}")
    if cfg.profile_cuda and device.type != "cuda":
        log("Warning: CUDA profiling requested but CUDA is unavailable; profiling disabled.")

    # Create model config
    model_config = getattr(MoEConfig, cfg.model_preset)()
    if cfg.num_experts is not None:
        model_config.num_experts = cfg.num_experts
    if cfg.num_experts_per_tok is not None:
        model_config.num_experts_per_tok = cfg.num_experts_per_tok
    if cfg.attention_type is not None:
        model_config.attention_type = cfg.attention_type
    if cfg.moe_kernel is not None:
        model_config.moe_kernel = cfg.moe_kernel

    log(f"Model config: {cfg.model_preset}")
    log(f"  num_experts: {model_config.num_experts}")
    log(f"  num_experts_per_tok: {model_config.num_experts_per_tok}")
    log(f"  attention_type: {model_config.attention_type}")
    log(f"  moe_kernel: {model_config.moe_kernel}")
    log(f"  hidden_size: {model_config.hidden_size}")
    log(f"  num_layers: {model_config.num_layers}")
    log(f"  Total params: ~{model_config.num_total_params / 1e6:.1f}M")
    log(f"  Active params: ~{model_config.num_active_params / 1e6:.1f}M")

    # Load tokenizer and update vocab size
    log(f"Loading tokenizer: {cfg.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, trust_remote_code=True)
    model_config.vocab_size = len(tokenizer)

    # Create model
    log("Creating model...")
    model = create_model(model_config)
    model = model.to(device=device, dtype=dtype)
    model.train()

    # Keep reference to original model for checkpointing
    model_for_checkpoint = model

    if cfg.compile_model and torch.cuda.is_available():
        log("Compiling model with torch.compile...")
        model.compile()

    log(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR scheduler
    lr_config = WSDConfig(
        peak_lr=cfg.peak_lr,
        floor_lr=cfg.floor_lr,
        warmup_steps=cfg.warmup_steps,
        sustain_tokens=0,
        decay_tokens=cfg.max_tokens or cfg.max_steps * cfg.pack_size * cfg.batch_size * cfg.gradient_accumulation,
    )
    scheduler = WSDScheduler(optimizer, lr_config)

    # Checkpointing
    checkpointer = Checkpointer(cfg.checkpoint_dir, keep_last=3)

    # Logging
    logger = setup_logging(
        log_dir=cfg.checkpoint_dir,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        console=True,
        console_every=cfg.log_every,
        config=cfg,
        rank=rank,
    )

    # Load dataset
    log(f"Loading dataset: {cfg.dataset_name}/{cfg.dataset_config}")
    hf_dataset = datasets.load_dataset(cfg.dataset_name, cfg.dataset_config, streaming=True, split="train")
    if dist.is_initialized():
        if hasattr(hf_dataset, "shard"):
            hf_dataset = hf_dataset.shard(num_shards=world_size, index=rank)
        else:
            raise ValueError("Dataset does not support sharding for distributed training.")
    if cfg.max_examples is not None:
        if hasattr(hf_dataset, "take"):
            hf_dataset = hf_dataset.take(cfg.max_examples)
        else:
            raise ValueError("Dataset does not support take(); remove max_examples or use a supported dataset")

    # Create packed dataset with prefetching
    dataset = PackedPretrainDataset(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        pack_size=cfg.pack_size,
        max_seq_len=cfg.max_seq_len,
        text_key=cfg.text_key,
        min_doc_len=cfg.min_doc_len,
        prefetch_batches=cfg.prefetch_batches,
        shuffle_buffer=cfg.shuffle_buffer,
        seed=cfg.seed,
        add_special_tokens=cfg.add_special_tokens,
    )

    # Resume from checkpoint if exists
    resume_step, resume_tokens = checkpointer.load(model_for_checkpoint, optimizer, scheduler)
    state = TrainState(step=resume_step, tokens_seen=resume_tokens)
    if resume_step > 0:
        log(f"Resumed from step {state.step}, tokens_seen {state.tokens_seen}")

    # Gradient scaler for mixed precision
    scaler = GradScaler() if cfg.dtype == "float16" else None

    log(f"Starting training for {cfg.max_steps} steps...")
    log(f"Pack size: {cfg.pack_size}, batch size: {cfg.batch_size}, grad accum: {cfg.gradient_accumulation}")

    loop_cfg = TrainLoopConfig(
        max_steps=cfg.max_steps,
        max_tokens=cfg.max_tokens,
        gradient_accumulation=cfg.gradient_accumulation,
        log_every=cfg.log_every,
        checkpoint_every=cfg.checkpoint_every,
        max_grad_norm=cfg.max_grad_norm,
    )

    def data_iter_factory():
        return iter(dataset)

    prefetch_config = PrefetchConfig(
        enabled=cfg.device_prefetch,
        pin_memory=cfg.prefetch_pin_memory,
        non_blocking=cfg.prefetch_non_blocking,
    )
    profiler_config = CudaProfilerConfig(
        enabled=cfg.profile_cuda,
        start_step=cfg.profile_start_step,
        profile_steps=cfg.profile_steps,
    )

    try:
        data_iter = data_iter_factory()

        state = train_loop(
            model=model,
            data_iter=data_iter,
            data_iter_factory=data_iter_factory,
            step_fn=lambda b: compute_loss(model, b, device, dtype),
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            checkpointer=checkpointer,
            cfg=loop_cfg,
            state=state,
            grad_scaler=scaler,
            autocast_dtype=None if cfg.dtype == "float32" else dtype,
            prefetch_config=prefetch_config,
            profile_config=profiler_config,
        )
    except KeyboardInterrupt:
        log("\nTraining interrupted by user")
    finally:
        dataset.stop()

    checkpointer.save(
        step=state.step,
        model=model_for_checkpoint,
        optimizer=optimizer,
        tokens_seen=state.tokens_seen,
        scheduler=scheduler,
    )
    checkpointer.wait()

    logger.close()
    log(f"Training complete. Final step: {state.step}, tokens_seen: {state.tokens_seen}")

    if distributed_enabled:
        cleanup_distributed()


if __name__ == "__main__":
    config = chz.entrypoint(TrainConfig, allow_hyphens=True)
    main(config)
