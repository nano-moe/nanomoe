from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import datasets
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from nanomoe.data.packed_dataset import PackedPretrainStreamGroup, cu_seqlens_to_packing_metadata
from nanomoe.model import MoEConfig, create_model
from nanomoe.optimizer import build_optimizer_adamw, build_optimizer_muon
from nanomoe.train import unified_loss
from nanomoe.train.metric_helper import build_run_dir, capture_hidden_metrics, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple packed pretraining runner for nanomoe.")
    parser.add_argument("--learning-rate", "--lr", type=float, default=3e-5, help="Learning rate for all param groups.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for optimizer weight groups.")
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "muon"),
        default="adamw",
        help="Optimizer to use for the run.",
    )
    parser.add_argument(
        "--iterations",
        "--num-iterations",
        "--steps",
        type=int,
        default=100,
        help="Number of optimizer steps to run.",
    )
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation micro-steps per optimizer step.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Number of LR warmup optimizer steps.")
    parser.add_argument(
        "--hidden-metrics-every",
        type=int,
        default=100,
        help="Capture hidden-state monitor metrics every N optimizer steps.",
    )
    parser.add_argument(
        "--log-dir",
        "--log-path",
        dest="log_dir",
        type=Path,
        default=Path("simple_pretrain_logs"),
        help="Base directory for .npy logs. A hyperparameter-named run directory is created inside it.",
    )
    parser.add_argument(
        "--use-depth-scaling",
        action="store_true",
        help="Enable residual depth scaling with residual_scale = 0.2 / sqrt(num_layers).",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.hidden_metrics_every < 1:
        parser.error("--hidden-metrics-every must be >= 1")
    return args


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    if args.optimizer == "adamw":
        return build_optimizer_adamw(
            model,
            lr_adamw=args.learning_rate,
            lr_layernorm=args.learning_rate,
            wd=args.weight_decay,
        )
    if args.optimizer == "muon":
        return build_optimizer_muon(
            model,
            lr_adamw=args.learning_rate,
            lr_muon=args.learning_rate,
            lr_layernorm=args.learning_rate,
            wd=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def zero_grad(optimizers) -> None:
    if isinstance(optimizers, list):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        return
    optimizers.zero_grad(set_to_none=True)


def step_optimizers(optimizers) -> None:
    if isinstance(optimizers, list):
        for optimizer in optimizers:
            optimizer.step()
        return
    optimizers.step()


def set_learning_rate(optimizers, lr: float) -> None:
    optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]
    for optimizer in optimizer_list:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def linear_warmup_decay_lr(step: int, total_steps: int, warmup_steps: int, peak_lr: float) -> float:
    if total_steps == 1:
        return peak_lr
    warmup_steps = min(warmup_steps, total_steps - 1)
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * float(step + 1) / float(warmup_steps)

    decay_steps = max(total_steps - warmup_steps - 1, 1)
    decay_progress = float(step - warmup_steps) / float(decay_steps)
    return peak_lr * max(0.0, 1.0 - decay_progress)


def compute_pretrain_loss(
    model: torch.nn.Module,
    batch,
    *,
    device: torch.device,
    aux_loss_coef: float = 0.0,
    return_router_logits: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    batch = batch.to(device, non_blocking=True)
    input_ids = batch.tokens.unsqueeze(0)
    position_ids = batch.position_ids.unsqueeze(0)
    doc_ids, seq_lens = cu_seqlens_to_packing_metadata(batch.cu_seqlens)

    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        packing_doc_ids=doc_ids,
        packing_seq_lens=seq_lens,
        use_cache=False,
        return_router_logits=return_router_logits,
    )

    if batch.labels is None:
        raise ValueError("Packed pretraining batches must include labels.")

    logits = outputs.logits[0]
    lm_loss = unified_loss(logits, batch.labels, batch.token_weights)
    aux_loss = outputs.aux_loss
    if not torch.is_tensor(aux_loss):
        aux_loss = torch.tensor(aux_loss, device=device, dtype=lm_loss.dtype)

    total_loss = lm_loss + aux_loss * aux_loss_coef
    token_count = int(batch.token_weights[:-1].abs().sum().item())
    metrics = {
        "loss": float(lm_loss.detach()),
        "aux_loss": float(aux_loss.detach()),
        "tokens": float(token_count),
    }

    if return_router_logits:
        router_expert_indices = outputs.router_expert_indices or []
        entropies = []
        for router_expert_index in router_expert_indices:
            flat = router_expert_index.reshape(-1)
            counts = torch.bincount(flat, minlength=model.config.num_experts).float()
            denom = float(flat.numel())
            if denom == 0.0:
                entropies.append(torch.tensor(float("nan"), device=counts.device))
                continue
            expert_load = counts / denom
            expert_entropy = -(expert_load * torch.log2(expert_load + 1e-12)).sum()
            entropies.append(expert_entropy)
        if entropies:
            entropy_t = torch.stack(entropies)
            metrics["expert_entropy"] = float(entropy_t.mean().detach())
            metrics["expert_entropy_by_layer"] = [float(x) for x in entropy_t.detach().cpu().tolist()]
        else:
            metrics["expert_entropy"] = float("nan")
            metrics["expert_entropy_by_layer"] = []

    return total_loss, metrics


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("simple_pretrain.py requires a CUDA GPU.")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset_name = "nvidia/Nemotron-CC-Math-v1"
    dataset_config = "4plus"
    tokenizer_name = "gpt2"
    # seq_len = 65536
    seq_len = 49152
    # seq_len = 32768
    max_seq_len = 2048
    batch_size = 1

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.model_max_length = max_seq_len

    model_config = MoEConfig.nano()
    model_config.vocab_size = len(tokenizer)
    model_config.max_position_embeddings = max_seq_len
    model_config.attention_type = "flex_attention"
    if args.use_depth_scaling:
        model_config.residual_scale = 0.2 / (model_config.num_layers**0.5)

    model = create_model(model_config).to(device=device, dtype=dtype)
    model.train()

    optimizers = build_optimizer(model, args)

    hf_dataset = datasets.load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    val_examples = 100
    train_hf_dataset = hf_dataset.skip(val_examples)
    val_hf_dataset = hf_dataset.take(val_examples)

    dataset = PackedPretrainStreamGroup(
        hf_dataset=train_hf_dataset,
        tokenizer=tokenizer,
        num_streams=batch_size,
        total_shards=batch_size,
        shard_base_index=0,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
        text_key="text",
        min_doc_len=64,
        prefetch_batches=2,
        seed=seed,
        add_special_tokens=False,
    )
    val_dataset = PackedPretrainStreamGroup(
        hf_dataset=val_hf_dataset,
        tokenizer=tokenizer,
        num_streams=batch_size,
        total_shards=batch_size,
        shard_base_index=0,
        seq_len=4096,
        max_seq_len=max_seq_len,
        text_key="text",
        min_doc_len=64,
        prefetch_batches=0,
        seed=seed,
        add_special_tokens=False,
    )

    effective_warmup_steps = min(args.warmup_steps, max(args.iterations - 1, 0))
    run_dir = build_run_dir(args)
    config: dict[str, Any] = {
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "iterations": args.iterations,
        "grad_accum": args.grad_accum,
        "warmup_steps": args.warmup_steps,
        "effective_warmup_steps": effective_warmup_steps,
        "hidden_metrics_every": args.hidden_metrics_every,
        "use_depth_scaling": args.use_depth_scaling,
        "residual_scale": model_config.effective_residual_scale,
        "seq_len": seq_len,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "tokenizer_name": tokenizer_name,
        "run_dir": str(run_dir),
    }
    train_records: list[dict[str, Any]] = []
    hidden_state_records: list[dict[str, Any]] = []

    print(
        "Starting simple pretrain: "
        f"optimizer={args.optimizer}, lr={args.learning_rate}, weight_decay={args.weight_decay}, "
        f"iterations={args.iterations}, grad_accum={args.grad_accum}, warmup_steps={effective_warmup_steps}, "
        f"depth_scaling={args.use_depth_scaling}, "
        f"residual_scale={model_config.effective_residual_scale:.6g}, run_dir={run_dir}"
    )

    data_iter = iter(dataset)
    try:
        print("Capturing hidden-state metrics at step=0")
        hidden_state_records.append(capture_hidden_metrics(model, val_dataset, step=0))

        for step in tqdm(range(args.iterations)):
            current_lr = linear_warmup_decay_lr(
                step,
                total_steps=args.iterations,
                warmup_steps=effective_warmup_steps,
                peak_lr=args.learning_rate,
            )
            set_learning_rate(optimizers, current_lr)
            zero_grad(optimizers)

            step_loss = 0.0
            step_aux_loss = 0.0
            step_tokens = 0
            step_router_monitor = []
            for _ in range(args.grad_accum):
                batch = next(data_iter)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss, metrics = compute_pretrain_loss(model, batch, device=device, return_router_logits=True)
                (loss / args.grad_accum).backward()
                step_loss += metrics["loss"]
                step_aux_loss += metrics["aux_loss"]
                step_tokens += int(metrics["tokens"])
                step_router_monitor += metrics.get("expert_entropy_by_layer", [])

            step_optimizers(optimizers)

            train_record = {
                "step": step + 1,
                "lr": current_lr,
                "loss": step_loss / args.grad_accum,
                "aux_loss": step_aux_loss / args.grad_accum,
                "tokens": step_tokens,
                "router_monitor": step_router_monitor,
            }
            train_records.append(train_record)

            if step == 0 or (step + 1) % 5 == 0:
                print(
                    f"step={step + 1} "
                    f"lr={current_lr:.6g} "
                    f"loss={train_record['loss']:.4f} "
                    f"aux_loss={train_record['aux_loss']:.4f} "
                    f"tokens={step_tokens}"
                )

            if (step + 1) % args.hidden_metrics_every == 0:
                print(f"Capturing hidden-state metrics at step={step + 1}")
                hidden_state_records.append(capture_hidden_metrics(model, val_dataset, step=step + 1))
    finally:
        dataset.stop()
        val_dataset.stop()
        save_metrics(run_dir, config, train_records, hidden_state_records)
        print(f"Saved .npy logs to {run_dir}")


if __name__ == "__main__":
    main()
