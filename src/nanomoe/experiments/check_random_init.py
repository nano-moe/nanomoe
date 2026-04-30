"""Inspect random init behavior for nanomoe models."""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import numpy as np
import torch
from transformers import AutoTokenizer

from nanomoe.data.packed_dataset import PackedPretrainStreamGroup
from nanomoe.model import MoEConfig, create_model
from nanomoe.monitors import hidden_state_cosine_similarities

SAVE_DIR = '/home/xwang457/work/nanomoe/logs'

@torch.no_grad()
def get_model_metrics(model: torch.nn.Module, dataset: PackedPretrainStreamGroup) -> dict[str, list[float] | list[tuple[float, float, float]]]:
    results = hidden_state_cosine_similarities(
        model=model,
        dataset=dataset,
        batch_size=1,
        max_batches=1,
        first_layer_as_reference=True,
    )

    all_avg_cos = [
        record.mean_off_diagonal_cosine
        for record in results.intra_sequence
    ]
    all_neighbour_layer_stats = [
        (record.mean_token_cosine, record.mean_rms_distance, record.mean_relative_rms_change)
        for record in results.neighbouring_layers
    ]
    all_first_layer_stats = [
        (record.mean_token_cosine, record.mean_rms_distance, record.mean_relative_rms_change)
        for record in results.first_layer_reference
    ]
    all_router_stats = [
        record.entropy for record in results.router_usage_entropy
    ]
    return {
        "all_avg_cos": all_avg_cos,
        "all_neighbour_layer_stats": all_neighbour_layer_stats,
        "all_first_layer_stats": all_first_layer_stats,
        "all_router_stats": all_router_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check random initialization statistics.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model + dataset.")
    parser.add_argument("--num-experts", type=int, default=64, help="Number of MoE experts.")
    parser.add_argument(
        "--attention-kv-head-ratio",
        type=int,
        default=1,
        help=(
            "Ratio of num_attention_heads to num_key_value_heads. "
            "For example, 4 with 32 attention heads uses 8 key/value heads."
        ),
    )
    parser.add_argument(
        "--use-depth-scaling",
        action="store_true",
        help="Enable depth scaling (uses constant=0.2, alpha=0.5).",
    )
    args = parser.parse_args()
    if args.attention_kv_head_ratio < 1:
        parser.error("--attention-kv-head-ratio must be >= 1")
    return args


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    num_layer = 32
    # seq_len_list = [2048, 4096, 8192, 16384, 32768]
    seq_len_list = [16384]

    dataset_specs = [
        ("nvidia/Nemotron-CC-Math-v1", "4plus"),
        ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ]

    dataset_split = "train"
    text_key = "text"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    constant, alpha = (0.2, 0.5) if args.use_depth_scaling else (1.0, 0.0)
    residual_scale = constant / (num_layer**alpha)
    num_attention_heads = 32
    if num_attention_heads % args.attention_kv_head_ratio != 0:
        raise ValueError(
            "num_attention_heads must be divisible by --attention-kv-head-ratio: "
            f"{num_attention_heads} % {args.attention_kv_head_ratio} != 0"
        )
    num_key_value_heads = num_attention_heads // args.attention_kv_head_ratio
    config = MoEConfig(
        hidden_size=2048,
        num_layers=num_layer,
        vocab_size=len(tokenizer),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=512,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts // 16,
        depth_alpha=None,
        residual_scale=residual_scale,
        tie_word_embeddings=False,
    )

    model = create_model(config).to(device).eval()
    print(
        " ".join(
            [
                f"device={device},",
                f"tokenizer_vocab={len(tokenizer):,},",
                f"parameters={model.num_parameters() / 1e6:,}M",
            ]
        )
    )
    output_dir = Path(SAVE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_config in dataset_specs:
        hf_dataset = datasets.load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
            streaming=True,
        )
        dataset_id = dataset_name.replace("/", "_")
        config_id = dataset_config.replace("/", "_")
        for seq_len in seq_len_list:
            max_seq_len = seq_len
            dataset = PackedPretrainStreamGroup(
                hf_dataset=hf_dataset,
                tokenizer=tokenizer,
                num_streams=1,
                total_shards=1,
                shard_base_index=0,
                seq_len=seq_len,
                max_seq_len=max_seq_len,
                text_key=text_key,
                min_doc_len=64,
                prefetch_batches=1,
                seed=args.seed,
                add_special_tokens=False,
            )

            print(f"dataset={dataset_name}:{dataset_config} seq_len={seq_len}")
            metrics = get_model_metrics(model, dataset)
            print(f"avg cos per layer (first 5): {metrics['all_avg_cos'][:5]}")

            output_name = (
                f"check_random_init_{dataset_id}_{config_id}_seq{seq_len}"
                f"_experts{args.num_experts}_headratio{args.attention_kv_head_ratio}"
                f"_seed{args.seed}_depth{int(args.use_depth_scaling)}.npz"
            )
            output_path = output_dir / output_name
            np.savez(
                output_path,
                all_avg_cos=np.array(metrics["all_avg_cos"], dtype=np.float64),
                all_neighbour_layer_stats=np.array(metrics["all_neighbour_layer_stats"], dtype=np.float64),
                all_first_layer_stats=np.array(metrics["all_first_layer_stats"], dtype=np.float64),
                all_router_stats=np.array(metrics["all_router_stats"], dtype=np.float64),
                seed=np.array([args.seed], dtype=np.int64),
                use_depth_scaling=np.array([int(args.use_depth_scaling)], dtype=np.int64),
                seq_len=np.array([seq_len], dtype=np.int64),
                num_attention_heads=np.array([num_attention_heads], dtype=np.int64),
                num_key_value_heads=np.array([num_key_value_heads], dtype=np.int64),
                attention_kv_head_ratio=np.array([args.attention_kv_head_ratio], dtype=np.int64),
                dataset_name=np.array([dataset_name]),
                dataset_config=np.array([dataset_config]),
            )
            print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
