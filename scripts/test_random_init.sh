set -euo pipefail

# for num_experts in 32 64 128; do
#     for seed in 0 1 2 3 4; do
#         uv run python -m nanomoe.experiments.check_random_init --num-experts "$num_experts" --seed "$seed"
#     done
# done

# for num_experts in 32 64 128; do
#     for seed in 0 1 2 3 4; do
#         uv run python -m nanomoe.experiments.check_random_init --num-experts "$num_experts" --seed "$seed" --use-depth-scaling
#     done
# done


for attention_kv_head_ratio in 2 4; do
    for seed in 0 1 2 3 4; do
        uv run python -m nanomoe.experiments.check_random_init --num-experts 128 --seed "$seed" --attention-kv-head-ratio "$attention_kv_head_ratio"
    done
done

for attention_kv_head_ratio in 2 4; do
    for seed in 0 1 2 3 4; do
        uv run python -m nanomoe.experiments.check_random_init --num-experts 128 --seed "$seed" --attention-kv-head-ratio "$attention_kv_head_ratio" --use-depth-scaling
    done
done
