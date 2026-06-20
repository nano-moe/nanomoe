#!/usr/bin/env bash
set -euo pipefail

# for lr in 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3; do
# for lr in 3e-4 1e-3 3e-3; do
# for lr in 1e-2 3e-2; do
#   for optimizer in adamw muon; do
#     lbatch -g 1 -t 8 \
#         -q 'h100,a100,nvl'  --name "moe_${optimizer}_lr${lr}" \
#         --memory 100 -c 16  --cmd \
#         uv run python -m nanomoe.train.simple_pretrain \
#         --optimizer ${optimizer} \
#             --learning-rate ${lr} \
#             --weight-decay 0.01 \
#             --iterations 2000 \
#             --grad-accum 16 \
#             --warmup-steps 100 \
#             --hidden-metrics-every 100 \
#             --log-dir /home/xwang457/work/nanomoe/pretrain_log \
#             --use-depth-scaling
#   done
# done


# for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
#   for optimizer in adamw muon; do
#     lbatch -g 1 -t 8 \
#         -q 'h100,a100,nvl'  --name "moe_no_depth_scaling_${optimizer}_lr${lr}" \
#         --memory 100 -c 16  --cmd \
#         uv run python -m nanomoe.train.simple_pretrain \
#         --optimizer ${optimizer} \
#             --learning-rate ${lr} \
#             --weight-decay 0.01 \
#             --iterations 2000 \
#             --grad-accum 16 \
#             --warmup-steps 100 \
#             --hidden-metrics-every 100 \
#             --log-dir /home/xwang457/work/nanomoe/pretrain_log
#   done
# done

for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
  for optimizer in adamw muon; do
    lbatch -g 1 -t 24 \
        -q 'a100,h100,nvl'  --name "moe_no_depth_scaling_${optimizer}_lr${lr}" \
        --memory 100 -c 16 -x c010  --cmd \
        uv run python -m nanomoe.train.simple_pretrain \
        --optimizer ${optimizer} \
            --learning-rate ${lr} \
            --weight-decay 0.01 \
            --iterations 2000 \
            --grad-accum 48 \
            --warmup-steps 100 \
            --hidden-metrics-every 100 \
            --dataset finewebedu --seed 42 \
            --log-dir /home/xwang457/work/nanomoe/pretrain_log_with_ckpt
  done
done

for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
  for optimizer in adamw muon; do
    lbatch -g 1 -t 24 \
        -q 'a100,h100,nvl'  --name "moe_${optimizer}_lr${lr}" \
        --memory 100 -c 16 -x c010  --cmd \
        uv run python -m nanomoe.train.simple_pretrain \
        --optimizer ${optimizer} \
            --learning-rate ${lr} \
            --weight-decay 0.01 \
            --iterations 2000 \
            --grad-accum 48 \
            --warmup-steps 100 \
            --hidden-metrics-every 100 \
            --use-depth-scaling \
            --dataset finewebedu --seed 42 \
            --log-dir /home/xwang457/work/nanomoe/pretrain_log_with_ckpt
  done
done

