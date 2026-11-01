#!/usr/bin/env python3
"""Offline capacity estimate from the accompanying checkpoint metadata; not a GPU benchmark."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-gib', type=float, default=48)
    p.add_argument('--utilization', type=float, default=0.85)
    p.add_argument('--overhead-gib', type=float, default=7)
    p.add_argument('--tp', type=int, choices=[1, 2], default=1)
    p.add_argument('--state-copies', type=int, default=1,
                   help='Live recurrent state plus retained checkpoint copies per sequence')
    args = p.parse_args()
    if args.gpu_gib <= 0 or not 0 < args.utilization <= 1 or args.overhead_gib < 0 or args.state_copies < 1:
        p.error('Invalid capacity, utilization, overhead, or state copy count')
    cfg = json.loads((ROOT / 'evidence/config.json').read_text())['text_config']
    weights = json.loads((ROOT / 'evidence/weight_summary.json').read_text())
    total = sum(weights['bytes'].values())
    index = json.loads((ROOT / 'evidence/model.safetensors.index.json').read_text())
    assert total == index['metadata']['total_size']
    assert total == sum(weights['parameters'].values()) * 2
    tp = args.tp
    # Conservative TP assumption: vision tower replicated, text parameters sharded.
    vision = weights['bytes']['vision']
    weight = ((total - vision) / tp + vision) / 2**30
    full = cfg['layer_types'].count('full_attention')
    linear = cfg['layer_types'].count('linear_attention')
    kv_per_token = 2 * full * cfg['num_key_value_heads'] * cfg['head_dim'] * 2 / tp
    conv_dim = 2 * cfg['linear_num_key_heads'] * cfg['linear_key_head_dim'] + cfg['linear_num_value_heads'] * cfg['linear_value_head_dim']
    recurrent = cfg['linear_num_value_heads'] * cfg['linear_key_head_dim'] * cfg['linear_value_head_dim'] * 4
    conv = conv_dim * (cfg['linear_conv_kernel_dim'] - 1) * 2
    state = linear * (recurrent + conv) / tp / 2**30 * args.state_copies
    budget = args.gpu_gib * args.utilization
    print('Analytical estimate only; assumes BF16 KV, FP32 recurrent state, no prefix sharing.')
    print(f'TP={tp}, per GPU: weights={weight:.6f} GiB, budget={budget:.3f} GiB')
    print(f'KV={kv_per_token:g} bytes/token, state={state:.6f} GiB/sequence')
    print(f'Assumed runtime overhead={args.overhead_gib:.3f} GiB; TP communication not derived.')
    print('| Total tokens/request | Cache GiB/request/GPU | B=1 | B=4 | B=8 | B=16 | Algebraic capacity |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for length in [8192, 16384, 32768, 65536, 131072, 262144]:
        cache = length * kv_per_token / 2**30 + state
        totals = [weight + args.overhead_gib + b * cache for b in [1, 4, 8, 16]]
        capacity = max(0, int((budget - weight - args.overhead_gib) // cache))
        print(f'| {length} | {cache:.3f} | ' + ' | '.join(f'{v:.3f}' for v in totals) + f' | {capacity} |')
    print('Capacity is not a certified max-num-seqs; profile images, blocks, state pools and prefill first.')


if __name__ == '__main__':
    main()
