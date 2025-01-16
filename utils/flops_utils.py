import torch
import argparse
import importlib
import os
import sys

from thop import profile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_flopsTransformerLayer(d_model, n_head, d_ff, seq_len):
    # Attention FLOPs
    flops_qkv = 3 * d_model * d_model * seq_len  # Q, K, V 投影
    flops_attention_scores = seq_len ** 2 * d_model  # Attention Score (QK^T)
    flops_attention_softmax = 2 * seq_len ** 2  # Softmax
    flops_attention_weighted_sum = seq_len ** 2 * d_model  # Weighted Sum
    flops_attention_output = d_model * d_model * seq_len  # Projection

    flops_attention = (
            n_head
            * (flops_qkv + flops_attention_scores + flops_attention_softmax + flops_attention_weighted_sum)
            + flops_attention_output
    )

    # FFN FLOPs
    flops_ffn = 2 * seq_len * d_model * d_ff + seq_len * d_ff  # 2 FC + Activation

    # Overall FLOPs
    total_flops = flops_attention + flops_ffn
    return total_flops


# note the early exits when measuring the flops
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    model_arg = args.model
    dataset_arg = args.dataset

    if model_arg == 'convnet':
        args.class_num = 10
        dummy_input = torch.randn(1, 3, 32, 32)
    elif model_arg == 'resnet18' and dataset_arg == 'cifar100':
        args.class_num = 100
        dummy_input = torch.randn(1, 3, 32, 32)
    elif model_arg == 'resnet18' and dataset_arg == 'tinyimagenet':
        args.class_num = 200
        dummy_input = torch.randn(1, 3, 64, 64)
    elif model_arg == 'transformer':
        args.class_num = 4
        dummy_input = torch.zeros(1, 200, dtype=torch.int)


    model_module = importlib.import_module(f'models.{model_arg}')
    model = getattr(model_module, model_arg)(args, None)

    flops, params = profile(model, (dummy_input,))


    if model_arg == 'transformer':
        layer_num = 4
        t_flops = calculate_flopsTransformerLayer(192, 4, 768, 200) * layer_num
        flops += t_flops

    print('flops: ', flops)
