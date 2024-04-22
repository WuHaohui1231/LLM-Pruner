import gc
import random
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):

    # print(args)

    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >=1.9 else False
    )

    # print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.dtype)
    if args.device != "cpu":
        model.half()

    # print("TOR", torch.cuda.device_count())
    # class MyDataParallel(nn.DataParallel):
    #     def __getattr__(self, name):
    #         try:
    #             return super().__getattr__(name)
    #         except AttributeError:
    #             return getattr(self.module, name)
    #
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # model = nn.parallel.DistributedDataParallel(model)

    model.to(args.device)

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))

    kwargs = {
        "importance": imp,
        "global_pruning": args.global_pruning,
        "iterative_steps": args.iterative_steps,
        "ch_sparsity": args.pruning_ratio,
        "ignored_layers":[],
        "channel_groups": {
        },
        "consecutive_groups": {
            layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
        },
        "customized_pruners": {
            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
        },
        "root_module_types": None,
        "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                          [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
    }
    logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
    logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

    print("ARGS", args)

    logger.log("Number of example for importance: {}".format(args.num_examples))
    logger.log("Seed: {}".format(args.seed))

    pruner = tp.pruner.MetaPruner(
        model,
        forward_prompts,
        **kwargs
    )
    model.zero_grad()

    print("DEVICE", next(model.parameters()).device)

    logger.log("Start Pruning")
    for i in range(args.iterative_steps):

        if pruner_type in ['taylor']:
            num_batch = int(args.num_batch)
            for _ in range(int(num_batch)):
                example_prompts = get_examples(args.dataset, tokenizer, 32, seq_len = 64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                # print(example_prompts.shape)
                outputs = model(example_prompts, labels=example_prompts)
                outputs.loss = outputs.loss / num_batch
                logger.log("Loss = {}".format(outputs.loss))
                outputs.loss.backward()
        pruner.step()

        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

        # modify inferece-related attributes
        for layer in model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

    # Clean the gradient in the model
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None

    path = 'imp_heat_s' + str(args.seed) + '_onePass_' + args.dataset[:3] + '_' + str(args.num_batch)
    pruner.visualize_importance(path)

    del pruner


    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        model.half()
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)
    
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    # model.config.pad_token_id = tokenizer.pad_token_id = 0
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    # parser.add_argument('dataset', type=str, )
    parser.add_argument('--dataset', help='dataset for estimating importance')
    parser.add_argument('--num_batch', help='number of 64-eg batch for estimating importance')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
