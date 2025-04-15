import argparse
import logging
import os
import sys
import time
from typing import Tuple

from transformers import Qwen2Config

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.amp import auto_mixed_precision
from mindspore.communication import get_group_size, get_rank, init
from mindspore.dataset import GeneratorDataset

# TODO: remove in future when mindway is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)


from dataset import TextQADataset
from optimizer import Muon

from mindway.transformers import Qwen2ForCausalLM
from mindway.utils.config import str2bool
from mindway.utils.logger import set_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5 Training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--model_name", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name.")
    parser.add_argument("-o", "--output_path", default="./output", help="Output directory to save the training result.")
    parser.add_argument("-d", "--dataset_name", default="pubmed_qa", help="Dataset Name.")

    parser.add_argument("--seed", default=42, type=int, help="Training seed.")
    parser.add_argument("--mode", default=1, choices=[0, 1], help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1).")
    parser.add_argument("--jit_level", default="O1", choices=["O0", "O1"], help="Jit Level")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel training.")
    parser.add_argument("--max_token_length", default=1024, type=int, help="Maximum token length.")
    parser.add_argument(
        "--load_weight", default=True, type=str2bool, help="Load pretrained weight or random initialization."
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp32", "bf16"],
        help="If it is not fp32, then train with auto mixed precision.",
    )
    parser.add_argument("--optim", default="adamw", type=str, choices=["adamw", "muon"], help="Optimizer name.")
    parser.add_argument("--lr", default=1e-4, type=float, help="The learning rate.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of total training epochs.")
    parser.add_argument("--batch_size", default=8, type=int, help="Training batch size.")
    parser.add_argument("--ckpt_max_keep", default=3, type=int, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--clip_grad", default=True, type=str2bool, help="Clip gradient.")
    parser.add_argument("--clip_grad_value", default=1.0, type=float, help="Clip gradient value.")
    args = parser.parse_args()
    return args


def init_env(args) -> Tuple[int, int]:
    ms.set_seed(args.seed)
    ms.set_context(mode=args.mode, jit_config=dict(jit_level=args.jit_level))
    if args.use_parallel:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=device_num
        )
    else:
        device_num, rank_id = 1, 0

    return device_num, rank_id


def main(args):
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # init env
    device_num, rank_id = init_env(args)
    set_logger(output_dir=os.path.join(args.output_path, "logs"), rank=rank_id)

    # prepare model
    logger.info("Creating model `%s`", args.model_name)
    config = Qwen2Config.from_pretrained(args.model_name)
    if args.load_weight:
        with nn.no_init_parameters():
            model = Qwen2ForCausalLM.from_pretrained(
                args.model_name, config=config, attn_implementation="flash_attention_2"
            )
    else:
        logger.info("Initialize network randomly.")
        model = Qwen2ForCausalLM(config, attn_implementation="flash_attention_2")
    model.set_train()

    if args.dtype != "fp32":
        logger.info("Using AMP with data type %s", args.dtype)
        dtype = ms.bfloat16 if args.dtype == "bf16" else ms.float16
        model = auto_mixed_precision(model, amp_level="auto", dtype=dtype)

    # prepare dataset
    logger.info("Creating dataset `%s`", args.dataset_name)
    dataset = TextQADataset(
        dataset_name=args.dataset_name, max_token_length=args.max_token_length, tokenizer_name=args.model_name
    )
    data_generator = GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        column_types=[ms.int64, ms.int64, ms.bool_],
        shuffle=True,
        num_parallel_workers=4,
        num_shards=device_num,
        shard_id=rank_id,
    )
    data_generator = data_generator.batch(args.batch_size, drop_remainder=True, num_parallel_workers=2)

    if args.optim == "adamw":
        optimizer = mint.optim.AdamW(model.trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Muon(model.trainable_params(), lr=args.lr, weight_decay=args.weight_decay)

    def forward_fn(*args, **kwargs):
        return model(*args, **kwargs).loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    reduce_func = nn.DistributedGradReducer(optimizer.parameters) if args.use_parallel else lambda x: x

    ds_iter = data_generator.create_dict_iterator(num_epochs=-1)
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    logger.info("Start training...")
    for epoch in range(1, args.epochs + 1):
        for step, data in enumerate(ds_iter):
            start_time_s = time.time()
            loss, grads = grad_fn(**data)
            grads = reduce_func(grads)
            if args.clip_grad:
                grads = ops.clip_by_global_norm(grads, clip_norm=args.clip_grad_value)
            optimizer(grads)
            step_time = time.time() - start_time_s
            logger.info(
                "epoch %d, step %d, loss %.8f, step time %.3fms",
                epoch,
                step,
                loss.item(),
                step_time * 1000,
            )
        ckpt_name = os.path.join(ckpt_dir, "last.ckpt")
        # FIXME: somehow the prefix of the parameter names are dropped after value_and_grad
        # here we just manually fix them
        for name, param in model.parameters_and_names():
            param.name = name
        ms.save_checkpoint(model.trainable_params(), ckpt_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)
