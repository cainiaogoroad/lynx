import argparse
import json
import logging
import math
import os
import random
import time
import numpy as np

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from functools import partial
from itertools import chain

import datasets
import matplotlib.pyplot as plt
import torch
import torch_xla
import transformers
import traceback
from datasets import load_dataset, load_from_disk
from instruction_dataset_utils import InstructionDataset
from matplotlib.ticker import MaxNLocator
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp import StateDictType
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
# , LlamaAttention, LlamaMLP
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.spmd.xla_sharding as xs
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
from torch_xla.distributed.fsdp import checkpoint_module
from torch_xla.amp import autocast as xla_autocast
from torch.amp import autocast as torch_autocast
import torch_xla.distributed.xla_backend
from torch_xla._internal import pjrt
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
from torch_xla.amp import syncfree
import importlib
from lynx.distributed.tensor_parallel import make_tensor_parallel,make_sequence_parallel
from lynx.distributed.expert_parallel import make_expert_parallel,make_gating_network_parallel
global USING_SPMD_FSDP
global SPMD_NUM_DEVICES
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")

logger = logging.getLogger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -U -r requirements.txt",
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
TRAINING_ARGS_NAME = "training_args.bin"

def dynamic_import_cls(import_name):
    """
    Dynamically import a class from a string.
    Args:
        import_name (str): The full import path to the class.
    Returns:
        cls: The imported class.
    """
    module_name, cls_name = import_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls

def _train_update(device, x, loss, rate):
    if is_main_process():
        test_utils.print_training_update(device, x, loss.item(), rate,
                                         rate * world_size())


def is_main_process():
    global USING_SPMD_FSDP
    if USING_SPMD_FSDP:
        return True
    return dist.get_rank() == 0


def is_local_main_process():
    global USING_SPMD_FSDP
    if USING_SPMD_FSDP:
        return True
    return int(os.environ["LOCAL_RANK"]) == 0


def local_rank():
    global USING_SPMD_FSDP
    if USING_SPMD_FSDP:
        return 0
    return int(os.environ["LOCAL_RANK"])


def world_size():
    global USING_SPMD_FSDP
    global SPMD_NUM_DEVICES
    if USING_SPMD_FSDP:
        return SPMD_NUM_DEVICES
    return dist.get_world_size()


def wait_for_everyone():
    torch.distributed.barrier()


def _goes_first(is_main):
    if is_main is False:
        wait_for_everyone()
    yield
    if is_main is True:
        wait_for_everyone()


@contextmanager
def main_process_first():
    yield from _goes_first(is_main_process())


def unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    try:
        return type(obj)(generator)
    except TypeError:
        # Some objects may not be able to instantiate from a generator directly
        return type(obj)(*list(generator))


def recursively_apply(
    func,
    data,
    *args,
    test_type=lambda t: isinstance(t, torch.Tensor),
    error_on_other_type=False,
    **kwargs,
):
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (recursively_apply(
                func,
                o,
                *args,
                test_type=test_type,
                error_on_other_type=error_on_other_type,
                **kwargs,
            ) for o in data),
        )
    elif isinstance(data, Mapping):
        return type(data)({
            k:
                recursively_apply(
                    func,
                    v,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                ) for k, v in data.items()
        })
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}.")
    return data


def gather(tensor):

    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        output_tensors = [
            tensor.clone() for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


def model_parameters_num(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return all_param, trainable_params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--not_save_model",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="A dir containing dataset with .arrow format.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="If passed, will set trust_remote_code=True when calling from_pretrained.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=0,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--total_train_batch_size",
        type=int,
        default=8,
        help="All batch size for the training dataloader. Equals to per_device_train_batch_size * world_size.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0,
        help="Clips gradient norm of an iterable of parameters.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--no_decay",
        nargs="*",
        default=["bias", "LlamaRMSNorm.weight"],
        help="No decay params.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="The number of sub-processes to use for the dataloader.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--ignore_dryrun_on_load_strategy",
        action="store_true",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=0,
        help="Log every X updates steps. Zero means do not logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"matplotlib"`, and `"all"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--logging_steps` bigger than 0."),
        choices=["all", "tensorboard", "matplotlib"],
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="If passed, will set ignore_mismatched_sizes=True when calling from_pretrained.",
    )
    parser.add_argument(
        "--distributed_method",
        default="ddp",
        choices=["ddp", "fsdp", "spmd_fsdp"],
        help="Choosing a Distributed Strategy",
    )
    parser.add_argument(
        "--fsdp_cpu_offload",
        action="store_true",
        help="If passed, offload model params to cpu memory.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16_amp", "fp16_amp", "bf16"],
        default="bf16_amp",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing or not.",
    )
    parser.add_argument(
        "--peft_type",
        type=str,
        default=None,
        help="Whether use peft and use what type of peft.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Lora attention dimension.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The alpha parameter for Lora scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="The dropout probability for Lora layers.",
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="*",
        default=["q_proj", "v_proj"],
        help="The names of the modules to apply Lora to.",
    )
    parser.add_argument(
        "--peft_task_type",
        type=str,
        default=TaskType.CAUSAL_LM,
        choices=[
            TaskType.SEQ_CLS, TaskType.SEQ_2_SEQ_LM, TaskType.CAUSAL_LM,
            TaskType.TOKEN_CLS
        ],
        help="Peft task type.",
    )
    parser.add_argument(
        "--fsdp_wrap_trainable_outmost",
        action="store_true",
        help="If fsdp would use wrap_trainable_outmost for peft model.",
    )
    parser.add_argument(
        "--random_log_n_training_samples",
        type=int,
        default=3,
        help="Log a few random samples from the training set.",
    )
    parser.add_argument(
        "--max_shard_size",
        default=None,
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`)."
            "`None` means no shard."),
    )
    parser.add_argument(
        "--enable_profiler",
        action="store_true",
        help="If passed, use torch.profiler.profile",
    )
    parser.add_argument(
        "--using_xla",
        action="store_true",
        help="If passed, using xla device",
    )
    parser.add_argument(
        "--init_emtpy_offload",
        action="store_true",
        help="If passed, use init_empty_weights_with_disk_offload. Should be used when training from scratch.",
    )

    parser.add_argument(
        "--spmd_model_axis_sharding",
        type=int,
        default=1,
        help="""Specifies the model axis sharding, spmd_model_axis_sharding=1 means that all tensors are split along the 0th dimension""",
    )
    parser.add_argument(
        "--spmd_add_xla_flash_attn_by_atorch",
        action="store_true",
        help="Using spmd with FlashAttention imported from atorch."
        "--fsdp-config-file",
        default="benchmarks/llm/fsdp_config.json",
    )
    parser.add_argument(
        "--fsdp-config-file",
        default="benchmarks/llm/fsdp_config.json",
        help="setting transformer_layer_cls options"
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="tensor parallel size of MLP/SelfAttention"
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=1,
        help="expert parallel size"
    )
    parser.add_argument(
        "--batch_chunk_size",
        type=int,
        default=1,
        help="batch chunk size"
    )
    parser.add_argument(
        "--enable_sequence_parallel",
        action="store_true",
        help="enable sequence parallel"
    )
    args = parser.parse_args()

    # Sanity checks
    if (args.dataset_name is None and args.train_file is None and
            args.validation_file is None and args.dataset_path is None):
        raise ValueError(
            "Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, json or txt file."

    return args


# for auto_accelerate
def optim_param_func(model, args):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in args.no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in args.no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


# for auto_accelerate
def my_loss_func(_, outputs):
    if isinstance(outputs, dict):
        return outputs["loss"]


# for auto_accelerate
def my_prepare_input(batch, device):
    batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
    return batch


def get_dataset(args):
    raw_datasets = None
    if is_local_main_process():
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    elif args.dataset_path is not None:
        raw_datasets = load_from_disk(args.dataset_path)
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(
            extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    return raw_datasets


def get_config(args):
    config = None
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")
    return config


def get_tokenizer(args):
    tokenizer = None
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer


def get_model(args, config):
    model = None
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=args.trust_remote_code,
        )

    if args.peft_type is not None:
        peft_config = get_peft_config(args)
        logger.info(f"Load Peft {args.peft_type} model ......")
        if args.gradient_checkpointing and args.peft_type == "lora":
            # Make Lora and gradient checkpointing compatible
            # https://github.com/huggingface/peft/issues/137
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
        model = get_peft_model(model, peft_config)
    return model


def get_peft_config(args):
    """
    Returns:
        config(PeftConfig)
    """
    if args.peft_type == "lora":
        peft_config = LoraConfig(
            task_type=args.peft_task_type,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
    else:
        raise NotImplementedError(f"Not support {args.peft_type}")
    return peft_config


def tokenize_dataset(args, model, raw_datasets, tokenizer):

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    with main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    return tokenized_datasets


def process_dataset(args, tokenized_datasets, tokenizer):
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i:i + block_size] for i in range(0, total_length, block_size)
            ] for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`.")
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    with main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    return lm_datasets


def compute_training_flops(
    batch_size,
    sequence_length,
    hidden_size,
    vocab_size,
    intermediate_size,
    num_layers,
    use_gradient_checkpointing=False,
    use_peft=False,
):
    """Returns:
    hardware flops
    model flops

    The source of formula:
    Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM's
    (APPENDIX: FLOATING-POINT OPERATIONS)

    Assuming that backward pass has twice FLOPs as many as forward pass. Only matrix multiplication FLOPs are computed.
    For use_peft, backward pass FLOPS is a little more than the forward pass. Assuming equal for simplicity here.
    """
    attention_forward_flops = (
        8 * batch_size * sequence_length * hidden_size**2 +
        4 * batch_size * sequence_length**2 * hidden_size)
    # llama2 use gate_proj, has 3 Linears
    two_mlps_forward_flops = 3 * 2 * batch_size * \
        sequence_length * hidden_size * intermediate_size
    logits_forward_flops = 2 * batch_size * \
        sequence_length * hidden_size * vocab_size
    decoder_layer_forward_flops = attention_forward_flops + two_mlps_forward_flops
    # forward FLOPs without gradient checkpointing
    forward_flops_wo_gc = num_layers * \
        decoder_layer_forward_flops + logits_forward_flops
    factor = 2 if use_peft else 3
    if not use_gradient_checkpointing:
        return forward_flops_wo_gc * factor, forward_flops_wo_gc * factor
    else:
        return (
            num_layers * decoder_layer_forward_flops * (factor + 1) +
            logits_forward_flops * factor,
            forward_flops_wo_gc * factor,
        )


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    train_prefix = "train_"
    train_prefix_len = len(train_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        elif k.startswith(train_prefix):
            new_d["train/" + k[train_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


try:
    import psutil

    PSUTILS_INSTALLED = True
except ImportError:
    PSUTILS_INSTALLED = False
    pass

try:
    from pynvml.smi import nvidia_smi

    PYNAMY_INSTALLED = True
except ImportError:
    nvidia_smi = None
    PYNAMY_INSTALLED = False


class ThroughputTimer:

    def __init__(
        self,
        batch_size,
        start_step=2,
        steps_per_output=50,
        monitor_memory=False,
        logging_fn=None,
    ):
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.batch_size = 1 if batch_size is None else batch_size
        self.start_step = start_step
        self.epoch_count = 0
        self.micro_step_count = 0
        self.global_step_count = 0
        self.total_elapsed_time = 0
        self.step_elapsed_time = 0
        self.steps_per_output = steps_per_output
        self.monitor_memory = monitor_memory
        self.logging = logging_fn
        if self.logging is None:
            import logging
            logger = logging.getLogger(__name__)
            self.logging = logger.info
        self.initialized = False

        if self.monitor_memory and not PSUTILS_INSTALLED:
            self.logging(
                "Unable to import `psutil`, please install package by `pip install psutil`. Set monitor_memory=False"
            )
            self.monitor_memory = False
        self.nvsmi = nvidia_smi.getInstance() if PYNAMY_INSTALLED else None

    def update_epoch_count(self):
        self.epoch_count += 1
        self.micro_step_count = 0

    def _init_timer(self):
        self.initialized = True

    def start(self):
        self._init_timer()
        self.started = True
        if self.global_step_count >= self.start_step:
            torch.cuda.synchronize()
            self.start_time = time.time()
        return self.start_time

    def stop(self, global_step=False, report_speed=True):
        if not self.started:
            return
        self.started = False
        self.micro_step_count += 1
        if global_step:
            self.global_step_count += 1
        if self.start_time > 0:
            torch.cuda.synchronize()
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.total_elapsed_time += duration
            self.step_elapsed_time += duration

            if global_step:
                if report_speed and self.global_step_count % self.steps_per_output == 0:
                    logging_infos = (
                        f"epoch={self.epoch_count}/micro_step={self.micro_step_count}/"
                        f"global_step={self.global_step_count}, RunningAvgSamplesPerSec={self.avg_samples_per_sec()},"
                        f" CurrSamplesPerSec={self.batch_size / self.step_elapsed_time},"
                        f" MemAllocated={round(torch.cuda.memory_allocated() / 1024**3, 2)}GB,"
                        f" MaxMemAllocated={round(torch.cuda.max_memory_allocated() / 1024**3, 2)}GB"
                    )
                    if PYNAMY_INSTALLED:
                        current_node_gpu_mem = []
                        nvsmi_gpu_memory_usage = self.nvsmi.DeviceQuery(
                            "memory.used, memory.total")["gpu"]
                        for gpu_id, memory_dict in enumerate(
                                nvsmi_gpu_memory_usage):
                            total_memory, used_memory, unit = (
                                memory_dict["fb_memory_usage"]["total"],
                                memory_dict["fb_memory_usage"]["used"],
                                memory_dict["fb_memory_usage"]["unit"],
                            )
                            current_node_gpu_mem.append(
                                f"GPU{gpu_id}:{int(used_memory)}/{int(total_memory)}{unit}"
                            )
                        nvismi_gpu_memory_infos = ",".join(current_node_gpu_mem)
                        logging_infos += ". " + nvismi_gpu_memory_infos
                    self.logging(logging_infos)
                    if self.monitor_memory:
                        virt_mem = psutil.virtual_memory()
                        swap = psutil.swap_memory()
                        self.logging(
                            f"epoch={self.epoch_count}/micro_step={self.micro_step_count}/"
                            f"global_step={self.global_step_count} virtual_memory %: {virt_mem.percent}, "
                            f"swap_memory %: {swap.percent}")
                self.step_elapsed_time = 0
        return self.end_time

    def avg_samples_per_sec(self):
        if self.global_step_count > 0:
            total_step_offset = self.global_step_count - self.start_step
            avg_time_per_step = self.total_elapsed_time / total_step_offset
            # training samples per second
            return self.batch_size / avg_time_per_step
        return float("-inf")


def main():

    args = parse_args()
    global USING_SPMD_FSDP
    global SPMD_NUM_DEVICES
    USING_SPMD_FSDP = args.distributed_method == "spmd_fsdp"
    # no gc this server;otherwise server will stop
    if args.distributed_method == "spmd_fsdp":
        xr.use_spmd()
        SPMD_NUM_DEVICES = xr.global_runtime_device_count()
        server = xp.start_server(9012)
        print("SPMD_NUM_DEVICES is", SPMD_NUM_DEVICES)
    elif args.using_xla:
        pjrt.initialize_multiprocess(os.environ["LOCAL_RANK"],
                                     os.environ["WORLD_SIZE"])
        device = xm.xla_device()
        server = xp.start_server(9012)
        dist.init_process_group('xla', init_method='xla://')

    else:
        device = torch.device("cuda:%d" % local_rank())
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    assert args.logging_steps >= 0, f"logging_steps must bigger or equal than 0 but got {args.logging_steps}."
    with_tracking = args.logging_steps > 0 and args.output_dir is not None
    if args.report_to is not None and not with_tracking:
        logger.info(
            f"Found args.logging_steps=={args.logging_steps} and args.output_dir=={args.output_dir}."
            "args.report_to will be ignored.")
    if args.output_dir is not None and is_main_process() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"output_dir is {args.output_dir}")

    config = get_config(args)
    model = get_model(args, config)

    tokenizer = get_tokenizer(args)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    num_params, num_trainable_params = model_parameters_num(model)
    if args.distributed_method == "spmd_fsdp" or is_local_main_process():
        logger.info(
            f"Model has {num_params} parameters and {num_trainable_params} "
            f"trainable parameters({100 * num_trainable_params / num_params:.3f}%)."
        )

    if "alpaca" in args.dataset_path:
        train_dataset = InstructionDataset(
            args.dataset_path,
            tokenizer,
            partition="train",
            max_words=args.block_size,
        )
        eval_dataset = InstructionDataset(
            args.dataset_path,
            tokenizer,
            partition="eval",
            max_words=args.block_size,
        )
    else:
        raw_datasets = get_dataset(args)
        tokenized_datasets = tokenize_dataset(args, model, raw_datasets,
                                              tokenizer)
        lm_datasets = process_dataset(args, tokenized_datasets, tokenizer)
        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(
            range(len(train_dataset)), args.random_log_n_training_samples):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    """
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
        pin_memory=True,
        drop_last=True,
    )
    """
    dataloader_args = {
        "shuffle": True,
        "collate_fn": default_data_collator,
        "batch_size": args.total_train_batch_size,
        "pin_memory": True,
        "num_workers": args.dataloader_num_workers,
        "persistent_workers": args.dataloader_num_workers > 0,
    }

    if "amp" in args.precision:
        pass
    if args.gradient_checkpointing:
        pass
        # strategy.append(("checkpoint", (LlamaDecoderLayer,)))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   **dataloader_args)
    transformer_layer_cls = ()
    if args.distributed_method == "spmd_fsdp":
        if args.spmd_add_xla_flash_attn_by_atorch:
            from atorch.modules.transformer.layers import LlamaAttentionFA
            from atorch.modules.transformer.inject import replace_module
            src_cls = getattr(LlamaAttentionFA, "_src_module_cls", LlamaAttentionFA.__base__)
            model = replace_module(model, src_cls, LlamaAttentionFA, need_src_module=True)

        device = xm.xla_device()
        model = model.to(device=device)
        optimizer = syncfree.AdamW(
            optim_param_func(model, args), lr=args.learning_rate)
        # optimizer = torch.optim.AdamW(optim_param_func(model, args), lr=args.learning_rate)
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))
        model_axis = args.spmd_model_axis_sharding
        mesh_shape = (num_devices // model_axis, model_axis)
        input_mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
        dp_mesh = xs.Mesh(device_ids, (num_devices//args.tp_size, args.tp_size), 
            ("fsdp", "model"))
        sharding_groups = dp_mesh.get_logical_mesh()
        if args.tp_size > 1 and num_devices>8:
            # TPx8 DPx2
            # data only shard on this machine
            this_rank = int(os.environ["RANK"])
            this_machine_device_ids = np.where(device_ids//8==this_rank)[0]
            mesh_2d = xs.Mesh(this_machine_device_ids, 
            (num_devices//args.tp_size, 
            args.tp_size), ("data", "model"))
            input_sharding=xs.ShardingSpec(
                mesh_2d,
                ("data", None))
            origin_forward = LlamaDecoderLayer.forward
            def new_forward(self,*args,**kwargs):
                for tensor in args:
                    if isinstance(tensor, torch.Tensor):
                        shape = tensor.shape
                        shard_shape = [None]*len(shape)
                        shard_shape[0] = "data"
                        xs.mark_sharding(tensor, mesh_2d, tuple(shard_shape))
                for tensor in kwargs.values():
                    if isinstance(tensor, torch.Tensor):
                        shape = tensor.shape
                        shard_shape = [None]*len(shape)
                        shard_shape[0] = "data"
                        xs.mark_sharding(tensor, mesh_2d, tuple(shard_shape))
                return origin_forward(self,*args,**kwargs)
            LlamaDecoderLayer.forward = new_forward
        elif args.tp_size > 1 and num_devices<=8:
            input_sharding=xs.ShardingSpec(
                xs.Mesh(device_ids, dp_mesh, ("data", "model")),
                (None, None))
        else:
            input_sharding=xs.ShardingSpec(
                xs.Mesh(device_ids, dp_mesh, ("data", "model")),
                ("data", None))
        train_dataloader = pl.MpDeviceLoader(
            train_dataloader,  # wraps PyTorch DataLoader
            device,
            # optional input_sharding field
            input_sharding=input_sharding)
        spmd_mesh = input_mesh
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2

        from torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel import (
            XlaFullyShardedDataParallel as FSDP_XLA)
        def shard_output(output, mesh):
            xs.mark_sharding(output.logits, mesh, ('fsdp', None, None))
        from torch_xla.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy)
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            })
        
        
        if args.tp_size > 1:
            make_tensor_parallel(model, num_devices, args.tp_size, parallel_choice=7)
        if args.enable_sequence_parallel:
            make_sequence_parallel(model, num_devices, args.tp_size)
        if args.ep_size>1:
            from lynx.distributed.expert_parallel import MoELayer
            from transformers.models.llama.modeling_llama import LlamaMLP
            # replace LlamaMLP to MoELayer
            # config
            expert_parallel_shape = (
                num_devices//args.ep_size,
                args.ep_size,
            )
            device_ids = np.array(range(num_devices))
            mesh = xs.Mesh(device_ids, expert_parallel_shape, 
                        ('data','expert'))
            replace_module(model,
                LlamaMLP,
                MoELayer,
                config=config,
                strict=False, #can't restore state dict
                num_experts=args.ep_size*2, # 2 experts per device
                mesh=mesh
            )
            
            model.to(device=device)
            for name, module in model.named_modules():
                # module should on xla device
                if isinstance(module, MoELayer):
                    make_expert_parallel(module,mesh)
                    make_gating_network_parallel(module.gate,mesh)
        # checkpoint_cls = [LlamaDecoderLayer]
        # for name, module in model.named_modules():
            # if isinstance(module, tuple(checkpoint_cls)):
                # checkpoint_module(module)
        wrapper_config = {}
        wrapper_config["auto_wrap_policy"] = wrap_policy
        #sharding_groups sharding_rank sharding_world_size
        # sharding_groups A list of list  Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        # wrapper_config["sharding_groups"] = sharding_groups
        # wrapper_config["sharding_rank"] = xm.get_ordinal()
        # wrapper_config["sharding_world_size"] = xr.global_runtime_device_count()
        # Currently, gradient checkpointing needs to be applied to the module before the FSDP wrapper
        # for i,block in 
        model = FSDPv2(model, mesh=spmd_mesh, shard_output=shard_output,)
        # model = FSDP_XLA(model, mesh=spmd_mesh, shard_output=shard_output,
        # **wrapper_config)
        if xm.get_ordinal()==0:
            print("model",model)
        for name, p in model.named_parameters():
            print(name,p.shape, torch_xla._XLAC._get_xla_sharding_spec(p))
    elif args.distributed_method == "ddp":
        model = model.to(device=device)
        model = DistributedDataParallel(
            model,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
            static_graph=True,
            broadcast_buffers=False)
        logger.info("Using DDP")
    else:
        if xm.get_ordinal()==0:
            os.environ["XLA_PERSISTENT_CACHE_READ_ONLY"] = "false"
        else:
            os.environ["XLA_PERSISTENT_CACHE_READ_ONLY"] = "true"
        wrapper_config = {}
        if args.using_xla:
            from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
            from torch_xla.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy, transformer_auto_wrap_policy)
            wrapper_config["compute_dtype"] = torch.bfloat16
            wrapper_config["auto_wrap_policy"] = wrap_policy
            # 
            if args.tp_size > 1:
                sharding_groups = []
                # tp_size=2, [[0,1,2,3],[4,5,6,7]]
                # tp_size=4, [[0,1,],[2,3],[4,5], [6,7]]
                shard_size = xr.global_runtime_device_count()//args.tp_size
                for i in range(args.tp_size):
                    sharding_groups.append(list(range(i*shard_size,(i+1)*shard_size)))
            else:
                sharding_groups = [list(range(xr.global_runtime_device_count()))]
            wrapper_config["sharding_groups"] = sharding_groups
            wrapper_config["sharding_rank"] = xm.get_ordinal()
            wrapper_config["sharding_world_size"] = xr.global_runtime_device_count()
        else:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            mixed_precision_config = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            wrapper_config["mixed_precision"] = mixed_precision_config
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            })
        
        print("wrapper_config",wrapper_config)
        model = FSDP(model, **wrapper_config).to(device=device)
        logger.info("Using FSDP")
    if args.distributed_method == "spmd_fsdp":
        pass
    elif args.using_xla:
        optimizer = syncfree.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    warmup_steps = 0
    num_training_steps = args.max_train_steps * args.gradient_accumulation_steps
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps * args.gradient_accumulation_steps
    elif args.warmup_ratio > 0.0:
        warmup_steps = int(num_training_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # create a summary_writer to write training metrics to tensorboard
    summary_writer = None

    report_to_tb = with_tracking and args.report_to in ("tensorboard", "all")
    report_to_tb = False
    if report_to_tb:
        tb_path = os.path.join(args.output_dir, default_logdir())
        summary_writer = SummaryWriter(tb_path)
        logger.info(f"Tensorboard eventfiles will be saved at {tb_path}")

    # Train!
    total_batch_size = args.total_train_batch_size * args.gradient_accumulation_steps

    if args.total_train_batch_size > 0:
        per_device_train_batch_size = int(args.total_train_batch_size /
                                          world_size())
        total_train_batch_size = args.total_train_batch_size
    elif args.per_device_train_batch_size > 0:
        per_device_train_batch_size = args.per_device_train_batch_size
        total_train_batch_size = per_device_train_batch_size * world_size()
    else:
        raise ValueError(
            f"per_device_train_batch_size must greater than 0 but got {per_device_train_batch_size}"
        )

    flops_per_gpu_per_iteration, _ = compute_training_flops(
        per_device_train_batch_size,
        args.block_size,
        config.hidden_size,
        config.vocab_size,
        config.intermediate_size,
        config.num_hidden_layers,
        args.gradient_checkpointing,
        args.peft_type is not None,
    )
    tput_timer = ThroughputTimer(
        total_train_batch_size, start_step=2, steps_per_output=50)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not is_local_main_process())
    completed_steps = 0
    completed_eval_steps = 0
    starting_epoch = 0

    # # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # Sorts folders by date modified, most recent checkpoint is the last
            path = dirs[-1]
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace(
                "step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    total_train_losses = [[], []]  # steps, loss
    total_eval_losses = [[], []]  # steps, loss
    all_results = {}
    training_time = 0
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        torch.cuda.synchronize()
        current_epoch_start_time = time.time()
        start_time = time.time()
        step = 0
        batch_num = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            optimizer.zero_grad()
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue
            if args.enable_profiler:
                if args.using_xla:
                    if step == 20:
                        xp.trace_detached('localhost:9012', 
                        "./llama_xla_trace", duration_ms=10000)
                        time.sleep(1)# wait trace init
                    context = xp.StepTrace('train_loop', step_num=step)
                else:
                    if step == 20:
                        context = torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            profile_memory=False,
                            with_stack=True,
                            with_modules=True,
                            record_shapes=True,
                        )
                    else:
                        context = nullcontext()
            else:
                context = nullcontext()
            if args.using_xla:
                autocast = xla_autocast(device, dtype=torch.bfloat16)
            else:
                autocast = torch_autocast("cuda", dtype=torch.bfloat16)
            with context as prof:
                # step_start_timestamp = tput_timer.start()
                batch = my_prepare_input(batch, device)
                if step == 0:
                    print("batch",{key:value.shape for key,value in batch.items()})
                    print("batch on spmd",{key:torch_xla._XLAC._get_xla_sharding_spec(value) for key,value in batch.items()})
                if args.batch_chunk_size > 0:
                    # key:tensor, torch.chunk(tensor, args.batch_chunk_size)
                    batches = []
                    splited_batch ={key:torch.chunk(value, args.batch_chunk_size) for key,value in batch.items()}
                    # {key:[tensor11,tensor12,tensor13],key2:[tensor21,tensor22,tensor23]}
                    # -> [{key:tensor11,key2:tensor21},{key:tensor12,key2:tensor22},{key:tensor13,key2:tensor23}]
                    for i in range(args.batch_chunk_size):
                        batches.append({key:value[i] for key,value in splited_batch.items()})
                        
                else:
                    batches = [batch]
                # run multi forward-backward
                for batch in batches:
                    with autocast:
                        outputs = model(**batch)
                        loss = outputs["loss"]
                    loss.backward()
                if args.using_xla:
                    # add this will cause loss=nan；if using syncfree optimizer,use this
                    # gradients = xm._fetch_gradients(optimizer)
                    # xm.all_reduce('sum', gradients, scale=1.0 / xm.xrt_world_size(),pin_layout=False)
                    found_inf = torch.isnan(loss).to(torch.float32)
                    optimizer.step(found_inf=found_inf)
                else:
                    optimizer.step()
                lr_scheduler.step()
            if step == 20 and not args.using_xla:
                prof.export_chrome_trace("./llama_trace%d.json" % local_rank())
            if step % 20 == 0:
                rate = per_device_train_batch_size / (time.time() - start_time)
                if args.using_xla:
                    xm.add_step_closure(
                        _train_update, args=(device, step, loss, rate))
                else:
                    _train_update(device, step, loss, rate)
                start_time = time.time()
            if step == batch_num - 2:
                print("now epoch down", epoch)
                break

        torch.cuda.synchronize()
        current_epoch_elapse_time = time.time() - current_epoch_start_time
        if is_main_process():
            logger.info(
                f"Training epoch {epoch} takes {current_epoch_elapse_time:.3f} seconds."
            )
        training_time += current_epoch_elapse_time
        continue
        tput_timer.update_epoch_count()
        model.eval()
        eval_losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = prepare_input(batch, device)
                outputs = model(**batch)

            loss = outputs.loss
            completed_eval_steps += 1
            gathered_loss = gather(loss.repeat(args.per_device_eval_batch_size))
            eval_losses.append(gathered_loss)
            if with_tracking:
                current_eval_step_aggregated_loss = torch.mean(gathered_loss)
                total_eval_losses[0].append(completed_eval_steps)
                total_eval_losses[1].append(
                    current_eval_step_aggregated_loss.cpu().item())
                eval_logs = {
                    "eval_loss": current_eval_step_aggregated_loss,
                    "epoch": epoch,
                }
                if report_to_tb and is_main_process():
                    eval_logs = rewrite_logs(eval_logs)
                    for key, value in eval_logs.items():
                        summary_writer.add_scalar(
                            f"{key}", value, global_step=completed_steps)

        eval_losses = torch.cat(eval_losses)
        try:
            eval_loss = torch.mean(eval_losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        if is_local_main_process():
            logger.info(
                f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}"
            )

        all_results[f"epoch{epoch}"] = {
            "eval_loss": eval_loss.cpu().item(),
            "perplexity": perplexity
        }

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
    return

    if with_tracking and args.report_to in ("all",
                                            "matplotlib") and is_main_process():
        fig, ax = plt.subplots(nrows=2, layout="constrained")
        ax[0].plot(
            total_train_losses[0], total_train_losses[1], label="train_loss")
        ax[0].set_xlabel("train_steps")
        ax[0].set_ylabel("train_loss")
        ax[0].set_title("Llama-2 train loss")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].plot(
            total_eval_losses[0], total_eval_losses[1], label="eval_loss")
        ax[1].set_xlabel("eval_steps")
        ax[1].set_ylabel("eval_loss")
        ax[1].set_title("Llama-2 eval loss")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        if args.output_dir is not None:
            fig_path = os.path.join(args.output_dir, "llama2_loss.png")
            fig.savefig(fig_path)
            logger.info(f"Loss curve has been saved at {fig_path}")

    if is_main_process():
        print("Training throughput is {:.3f} samples/s".format(
            args.max_train_steps * args.total_train_batch_size /
            training_time,))

    if args.output_dir is not None:
        wait_for_everyone()
        if not args.not_save_model:
            model_state_dict = None
            if isinstance(model, FSDP):
                fsdp_save_policy = FullStateDictConfig(
                    offload_to_cpu=world_size() > 1,
                    rank0_only=world_size() > 1)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                          fsdp_save_policy):
                    model_state_dict = model.state_dict()

            unwrapped_model = unwrap_model(model)

            if args.max_shard_size is None:
                max_shard_size_gb = math.ceil(4 * num_params / 1e9) + 1
                max_shard_size = f"{max_shard_size_gb}GB"
            else:
                max_shard_size = args.max_shard_size
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=is_main_process(),
                state_dict=model_state_dict,
                save_function=torch.save,
                max_shard_size=max_shard_size,
            )

        if is_main_process():
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "all_results.json"),
                      "w") as f:
                json.dump(all_results, f)

            torch.save(args, os.path.join(args.output_dir, TRAINING_ARGS_NAME))
            logger.info(
                f"Ckpts and other configs have been saved at {args.output_dir}")


def _init_xla():

    import os
    print("os.environ", os.environ)

    import torch.nn.parallel.distributed
    import torch.distributed.utils
    torch.distributed.utils._verify_param_shape_across_processes = torch.nn.parallel.distributed._verify_param_shape_across_processes = lambda process_group, tensors, logger=None: None


if __name__ == "__main__":
    """
    set -exo pipefail
DATASET_PATH=./datasets/alpaca/alpaca_data.json
PRETRAINED_MODEL_DIR=./pretrained_models/Llama-2-1.7b-hf
PJRT_DEVICE=CUDA
PER_DEVICE_TRAIN_BATCH_SIZE=6
GPU_NUM_DEVICES=8
export XLA_GPU_MEMORY_FRACTION=0.7
export XLA_GPU_MEMORY_PREALLOCATE=false
export XLA_GPU_MEMORY_ALLOCATOR_KIND=3
export PJRT_DEVICE=CUDA
export PJRT_ALLOCATOR_PREALLOCATE=false
export PJRT_ALLOCATOR_FRACTION=0.7
export PJRT_ALLOCATOR_CUDA_ASYNC=true
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_VMODULE="hlo_pass_pipeline=5"
export XLA_FLAGS="--xla_gpu_enable_linear_program_scheduler=true --xla_gpu_enable_analytical_latency_estimator=true --xla_gpu_enable_xla_runtime_executable=false --xla_cpu_enable_fast_math=false --xla_gpu_simplify_all_fp_conversions=false --xla_gpu_force_compilation_parallelism=8  --xla_gpu_enable_pipelined_collectives=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_async_collectives=true --xla_gpu_enable_latency_hiding_scheduler=true --xla_disable_hlo_passes=rematerialization,gpu-convert-async-collectives-to-sync"
torchrun --nnodes 1 --nproc-per-node 8 \
        benchmarks/llm/llama_benchmark.py \
    --not_save_model \
    --dataset_path $DATASET_PATH \
    --config_name $PRETRAINED_MODEL_DIR \
    --tokenizer_name $PRETRAINED_MODEL_DIR \
    --num_train_epochs 6 \
    --block_size 512 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --per_device_eval_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --seed 42 \
    --preprocessing_num_workers 2 \
    --dataloader_num_workers 4 \
    --ignore_mismatched_sizes \
    --ignore_dryrun_on_load_strategy \
    --output_dir ./outputs \
    --random_log_n_training_samples 0 \
    --logging_steps 10 \
    --report_to all --distributed_method fsdp --using_xla

    """
    _init_xla()
    main()
