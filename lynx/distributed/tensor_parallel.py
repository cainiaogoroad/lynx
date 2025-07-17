from typing import Optional, Callable
import torch.nn as nn
import torch_xla.core.xla_model as xm
import numpy as np
import enum
import torch_xla.distributed.spmd.xla_sharding as xs
import torch_xla
import logging
import torch
logger = logging.getLogger(__name__)
class TPMode(enum.Enum):
    TP_AND_FSDP = 0 # shard at the data dimension
    TP_AND_DP = 1 # duplicate at the data dimension
class ParallelChoice:
    q = 1
    k = 2
    v = 4
    o = 8
def make_tensor_parallel(
    module:nn.Module, 
    num_devices:int,
    tensor_parallel_size:int,
    mode:TPMode=TPMode.TP_AND_FSDP,
    parallel_choice:int=0,# qkv and o ,2^4=16
    ):
    """
    https://github.com/pytorch-tpu/transformers/blob/llama2-google-next-training/examples/pytorch/language-modeling/run_clm.py#L538
    it say: qkv should be column_parallel_mode, o should be row_parallel_mode.
    I have examined the code, parallel_choice=1000 have 22020096.0 communication cost and 3 ops;
    1111 and 1101 have 13762560.0(half than upper) and 3 ops
    """
    device_ids = np.array(range(num_devices))
    mesh_2d = xs.Mesh(device_ids, (num_devices//tensor_parallel_size, tensor_parallel_size), ("data", "model"))
    if mode == TPMode.TP_AND_FSDP:
        # data means the tensor is split along the data dimension
        row_parallel_mode = ("model", "data")
        column_parallel_mode = ("data", "model")
    else:
        # None means duplicate
        row_parallel_mode = ("model", None)
        column_parallel_mode = (None, "model")
    if parallel_choice & ParallelChoice.q:
        q_parallel_mode = row_parallel_mode
    else:
        q_parallel_mode = column_parallel_mode
    if parallel_choice & ParallelChoice.k:
        k_parallel_mode = row_parallel_mode
    else:
        k_parallel_mode = column_parallel_mode
    if parallel_choice & ParallelChoice.v:
        v_parallel_mode = row_parallel_mode
    else:
        v_parallel_mode = column_parallel_mode
    if parallel_choice & ParallelChoice.o:
        o_parallel_mode = row_parallel_mode
    else:
        o_parallel_mode = column_parallel_mode
    for name, param in module.named_parameters():
        if ("gate_proj.weight" in name or 
            "up_proj.weight" in name or 
            "dense_h_to_4h.weight" in name): # MLP
            xs.mark_sharding(param, mesh_2d, row_parallel_mode)
            logger.info("%s have be shard to %s",name, torch_xla._XLAC._get_xla_sharding_spec(param))
        elif ("down_proj.weight" in name 
            or "dense_4h_to_h.weight" in name): # MLP
            xs.mark_sharding(param, mesh_2d, column_parallel_mode)
            logger.info("%s have be shard to %s",name, torch_xla._XLAC._get_xla_sharding_spec(param))
        elif ("query_key_value.weight" in name
        ): # Self Attention QKV
            xs.mark_sharding(param, mesh_2d, column_parallel_mode)
            logger.info("%s have be shard to %s",name, torch_xla._XLAC._get_xla_sharding_spec(param))
        elif "q_proj.weight" in name:
            xs.mark_sharding(param, mesh_2d, q_parallel_mode)
            logger.info("%s have be shard to %s",name, torch_xla._XLAC._get_xla_sharding_spec(param))
        elif "k_proj.weight" in name:
            xs.mark_sharding(param, mesh_2d, k_parallel_mode)
            logger.info("%s have be shard to %s",name, torch_xla._XLAC._get_xla_sharding_spec(param))
        elif "v_proj.weight" in name:
            xs.mark_sharding(param, mesh_2d, v_parallel_mode)
            logger.info("%s have be shard to %s",name, torch_xla._XLAC._get_xla_sharding_spec(param))
        elif ("dense.weight" in name or
         "o_proj.weight" in name):# Self Attention Output
            xs.mark_sharding(param, mesh_2d, o_parallel_mode)
            logger.info("%s have be shard to %s",name, torch_xla._XLAC._get_xla_sharding_spec(param))
        else:
            logger.info("%s have no shard plan",name)
    return module
def wrap_module_input(module:nn.Module, mesh, sharding_spec_fn):
    old_forward = module.forward
    if hasattr(module, "is_input_wrapped"):
        return module
    
    def new_forward(self, *args, **kwargs):
        logger.info("wrap input,before forward:%s",self)
        for arg in args:
            sharding_spec = sharding_spec_fn(arg)
            if sharding_spec is not None:
                xs.mark_sharding(arg, mesh, sharding_spec)
        for key,value in kwargs.items():
            sharding_spec = sharding_spec_fn(value)
            if sharding_spec is not None:
                xs.mark_sharding(value, mesh, sharding_spec)
        logger.info("wrap input,after warpper",[torch_xla._XLAC._get_xla_sharding_spec(arg) for arg in args],
        {key:torch_xla._XLAC._get_xla_sharding_spec(value) for key,value in kwargs.items()})
        # TODO: output shard?
        return old_forward(*args, **kwargs)
    module.forward = new_forward
    module.is_input_wrapped = True

    return module
def _prepare_sp_sharding_spec(param):
    if not isinstance(param, torch.Tensor):
        return None
    shape = param.shape
    if len(shape) < 2:
        # do nothing
        return None
    partition_spec = [None] * len(shape)
    partition_spec[1] = "sequence"
    return tuple(partition_spec)
def make_sequence_parallel(
    module:nn.Module,
    num_devices:int,
    sequence_parallel_size:int,
    ):
    """
    sequence parallel on Layernorm and Dropout
    Pytorch: torch/distributed/tensor/parallel/style.py
    make param is replicate
    TODO: support SP+DP
    """
    device_ids = np.array(range(num_devices))
    mesh_1d = xs.Mesh(device_ids, (num_devices//sequence_parallel_size, sequence_parallel_size), ("data","sequence"))
    for name,param in module.named_parameters():
        if "layernorm.weight" in name:
            xs.mark_sharding(param, mesh_1d, (None,))
            logger.info("%s have be shard to %s",name, torch_xla._XLAC._get_xla_sharding_spec(param))
    
    
    for name,child in module.named_children():
        if "input_layernorm" in name or "post_attention_layernorm" in name:# llama
            wrap_module_input(child, mesh_1d, _prepare_sp_sharding_spec)