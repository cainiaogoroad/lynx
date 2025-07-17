import torch
from torch import Tensor,nn
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import torch_xla.distributed.spmd.xla_sharding as xs
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla
import torch_xla.debug.profiler as xp
from lynx.distributed.tensor_parallel import make_tensor_parallel, TPMode,make_sequence_parallel
import time
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention,LlamaConfig,LlamaDecoderLayer,LlamaRMSNorm
from glob import glob
import os
import unittest
import logging
class MultiAttention(nn.Module):
    def __init__(self,nlayers,config):
        super().__init__()
        self.attentions = nn.ModuleList([LlamaAttention(config) for _ in range(nlayers)])
    def forward(self,hidden_states,attention_mask,position_ids):
        for attention in self.attentions:
            hidden_states,_,_ = attention(hidden_states,attention_mask,position_ids)
        return hidden_states
class OneNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.layernorm = LlamaRMSNorm(hidden_size, eps)
    def forward(self,x):
        return self.layernorm(x)
class TestTP(unittest.TestCase):
    def setUp(self):
        xr.use_spmd()
        logging.basicConfig(level=logging.INFO)

    def test_mlp_tp(self):
        server = xp.start_server(9012)
        device = xm.xla_device()
        world_size = xr.global_runtime_device_count()
        print("world_size", world_size)
        tp_size = world_size
        config = LlamaConfig(hidden_size=1024, intermediate_size=4096)
        mlp = LlamaMLP(config)
        mlp.to(device)
        make_tensor_parallel(mlp, world_size, tp_size)
        x = torch.randn(8, 1024)
        x = x.to(device)
        
        for i in range(10):
            context = xp.StepTrace('train_loop', step_num=i)
            with context:
                output = mlp(x)
            # xp.trace_detached('localhost:9012', "./mlp_xla_trace")
        time.sleep(5)
    def test_attention_tp(self):
        server = xp.start_server(9012)
        device = xm.xla_device()
        world_size = xr.global_runtime_device_count()
        print("world_size", world_size)
        tp_size = world_size
        config = LlamaConfig(hidden_size=1024, intermediate_size=4096, 
                             num_attention_heads=16,num_key_value_heads=8,
                             max_position_embeddings=768)
        parallel_choice = int(os.environ.get("PARALLEL_CHOICE",0))
        attention = LlamaAttention(config)
        attention.to(device)
        make_tensor_parallel(attention, world_size, tp_size,parallel_choice=parallel_choice)
        # 
        (batch_size, query_length, key_value_length) = (8, 768, 768)
        hidden_states = torch.randn(batch_size, query_length, config.hidden_size)
        attention_mask = torch.ones(batch_size, 1, query_length,key_value_length)
        position_ids = torch.arange(0, query_length).repeat(batch_size, 1)
        hidden_states = hidden_states.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
        # for i in range(10):
            # xp.trace_detached('localhost:9012', "./ut_xla_profiler")
        context = xp.StepTrace('train_loop', step_num=0)
        with context:
            output = attention(hidden_states,attention_mask,position_ids)
        hlo_path = "dump_mlp_tp/%s/*sm_9.0_gpu_after_optimizations.txt"%parallel_choice
        for filename in glob(hlo_path):
            # grep replica_groups,means how many communication
            communication_count = 0

            with open(filename,'r') as f:
                for line in f:
                    if "replica_groups" in line:
                        communication_count +=1
            print("parallel_choice",parallel_choice,filename,"communication_count:",communication_count)
            # time.sleep(5)
    def test_atttention_tp_microbatch(self):
        server = xp.start_server(9012)
        device = xm.xla_device()
        world_size = xr.global_runtime_device_count()
        print("world_size", world_size)
        tp_size = world_size
        config = LlamaConfig(hidden_size=1024, intermediate_size=4096, 
                             num_attention_heads=16,num_key_value_heads=8,
                             max_position_embeddings=768)
        attention = MultiAttention(2, config)
        attention.to(device)
        # 7=0b0111, o is column_parallel_mode, qkv is row_parallel_mode
        make_tensor_parallel(attention, world_size, tp_size, parallel_choice=7)
        # 
        (batch_size, query_length, key_value_length) = (16, 4096, 4096)
        hidden_states = torch.randn(batch_size, query_length, config.hidden_size)
        attention_mask = torch.ones(batch_size, 1, query_length,key_value_length)
        position_ids = torch.arange(0, query_length).repeat(batch_size, 1)
        hidden_states = hidden_states.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
        batch = {
            "hidden_states":hidden_states,
            "attention_mask":attention_mask,
            "position_ids":position_ids
        }
        batches = []
        batch_chunk_size = 2
        splited_batch ={
            key:torch.chunk(value, batch_chunk_size) for key,value in batch.items()}
        xm.mark_step()

        # {key:[tensor11,tensor12,tensor13],key2:[tensor21,tensor22,tensor23]}
        # -> [{key:tensor11,key2:tensor21},{key:tensor12,key2:tensor22},{key:tensor13,key2:tensor23}]
        for i in range(batch_chunk_size):
            batches.append({key:value[i] for key,value in splited_batch.items()})
        for i in range(10):
            xp.trace_detached('localhost:9012', "./ut_xla_profiler")
            if i == 0:
                time.sleep(1)#wait trace init finish
            context = xp.StepTrace('train_loop', step_num=i)
            with context:# include mark_step,so two batch can be run in on graph
                outputs = []
                for batch in batches:
                    output = attention(**batch)
                    outputs.append(output)
                torch.cat(outputs,dim=0)
            xm.mark_step()
        time.sleep(5)
    def test_sequence_parallel(self):
        """sequence parallel on Layernorm and Dropout"""
        server = xp.start_server(9012)
        device = xm.xla_device()
        world_size = xr.global_runtime_device_count()
        print("world_size", world_size)
        tp_size = world_size
        config = LlamaConfig(hidden_size=1024, intermediate_size=4096)

        model = LlamaDecoderLayer(config)
        model.to(device)
        make_sequence_parallel(model, world_size, tp_size)
        make_tensor_parallel(model, world_size, tp_size,parallel_choice=7)
        batch,seq_len,embed_dim = 8,768,1024
        x = torch.randn(batch,seq_len,embed_dim)
        attention_mask = torch.ones(batch,1,seq_len,seq_len)
        position_ids = torch.arange(0, seq_len).repeat(batch, 1)
        x = x.to(device)
        
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
        mesh = xs.Mesh(np.arange(world_size), (world_size//tp_size,tp_size,1), ("data","sequence","model"))
        xs.mark_sharding(x, mesh, (None,None,None))
        xs.mark_sharding(attention_mask, mesh, (None,None,None,None))
        xs.mark_sharding(position_ids, mesh, (None,None))
        for i in range(10):
            xp.trace_detached('localhost:9012', "./ut_xla_profiler",duration_ms=2000)
            context = xp.StepTrace('train_loop', step_num=i)
            with context:
                output = model(x,attention_mask,position_ids)
if __name__ == "__main__":
    unittest.main()