import torch
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention,LlamaConfig
from argparse import ArgumentParser
import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast as xla_autocast
import contextlib
import os
@contextlib.contextmanager
def nosave_xla_context():
    """
    unset env in contextlib
    """
    if "XLA_FLAGS" in os.environ:
        old_xla_flags = os.environ["XLA_FLAGS"]
        del os.environ["XLA_FLAGS"]
    else:
        old_xla_flags = None
    yield
    if old_xla_flags:
        os.environ["XLA_FLAGS"] = old_xla_flags
def trace_MLP(mlp):
    
    device = xm.xla_device()
    mlp.to(device)
    # (B, S, H)
    x = torch.randn(batch_size,seq_len,hidden_size).to(device)
    for name,param in mlp.named_parameters():
        print(name, param.shape)
    # using xla autocast or torch autocast?
    xm.mark_step()
    with xla_autocast(device=device,dtype=torch.bfloat16):
        with nosave_xla_context():
            # try no save forward grad
            y = mlp(x)
            loss = y.sum()
            xm.unlazy([loss])
        # xm.mark_step()
        loss.backward()
        grads = [p.grad for p in mlp.parameters()]
        xm.unlazy(grads)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["trace", "run"])
    parser.add_argument("--ptx-file", type=str)

    batch_size,seq_len,hidden_size = 8,768,1024

    config = LlamaConfig(hidden_size=hidden_size,
    intermediate_size=hidden_size*4,
    num_attention_heads=16,
    num_hidden_layers=12)
    mlp = LlamaMLP(config)
    
    args = parser.parse_args()
    if args.mode == "trace":
        torch.save(mlp.state_dict(), "mlp.pth")
        print("save mlp to mlp.pth")
        trace_MLP(mlp)

    else:
        mlp.load_state_dict(torch.load("mlp.pth"))