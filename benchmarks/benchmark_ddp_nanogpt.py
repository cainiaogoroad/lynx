"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import argparse
import os
import time
import math
import pickle
import logging
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim import lr_scheduler

from model import GPTConfig, GPT



parser = argparse.ArgumentParser(description="NanoGPT")
# I/O
parser.add_argument("--out_dir", type=str, default="out", help="The output dir.")
parser.add_argument("--eval_interval", type=int, default=1000, help="Eval interval")
parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
parser.add_argument("--eval_iters", type=int, default=200, help="Eval iterations")
parser.add_argument("--eval_only", action="store_true", help="If True, script exits right after the first eval.")
parser.add_argument("--always_save_checkpoint", action="store_true", help="If True, always save a checkpoint after each eval.")
parser.add_argument("--init_from", type=str, default="scratch", help="If the training should continue from a checkpoint folder.",
                    choices=["scratch", "resume", "gpt2*", "init-mean-std-only", "resume-model-only"])
# wandb logging
parser.add_argument("--wandb_log", action="store_true", help="If True, use wandb log.")
parser.add_argument("--wandb_project", type=str, default="owt", help="wandb project.")
parser.add_argument("--wandb_run_name", type=str, default="gpt2", help="wandb project.")
# data related
parser.add_argument("--dataset", type=str, default="openwebtext", help="Dataset.")
# train related
parser.add_argument("--max_iters", type=int, default=600000, help="total number of training iterations.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=40, help="Used to simulate larger batch sizes.")
parser.add_argument("--batch_size", type=int, default=12, help="If gradient_accumulation_steps > 1, this is the micro-batch size")
parser.add_argument("--block_size", type=int, default=1024, help="Block size (sequence length).")
# model related
parser.add_argument("--tied", action="store_true", help="If True, weight tying will be used.")
parser.add_argument("--n_layer", type=int, default=12, help="Layer Num.")
parser.add_argument("--n_head", type=int, default=12, help="Head Num.")
parser.add_argument("--n_embd", type=int, default=768, help="Embedding Dim.")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout ratio. For pretraining 0 is good, for finetuning try 0.1+.")
parser.add_argument("--bias", action="store_true", help="Do we use bias inside LayerNorm and Linear layers?")
# optimizer related
parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer.")
parser.add_argument("--lr", "--learning_rate", type=float, default=6e-4, dest="learning_rate", help="Learning rate.")
parser.add_argument("--wd", "--weight_decay", type=float, default=0.1, dest="weight_decay", help="Weight Decay.")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta1.")
parser.add_argument("--beta2", type=float, default=0.95, help="Beta2.")
parser.add_argument("--eps", type=float, default=1e-8, help="epsilon.")
parser.add_argument("--grad_clip", type=float, default=1.0, help="clip gradients at this value, or disable if == 0.0.")
parser.add_argument("--update_clip", type=float, default=None, help="update clip.")
parser.add_argument("--win", action="store_true", help="The parameter for agd optimizer.")
# learning rate scheduler
parser.add_argument("--warmup_iters", type=int, default=2000, help="How many steps to warm up for.")
parser.add_argument("--lr_decay_iters", type=int, default=600000, help="Should be ~= max_iters per Chinchilla.")
parser.add_argument("--min_lr", type=float, default=None, help="Minimum learning rate. Default to be ~= learning_rate/10 per Chinchilla.")
# system
parser.add_argument("--backend", type=str, default="nccl", help="Backend. 'nccl', 'gloo', etc.")
parser.add_argument("--device", type=str, default="cuda", help="Examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks.")
parser.add_argument("--dtype", type=str, default="float16", help="'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler")
parser.add_argument("--compile", action="store_true", help="Use PyTorch 2.0 to compile the model to be faster")
# mup related
parser.add_argument("--load_base_shapes", type=str, default=None, help="Load the base shape file to apply mup.")
parser.add_argument("--scaled_wd", action="store_true", help="If True, weight decay will be scaled in mup.")
parser.add_argument("--base_dk", type=float, default=64.0, help="A parameter to control the attn_mult in mup. Usually set to base_shape/n_head.")
# logging
parser.add_argument("--log_file", type=str, default="None", help="log file name")

args = parser.parse_args()

if args.min_lr is None:
    args.min_lr = args.learning_rate / 10
if args.log_file.lower() == "none":
    args.log_file = None

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=args.log_file, level=logging.INFO, format=LOG_FORMAT)
logging.captureWarnings(True)

config = {arg: getattr(args, arg) for arg in vars(args)}

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert args.gradient_accumulation_steps % ddp_world_size == 0
    args.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    for k, v in config.items():
        logging.info(f"{k}: {v}")
    os.makedirs(args.out_dir, exist_ok=True)
torch.distributed.barrier()

tokens_per_iter = args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.block_size
logging.info(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data_', args.dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    logging.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                  bias=args.bias, vocab_size=None, dropout=args.dropout) # start with model_args from command line
if args.init_from == 'scratch':
    # init a new model from scratch
    logging.info("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        logging.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, load_base_shapes=args.load_base_shapes, tied=args.tied, base_dk=args.base_dk)
    # mup related
    logging.info(f'standard parameterization')
    model.apply(model._init_weights)
elif args.init_from.startswith('resume'):
    logging.info(f"Resuming training from {args.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, load_base_shapes=args.load_base_shapes, tied=args.tied, base_dk=args.base_dk)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if args.init_from == 'resume':
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
elif args.init_from == 'init-mean-std-only':
    logging.info(f"Initializing a new model from {args.out_dir}")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        logging.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, load_base_shapes=args.load_base_shapes, tied=args.tied, base_dk=args.base_dk)
    # extract mean and std from a checkpoint.
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    init_mean_std_dict = {}
    for key, v in state_dict.items():
        init_mean_std_dict[key] = (v.mean().item(), v.std().item())
    logging.info(f'standard parameterization')
    model.init_weights_by_mean_and_std(init_mean_std_dict)
elif args.init_from.startswith('gpt2'):
    logging.info(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=args.dropout)
    model = GPT.from_pretrained(args.init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if args.block_size < model.config.block_size:
    model.crop_block_size(args.block_size)
    model_args['block_size'] = args.block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    args.weight_decay,
    args.learning_rate,
    (args.beta1, args.beta2),
    device_type,
    args.win,
    opt_name=args.optimizer,
    eps=args.eps,
    update_clip=args.update_clip,
    scaled_wd=args.scaled_wd
)
if args.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if args.compile:
    logging.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return 1.0 * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr / args.learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return (args.min_lr + coeff * (args.learning_rate - args.min_lr)) / args.learning_rate

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

# logging
if args.wandb_log and master_process:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config, mode="offline")

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % args.eval_interval == 0 and master_process:
        losses = estimate_loss()
        logging.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if args.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or args.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                logging.info(f"saving checkpoint to {args.out_dir}")
                torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
    if iter_num == 0 and args.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(args.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % args.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * args.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        logging.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, lr {scheduler.get_last_lr()}")
    iter_num += 1
    local_iter_num += 1
    scheduler.step()

    # termination conditions
    if iter_num > args.max_iters:
        break

if ddp:
    destroy_process_group()
