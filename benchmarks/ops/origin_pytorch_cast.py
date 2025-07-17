# coding=utf-8
from __future__ import unicode_literals, absolute_import
import torch
from argparse import ArgumentParser
import triton
import triton.language as tl
from contextlib import nullcontext
@triton.jit
def cast_kernel(input_ptr,  # *Pointer* to first input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)
def torch_fn(input_tensor):
    return input_tensor.to(torch.float16)

def triton_fn(input_tensor):
    # Allocate space for the output tensor.
    output_tensor = torch.empty_like(input_tensor, dtype=torch.float16)
    # Define the kernel launch configuration.
    # We use a 1D grid where each block is responsible for processing BLOCK_SIZE elements.
    # We need to ensure that the grid is large enough to cover all the elements in the input tensor.
    # We also need to ensure that the grid is a multiple of the block size so that all the elements are covered.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    BLOCK_SIZE = 1024
    grid = (input_tensor.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel.
    cast_kernel[(grid,)](input_tensor, output_tensor, input_tensor.numel(), BLOCK_SIZE)
    return output_tensor
def main():
    parser = ArgumentParser()
    parser.add_argument("method", choices=["torch", "triton"])
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.method == "torch":
        fn = torch_fn
    else:
        fn = triton_fn
    tensor = torch.randn(64*1024*1024, dtype=torch.float32,device="cuda")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(100):
        fn(tensor)
    start_event.record()
    repeat_times = 100
    if args.trace:
        context = torch.profiler.profile(
        activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
        record_shapes=True,
        with_flops=True,
        with_stack=True,
    ) 
    else:
        context = nullcontext()
    with context as prof:
        for i in range(repeat_times):
            fn(tensor)
    end_event.record()
    torch.cuda.synchronize()
    if args.trace:
        prof.export_chrome_trace("./tensor_to_float16_%s.json"%args.method)
    bytes_number = tensor.element_size() * tensor.numel()
    cost_time = start_event.elapsed_time(end_event)
    print("%s gpu time=%.3fms"%(args.method,cost_time/repeat_times))
    print("gpu throughput=%.3f MB/s"%(bytes_number*1e-6/(cost_time*1e-3/repeat_times)))




if __name__ == '__main__':
    main()