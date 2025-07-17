import torch
from torch.autograd import Function
import lynx.models.cuda_extension

class MyFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = lynx.models.cuda_extension.forward_cuda(input)
        return output

    # @staticmethod
    # def backward(ctx, grad_output):
    #     input, = ctx.saved_tensors
    #     grad_input = lynx.models.cuda_extension.backward_cuda(grad_output, input)
    #     return grad_input
if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--ptx", type=str, default="custom_call.ptx")
    args.add_argument("--func", type=str, default="wrapped_convert")
    args = args.parse_args()
    torch.cuda.set_device(0)
    print("load ptx finish")
    manager = lynx.models.cuda_extension.PTXLoader(args.ptx,
               args.func.split(","))
    tensor = torch.randn(8,768,1024,device="cuda")
    tensor2 = torch.randn(4096,1024,device="cuda")
    output = torch.zeros_like(tensor,device="cuda",dtype=torch.bfloat16)
    output2 = torch.zeros_like(tensor2,device="cuda",dtype=torch.bfloat16)

    manager.forward_cuda("wrapped_convert",[tensor],[output],128,1248)
    manager.forward_cuda("wrapped_convert_2",[tensor2],[output2],128,1248)
    #print("output",output)
    torch.testing.assert_close(output, tensor.to(torch.bfloat16))
    torch.testing.assert_close(output2, tensor2.to(torch.bfloat16))
    params = [
        torch.randn(6144,4096,device="cuda",dtype=torch.float32),
        torch.randn(6144,4096,device="cuda",dtype=torch.float32),
        torch.randn(6144,4096,device="cuda",dtype=torch.float32),
    ]
    outputs = [
        torch.zeros(8,768,4096,device="cuda",dtype=torch.float32),
        torch.zeros(8,768,4096,device="cuda",dtype=torch.float32),
        torch.zeros(8,768,4096,device="cuda",dtype=torch.float32),
    ]
    print("loop_multiply_fusion")
    # why xla convert to fp16
    # Launching gemm_fusion_dot_68_0
    # Arg: alloc #3, offset: 0: 0x7efbec800200 (12582912B) #(8,768,1024)
    # Arg: alloc #8, offset: 251658752: 0x7efbfc800400 (50331648B) #(8,768,4096)
    # Arg: alloc #1, offset: 0: 0x7efbea800200 (16777216B)
    # dims: blocks: {512, 1, 1}, threads/block: {128, 1, 1}
    
    # kernel = loop_multiply_fusion, launch dimensions = blocks: {49152, 1, 1}, threads/block: {128, 1, 1}
    manager.forward_cuda("loop_multiply_fusion",params,outputs,128,49152)
    for tensor in outputs:
        assert not torch.allclose(tensor, torch.zeros_like(tensor))
    # assert not torch.allclose(output, torch.zeros_like(output))
    # torch.testing.assert_close(output, pytorch_out)
    input1 = torch.randn(6144,1024, device="cuda",dtype=torch.float32)
    input2 = torch.randn(4096,1024, device="cuda",dtype=torch.float32)
    output = torch.zeros(6144,4096, device="cuda",dtype=torch.float32)
    manager.run_gemm(output,input1,input2,1,1)
    pytorch_out = input1@input2
    torch.testing.assert_close(output, pytorch_out)
