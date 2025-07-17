#include <torch/extension.h>
#include <cuda.h>
#include <vector>
#include <unordered_map>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda_runtime.h>
using namespace at::native;
using namespace at;
c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(const Tensor& tensor, bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}
enum GEMMAndBiasActivationEpilogue {
  None,
  RELU,
  GELU,
};
c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}
c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, true);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}
struct cublasCommonArgs {
  cublasCommonArgs(const Tensor& mat1, const Tensor& mat2, Tensor& c) {
    bool transpose_result, transpose_mat1, transpose_mat2;
    result = prepare_matrix_for_cublas(c, transpose_result);
    mata = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
    matb = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);
    auto mat1_sizes = mat1.sizes();
    auto mat2_sizes = mat2.sizes();
    if (transpose_result) {
      transpose_mat1 = !transpose_mat1;
      transpose_mat2 = !transpose_mat2;
      mat1_sizes = mata->sizes();
      mat2_sizes = matb->sizes();
    }

    m = mat1_sizes[transpose_result ? 1 : 0];
    k = mat1_sizes[transpose_result ? 0 : 1];
    n = mat2_sizes[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_mat1 == transpose_result) ? 1 : 0);
    ldb = matb->stride((transpose_mat2 == transpose_result) ? 1 : 0);
    result_ld = result->stride(transpose_result ? 0 : 1);
    transa = transpose_mat1 ?  mata->is_conj() ? 'c' : 't' : 'n';
    transb = transpose_mat2 ?  matb->is_conj() ? 'c' : 't' : 'n';
  }
  char transa, transb;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<Tensor> mata, matb, result;
};

class PTXLoader {
    private:
        std::unordered_map<std::string, CUmodule> filename2module;
        std::unordered_map<std::string, CUfunction> kernel_map;
        CUcontext current_context;

    public:
    PTXLoader(std::string ptx_file, std::vector<std::string> kernel_names) {
        load_ptx_module(ptx_file, kernel_names);
    }
    ~PTXLoader() {
        for (auto& [_, module] : filename2module) {
            cuModuleUnload(module);
        }
    }
    void load_ptx_module(std::string ptx_file, std::vector<std::string> kernel_names);

    void forward_cuda(std::string kernel_name, 
            std::vector<torch::Tensor> args,
            std::vector<torch::Tensor> outputs,
            size_t threads,size_t blocks);
    void run_gemm(
                  torch::Tensor lhs,
                  torch::Tensor rhs,
                  torch::Tensor output,
                  int lhs_contracting_dim,
                  int rhs_contracting_dim,
                  );
};
void PTXLoader::load_ptx_module(std::string ptx_file, std::vector<std::string> kernel_names) {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is not available");
    
    CUresult result = cuCtxGetCurrent(&current_context);
    TORCH_CHECK(result == CUDA_SUCCESS,
                "No current CUDA context:"+std::to_string(result));
    if (current_context == nullptr) {
        result = cuInit(0);
        TORCH_CHECK(result == CUDA_SUCCESS, "Failed to initialize CUDA driver");
        
        CUdevice device;
        result = cuDeviceGet(&device, c10::cuda::current_device());
        TORCH_CHECK(result == CUDA_SUCCESS, "Failed to get CUDA device");
        
        result = cuCtxCreate(&current_context, CU_CTX_SCHED_AUTO, device);
        TORCH_CHECK(result == CUDA_SUCCESS, "Failed to create CUDA context");
    }
    result = cuCtxSetCurrent(current_context);
    TORCH_CHECK(result == CUDA_SUCCESS, "Failed to set current context");
    CUmodule module;
    if(filename2module.find(ptx_file) == filename2module.end()) {
        result = cuModuleLoad(&module, ptx_file.c_str());
        filename2module[ptx_file] = module;
        if (result != CUDA_SUCCESS) {
            TORCH_CHECK(false, 
            "Failed to load PTX module: " + ptx_file + ":" + std::to_string(result));
        }
    } else {
        module = filename2module[ptx_file];
    }
   
    for (const auto& kernel_name : kernel_names) {
        CUfunction kernel;
        result = cuModuleGetFunction(&kernel, module, kernel_name.c_str());
        TORCH_CHECK(result == CUDA_SUCCESS, "Failed to get function: " + kernel_name + ":" + std::to_string(result));
        //using python print
        std::cout << "Loaded kernel: " << kernel_name << std::endl;
        kernel_map[kernel_name] = kernel;
    }
}


void PTXLoader::run_gemm(
                  torch::Tensor lhs,
                  torch::Tensor rhs,
                  torch::Tensor output,
                  int lhs_contracting_dim,
                  int rhs_contracting_dim
                  ) {
    TORCH_CHECK(lhs.is_cuda(), "LHS must be a CUDA tensor");
    TORCH_CHECK(rhs.is_cuda(), "RHS must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");
    cublasCommonArgs args(lhs, rhs, output);
    auto scalar_type = output.scalar_type();
    auto activation_epilogue = at::cuda::blas::GEMMAndBiasActivationEpilogue::None;
    // useLtInterface
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "lynx_run_gemm",
        [&] {
          scalar_t alpha = 1.0f;

          at::cuda::blas::gemm_and_bias<scalar_t>(
              args.transa == 't',
              args.transb == 't',
              args.m,
              args.n,
              args.k,
              alpha,
              args.mata->const_data_ptr<scalar_t>(),
              args.lda,
              args.matb->const_data_ptr<scalar_t>(),
              args.ldb,
              nullptr,
              args.result->data_ptr<scalar_t>(),
              args.result_ld,
              activation_epilogue
          );
        });

}
void PTXLoader::forward_cuda(std::string kernel_name, 
        std::vector<torch::Tensor> args,
        std::vector<torch::Tensor> outputs,
        size_t threads,size_t blocks) {
    if(kernel_map.find(kernel_name) == kernel_map.end()) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        TORCH_CHECK(false, "Kernel not found: " + kernel_name);
    }
    CUfunction kernel = kernel_map[kernel_name];
    TORCH_CHECK(current_context != nullptr, "Current context is nullptr");
    CUresult status = cuCtxSetCurrent(current_context);
    TORCH_CHECK(status == CUDA_SUCCESS, "Failed to set current context:"+std::to_string(status));
    std::vector<void*> argument_addresses;
    argument_addresses.reserve(args.size()+outputs.size());
    for(auto input_tensors:args){
        TORCH_CHECK(input_tensors.is_cuda(), "Input must be a CUDA tensor");
        void* input_ptr=input_tensors.data_ptr();
        argument_addresses.push_back(&input_ptr);
    }
    for(auto output_tensors:outputs){
        TORCH_CHECK(output_tensors.is_cuda(), "Output must be a CUDA tensor");
        void* output_ptr=output_tensors.data_ptr();
        argument_addresses.push_back(&output_ptr);
    }
    int smem_size=0;
    CUstream hStream = at::cuda::getCurrentCUDAStream().stream();
    printf("Launch kernel: %s,kernel ptr: %p, blocks: %d, threads: %d, smem_size: %zu\n", 
           kernel_name.c_str(),kernel, blocks, threads, smem_size);
    status = cuLaunchKernel(kernel, 
                           blocks, 1, 1,    // grid dimensions
                           threads, 1, 1,    // block dimensions
                           smem_size,        // shared memory size
                           hStream, 
                           (void**)(argument_addresses.data()), 
                           0);
                           
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error after launch: %s\n", cudaGetErrorString(error));
    }
    TORCH_CHECK(status == CUDA_SUCCESS, "Failed to launch kernel: " + kernel_name+":"+std::to_string(status));

}
class CUDAGraphExtension: public at::cuda::CUDAGraph {
    public:
        void dump(std::string path);
        void restore(std::string path);
        
};
//TODO: using CUDAGraphExtension 
void CUDAGraphExtension::dump(std::string path) {
    std::vector<cudaGraphNode_t> nodes;
    size_t num_nodes;
    auto status = cudaGraphGetNodes(graph_, nodes.data(), &num_nodes);
    TORCH_CHECK(status == CUDA_SUCCESS, "Failed to get nodes:"+std::to_string(status));
    for(int i=0;i<nodes.size();i++) {
        cudaGraphNode_t node = nodes[i];
        printf("Node %s\n",node);
    }
    std::vector<cudaGraphNode_t> froms;
    std::vector<cudaGraphNode_t> tos;
    size_t num_edges;
    cudaGraphGetEdges(graph_, froms.data(), tos.data(), &num_edges);
    for(int i=0;i<num_edges;i++) {
        printf("Edge %d: src=%d, dst=%d\n",i,froms[i],tos[i]);
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    py::class_<PTXLoader>(m, "PTXLoader")
        .def(py::init<std::string, std::vector<std::string>>())
        .def("forward_cuda", &PTXLoader::forward_cuda, "Forward op, forward_cuda(kernel_name,tensor_list_in,tensor_list_out,threads,blocks)")
        .def("run_gemm", &PTXLoader::run_gemm, "Run Gemm run_gemm(output,lhs,rhs,lhs_contracting_dim,rhs_contracting_dim)")
        .def("load_ptx_module",[](PTXLoader& self, std::string ptx_file, std::vector<std::string> kernel_names){
            self.load_ptx_module(ptx_file, kernel_names);
        }, "Load PTX module");
    py::class_<CUDAGraphExtension>(m, "CUDAGraphExtension")
        .def(py::init())
        .def("capture_begin", &CUDAGraphExtension::capture_begin, "Capture begin")
        .def("capture_end", &CUDAGraphExtension::capture_end, "Capture end")
        .def("replay", &CUDAGraphExtension::replay, "Replay")
        .def("reset", &CUDAGraphExtension::reset, "Reset")
        .def("dump", &CUDAGraphExtension::dump, "Dump cudagraph");
}