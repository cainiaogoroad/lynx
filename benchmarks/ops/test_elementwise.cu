#include "elementwise.cuh"
#include <vector>
#include <numeric>
#include <functional>
#include <utility>
#include <stdio.h>
#include <iostream>

#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <cmath>
// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <curand_kernel.h>
#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32
template<typename T>
struct CastFP2HalfFunctor {
  __device__ half operator()(T x) const {
    // const T zero_val = ; 
    return __float2half(x);
  }
};
//PER_THREAD_DATA: how many elements are processed by each thread
template <uint PER_THREAD_DATA>
__global__ void castfp2half(float *d_in, half *d_out){
    
      uint16_t block_offset = blockIdx.x * blockDim.x;
      int idx = block_offset + threadIdx.x;
      float2 d_in_float2=make_float2(d_in[idx],d_in[idx+1]);
      float2 d_in_float2_1=make_float2(d_in[idx+2],d_in[idx+3]);
      half2 out_half = __float22half2_rn(d_in_float2);
      half2 out_half_1 = __float22half2_rn(d_in_float2_1);
      d_out[idx] = out_half.x;
      d_out[idx+1] = out_half.y;
      d_out[idx+2] = out_half_1.x;
      d_out[idx+3] = out_half_1.y;
      // d_out[idx] = __float2half(d_in[idx]);
      // d_out[idx+1] = __float2half(d_in[idx+1]);
      // d_out[idx+2] = __float2half(d_in[idx+2]);
      // d_out[idx+3] = __float2half(d_in[idx+3]);
      printf("blockIdx.x=%d blockDim.x=%d threadIdx.x=%d idx=%d input=%.3f output=%.3f\n",blockIdx.x,
          blockDim.x,
          threadIdx.x,
          idx,d_in[idx], d_out[idx]);
      __syncwarp();
}

bool half_allclose(float x, half y,float atol=1e-5,float rtol=1e-3)
{
    float y_float = static_cast<float>(y);
    float abs_diff = std::abs(x - y_float);
    return abs_diff <= atol + rtol * std::abs(y_float);
}

inline bool half_almost_equal(float a,half b){
  return std::abs(a-static_cast<float>(b))<=std::numeric_limits<float>::epsilon();

}
bool cpu_validate(float* input,half* output, int N){
  int invalid_number = 0;
  for(int i=0;i<N;i++){
    
    if(!half_allclose(input[i],output[i])){
      printf("i=%d,input=%.3f,output=%.3f\n",i,input[i],static_cast<float>(output[i]));
      invalid_number++;
    }
  }
  if(invalid_number>0){
    printf("invalid_number=%d,rate=%.3f\n",invalid_number,invalid_number*1.0/N);
    return false;
  }
  return true;
}
using namespace std;
using namespace oneflow;
int main(int argc, char **argv) {
    
    // random a tensor 
    int number;
    if(argc!=2){
        printf("usage:./elementwise <number>\n");
        number = 1;
    }
    else{
        number = atoi(argv[1]);
    }
    const int N = 64; // 64M*4B=256MB
    float *gpu_data;
    
    uint bytes = N*sizeof(float);
    
    half *gpu_out;
    // cudaMalloc((void **)&gpu_out, (block_num) * sizeof(float));

    cudaMallocManaged(&gpu_data, N * sizeof(float));
    cudaMallocManaged(&gpu_out, N * sizeof(half));

    // make random data using seed
    std::normal_distribution<float> d{0, 2};
    std::mt19937 gen {1241};

    for(int i=0;i<N;i++){
          gpu_data[i] = d(gen);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    uint repeattimes = 1;
    uint warmtime=1;
    cudaStream_t defaultStream = 0;
    const uint PER_THREAD_DATA = 4;
    // const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    for (size_t i = 0; i < warmtime; i++)
    {
        /* code */
        castfp2half<PER_THREAD_DATA><<<N/WARP_SIZE/PER_THREAD_DATA, WARP_SIZE>>>(gpu_data, gpu_out);
        
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (size_t i = 0; i < repeattimes; i++)
    {
        /* code */
        switch(number){
          case 1:{
            castfp2half<PER_THREAD_DATA><<<N/WARP_SIZE, WARP_SIZE>>>(gpu_data, gpu_out);
            break;
          }
          case 2:{
            cuda::elementwise::Unary(CastFP2HalfFunctor<float>(), N,gpu_out, gpu_data, defaultStream);
          }
        }
        
        
    }
    auto err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    auto speed = bytes*1e-6/(1e-3*milliseconds/repeattimes);
    
    printf("gpu time=%.3f ms\n",milliseconds/repeattimes);
    printf("gpu speed=%.3f MB/s\n",speed);

    cudaDeviceSynchronize();
    if(cpu_validate(gpu_data,gpu_out,N)){
      printf("result success");
    }
    cudaFree(gpu_data);
    cudaFree(gpu_out);
    return 0;
}