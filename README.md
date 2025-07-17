# Lynx
Lynx is a distributed deep learning system that supports automatic parallelism. It enables optimization through whole-graph compilation to enhance the utilization of single devices, while also addressing the challenges of large-scale training and distributed scaling.

ww## Build From Source
```shell
# Run from the official image


# Clone the code from the github repository


# Update the submodules with the latest version recursively
# git submodule update --init --recursive

# Install the environment
scripts/build.sh install_environment

# Build PyTorch and install the wheel package
scripts/build.sh install_torch

# Build TorchXLA and install the wheel package
scripts/build.sh install_torch_xla
# Build Lynx Runtime
scripts/build.sh build_lynx_runtime

# Test if the installation is successful
GPU_NUM_DEVICES=2 PJRT_DEVICE=CUDA python benchmarks/test_train_mp_mnist.py --fake_data --num_epochs=1

# Test Lynx runtime binaries and You will see the message:
# I Love Lynx!
bazel-bin/lynx/csrc/runtime/test/test

# Test if the installation is successful
export PJRT_ALLOCATOR_PREALLOCATE=false
export PJRT_ALLOCATOR_FRACTION=0.75
export PJRT_ALLOCATOR_CUDA_ASYNC=false
XLA_FLAGS="--xla_dump_to=/home/log/beit_xla_hlo --xla_dump_hlo_as_html --xla_dump_hlo_as_dot --xla_dump_fusion_visualization" \
GPU_NUM_DEVICES=2 PJRT_DEVICE=CUDA TF_CPP_MAX_VLOG_LEVEL=2 python benchmarks/test_train_mp_mnist.py --fake_data --num_epochs=2 --ddp

# feature: auto reorder
XLA_FLAGS="--xla_gpu_enable_linear_program_scheduler --xla_gpu_enable_analytical_latency_estimator --xla_gpu_enable_xla_runtime_executable" GPU_NUM_DEVICES=8 PJRT_DEVICE=CUDA TF_CPP_MAX_VLOG_LEVEL=2 python benchmarks/test_train_mp_mnist.py --fake_data --num_epochs=1
# for compare, analytical_latency_estimator
XLA_FLAGS=--xla_gpu_enable_analytical_latency_estimator GPU_NUM_DEVICES=2 PJRT_DEVICE=CUDA TF_CPP_MAX_VLOG_LEVEL=2 python benchmarks/test_train_mp_mnist.py --fake_data --num_epochs=1
```
## Development

After modify openxla code, you should want torch_xla using new openxla code, you can go to torch_xla/WORKSPACE do:

```
# For development, one often wants to make changes to the OpenXLA repository as well
# as the PyTorch/XLA repository. You can override the pinned repository above with a
# local checkout by either:
# a) overriding the OpenXLA repository on the build.py command line by passing a flag
#    like:
#    bazel --override_repository=xla=/path/to/openxla
#    or
# b) by commenting out the http_archive above and uncommenting the following:
local_repository(
   name = "xla",
   path = "/path/to/openxla",
)
```


# Tools

## subgraph dot tools
```
# use this tool to pick subgraph to dot file,then draw graphviz
python /root/lynx/lynx/tools/pick_subgraph.py cluster_3869814064 module_0205.SyncTensorsGraph.33142.sm_8.0_gpu_after_optimizations.dot loop_convert2.dot
found subgraph: cluster_3869814064
subgraph closure: cluster_3869814064
found 2583 subgraphs, and 58 edges belong to subgraph:cluster_3869814064

dot -Tpng loop_convert2.dot -o loop_convert.png

```


# Build and Run third test

## torch_xla
```
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE="auto_reorder=5,auto_reorder_solver=5,gpu_collective_performance_model=5,offline_sqlite_pgle=5,gpu_performance_model=5,convert_xplane=5" bazel run --compilation_mode=dbg torch_xla/csrc/runtime:cache_test --incompatible_strict_action_env --action_env=USE_CUDA --action_env=XLA_CUDA --jobs=8
```


## local solve


