PRETRAINED_MODEL_DIR=./pretrained_models/Llama-2-7b-hf
DATASET_PATH=./datasets/alpaca/alpaca_data.json
PER_DEVICE_TRAIN_BATCH_SIZE=16
export GPU_NUM_DEVICES=8
export PJRT_DEVICE=CUDA
export WHICH_DEVICE="xla"
export NCCL_DEBUG=TRACE

# open preallocate gpu memory
export XLA_GPU_MEMORY_FRACTION=0.95
export XLA_GPU_MEMORY_PREALLOCATE=true
export XLA_GPU_MEMORY_ALLOCATOR_KIND=3

# set intra and inner devices bw
export XLA_INTERNODE_BW=200
export XLA_INNERNODE_BW=800

export XLA_AUTOREORDER_TIMEOUT=300
export XLA_AUTOREORDER_WORKER=36
# open debug log
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TF_CPP_MIN_LOG_LEVEL=0
#TF_CPP_VMODULE="hlo_pass_pipeline=5,lazy_graph_executor=4,xla_graph_executor=5"
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
    --xla_gpu_enable_analytical_latency_estimator=true \
    --xla_gpu_mlir_emitter_level=0\
    --xla_gpu_enable_linear_program_scheduler=true \
    --xla_cpu_enable_fast_math=false \
    --xla_gpu_force_compilation_parallelism=8  \
    --xla_gpu_enable_pipelined_collectives=true \
    --xla_gpu_enable_pipelined_all_reduce=true \
    --xla_disable_hlo_passes=post-scheduling-passes,gpu-schedule-postprocessing,fusion_wrapper,remat-pipeline
"

python benchmarks/llm/llama_benchmark.py \
    --not_save_model \
    --dataset_path $DATASET_PATH \
	--model_name_or_path $PRETRAINED_MODEL_DIR \
    --num_train_epochs 6 \
    --block_size 512 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
	--total_train_batch_size $((PER_DEVICE_TRAIN_BATCH_SIZE * GPU_NUM_DEVICES)) \
	--per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
	--per_device_eval_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --seed 42 \
    --preprocessing_num_workers 6 \
    --dataloader_num_workers 10 \
    --ignore_mismatched_sizes \
    --ignore_dryrun_on_load_strategy \
    --output_dir ./outputs \
    --random_log_n_training_samples 0 \
    --logging_steps 10 \
    --report_to all --using_xla --distributed_method spmd_fsdp
