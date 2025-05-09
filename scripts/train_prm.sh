# python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --num_cpu_threads_per_process=6 train_main.py \
#                                                                                             --loss-type='rank' \
#                                                                                             --logger='none' \

export WANDB_PROJECT=PQM
export TMPDIR="/storage/group/renkan/luao/tmp2"
export TOKENIZERS_PARALLELISM=False

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --config_file=accelerate_configs/ddp_4.yaml --num_cpu_threads_per_process=6 --main_process_port=29600 train_main.py \
                                                                                            --loss-type='mse' \
                                                                                            --logger='wandb' \
                                                                                            --run-name='mse_qwen3B' \
                                                                                            --model-path='/storage/group/renkan/luao/pretrain/Qwen2.5-3B-Instruct' \
                                                                                            --seed=1777 \
