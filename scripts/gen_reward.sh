CUDA_VISIBLE_DEVICES=0 deepspeed gen_reward.py \
                                        --backbone-path '/storage/group/renkan/luao/pretrain/Qwen2.5-3B-Instruct' \
                                        --model-path '/storage/group/renkan/luao/PQM/con_qwen3B/checkpoint-663' \
                                        --data-name 'math' \
                                        --data-file 'bon_test_set/math500.json' \
