CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed gen_reward.py \
                                        --backbone-path '/storage/group/renkan/luao/pretrain/Qwen2.5-3B-Instruct' \
                                        --model-path '/storage/group/renkan/luao/PQM/con_qwen3B/checkpoint-199' \
                                        --data-name 'math' \
                                        --data-file 'bon_test_set/math-metamath-mistral-128.json' \
