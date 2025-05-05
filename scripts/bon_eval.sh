export CUDA_VISIBLE_DEVICES=0 
deepspeed --num_gpus=1 bon_eval_hf.py --data-name 'math' --data-file "bon_test_set/math500.json"