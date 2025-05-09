import pandas as pd
import numpy as np
import json
import random
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from value_model import AutoModelForCausalLMWithValueHead
import torch
from tqdm import tqdm
import re
import deepspeed
from copy import deepcopy
from safetensors import safe_open
from collections import Counter
from utils import rm_instruction_format
import os

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--loss-type", type=str, default='con',
    #                     choices=['con', 'rank', 'orm', 'mse', 'bce'])
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--backbone-path", type=str, default="/storage/group/renkan/luao/pretrain/deepseek-math-7b-base")
    parser.add_argument("--model-path", type=str, default="/storage/group/renkan/luao/PQM/orm/checkpoint-596")
    parser.add_argument("--data-name", type=str,choices=['math','gsm8k'])
    parser.add_argument("--data-file", type=str,required=True)

    args = parser.parse_args()
    print(args)

    seed_everything(0)
    accelerator = Accelerator()
    data_name = args.data_name

    backbone_path = args.backbone_path
    model = AutoModelForCausalLM.from_pretrained(backbone_path,
                                                    torch_dtype=torch.bfloat16)
    # tokenizer.add_special_tokens({'additional_special_tokens':[prm_token]})
   
    prm_token = '[PRM]'
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    prm_token_id = tokenizer.encode(prm_token, add_special_tokens=False)[-1]

    model_info = args.model_path.split('/')
    model_type = model_info[-2]
    checkpoint = model_info[-1]
    loss_choices = ['con', 'rank', 'orm', 'mse', 'bce']
    if 'con' in model_type:
        loss_type = 'con'
    elif 'rank' in model_type:
        loss_type = 'rank'
    elif 'orm' in model_type:
        loss_type = 'orm'
    elif 'mse' in model_type:
        loss_type = 'mse'
    elif 'bce' in model_type:
        loss_type = 'bce'
    else:
        raise ValueError('loss type not found')

    model.resize_token_embeddings(len(tokenizer))
    model = AutoModelForCausalLMWithValueHead(model, loss_type=loss_type)
    
    print('loading model weights from', args.model_path)
    if '.safetensor' in args.model_path:
        state_dict = {}
        with safe_open(args.model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(args.model_path+"/pytorch_model.bin", weights_only=True)
    model.load_state_dict(state_dict)

    print("Init deepspeed engine")
    deepspeed_config = json.load(open('accelerate_configs/deepspeed_3.json'))
    ds_engine = deepspeed.init_inference(model,
                                            tensor_parallel={"tp_size": 1},
                                            dtype=torch.bfloat16)

    model = ds_engine.module
    model.eval()

    def data_collator(example, tokenizer=tokenizer):
        inputs = []
        special_ids = []
        step_labels = []
        orm_ids = []
        idx,reward_idx = [],[]
        template = '{query}\n{answer}'
        for d in example:
            input_ids = tokenizer.encode(template.format(query=d['query'],answer=d['answer']),
                                            add_special_tokens=False)
            inputs.append(torch.tensor(input_ids))

            cur_special_ids = []
            for ii,id in enumerate(input_ids):
                if id==prm_token_id:
                    cur_special_ids.append(ii)
            # assert len(cur_special_ids)==len(d['labels'])
            special_ids.append(torch.tensor(cur_special_ids))
            orm_ids.append(cur_special_ids[-1])
            # step_labels.append(torch.tensor(d['labels']))
            idx.append(d['idx'])
            reward_idx.append(d['reward_idx'])

        inputs = pad_sequence(inputs, padding_value=tokenizer.pad_token_id, batch_first=True)
        attention_mask = (inputs!=tokenizer.pad_token_id)
        special_ids = pad_sequence(special_ids, padding_value=-100, batch_first=True)
        # step_labels = pad_sequence(step_labels, padding_value=-100, batch_first=True)

        return {
            'input_ids': inputs.int().to(accelerator.device),
            'attention_mask': attention_mask.int().to(accelerator.device),
            'special_tokens':special_ids.to(accelerator.device),
            'orm_tokens': torch.tensor(orm_ids).to(accelerator.device),
            'idx':torch.tensor(idx).to(accelerator.device),
            'reward_idx':torch.tensor(reward_idx).to(accelerator.device)
        }

    if data_name == 'gsm8k':
        file_list = [
            args.data_file,
        ]
        queries = []
        cur_queries = []
        origin_dataset = load_dataset('qintongli/GSM-Plus')['testmini']
        for file_name in file_list:
            cur_data = json.load(open(file_name))
            if len(cur_queries) == len(cur_data):
                for cur_q, cur_d in zip(cur_queries, cur_data):
                    cur_q['responses'].extend(cur_d['responses'])
            else:
                cur_queries = deepcopy(cur_data)

        assert len(origin_dataset) == len(cur_queries), (len(origin_dataset), len(queries))
        for idx, (data, ori) in enumerate(zip(cur_queries, origin_dataset)):
            assert data['question'] == ori['question']
            assert len(data['responses']) == 128
            for response_dict in data['responses']:
                queries.append({
                    'idx': idx,
                    'prompt': data['question'],
                    'response': response_dict['text'],
                    'solution': ori['answer'],
                    'logprobs': 0,
                })
    elif data_name == 'math':
        file_list = [
            args.data_file,
        ]
        queries = []
        cur_queries = []
        path = '/storage/group/renkan/luao/original_datasets/mathQA-datasets/math500/test.jsonl'
        with open(path) as f:
            origin_dataset = [json.loads(line) for line in f]
        for file_name in file_list:
            cur_data = json.load(open(file_name))
            if len(cur_queries) == len(cur_data):
                for cur_q, cur_d in zip(cur_queries, cur_data):
                    cur_q['responses'].extend(cur_d['responses'])
            else:
                cur_queries = deepcopy(cur_data)

        assert len(origin_dataset) == len(cur_queries), (len(origin_dataset), len(queries))
        for idx, (data, ori) in enumerate(zip(cur_queries, origin_dataset)):
            assert data['question'] == ori['problem']
            # assert len(data['responses']) % 128==0
            for response_dict in data['responses']:
                queries.append({
                    'idx': idx,
                    'prompt': data['question'],
                    'response': response_dict['text'],
                    'solution': ori['solution'],
                    'logprobs': 0,
                })

    for idx, data in enumerate(queries):
        data['reward_idx'] = idx
        data["query"] = rm_instruction_format(data["prompt"])
        steps = re.split('Step \d+:', data['response'])
        steps = [f'Step {id + 1}: ' + step.strip() for id, step in enumerate(steps) if step.strip()!='']
        data["answer"] = f" {prm_token}\n".join(steps) + f" {prm_token}"
        if(idx <= 1):
            print(data['answer'])


    dataset = Dataset.from_pandas(pd.DataFrame.from_records(queries))
    dataloader = DataLoader(dataset,batch_size=4,shuffle=False,collate_fn=data_collator)

    for i, inputs in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            _, _, rewards = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        cur_index = torch.where(inputs['special_tokens']==-100,0,inputs['special_tokens'])
        if loss_type != 'orm':
            rewards = rewards.gather(dim=-1, index=cur_index)
        else:
            rewards = rewards.gather(dim=-1, index=inputs['orm_tokens'][...,None])
        for step_reward, reward_idx in zip(rewards.tolist(),inputs['reward_idx'].tolist()):
            queries[int(reward_idx)]['step_reward'] = [r for r in step_reward if r!=1e5]
        # Free up the memory for rewards
        if i <= 0:
            # Check the device of the rewards
            print(rewards.device)
        del rewards

    save_file = f"./bon_result/{model_type}/{checkpoint}.json"
    print("Saving the results to", save_file)
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file,'w') as f:
        json.dump(queries,f)





