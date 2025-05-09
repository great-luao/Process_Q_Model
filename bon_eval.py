import pandas as pd
import numpy as np
import json
import random
from datasets import load_dataset, Dataset
import torch
from bon_eval_utils import eval_gsm8k, eval_math_prm
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
from copy import deepcopy

LOSS_TYPE : str = None

def split_query(completions, n, N=16): # extract top-n logprob completion for each query
    splitted_completions = []
    for idx in range(int(len(completions) / N)):
        samples = [sample for sample in completions if sample["idx"] == idx]
        samples = sorted(samples, key=lambda x: x["logprobs"], reverse=True)
        splitted_completions.append(samples[:n])
    return splitted_completions

def best_of_n(splitted_completions, type:str):
    selected_completions = []
    if type == 'min':
        for n_completions_per_query in splitted_completions:
            n_completions_per_query = sorted(n_completions_per_query, key=lambda x: min(x["step_reward"]), reverse=True)
            assert all([min(n_completions_per_query[0]["step_reward"]) >= min(completion["step_reward"]) for completion in n_completions_per_query])
            selected_completions.append(n_completions_per_query[0])
    elif type == 'last':
        for n_completions_per_query in splitted_completions:
            n_completions_per_query = sorted(n_completions_per_query, key=lambda x: x["step_reward"][-1], reverse=True)
            assert all([n_completions_per_query[0]["step_reward"][-1] >= completion["step_reward"][-1] for completion in n_completions_per_query])
            selected_completions.append(n_completions_per_query[0])
    elif type == 'max':
        for n_completions_per_query in splitted_completions:
            n_completions_per_query = sorted(n_completions_per_query, key=lambda x: max(x["step_reward"]), reverse=True)
            assert all([max(n_completions_per_query[0]["step_reward"]) >= max(completion["step_reward"]) for completion in n_completions_per_query])
            selected_completions.append(n_completions_per_query[0])
    elif type == 'con':
        for n_completions_per_query in splitted_completions:
            # TODO: Compute the series probability of the step_reward
            for completion in n_completions_per_query:
                probs = torch.tensor(completion["step_reward"])
                # probs = torch.sigmoid(logits)
                log_U = torch.log(1 - probs + 1e-10).sum(dim=0)  # (steps)
                c_H = 1 - torch.exp(log_U)
                completion["reward"] = c_H.item()  # Predicted probabilities (success probability)

            n_completions_per_query = sorted(n_completions_per_query, key=lambda x: x["reward"], reverse=True)
            assert all([n_completions_per_query[0]["reward"] >= completion["reward"] for completion in n_completions_per_query])
            selected_completions.append(n_completions_per_query[0])
    else:
        raise ValueError(f"Unknown type: {type}")
    return selected_completions

def compute_metrics(dataset_name, scored_results):
    metrics = {}
    # sample_nums = [1, 8, 16, 32, 64, 128]
    sample_nums = [1, 8, 16]

    if dataset_name == 'gsm8k':
        original_dataset = load_dataset('qintongli/GSM-Plus')['testmini']
    else:
        path = '/storage/group/renkan/luao/original_datasets/mathQA-datasets/math500/test.jsonl'
        with open(path) as f:
            original_dataset = [json.loads(line) for line in f]

    for n in sample_nums:
        results = deepcopy(scored_results)
        splitted_completions = split_query(results, n)
        
        selected_completions = best_of_n(splitted_completions, type="last")
        # print("Length of selected_completions: ", len(selected_completions))
        # print("Length of original_dataset: ", len(original_dataset))
        assert len(original_dataset) == len(selected_completions)
        assert dataset_name == 'math'
        acc, _, _ = eval_math_prm([{'response': query['response']} for query in selected_completions],
                                    all_problems=[{'solution': data['solution'], 'question': data['problem']} for
                                                data in original_dataset], is_extract=False)
        metrics[n] = acc
        print('*********')
        print(n, acc)
        print('*********')
        # else:
        #     selected_completions = []
        #     for comps in splitted_completions:
        #         selected_completions += comps
        #     if dataset_name == 'math':
        #         # acc, _, _ = eval_math_prm([{'response':query['response']} for query in selected_completions],all_problems=[{'solution':data['question']['ground_truth_answer'],'question':data['question']['problem']} for data in original_dataset],is_extract=True)
        #         acc, acc_list, output_list = eval_math_prm([{'response': query['response']} for query in selected_completions],
        #                                   all_problems=[{'solution': data['solution'], 'question': data['problem']} for
        #                                                 data in original_dataset for _ in range(n)], is_extract=False)
        #     else:
        #         acc, acc_list, output_list = eval_gsm8k([{'response': query['response']} for query in selected_completions],
        #                                answers=[data['answer'] for data in original_dataset for _ in range(n)],is_extract=True)
        #     total_index = int(len(acc_list) / n)
        #     if args.baseline:
        #         pass_k = sum([1 for ii in range(total_index) if True in acc_list[ii*n:(ii+1)*n]])/total_index
        #         consistent_outputs = [Counter(output_list[ii*n:(ii+1)*n]).most_common(1)[0][0] for ii in range(total_index)]  # (num_instructions, )
        #         position_of_consistent_outputs = [output_list[ii*n:(ii+1)*n].index(consistent_outputs[ii]) for ii in range(total_index)]  # (num_instructions, )
        #         acc_of_consistency = [acc_list[ii*n:(ii+1)*n][idx_of_split] for ii, idx_of_split in enumerate(position_of_consistent_outputs)]
        #         sc = sum(acc_of_consistency)/total_index
        #         print('*********')
        #         print(n,pass_k,sc)
        #         print('*********')
        #     else:
        #         correct,sumv = 0,0
        #         for ii in range(total_index):
        #             answer_dict = {k:0 for k in set(output_list[ii*n:(ii+1)*n])}
        #             reward_list = [ele['reward'] for ele in selected_completions[ii*n:(ii+1)*n]]
        #             for ele,reward in zip(output_list[ii*n:(ii+1)*n],reward_list):
        #                 answer_dict[ele]+=torch.sigmoid(torch.tensor(reward)).item()
        #             select_answer = sorted(answer_dict.items(),key=lambda x:x[1],reverse=True)[0][0]
        #             correct += acc_list[ii*n:(ii+1)*n][output_list[ii * n:(ii + 1) * n].index(select_answer)]
        #             sumv+=1
        #         print('*********')
        #         print(n,correct/sumv)
        #         print('*********')
    return metrics

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--model-path", type=str, default="/storage/group/renkan/luao/pretrain/PQM/zeta-4/model.safetensors")
    parser.add_argument("--data-name", type=str,choices=['math','gsm8k'])
    parser.add_argument("--reward-file", type=str,default="./bon_result/con-prm-data.json")

    args = parser.parse_args()
    print(args)

    loss_choices = ['con', 'rank', 'orm', 'mse', 'bce']
    for loss in loss_choices:
        # Detect if the loss name in the reward file
        if loss in args.reward_file:
            LOSS_TYPE = loss
            break

    if LOSS_TYPE is None:
        raise ValueError("No valid loss type found in the reward file name.")
    print(f"Using loss type: {LOSS_TYPE}")

    queries = json.load(open(args.reward_file))
    compute_metrics(args.data_name, queries)