import pandas as pd
import numpy as np
import json
import random
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from value_model import AutoModelForCausalLMWithValueHead
import torch
import os
import re
import multiprocessing
import torch.nn as nn
import argparse
from sklearn.metrics import roc_auc_score
from utils import rm_instruction_format
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import (
    is_sagemaker_mp_enabled,
)
from transformers.trainer_pt_utils import (
    nested_detach,
)

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PRMTrainer(Trainer):
    def __init__(self, model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 processing_class=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None, ):
        super().__init__(model=model,
                         args=args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         processing_class=processing_class,
                         model_init=model_init,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics, )
        self.loss_type = model.config.loss_type
        if self.loss_type == 'nce' or self.loss_type == 'orm':
            self.loss_fn = nn.BCELoss(reduction='none')
        elif self.loss_type=='mse':
            self.loss_fn = nn.MSELoss(reduction='none')

    def ranking_loss(self,rewards,labels,has_neg):
        """
        rewards: 模型在每个token位置预测的值, B*S
        labels: B*S
        has_neg: B, 有neg标签就为1, 没有就是0.
        """
        pos_rewards_exp = torch.where(labels == 1, (rewards).exp(), 0) # Q_c
        neg_rewards_exp = torch.where(labels == 0, (rewards+args.zeta).exp(), 0).flip(dims=[-1]) # Q_w,越靠后的错误reward应该越小
        neg_reward_sum = neg_rewards_exp.sum(-1)

        pos_rewards_cumsum = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), pos_rewards_exp],
                                       dim=1).cumsum(-1)[:, :-1]
        pos_rewards_cumsum = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device), pos_rewards_cumsum],
                                       dim=-1)

        reward_exp_cur = torch.where(labels == 1, pos_rewards_exp, 1)
        reward_exp_cur = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), reward_exp_cur], dim=-1)

        loss = -torch.log(reward_exp_cur / (reward_exp_cur + pos_rewards_cumsum + neg_reward_sum[..., None] + 1e-5))

        labels = torch.cat([has_neg[..., None], labels], dim=-1)
        loss = (torch.where(labels == 1, loss, 0).sum(-1) / torch.where(labels == 1, 1, 0).sum(-1)).mean()
        return loss

    def conditional_loss(self, logits, correctness, return_outputs=False):
        log_U = torch.log(1 - logits + 1e-10).sum(dim=1)
        
        loss_true = -torch.log(1 - torch.exp(log_U) + 1e-10)
        loss_false = -log_U
        
        loss_all = torch.where(correctness.bool(), loss_true, loss_false)
        # print("Loss true:", loss_true, " Loss false:", loss_false)
        # print("Logits shape:", logits.shape)
        # print("Logits", logits[:1])
        # print("Correctness", correctness[:32])
        # print("Losses", loss_all[:10])
        loss = loss_all.mean()

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        _,_,rewards = model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])

        if self.loss_type=='nce':
            rewards = rewards.gather(dim=-1, index=inputs['special_tokens'])
            # rewards = rewards.sigmoid()
            loss = (self.loss_fn(rewards, torch.where(inputs['step_labels']!=-100,inputs['step_labels'],0).bfloat16()) * (inputs['step_labels']!=-100)).sum()/(inputs['step_labels']!=-100).sum()
        elif self.loss_type=='mse':
            rewards = rewards.gather(dim=-1, index=inputs['special_tokens'])
            # rewards = rewards.sigmoid()
            loss = (self.loss_fn(rewards,
                                 torch.where(inputs['step_labels'] != -100, inputs['step_labels'], 0).bfloat16()) * (
                                inputs['step_labels'] != -100)).sum() / (inputs['step_labels'] != -100).sum()
        elif self.loss_type=='orm':
            rewards = rewards.gather(dim=-1, index=inputs['orm_tokens'][...,None])
            # rewards = rewards.sigmoid()
            loss = self.loss_fn(rewards.squeeze(1),1-inputs['has_neg'].bfloat16()).mean()
        elif self.loss_type=='rank':
            rewards = rewards.gather(dim=-1, index=inputs['special_tokens'])
            loss = self.ranking_loss(rewards,inputs['step_labels'],inputs['has_neg'])
        elif self.loss_type=='con':
            rewards = rewards.gather(dim=-1, index=inputs['special_tokens'])
            # print("Special tokens:", inputs['special_tokens'][:1])
            # print("Rewards here:", rewards[:1])
            rewards = torch.where(inputs['special_tokens']!=0,rewards,0).bfloat16()
            # rewards = rewards.sigmoid() We move the sigmoid to the model
            loss = self.conditional_loss(rewards,1-inputs['has_neg'])

        if return_outputs:
            return loss, rewards, 1-inputs['has_neg']
        
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        # has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # # For CLIP-like models capable of returning loss values.
        # # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # # is `True` in `model.forward`.
        # return_loss = inputs.get("return_loss", None)
        # if return_loss is None:
        #     return_loss = self.can_return_loss
        # loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raise NotImplementedError()
            else:
                with self.compute_loss_context_manager():
                    loss, logits, labels = self.compute_loss(model, inputs, return_outputs=True)
                    # print("Loss after compute_loss:", loss)
                loss = loss.mean().detach()

        if self.model.config.loss_type != 'con' or prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        # print("Logits in prediction step:", logits[:1])

        # print("Labels collected in prediction step:", labels[:10])

        return (loss, logits, labels)

def compute_metrics(eval_preds):
    probs, correctness = eval_preds
    
    probs = np.array(probs)
    correctness = np.array(correctness)

    print("Start computing metrics-----------")

    # Replace all -100 in probs with 0
    probs[probs == -100] = 0.

    print("Probs shape:", probs.shape)
    # print("Probs is",probs[:5])

    # with torch.no_grad():        
    log_U = np.log(1 - probs + 1e-10)
    log_U = np.sum(log_U, axis=1)  # (batch_size,)
    c_H = 1 - np.exp(log_U)
    
    loss_true = -np.log(1 - np.exp(log_U) + 1e-10)
    loss_false = -log_U

    # Replace correctness.bool() with an np function
    # loss = np.where(correctness.bool(), loss_true, loss_false)
    loss = np.where(correctness.astype(bool), loss_true, loss_false)
    bce = loss.mean()
    
    # ---- 2) Brier Score ----Brier = mean( (y - p)^2 )
    brier_mse = ((correctness - c_H)**2).mean()

    # AUROC computation
    y_true = correctness  # Ground truth labels (0 or 1)
    y_pred = c_H  # Predicted probabilities (success probability)
    auroc = roc_auc_score(y_true, y_pred)  # Compute AUROC score

    # print("BCE Loss:", bce.item())
    # print("Brier MSE:", brier_mse.item())
    # print("AUROC:", auroc)

    return {
        "nll": bce.item(),
        "brier_mse": brier_mse.item(),
        "auroc": auroc
    }

num_process = multiprocessing.cpu_count() - 4

def generate_dataset(prm_token, tokenizer, max_length=512):
    ds = load_dataset("json", data_dir=args.dataset_path, num_proc=6)['train']
    # 测试用：仅选择前1000行
    # ds = ds.select(range(10000))

    def process_example(example, prm_token, tokenizer, max_length):
        # 分割输入文本
        input_text = example['input']
        steps = re.split(r'Step \d+:', input_text)
        steps = [s for s in steps if s.strip() != '']
        
        # 跳过只有一个步骤的样本
        if len(steps) <= 1:
            return None
        
        # 提取问题和步骤
        question = steps[0]
        steps = [
            f'Step {i + 1}: ' + step.strip().replace('ки', '').strip()
            for i, step in enumerate(steps[1:])
            if step.strip() != ''
        ]
        
        # 处理标签
        label_steps = re.split(r'Step \d+:', example['label'])
        label_steps = [s.strip() for s in label_steps[1:] if s.strip() != '']
        
        # 验证标签以 '+' 或 '-' 结尾
        try:
            for s in label_steps:
                assert s[-1] in ['+', '-'], f"Invalid label format: {label_steps}"
        except AssertionError:
            return None
        
        # 提取步骤标签
        step_labels = [1 if l[-1] == '+' else 0 for l in label_steps]
        
        # 验证步骤和标签数量匹配
        try:
            assert len(steps) == len(step_labels)
        except AssertionError:
            return None
        
        # 构造查询
        query = {
            "query": rm_instruction_format(question), 
            "answer": f" {prm_token}\n".join(steps) + f" {prm_token}",
            "labels": step_labels,
        }
        
        # 计算编码长度
        encoded = tokenizer.encode(query['query'] + query['answer'])
        query_length = len(encoded)
        
        # 分类：普通、较长、最长
        if query_length <= max_length:
            query["length_category"] = "normal"
        else:
            query["length_category"] = "longer"
        
        return query

    processed_ds = ds.map(
        process_example,
        fn_kwargs={
            "prm_token": prm_token,
            "tokenizer": tokenizer,
            "max_length": max_length,
        },
        batched=False, 
        num_proc=6,    
        remove_columns=ds.column_names,
        desc="Processing dataset"
    )

    # 过滤掉无效样本（返回 None 的行）
    processed_ds = processed_ds.filter(lambda x: x is not None, num_proc=6)

    # print("Type of processed_ds:", type(processed_ds))

    # dataset = processed_ds.train_test_split(test_size=0.05, seed=args.seed, shuffle=True)
    # train_dataset = dataset['train']
    # eval_dataset = dataset['test']

    # 分割数据集
    queries = processed_ds.filter(lambda x: x["length_category"] == "normal", num_proc=6)
    longer_queries = processed_ds.filter(lambda x: x["length_category"] == "longer", num_proc=6)

    # Turn these queries into a list of dictionaries
    queries = [{"query": q["query"], "answer": q["answer"], "labels": q["labels"]} for q in queries]
    longer_queries = [{"query": q["query"], "answer": q["answer"], "labels": q["labels"]} for q in longer_queries]

    # 主进程打印调试信息
    if accelerator.is_main_process:
        if len(queries) > 0:
            print(f"Train Data Examples:\n{queries[0]}\n{queries[-1]}")
        print(f"Train Dataset Lengths: Normal={len(queries)}, Longer={len(longer_queries)}")

    # Randomly split each querie into train and eval
    from utils import split_queries
    train_queries, eval_queries = split_queries(queries, test_size=0.005, seed=args.seed)
    # longer_train_queries, longer_eval_queries = split_queries(longer_queries, test_size=0.01, seed=args.seed)

    train_dataset = TrainDataset(train_queries, longer_queries, args.train_batch_size)
    # eval_dataset = EvalDataset(eval_queries, args.eval_batch_size)
    eval_dataset = EvalDataset(eval_queries, args.eval_batch_size)

    # Check if eval dataset has only one kind of label
    eval_labels = [q['labels'][-1] for q in eval_queries]
    if len(set(eval_labels)) < 2: # 0, 1
        print(set(eval_labels))
        raise ValueError("Eval dataset should have all labels, please check your dataset.")

    return train_dataset, eval_dataset

class TrainDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_size=64):
        iter_1_step = batch_size
        iter_2_step = batch_size // 4
        self.iteration_1 = math.ceil(len(dataset1)/iter_1_step)
        self.iteration_2 = math.ceil(len(dataset2)/iter_2_step)
        self.dataset = []
        for i in range(self.iteration_2):
            self.dataset.append(dataset2[i*iter_2_step:(i + 1)*iter_2_step])
        for i in range(self.iteration_1):  # 更多操作在这里完成
            self.dataset.append(dataset1[i*iter_1_step:(i+1)*iter_1_step])

    def __getitem__(self,index):
        return self.dataset[index]

    def __len__(self):
        return self.iteration_1+self.iteration_2

# TODO: This doesn't work well with gather_function, batch size will be reset to 1 after gather.
class EvalDataset(Dataset):
    def __init__(self, dataset, batch_size=64):
        self.iteration = math.ceil(len(dataset)/batch_size)
        self.dataset = []
        for i in range(self.iteration - self.iteration % 4):
            self.dataset.append(dataset[i*batch_size:(i+1)*batch_size])

    def __getitem__(self,index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="/storage/group/renkan/luao/reward_datasets/math-shephered/")
    parser.add_argument("--model-path", type=str, default="/storage/group/renkan/luao/pretrain/deepseek-math-7b-base")
    # parser.add_argument("--save-path", type=str, default="/storage/group/renkan/luao/PQM/orm")
    parser.add_argument("--train-batch-size", type=int, default=56)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="Test")
    parser.add_argument("--zeta", type=int, default=4)
    parser.add_argument("--loss-type", type=str, default='rank',
                        choices=['con', 'rank', 'orm', 'mse', 'bce'])
    parser.add_argument("--logger", type=str, default='none')
    args = parser.parse_args()

    seed_everything(args.seed)
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print(args)
    
    prm_token = '[PRM]'
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens':[prm_token]})
    prm_token_id = tokenizer.encode(prm_token, add_special_tokens=False)[-1]
    train_dataset, eval_dataset= generate_dataset(prm_token, tokenizer)

    if accelerator.is_local_main_process:
        print("Dataset loaded successfully")
        print(f"Dataset length: {len(train_dataset)}")
        print(f"Eval dataset length: {len(eval_dataset)}")

    def data_collator(example, tokenizer=tokenizer):
        inputs = []
        special_ids = []
        step_labels = []
        outcome_labels = []
        orm_tokens,orm_labels = [],[]
        has_neg = []
        template = '{query}\n{answer}'
        example = example[0]
        for d in example:
            input_ids = tokenizer.encode(template.format(query=d['query'],answer=d['answer']),
                                          add_special_tokens=False)
            inputs.append(torch.tensor(input_ids))

            cur_special_ids = []
            for ii,id in enumerate(input_ids):
                if id==prm_token_id:
                    cur_special_ids.append(ii)
            assert len(cur_special_ids)==len(d['labels'])
            special_ids.append(torch.tensor(cur_special_ids))
            step_labels.append(torch.tensor(d['labels']))
            orm_tokens.append(cur_special_ids[-1])
            has_neg.append(1 if 0 in d['labels'] else 0)

        inputs = pad_sequence(inputs, padding_value=tokenizer.pad_token_id, batch_first=True)
        attention_mask = (inputs!=tokenizer.pad_token_id)
        special_ids = pad_sequence(special_ids, padding_value=0, batch_first=True)
        step_labels = pad_sequence(step_labels, padding_value=-100, batch_first=True)


        return {
            'input_ids': inputs.int(),
            'attention_mask': attention_mask.int(),
            'special_tokens':special_ids,
            'step_labels':step_labels,
            'orm_tokens':torch.tensor(orm_tokens),
            'has_neg':torch.tensor(has_neg),
        }

    deepspeed_config = json.load(open('accelerate_configs/deepspeed_3.json'))
    deepspeed_config["scheduler"]["params"] = {
        "warmup_min_lr": 0,
        "warmup_max_lr": 'auto',
        "warmup_num_steps": 'auto',
        "total_num_steps": 'auto',
    }

    # create dir for save_path
    save_path = "/storage/group/renkan/luao/PQM/" + args.run_name
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True if args.run_name != 'TEST' else False)

    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,

        optim="adamw_torch",
        learning_rate=1e-6, # 2e-6 for 8 GPUs
        seed=args.seed,
        lr_scheduler_type="cosine",
        # warmup_steps = 150,
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        num_train_epochs=1,
        gradient_accumulation_steps=4, #4 for 8 GPUs
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_steps=0.01,
        save_strategy="no" if args.run_name == 'TEST' else "steps",
        save_steps=0.3,
        run_name=args.run_name,
        report_to=args.logger,
        remove_unused_columns=False,
        bf16=True,
        fp16_backend="auto",
        save_safetensors=False,
        # group_by_length = True,
        deepspeed=deepspeed_config,
        # sharded_ddp="zero_dp_2",

        eval_strategy="steps",
        eval_steps=0.001,
        prediction_loss_only=True if args.loss_type != 'con' else False,
        # batch_eval_metrics=True,
    )

    ###############
    # Load Model
    ###############
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",
                                                 use_cache=False)
    model.resize_token_embeddings(len(tokenizer))
    reward_model = AutoModelForCausalLMWithValueHead(model, loss_type=args.loss_type)
    if accelerator.is_local_main_process:
        print("Model loaded successfully")

    trainer = PRMTrainer(
        reward_model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.loss_type == 'con' else None,
    )

    ###############
    # Training loop
    ###############
    if accelerator.is_main_process:
        print("*** Train ***")
    trainer.train()

    ###############
    # Evaluation loop
    ###############
    metrics = trainer.evaluate()
    if accelerator.is_main_process:
        print("Initial Evaluation metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
