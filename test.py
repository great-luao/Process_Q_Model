
from transformers import AutoModelForCausalLM
from value_model import AutoModelForCausalLMWithValueHead
import torch
model_path = "/storage/group/renkan/luao/pretrain/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",
                                                 use_cache=False)
reward_model = AutoModelForCausalLMWithValueHead(model,
                                                 prob=True)