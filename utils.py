from typing import List, Dict, Tuple
import random
import os

def rm_instruction_format(s):
    # return f'[INST] {s} [/INST]'
    return f'{s}'

def split_queries(queries: List[Dict], test_size: float = 0.2, seed: int = None) -> Tuple[List[Dict], List[Dict]]:
    """
    按比例随机分配 queries 列表为训练集和测试集。
    
    Args:
        queries (List[Dict]): 包含 {"query": str, "answer": str, "labels": any} 的字典列表。
        test_size (float): 测试集比例（0 到 1），默认 0.2。
        seed (int): 随机种子，默认 None。
    
    Returns:
        Tuple[List[Dict], List[Dict]]: 训练集和测试集的 queries 列表。
    """
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
    
    # 复制 queries 并打乱
    shuffled_queries = queries.copy()
    random.shuffle(shuffled_queries)
    
    # 计算分割点
    total = len(queries)
    test_count = int(total * test_size)
    train_count = total - test_count
    
    # 划分训练集和测试集
    train_queries = shuffled_queries[:train_count]
    test_queries = shuffled_queries[train_count:]
    
    return train_queries, test_queries