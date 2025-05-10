#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# === Configuration (edit these values as needed) ===
DATA_NAME    = "math"
REWARD_FILE    = "bon_test_set/math-metamath-mistral-128.json"
REWARD_DIRS = [
    # "reward_result/con_qwen3B_math",
    "reward_result/bce_qwen3B_math",
    # "reward_result/orm_qwen3B_math",
    # "reward_result/rank_qwen3B_math",
    # "reward_result/mse_qwen3B_math",
]
BON_TYPES = {
    'con': 'last',
    'rank': 'min',
    'orm': 'min',
    'mse': 'min',
    'bce': 'min'
} 
# ==================================================

def find_jsons(model_dir: Path):
    """Return sorted list of *.json subdirectories under a model folder."""
    return sorted(p for p in model_dir.iterdir() if p.name.endswith(".json"))

def bon_for_json(model_dir, json_path: Path):
    """Invoke python bon_eval.py for a single json."""
    env = os.environ.copy()

    bon_type : str = None
    for key in BON_TYPES.keys():
        if key in model_dir.name:
            bon_type = BON_TYPES[key]
            break
    if bon_type is None:
        print(model_dir.name)
        print(f"[ERROR] No valid loss type found in the reward dir name: {model_dir.name}")
        return

    cmd = [
        "python", "bon_eval.py",
        "--reward-file", str(json_path),
        "--data-name", DATA_NAME,
        "--bon-type", bon_type, 
    ]

    print(f"[INFO] Generating bon results for {json_path} ...")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed on {json_path}: {e}")

def main():
    for reward_path in REWARD_DIRS:
        reward_dir = Path(reward_path)
        if not reward_dir.is_dir():
            print(f"[ERROR] Model directory not found: {reward_dir}, skipping")
            continue

        json_files = find_jsons(reward_dir)
        if not json_files:
            print(f"[WARN] No checkpoints in {reward_dir}, skipping")
            continue

        # generate reward for each checkpoint in this model folder
        for json in json_files:
            bon_for_json(reward_dir, json)

if __name__ == "__main__":
    main()