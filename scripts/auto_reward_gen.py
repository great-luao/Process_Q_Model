#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# === Configuration (edit these values as needed) ===
BACKBONE_PATH = "/storage/group/renkan/luao/pretrain/Qwen2.5-3B-Instruct"
# List only the specific model folders you want to process
MODEL_DIRS = [
    # "/storage/group/renkan/luao/PQM/con_qwen3B",
    "/storage/group/renkan/luao/PQM/rank_qwen3B",
    "/storage/group/renkan/luao/PQM/orm_qwen3B",
    "/storage/group/renkan/luao/PQM/mse_qwen3B",
    "/storage/group/renkan/luao/PQM/bce_qwen3B",
    # add more paths here...
]
DATA_NAME    = "math"
DATA_FILE    = "bon_test_set/math-metamath-mistral-128.json"
CUDA_DEVICES = "0,1,2,3"
# ==================================================

def find_checkpoints(model_dir: Path):
    """Return sorted list of checkpoint-* subdirectories under a model folder."""
    return sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-"))

def gen_reward_for_checkpoint(ckpt_path: Path):
    """Invoke deepspeed gen_reward.py for a single checkpoint."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES

    cmd = [
        "deepspeed", "gen_reward.py",
        "--backbone-path", BACKBONE_PATH,
        "--model-path", str(ckpt_path),
        "--data-name", DATA_NAME,
        "--data-file", DATA_FILE,
    ]

    print(f"[INFO] Generating rewards for {ckpt_path} ...")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed on {ckpt_path}: {e}")

def main():
    for model_path in MODEL_DIRS:
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            print(f"[ERROR] Model directory not found: {model_dir}, skipping")
            continue

        checkpoints = find_checkpoints(model_dir)
        if not checkpoints:
            print(f"[WARN] No checkpoints in {model_dir}, skipping")
            continue

        # generate reward for each checkpoint in this model folder
        for ckpt in checkpoints:
            gen_reward_for_checkpoint(ckpt)

if __name__ == "__main__":
    main()