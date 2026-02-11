"""
Run evaluation on all checkpoints for a given model tag.

Usage:
python -m scripts.eval_all_checkpoints --model-tag <model_tag>
"""
import os
import sys
import glob
import re
import subprocess
import argparse
from nanochat.common import get_base_dir, print0



def main():
    parser = argparse.ArgumentParser(description="Run evaluation on all checkpoints")
    parser.add_argument('--model-tag', type=str, default=None, help='Model tag to evaluate (e.g. d12)')
    args = parser.parse_args()

    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")

    # Resolve model tag
    model_tag = args.model_tag
    if model_tag is None:
        from nanochat.checkpoint_manager import find_largest_model
        try:
            model_tag = find_largest_model(checkpoints_dir)
            print0(f"No model tag provided, defaulting to: {model_tag}")
        except FileNotFoundError:
            print0(f"No checkpoints found in {checkpoints_dir}")
            sys.exit(1)

    model_dir = os.path.join(checkpoints_dir, model_tag)
    if not os.path.exists(model_dir):
        print0(f"Model directory not found: {model_dir}")
        sys.exit(1)

    # Find all checkpoints
    checkpoint_files = glob.glob(os.path.join(model_dir, "model_*.pt"))
    steps = []
    for f in checkpoint_files:
        match = re.search(r"model_(\d+).pt", f)
        if match:
            step = int(match.group(1))
            steps.append(step)
    
    steps.sort()
    
    if not steps:
        print0(f"No checkpoints found in {model_dir}")
        sys.exit(0)

    print0(f"Found {len(steps)} checkpoints for {model_tag}: {steps}")
    
    for step in steps:
        print0(f"\n{'='*40}")
        print0(f"Evaluating step {step}...")
        print0(f"{'='*40}\n")
        
        cmd = [
            sys.executable, "-m", "scripts.base_eval",
            "--model-tag", model_tag,
            "--step", str(step),
            "--max-per-task", "-1" # Evaluate on whole set
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print0(f"Error evaluating step {step}: {e}")
            # We continue to the next checkpoint even if one fails
            continue

if __name__ == "__main__":
    main()
