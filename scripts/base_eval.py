"""
Evaluate the CORE metric for a given model.

Run on a single GPU:
python -m scripts.base_eval

Run with torchrun on e.g. 8 GPUs:
torchrun --nproc_per_node=8 -m scripts.base_eval

The script will print the CORE metric to the console.
"""
import os
import csv
import glob
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
from contextlib import nullcontext

import wandb
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock, DummyWandb
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import find_largest_model, find_last_step, build_model
from nanochat.core_eval import evaluate_task

# -----------------------------------------------------------------------------
# nanochat specific function dealing with I/O etc.

# ~162MB of data needed to evaluate the CORE metric
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    # here file_path is the path to the eval_bundle.zip file
    # we need to unzip it and place it in the base directory
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    """
    # Load config and task metadata
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle to disk (and unzip if needed)
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values from eval metadata
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        # Load data for this task
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the ability to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # run the evaluation for this task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# HuggingFace loading utilities and light wrappers for a model

class ModelWrapper:
    """Lightweight wrapper for a HuggingFace model"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits

def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    # Load the model
    from transformers import AutoModelForCausalLM
    import torch.distributed as dist
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    if world_size == 1:
        model = AutoModelForCausalLM.from_pretrained(hf_path, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype=torch.bfloat16)
        model.to(device)
        
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # Load the tokenizer
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

# -----------------------------------------------------------------------------
def find_all_steps(checkpoint_dir):
    """List every checkpoint step available in a checkpoint_dir, sorted ascending."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    steps = sorted(int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return steps

def evaluate_one_checkpoint(args, device, autocast_ctx, ddp_rank, checkpoint_dir=None, step=None):
    """Load one checkpoint (or the HF model), evaluate it, write its CSV, return summary fields for wandb."""
    if args.hf_path is not None:
        # atm assume that if a path is given, it's a huggingface model path
        hf_path = args.hf_path
        print0(f"Loading huggingface model from: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path # just for logging
        model_slug = hf_path.replace("/", "-") # for the output csv file
        logged_step = None
    else:
        # load directly from the resolved checkpoint directory (either an explicit
        # --checkpoint-dir, or one derived from base_checkpoints/<model_tag>)
        model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase="eval")
        dir_name = os.path.basename(checkpoint_dir.rstrip('/'))
        model_name = f"{dir_name} (step {meta['step']})" # just for logging
        model_slug = f"{dir_name}_{meta['step']:06d}" # for the output csv file
        logged_step = meta["step"]

    # Evaluate the model
    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)
    del model

    # Write out the results to a csv file
    core_metric = None
    centered_results = {}
    results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        # Print the content of the csv file to console too
        print0("="*80)
        print0(f"Model: {model_name}")
        print0("="*80)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            print0(f.read())

    # Log to report
    from nanochat.report import get_report
    get_report().log(section="Base model evaluation", data=[
        {
            "Model": model_name,
            "CORE metric": core_metric,
        },
        centered_results, # the full table
    ])

    return {
        "logged_step": logged_step,
        "core_metric": core_metric,
        "centered_results": centered_results,
        "results": results,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path to evaluate')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per task to evaluate (-1 = disable)')
    parser.add_argument('--model-tag', type=str, default=None, help='optional model tag for the output directory name')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='exact directory containing model_<step>.pt/meta_<step>.json (bypasses --model-tag / base_checkpoints resolution, useful when checkpoints for different runs live in separate folders)')
    parser.add_argument('--step', type=int, default=None, help='optional model step for the output directory name')
    parser.add_argument('--steps', type=str, default=None, help='comma-separated list of steps to evaluate, or "all" to evaluate every checkpoint found in the checkpoint dir; overrides --step')
    parser.add_argument('--run', type=str, default='dummy', help='wandb run name ("dummy" = no wandb logging)')
    args = parser.parse_args()

    # distributed / precision setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    master_process = ddp_rank == 0

    # wandb logging init (once, regardless of how many checkpoints get evaluated below)
    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=vars(args))

    # Resolve the checkpoint directory (None when using --hf-path, which doesn't need one)
    checkpoint_dir = None
    if args.hf_path is None:
        if args.checkpoint_dir is not None:
            checkpoint_dir = args.checkpoint_dir
        else:
            base_dir = get_base_dir()
            model_tag = args.model_tag or find_largest_model(os.path.join(base_dir, "base_checkpoints"))
            checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)

    # Resolve which steps to evaluate
    if args.hf_path is not None:
        steps_to_eval = [None]
    elif args.steps is not None:
        if args.steps == "all":
            steps_to_eval = find_all_steps(checkpoint_dir)
            print0(f"Found {len(steps_to_eval)} checkpoints in {checkpoint_dir}: {steps_to_eval}")
        else:
            steps_to_eval = [int(s) for s in args.steps.split(",")]
    elif args.step is not None:
        steps_to_eval = [args.step]
    else:
        steps_to_eval = [find_last_step(checkpoint_dir)]

    for step in steps_to_eval:
        summary = evaluate_one_checkpoint(args, device, autocast_ctx, ddp_rank, checkpoint_dir=checkpoint_dir, step=step)
        if not use_dummy_wandb:
            log_data = {
                "core_metric": summary["core_metric"],
                **{f"centered_results.{k}": v for k, v in summary["centered_results"].items()},
                **{f"results.{k}": v for k, v in summary["results"].items()},
            }
            if summary["logged_step"] is not None:
                log_data["step"] = summary["logged_step"]
            wandb_run.log(log_data)

    wandb_run.finish()
    compute_cleanup()

if __name__ == "__main__":
    main()
