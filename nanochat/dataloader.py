from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files, Fineweb2pl
from nanochat.tokenizer import get_tokenizer

def state_tracking_collate(batch):
    """
    batch is a list of tuples: [(tensor, pq, rg), (tensor, pq, rg), ...]
    We want to stack the tensors, but just keep the LAST state for tracking.
    """
    # Unzip the batch
    tensors = [item[0] for item in batch]
    
    # We take the state from the LAST item in the batch.
    # This is an approximation, but sufficient for checkpoints.
    last_pq_idx = batch[-1][1]
    last_rg_idx = batch[-1][2]
    
    # Stack tensors into [B, T+1]
    stacked_tensors = torch.stack(tensors)
    
    return stacked_tensors, {"pq_idx": last_pq_idx, "rg_idx": last_rg_idx}


def tokenizing_distributed_data_loader_with_state(B, T, split, device="cuda", num_workers=4, resume_state=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    
    dataset = Fineweb2pl(
        T=T,
        ddp_world_size=ddp_world_size,
        ddp_rank=ddp_rank,
        split=split,
        resume_state_dict=resume_state
    )
    
    loader = DataLoader(
        dataset,
        batch_size=B,
        num_workers=num_workers, # Multiprocessing enabled!
        collate_fn=state_tracking_collate, # Handles the state extraction
        pin_memory=(device=="cuda"),
        drop_last=True
    )
    
    for batch_tensor, state_dict in loader:
        # batch_tensor is [B, T+1]
        batch_tensor = batch_tensor.to(device, non_blocking=True)
        
        inputs = batch_tensor[:, :-1].contiguous()
        targets = batch_tensor[:, 1:].contiguous()
        
        # We yield the inputs, targets, AND the state of where we are
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
