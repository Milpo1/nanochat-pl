from torch.utils.data import IterableDataset
import pyarrow.parquet as pq
from nanochat.tokenizer import get_tokenizer
from nanochat.common import get_dist_info
from collections import deque

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

class FinewebIterableDataset(IterableDataset):
    def __init__(self, T, ddp_world_size, ddp_rank, split,  resume_state_dict=None):
        super(FinewebIterableDataset).__init__()
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.split = split
        self.resume_state_dict = resume_state_dict
        self.T = T

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            process_id = 0
            num_workers = 1
        else:
            process_id = worker_info.id
            num_workers = worker_info.num_workers
        
        world_size = self.ddp_world_size * num_workers
        local_rank = self.ddp_rank + self.ddp_world_size * process_id


        needed_tokens = self.T + 1 # +1 is because we also need the target at the last token
        tokenizer = get_tokenizer()
        bos_token = tokenizer.get_bos_token_id()

        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if self.split == "train" else parquet_paths[-1:]

        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None

        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        token_buffer = deque()

        while True: # iterate infinitely (multi-epoch)
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // self.ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * self.ddp_world_size + local_rank
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = local_rank 

                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    token_lists = tokenizer.encode(batch, prepend=bos_token)

                    for tokens in token_lists:
                        token_buffer.extend(tokens)

                    while len(token_buffer) >= needed_tokens:
                        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
                        yield (torch.tensor(tokens, dtype=torch.long), pq_idx, rg_idx)
                    rg_idx += world_size 

                pq_idx += 1 # advance to the next parquet file
            pq_idx = 0


            