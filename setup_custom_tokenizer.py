# %%
import os
import torch
from tokenizers import Tokenizer
from nanochat.common import get_base_dir

# 1. Define paths
my_tokenizer_path = "./tokenizer/herbert-tokenizer.json" # <--- POINT TO YOUR FILE
base_dir = "./"
target_dir = os.path.join(base_dir, "tokenizer")
os.makedirs(target_dir, exist_ok=True)

# 2. Load your tokenizer
print(f"Loading {my_tokenizer_path}...")
tokenizer = Tokenizer.from_file(my_tokenizer_path)

# 3. Add Nanochat Special Tokens
# These are required for the chat templates and state machine
SPECIAL_TOKENS = [
    "<|bos|>",            # Document delimiter
    "<|user_start|>",     # User message start
    "<|user_end|>",       # User message end
    "<|assistant_start|>",# Assistant message start
    "<|assistant_end|>",  # Assistant message end
    "<|python_start|>",   # Tool use start
    "<|python_end|>",     # Tool use end
    "<|output_start|>",   # Tool output start
    "<|output_end|>",     # Tool output end
]

print("Adding special tokens...")
# Determine which are missing
existing = set(tokenizer.get_vocab().keys())
to_add = [t for t in SPECIAL_TOKENS if t not in existing]

if to_add:
    tokenizer.add_special_tokens(to_add)
    print(f"Added {len(to_add)} special tokens.")
else:
    print("All special tokens already present.")

# 4. Save the modified tokenizer.json to the nanochat cache dir
target_json = os.path.join(target_dir, "tokenizer.json")
tokenizer.save(target_json)
print(f"Saved modified tokenizer to {target_json}")

# 5. Generate token_bytes.pt
# This maps every token ID to its length in bytes (utf-8).
# This is required for the 'bpb' (bits per byte) metric in training logs.
print("Generating token_bytes.pt...")

vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")

# Get mapping of id -> token string
# Note: HF Tokenizers decoder might not be perfect for getting raw bytes 
# for every single token if using BPE with byte_fallback, but this is a close approximation.
token_bytes_list = []

# We iterate 0..vocab_size. 
# Special tokens are counted as 0 bytes for bpb calculation.
special_ids = set(tokenizer.token_to_id(t) for t in SPECIAL_TOKENS if tokenizer.token_to_id(t) is not None)

for i in range(vocab_size):
    if i in special_ids:
        token_bytes_list.append(0)
        continue
        
    # Attempt to decode
    decoded = tokenizer.decode([i], skip_special_tokens=False)
    
    # If decode results in empty string (unknown or special), check if we can get raw content
    if not decoded:
        # Fallback logic depends on specific tokenizer model (BPE/WordPiece)
        # For bpb calculation, 0 is safer than crashing if unknown
        token_bytes_list.append(0)
    else:
        # Calculate UTF-8 bytes length
        token_bytes_list.append(len(decoded.encode("utf-8")))

token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
token_bytes_path = os.path.join(target_dir, "token_bytes.pt")
torch.save(token_bytes_tensor, token_bytes_path)

print(f"Saved token_bytes.pt to {token_bytes_path}")
print("Done! You can now run training.")