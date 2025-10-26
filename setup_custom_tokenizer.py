import os
import torch
from nanochat.common import get_base_dir
from nanochat.tokenizer import HuggingFaceTokenizer, SPECIAL_TOKENS

# 1. --- User Configuration ---
# Set this to the Hugging Face Hub name of your desired tokenizer.
# For example: "gpt2", "bert-base-uncased", "EleutherAI/gpt-neo-125M"
HF_TOKENIZER_NAME = "allegro/herbert-klej-cased-tokenizer-v1"

# 2. --- Load, Add Special Tokens, and Save Tokenizer ---
print(f"Loading tokenizer '{HF_TOKENIZER_NAME}' from Hugging Face...")
tokenizer_wrapper = HuggingFaceTokenizer.from_pretrained(HF_TOKENIZER_NAME)

# Add nanochat's special tokens (like <|bos|>) if they don't already exist.
# This is crucial for the dataloader and other parts of the code.
# The `tokenizer.get_added_tokens_decoder()` returns a map of special token IDs to their string representation.
current_special_tokens = {t.content for t in tokenizer_wrapper.tokenizer.get_added_tokens_decoder().values()}
new_special_tokens = [st for st in SPECIAL_TOKENS if st not in current_special_tokens]
if new_special_tokens:
    print(f"Adding new special tokens to tokenizer: {new_special_tokens}")
    tokenizer_wrapper.tokenizer.add_special_tokens(new_special_tokens)
    # Important: After adding tokens, the model's embedding layer needs to be resized.
    # The training script handles this by re-reading the vocab size from the tokenizer.

# Save the tokenizer to the directory where nanochat expects to find it.
# This will create a `tokenizer.json` file.
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer_wrapper.save(tokenizer_dir)
print(f"Saved tokenizer to {tokenizer_dir}")


# 3. --- Generate token_bytes.pt helper file ---
# This file is required for calculating the bits-per-byte (bpb) metric during evaluation.
print("Generating token_bytes.pt...")

# Re-load the tokenizer to ensure all tokens (including new special ones) are correctly indexed.
tokenizer_wrapper = HuggingFaceTokenizer.from_directory(tokenizer_dir)
vocab_size = tokenizer_wrapper.get_vocab_size()
special_set = set(tokenizer_wrapper.get_special_tokens())

token_bytes_list = []
for token_id in range(vocab_size):
    token_str = tokenizer_wrapper.id_to_token(token_id)

    if token_str is None: # Handle undefined tokens
        token_bytes_list.append(0)
        continue

    if token_str in special_set:
        token_bytes_list.append(0)  # Special tokens don't count towards bpb
    else:
        num_bytes = len(token_str.encode("utf-8"))
        token_bytes_list.append(num_bytes)

token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
torch.save(token_bytes_tensor, token_bytes_path)
print(f"Saved token_bytes to {token_bytes_path}. Setup complete.")