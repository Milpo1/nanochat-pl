import os
import torch
from tokenizers import Tokenizer

# 1. Define paths
my_tokenizer_path = "./tokenizer/our_tokenizer6.json" 
target_dir = "./tokenizer"
os.makedirs(target_dir, exist_ok=True)

# 2. Load tokenizer
print(f"Loading {my_tokenizer_path}...")
tokenizer = Tokenizer.from_file(my_tokenizer_path)

# 3. Save tokenizer.json to target dir
target_json = os.path.join(target_dir, "tokenizer.json")
tokenizer.save(target_json)
print(f"Saved tokenizer to {target_json}")

# 4. Generate token_bytes.pt
print("Generating token_bytes.pt...")
vocab_size = tokenizer.get_vocab_size()
token_bytes_list = []

for i in range(vocab_size):
    # Decode to find byte length
    decoded = tokenizer.decode([i], skip_special_tokens=False)
    
    # If decode yields empty (unknown/special) use 0, otherwise UTF-8 len
    if not decoded:
        token_bytes_list.append(0)
    else:
        token_bytes_list.append(len(decoded.encode("utf-8")))

token_bytes_path = os.path.join(target_dir, "token_bytes.pt")
torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)

print(f"Saved token_bytes.pt to {token_bytes_path}")
print("Done.")