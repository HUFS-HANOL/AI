from transformers import PreTrainedTokenizerFast
import torch
from config import model_name, max_length, data_path, tokenized_data_path

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

inputs = []
for line in lines:
    if line.strip() == "": continue
    tokenized = tokenizer.encode(line.strip(), truncation=True, max_length=max_length)
    inputs.append(torch.tensor(tokenized))

torch.save(inputs, tokenized_data_path)
print(f"✅ Tokenized {len(inputs)} poems and saved to {tokenized_data_path}")