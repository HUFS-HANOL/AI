import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from config import model_name, save_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.load_state_dict(torch.load(f"{save_path}/kogpt2_poem_epoch5.pt"))
model.to(device)
model.eval()

prompt = "비 오는 날"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(input_ids, max_length=100, num_beams=5, do_sample=True, top_k=50)
    print(tokenizer.decode(output[0], skip_special_tokens=True))