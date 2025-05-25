import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from config import *
from dataset import PoemDataset
from model import CustomKoGPT2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PoemDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate(x))

model = CustomKoGPT2(model_name).to(device)
optimizer = AdamW(model.parameters(), lr=lr)

def collate(batch):
    input_ids = [item['input_ids'] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    return {'input_ids': input_ids, 'labels': input_ids.clone()}

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader)}")
    torch.save(model.state_dict(), f"{save_path}/kogpt2_poem_epoch{epoch+1}.pt")
