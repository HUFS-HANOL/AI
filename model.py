from transformers import GPT2LMHeadModel
import torch.nn as nn

class CustomKoGPT2(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_ids, labels=None):
        output = self.model(input_ids=input_ids, labels=labels)
        return output  # loss + logits


# loss function 커스터마이징을 하고 싶다면 아래와 같이 가능
# loss = outputs.loss
# logits = outputs.logits
# custom_loss = loss * weight_map[labels]
