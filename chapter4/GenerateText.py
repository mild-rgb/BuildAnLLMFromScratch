import torch
import tiktoken
import torch
import torch.nn as nn
from chapter4.config import GPT_CONFIG_124M
from chapter4.GPTModel import GPTModel
from chapter4.LayerNorm import LayerNorm
from chapter4.FeedForward import FeedForward
from chapter4.ExampleDeepNeuralNetwork import ExampleDeepNeuralNetwork
from chapter4.TransformerBlock import TransformerBlock

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] #crops current context to maximum size
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] #this is because the model outputs predictions based on each token. example - given tokens 1, 2, 3, 4, 5, the model will output predictions for 2, 3, 4, 5, 6. only the prediction for token 6 is needed
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

'''
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
start_context = "i am cheese"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)


model = GPTModel(GPT_CONFIG_124M)
model.eval()

out = generate_text_simple(
    model=model,
idx=encoded_tensor,
max_new_tokens=6,
context_size=GPT_CONFIG_124M["context_length"]
)

print("output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
'''