import tiktoken
import torch
import torch.nn as nn
from config import GPT_CONFIG_124M
from GPTModel import DummyGPTModel
from LayerNorm import LayerNorm

#4.1
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
#batch is just equivalent to the tokenised verions of txt

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
#logits is just equivalent to the tokenised versions of txt
print(logits)

#4.2
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True) #dim = -1 -> calculate across last dimensions. dim = 0 -> calculate across first dimension etc. dim = 1 is columns in the case of a 2d vector
var = out.var(dim=-1, keepdim=True) #keepdim = True -> maintain input dimensions
print("Mean:\n", mean)
print("Variance:\n", var)


out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)

#this normalises each layer to have a mean value of 0 and standard deviation of 1
#this is important because of the vanishing/exploding gradient problem
#computers can make mistakes with very small or large numbers
#it is possible to reach these if the layers have very large or small gradients

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)