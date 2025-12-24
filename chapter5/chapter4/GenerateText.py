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

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


