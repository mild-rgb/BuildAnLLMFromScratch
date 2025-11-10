import tiktoken
import torch
import torch.nn as nn
from config import GPT_CONFIG_124M
from GPTModel import GPTModel
from LayerNorm import LayerNorm
from FeedForward import FeedForward
from ExampleDeepNeuralNetwork import ExampleDeepNeuralNetwork
from TransformerBlock import TransformerBlock
from GenerateText import generate_text_simple

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
model = GPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
#logits is just the outputs of txt
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

#4.3

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)

model_without_shortcut = ExampleDeepNeuralNetwork(
layer_sizes, use_shortcut=False
)


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss() #mean squared error loss
    loss = loss(output, target) #target is all 0s
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

print_gradients(model_without_shortcut, sample_input)
#this shows the vanishing gradients as small gradients are multiplied by even smaller gradients

print("the model below uses shortcuts")

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)

print_gradients(model_with_shortcut, sample_input)

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

print("printing shapes of full gpt model input/output \n")
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

print("printing params of full gpt model \n")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}") #there is a discrepancy between the number of params declared in config (124 M) and printed here (163 M)
#this is because there are different layers for casting in and out of token_ids. using the same layer is called 'weight tying'. using it can decrease performance but does decrease memory performance
#as further insight, out head is equal to (vocab_size * embedding_dimensions)

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

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
