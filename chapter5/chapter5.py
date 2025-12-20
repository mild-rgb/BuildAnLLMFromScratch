import tiktoken
import torch
from chapter4.GPTModel import GPTModel
from chapter4.GenerateText import generate_text_simple

GPT_CONFIG_124M = {
	"vocab_size": 50257,
	"context_length": 256,
	"emb_dim": 768,
	"n_heads": 12,
	"n_layers": 12,
	"drop_rate": 0.1,
	"qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def text_to_token_ids(text, tokenizer):
	encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
	encoded_tensor = torch.tensor(encoded).unsqueeze(0)
	return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
	flat = token_ids.squeeze(0)
	return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
	model=model,
	idx=text_to_token_ids(start_context, tokenizer),
	max_new_tokens=10,
	context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
# ["every effort moves", "I really like"] #both of these are shifted by one token

targets = torch.tensor([[3626, 6100, 345 ], [1107, 588, 11311]])
# [" effort moves you", " really like chocolate"]

with torch.no_grad():              #disables gradient tracking
	logits = model(inputs)     #logits are floats essentially 
probas = torch.softmax(logits, dim=-1) #this gets cast to probabilities which sum to 1 
print(probas.shape) #this is a 2,3,50257
#i.e, 2 inputs of 3 words, with 50257 predictions about what could follow each

token_ids = torch.argmax(probas, dim=-1, keepdim=True) #choose the highest values of probas from the last dimension and preserve the other dimensions of the 'slice' with the highest value. for example, see below
print("Token IDs:\n", token_ids) #this has a trailing dimension. despite being a (2,2), it has 3 brackets. the highest values of the 50257 have been taken for each other dimension. this collapses the matrix to a 2,3,1.

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]] #from probas, select from the first batch, all(in this case, all means 3) of the tokens, and then the values corresponding to the correct answers. 
print("Text 1:", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]] #same here
print("Text 2:", target_probas_2)
#each probas contains the probabilities of the correct answer being picked

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2))) #make a tensor of length 6 by concating the correct possibilities of the 2 strings in the batch and take the log. the log is used because it converts what would be multiplication of very small/large numbers into addition of moderately sized numbers
print(log_probas)

avg_log_probas = torch.mean(log_probas) #take the mean of the logs 
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1 #make it a psoitive number (the aim will be to make it smaller) 
print(neg_avg_log_probas)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)


file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
	text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)


from chapter28.dataloader_v1 import create_dataloader #at this moment, i realise jupyter is a good idea

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


train_loader = create_dataloader(
	train_data,
	batch_size=2,
	max_length=GPT_CONFIG_124M["context_length"],
	stride=GPT_CONFIG_124M["context_length"],
	drop_last=True,
	shuffle=True,
	num_workers=0
)

val_loader = create_dataloader(
	val_data,
	batch_size=2,
	max_length=GPT_CONFIG_124M["context_length"],
	stride=GPT_CONFIG_124M["context_length"],
	drop_last=False,
	shuffle=False,
	num_workers=0
)

print("Train loader:")
for x, y in train_loader:
	print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
	print(x.shape, y.shape)
