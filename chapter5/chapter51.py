import tiktoken
from chapter4.GenerateText import generate_text_simple
from chapter4.GPTModel import GPTModel
from chapter4.config import GPT_CONFIG_124M
import torch

torch.manual_seed(123)
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)
model.eval()

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves"]
[40, 1107, 588]]) # ["I really like"]

targets = torch.tensor([[3626, 6100, 345 ],
[1107, 588, 11311]])
# [" effort moves you",
# " really like chocolate"]

with torch.no_grad():
    logits = model(inputs) #this produces 2 rows (batch size), 3 tokens, 50257 possible options. this shows an unadjusted row of floats
probas = torch.softmax(logits, dim=-1) #casting to probabilities which sum to 1

token_ids = torch.argmax(probas, dim=-1, keepdim=True) #choosing most likely tokens
print("Token IDs:\n", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)
#for a given text, at each position (0,1,2), extract the probability the model assigned to the correct token at that position


avg_log_probas = torch.log(torch.cat((target_probas_1, target_probas_2))) #then concatenate these probabilities and take the log of them
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("loss:", loss)

perplexity = torch.exp(loss)
print("perplexity:", perplexity)

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)
