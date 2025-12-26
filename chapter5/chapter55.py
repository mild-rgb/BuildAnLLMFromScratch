import torch
import tiktoken
from chapter5 import (
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    generate_and_print_sample,
    train_model_simple,
    plot_losses,
    print_sampled_tokens,
    softmax_with_temperature,
    generate
)
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


print("imports completed")


tokenizer = tiktoken.get_encoding("gpt2")
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

from chapter28.dataloader_v1 import create_dataloader  # at this moment, i realise jupyter is a good idea

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
eval_data = text_data[split_idx:]

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
    eval_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_under_test = torch.device('cuda')
free, total = torch.cuda.mem_get_info(device_under_test)
mem_used_MB = (total - free) / 1024 ** 2
print("mem used after before train 2", mem_used_MB)


torch.cuda.empty_cache()
checkpoint = torch.load("model_and_optimizer.pth", map_location='cpu')#this is because of a known issue https://discuss.pytorch.org/t/out-of-memory-error-when-resume-training-even-though-my-gpu-is-empty/30757/5. workaround is to push it through cpu first
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()

train_losses, val_losses, tokens_seen = train_model_simple(
model, train_loader, val_loader, optimizer, device,
num_epochs=1, eval_freq=5, eval_iter=5,
start_context="Every effort moves you", tokenizer=tokenizer
)

model.eval()
token_ids = generate(
model=model,
idx=text_to_token_ids("donkey", tokenizer).to(device),
max_new_tokens=70,
context_size=GPT_CONFIG_124M["context_length"],
top_k=25,
temperature=1.8
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))