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

#note to self - this is an implementation of cross-entropy. it works out the probability of the correct token being picked over a batch by selecting the token ids of the correct tokens from logits.

neg_avg_log_probas = avg_log_probas * -1 #make it a positive number (the aim will be to make it smaller)
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

#print("Train loader:")
#for x, y in train_loader:
#    print(x.shape, y.shape) #9 2x256

#print("\nValidation loader:")
#for x, y in val_loader:
#    print(x.shape, y.shape) #1 2x256



def calc_loss_batch(input_batch, target_batch, model, device): #wrapper function for loss calculation
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()   #check the chances of the correct tokens being picked by looking at the logits. see above for implementation
    ) #check the chances of the correct tokens being picked by looking at the logits. see above for implementation
     #syntax -> pass flattened logits and targets
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None): #wraps calc_loss_batch and iterates over loader
        total_loss = 0
        if len(data_loader) == 0:
                return float("nan") #error handling

        elif num_batches is None:
                num_batches = len(data_loader) #data_loader

        else:
                num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader): #iterates over batches in data_loader. data_loader is a construct that wraps dataset
                if i < num_batches: 
                        loss = calc_loss_batch(
                                input_batch, target_batch, model, device
                        ) #calculate loss per batch
                        total_loss += loss.item()
                else:
                        break
        return total_loss / num_batches #provide average loss per batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() #sets internal training flag to false (i.e - turns dropout off)
    with torch.no_grad(): #turns off gradient tracking so no log of activations is kept. this is faster
        train_loss = calc_loss_loader( #see above
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device) #start_context is tokenised and assigned to encoded
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train() #puts model into training mode by enabling dropout
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() #resets optimiser from last run
            loss = calc_loss_batch(input_batch, target_batch, model, device) #see comments above
            loss.backward() #work out backprop of loss to see what caused the loss
            optimizer.step() #and do something about the loss
            tokens_seen += input_batch.numel() #.numel -> number of elements in input batch
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
                model, tokenizer, device, start_context
        ) #generate sample text

    return train_losses, val_losses, track_tokens_seen

#torch.nn.cross_entropy (calculates cross entropy) -> calculate_loss_batch (calculates logits for batch and feeds into cross_ent)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)

num_epochs = 10 #go over dataset 10x
train_losses, val_losses, tokens_seen = train_model_simple(
model, train_loader, val_loader, optimizer, device,
num_epochs=num_epochs, eval_freq=5, eval_iter=5,
start_context="Every effort moves you", tokenizer=tokenizer
)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="validation loss"
    )
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
#epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
#plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
#when training loss drops beneath validation loss, the model is overfitting
#this can be intuitively understood as the feeling when questions you don't expect appear on the test

model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("every effort moves you ", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("output text:\n", token_ids_to_text(token_ids, tokenizer))

vocab = {
"closer": 0,
"every": 1,
"effort": 2,
"forward": 3,
"inches": 4,
"moves": 5,
"pizza": 6,
"toward": 7,
"you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()} #this loops over vocab (for k,v in vocab.items) and writes it backwards into a new dictionary (v:k)

next_token_logits = torch.tensor(
[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0) #softmax to get probabilities
next_token_id = torch.argmax(probas).item() #choose most likely
print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item() #samples according to distribution
print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
        for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
print_sampled_tokens(probas)

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature #when temp is v low, behaviour approaches that of argmax
    return torch.softmax(scaled_logits, dim=0)

# exercise 5.1 - chance of pizza being returned below
# pizza is at position 6 so just read out values

temp_altered = next_token_logits/5
probs = torch.softmax(temp_altered, dim=0)
print("exercise 5.1 ans:", probs[6]*100)

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k) #returns keys and values of 3 highest  from each dimension
print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1], #create a boolean mask for all values that are top_logits[-1] (-1 takes the last value and top_k automatically sorts by size)
    input=torch.tensor(float('-inf')), #cover with -inf tensors if true
    other=next_token_logits #carry out on next_token_logits
)
print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens): #iterate max_new_tokens times
        idx_cond = idx[:, -context_size:] #this is the context length cut - from the start to -context_length, take all values
        with torch.no_grad(): #if ^^ was not included, the model would be given something longer than context length.
            logits = model (idx_cond)
        logits = logits[:, -1, :] #model takes sequences of tokens sequentially from batch. model outputs predictions of the next token for all tokens in sequence. example for 1, 2, 3, 4 -> model will output predictions for tokens in 2, 3, 4, 5

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits/temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

torch.manual_seed(123)
token_ids = generate(
model=model,
idx=text_to_token_ids("every step moves you forward", tokenizer),
max_new_tokens=70,
context_size=GPT_CONFIG_124M["context_length"],
top_k=25,
temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

#exercise 5.2 - high temp is better for situations where repeatability is bad - for example, roleplaying
#exercise 5.3 - low temp is closest to argmax

torch.save(model.state_dict(), "model.pth")

torch.save({
"model_state_dict": model.state_dict(),
"optimizer_state_dict": optimizer.state_dict(),
},
"model_and_optimizer.pth"
)