import torch

#INPUTS IS TOKENS AFTER EMBEDDING
inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your
[0.55, 0.87, 0.66], # journey
[0.57, 0.85, 0.64], # starts
[0.22, 0.58, 0.33], # with
[0.77, 0.25, 0.10], # one
[0.05, 0.80, 0.55]] # step
)

#this is a way of generating attention scores for all
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

#much faster way of generating attention scores
attn_scores = inputs @ inputs.T #@ is used for matrix multiplication
print(attn_scores)

#just softmaxxing to normalise and get weights
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

#matrix multiplication between attn_weights and inputs
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)