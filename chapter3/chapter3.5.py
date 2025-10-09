from listing31 import *
from listing33 import *

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your
[0.55, 0.87, 0.66], # journey
[0.57, 0.85, 0.64], # starts
[0.22, 0.58, 0.33], # with
[0.77, 0.25, 0.10], # one
[0.05, 0.80, 0.55]] # step
)

torch.manual_seed(789)
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2


sa_v2 = SelfAttention_v2(d_in, d_out)


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

#as this llm is intended for generation, all tokens that are after the current token can be masked out. this is so the model learns to predict as this data is not provided.

context_length = attn_scores.shape[0]
#mask_simple = torch.tril(torch.ones(context_length, context_length))
#print(mask_simple)

#masked_simple = attn_weights*mask_simple
#print(masked_simple)

#however, this is a bit computationally inefficient.

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
#this is much fewer operations but still needs to be softmaxxed
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

#dropout is useful because it forces the model to 'think outside the box' and learn the relationships between tokens as opposed to memorising
#it is disabled during actual use

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

torch.manual_seed(123)
print(dropout(attn_weights))

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)