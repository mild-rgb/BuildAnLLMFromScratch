import torch

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your
[0.55, 0.87, 0.66], # journey
[0.57, 0.85, 0.64], # starts
[0.22, 0.58, 0.33], # with
[0.77, 0.25, 0.10], # one
[0.05, 0.80, 0.55]] # step
)

query = inputs[1] #the embedding of the word 'journey'
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs): #provides each row of inputs and its index
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

#attn_scores_2 is the dot product between the query [inputs [1]] and all other vectors in inputs


attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

#attn_weights_2 is the softmax of the dot product between inputs[1] and all other rows in inputs

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
    #this is the dot product of attn_weights with every other entry
print(context_vec_2) #why is this 1x3?
                    #because it was embedded into three dimension vectors

