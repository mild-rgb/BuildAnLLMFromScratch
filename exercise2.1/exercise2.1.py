import tiktoken

text = "akwirw ier"
tokenizer = tiktoken.get_encoding("gpt2")

integers = tokenizer.encode(text)
print(integers)

for i in integers:
    print(tokenizer.decode_single_token_bytes(i))

print(tokenizer.decode(integers))

#yes, it can reconstruct the original string

