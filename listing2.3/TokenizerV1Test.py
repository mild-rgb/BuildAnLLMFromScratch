from SimpleTokenizerV1 import SimpleTokenizerV1 
import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with"""
ids = tokenizer.encode(text)
print(ids)
text = tokenizer.decode(ids)
print(text)
