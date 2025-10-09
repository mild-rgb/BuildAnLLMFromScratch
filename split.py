import re
text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text)
print(result)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


print(len(preprocessed))

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
	print(item)
	if i > 50:
		break
