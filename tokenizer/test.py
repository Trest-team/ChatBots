import tokenizers
from transformers import (BartTokenizerFast, BartTokenizer)

merges_file = '' # ...merges.txt file의 path
vocab_path = '' # ...vocab.json file의 path

tokenizer = BartTokenizer(vocab_file = vocab_path, merges_file = merges_file, do_lower_case = False) # BartTokenizerFast는 동작하지 않음

test_str = ' [CLS] 안녕하세요, 저는 JminJ 입니다. [SEP]'
print('테스트 문장 :', test_str)

encoded_str = tokenizer.encode(test_str, add_special_tokens=False)
print('문장 인코딩 :', encoded_str)

decoded_str = tokenizer.decode(encoded_str)
print('문장 디코딩 :',decoded_str)