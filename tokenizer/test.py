from transformers import BertTokenizerFast

tokenizer_for_load = BertTokenizerFast.from_pretrained('ChatBots/tokenizer/made_tokenizers_wordpiece',
                                                       strip_accents=False,  # Must be False if cased model
                                                       lowercase=False)  # 로드

print('vocab size : %d' % tokenizer_for_load.vocab_size)
# tokenized_input_for_pytorch = tokenizer_for_load("i am very hungry", return_tensors="pt")
tokenized_input_for_pytorch = tokenizer_for_load("나는 오늘 아침밥을 먹었다.", return_tensors="pt")
# tokenized_input_for_tensorflow = tokenizer_for_load("나는 오늘 아침밥을 먹었다.", return_tensors="tf")

print("Tokens (str)      : {}".format([tokenizer_for_load.convert_ids_to_tokens(s) for s in tokenized_input_for_pytorch['input_ids'].tolist()[0]]))
print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids'].tolist()[0]))
print("Tokens (attn_mask): {}\n".format(tokenized_input_for_pytorch['attention_mask'].tolist()[0]))