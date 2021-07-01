import os
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(strip_accents=False,  # Must be False if cased model
                                   lowercase=False)

corpus_file = ['ChatBots/datasets/mecab_kowiki.txt'] # mecab을 적용한 파일을 사용
vocab_size = 32000
limit_alphabet = 6000
output_path = 'higging_%d'%(vocab_size)
min_frequency = 6

tokenizer.train(files = corpus_file,
                vocab_size = vocab_size,
                min_frequency = min_frequency,
                limit_alphabet = limit_alphabet, # ByteLevelBPETokenizer 학습시에는 주석처리
                show_progress = True)

print('train complete')

# example
sentence = '나는 JminJ라고 한다. 지금 매우 혼란스럽다.. 학습이 잘 되면 좋겠다!'
output = tokenizer.encode(sentence)
print(sentence)
print('=>idx    : %s'%output.ids)
print('=>tokens : %s'%output.tokens)
print('=>offsets: %s'%output.offsets)
print('=>decode : %s'%tokenizer.decode(output.ids))

sentence = 'I want to go my hometown'
output = tokenizer.encode(sentence)
print(sentence)
print('=>idx   : %s'%output.ids)
print('=>tokens: %s'%output.tokens)
print('=>offset: %s'%output.offsets)
print('=>decode: %s\n'%tokenizer.decode(output.ids))

# save tokenizer
hugging_model_path = 'ChatBots/tokenizer/made_tokenizers_wordpiece'
tokenizer.save_model(hugging_model_path)