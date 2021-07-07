from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
import numpy

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)
model = GPT2LMHeadModel.from_pretrained('./kogpt2_repo/model_-epoch=03-train_loss=23.42.ckpt')

def chat(user_ans, sent = '0'):
    tok = TOKENIZER
    with torch.no_grad():
        a = ''
        input_ids = torch.LongTensor(tok.encode(U_TKN + user_ans + SENT + sent + S_TKN + a))
        pred = model(input_ids, return_dict = True).logits
        gen = tok.convert_ids_to_tokens(
            torch.argmax(
                pred,
                dim = -1).sqeeze().numpy().tolist())[-1]
        a += gen.replace('_', ' ')
    
    return a.strip()

print(chat('안녕하세요'))