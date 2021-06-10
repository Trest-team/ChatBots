# -*- coding: utf-8 -*-
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='model_chp/model_-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# spetial token들
U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

# 기존의 학습된 tokenizer를 사용
TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER 

    # data의 길이 확인
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # data에서 index번째의 요소를 가져오고 이를 Q, A로 나눈다.
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        # label을 sentiment로 설정한다.
        sentiment = str(turn['label'])

        # q앞에 'U_TKN'을 붙이고 뒤에 'SENT'와 sentiment를 붙여 이를 tokenizer에 넣는다.
        q_toked = self.tokenizer.tokenize(self.q_token + q + \
                                          self.sent_token + sentiment)  
        # tokeinze를 수행한 뒤의 token 개수 
        q_len = len(q_toked)
        # a 또한 유사하게 'S_TKN'을 앞에 붙이고 a 뒤에 'EOS'를 붙여 tokenizer에 넣는다.
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        # tokenize를 수행한 뒤의 token 개수
        a_len = len(a_toked)

        # 만약 max_len보다 q_len과 a_len을 더한 값이 더 크다면 max_len에서 q_len을 뺸 나머지 길이를 a_len이 갖는다.
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            # 이떄 a_len이 0보다 작거나 같아진다면 q_toked -(int(self.max_len/2))부터 끝 까지의 q_toked를 가져온다.
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                # 수정한 q_len을 max_len에서 빼줌으로써 a_len을 구한다.
                a_len = self.max_len - q_len
                # a_len이 0보다 작으면 asserterror를 발생시킨다.
                assert a_len > 0
            # a_token을 a_len-1까지 받아온다.
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            # a_len이 len(a_toked)와 다르면 asserterror를 호출
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        # a_toked가 아닌 부분을 mask로 처리하고 bos부터 a_toked를 시작한다.
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        # self.first가 True면 아래 내용을 출력해주고 self.first를 False로 한다.
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        # mask를 만든다(q_len은 0, a_len은 1, max_len을 맞추기 위해 max_len에서 q_len과 a_len을 뺀 값을 0으로 만들어 준다).
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        # labels를 tokenize를 해 labels_ids를 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        # label_ids가 max_len보다 작을 동안 mask id를 추가해 준다.
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        # q_toked와 a_toked를 합쳐 tokenize를 시켜 token_ids를 만든다.
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        # token_ids의 길이가 max_len보다 작을 떄 까지 mask id를 추가해 준다.
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),
               labels_ids)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        # ?
        self.neg = -1e18
        # pretrain된 모델을 가져온다.
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        # CrossEntropyLoss로 loss_function을 정의한다.
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    # argument들을 추가해 준다.
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # gpt2 모델을 학습시켜 output을 얻는다.
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        # batch에서 token_ids, mask, label을 얻는다.
        token_ids, mask, label = batch
        # forward를 실행
        out = self(token_ids)
        # mask의 2차원에 한 차원을 추가하고 repeat_interleave를 통해 2차원에서 out의 2차원의 수(seq_len)만큼 반복한다.
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        # mask_3d가 1인 부분은 out을 그대로 유지하고, 1이 아닌 부분은 self.neg으로 변환한다.
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        # mask_out의 1차원과 2차원을 치환해주고 label과 비교해 loss를 구한다.
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # param_optimizer에서 no_decay에 없는 것은 'weight_decay를 0.01로 주고, no_decay에 포함되는 것은 0.0을 준다.
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # AdamW를 사용해 optimize를 수행한다.
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        # max_epochs에 train_dataloader의 길이를 곱해 num_train_step을 구한다.
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        # num_train_step에 warmup_ratio를 곱해 warmup_step을 구한다.
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        # get_cosine_schedular_with_warmup 함수를 통해 schedular를 선언.
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        # batch의 item들 중 0번째 값은 data로, 1번째 값은 mask로, 2번째 값은 label로 설정하고 torch.LongTensor로 변환해 return한다.
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        # 학습 데이터를 읽어온다.
        data = pd.read_csv('Chatbot_data/ChatbotData.csv')
        # ChatDataset Class를 통해 self.train_set을 설정한다.
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        # DataLoader를 사용해 train_dataloader를 선언.
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)

        return train_dataloader

    def chat(self, sent='0'):
        # tokenizer 선언
        tok = TOKENIZER
        # sent를 tokenize한다.
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            while 1:
                # input을 받는다.
                q = input('user > ').strip()
                # 만약 'quit'과 q가 같다면 break를 한다.
                if q == 'quit':
                    break
                a = ''
                while 1:
                    # U_TKN + q + SENT + sent + S_TKN + a를 tokenizer로 encode하고 0차원에 빈 차원을 추가한 뒤, torch.LongTensor로 변환한다.
                    input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                    # input_ids를 사용해 예측(logits가 나오는 건가?)
                    pred = self(input_ids)
                    # 예측한 token id들을 token들로 변환
                    gen = tok.convert_ids_to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    # 현재 time step의 gen이 EOS와 같다면 break.
                    if gen == EOS:
                        break
                    # a에 생성한 gen을 넣고 '_'를 ' '로 치환한다.
                    a += gen.replace('▁', ' ')
                print("Simsimi > {}".format(a.strip()))

# 각 class에서 요구하는 argument들을 parser에 추가
parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    if args.train:
        # checkpoint_callback을 선언한다.
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            prefix='model_'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        model = KoGPT2Chat(args)
        model.train()
        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    if args.chat:
        # args.chat이 True일 때 args.model_params의 model을 불러온다.
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        # 문장을 생성후 결과 줄력
        model.chat()
