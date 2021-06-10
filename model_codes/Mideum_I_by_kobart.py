import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset
from transformers import (BartForConditionalGeneration, BartTokenizer)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser(description = "Trest's Mideom_I")
parser.add_argument('--checkpoint_path', type = str, help = 'checkpoint path')
parser.add_argument('--chat', action = 'store_true', default = False, help = 'response generation on given user input')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ArgsBase():
    @staticmethod
    def add_level_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents = [parent_parser], add_help = False)
        parser.add_argument('--train_file',
                            type = str,
                            default = '',
                            help = 'train file')

        parser.add_argument('--test_file',
                            type = str,
                            default = '',
                            help = 'test file')

        parser.add_argument('--tokenizer_path',
                            type = str,
                            default = '',
                            help = 'tokenizer')

        parser.add_argument('--batch_size',
                            type = int,
                            default = 14,
                            help = 'batch size')

        parser.add_argument('--max_seq_len',
                            type = int,
                            default = 36,
                            help = 'max seq len')

        return parser

class ChatDataset(Dataset):
    def __init__(self, filepath, vocab_path, merges_file, max_seq_len = 128) -> None: # vocab_path, merges_file 파라미터 추가해야 함
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len
        # self.tokenizer = PreTrainedTokenizerFast(
        #     tokenizer_file = os.path.join(tokenizer_path, 'model.json'),
        #     bos_token = self.bos_token,
        #     eos_token = self.eos_token,
        #     unk_token = '<unk>',
        #     pad_token = '<pad>',
        #     mask_token = '<mask>'
        # )
        self.tokenizer = BartTokenizer(
            vocab_file = vocab_path, 
            merges_file = merges_file,
            bos_token = self.bos_token,
            eos_token = self.eos_token,
        )
        
    # data file의 길이를 받아온다
    def __len__(self):
        return len(self.data)

    def mask_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        # max 길이보다 input_id의 길이가 작다면 self.max_seq_len까지 input_id에 pad_token_id를 연결, attention mask에 [0]을 연결
        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        # max 길이보다 input_id의 길이가 길다면 max_seq_len - 1 까지 input_id를 받아오고 eos_token_id를 연결
        else:
            input_id = input_id[:self.max_seq_len - 1] + [self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return input_id, attention_mask

    def __getitem__(self, index):
        # data에서 index 번째 요소를 가져온다.
        record = self.data.iloc[index]
        q, a = record['Q'], record['A']
        # 문장 token들에 bos_token과 eos_token을 연결해준다.
        q_tokens = [self.bos_token] + self.tokenizer.tokenize(q) + [self.eos_token]
        a_tokens = [self.bos_token] + self.tokenizer.tokenize(a) + [self.eos_token]
        # make_input_id_mask 함수를 통해 masking된 문장 token들의 id와 attention token들을 받아온다.
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(q_tokens, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(a_tokens, index)
        # a_tokens에서 bos_token 다음의 token부터 self.max_seq_len까지의 id들을 가져온다.
        labels = self.tokenizer.convert_tokens_to_ids((a_tokens[1:(self.max_seq_len + 1)]))
        
        # cross entropy loss를 위한 masking
        if len(labels) < self.max_seq_len: 
            while len(labels) < self.max_seq_len:
                labels += [-100]

        return {'input_ids' : np.array(encoder_input_id, dtype = np.int_),
                'attention_mask' : np.array(encoder_attention_mask, dtype = np.float_),
                'decoder_input_ids' : np.array(decoder_input_id, dtype = np.int_),
                'decoder_attention_mask' : np.array(decoder_attention_mask, dtype = np.float_),
                'labels' : np.array(labels, dtype = np.int_)}

class ChatDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, vocab_path, merges_file, max_seq_len = 128, batch_size = 32, num_workers = 5): # vocab_path, merges_file 파라미터를 추가함
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        # self.tokenizer_path = tokenizer_path
        self.vocab_path = vocab_path 
        self.merges_file = merges_file 
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        # parent_parser ArgumentParser에 '--numworkers'인자를 추가한다.
        parser = argparse.ArgumentParser(parents = [parent_parser], add_help = False)
        parser.add_argument('--num_workers',
                            type = int,
                            default = 5,
                            help = 'num of worker for dataloader')
        return parser

    def setup(self, stage):
        # train과 test data를 ChatDataset class에 넣는다.
        self.train = ChatDataset(self.train_file_path, self.vocab_path, self.merges_file, self.max_seq_len)
        self.test = ChatDataset(self.test_file_path, self.vocab_path, self.merges_file, self.max_seq_len)

    def train_dataloader(self):
        # ChatDataset에 넣은 self.train을 DataLoader로 만든다.
        train = DataLoader(self.train, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)

        return train

    def val_dataloader(self):
        # ChatDataset에 넣은 self.test를 DataLoader로 만든다.
        val = DataLoader(self.val, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)

        return val

    def test_dataloader(self):
        # ChatDataset에 넣은 self.test를 DataLoader로 만든다.
        test = DataLoader(self.test, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)

        return test

class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # parent_parser에 추가적인 인자들을 추가한다.
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument('--batch-size', type = int, default = 14, help = 'batch size for training (default: 96)')

        parser.add_argument('--lr', type = float, default = 5e-5, help = 'The initial learning rate')

        parser.add_argument('--warmup_ratio', type = float, default = 0.1, help = 'warmup ratio')

        parser.add_argument('--model_path', type = str, default = None, help = 'kobart model path')

        return parser

    def configure_optimizers(self):
        # model의 parameter들을 모두 받아온다.
        param_optimizer = list(self.model.named_parameters())
        # 가중치가 감소되지 않을 parameter
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # parameter 중 no_decay에 들어있는 건 'weight_decay'가 0.0, no_decay에 없는 건 0.1
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # AdamW를 사용해 optimizer들을 최적화 한다.
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.hparams.lr, correct_bias = False)

        # self.hparams.gpus와 self.hparams.num_nodes를 곱해 num_workers를 만든다
        num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (self.hparams.num_nodes if self.hparams.num_node is not None else 1)
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        # data_len을 self.hparams.batch_size * num_worker로 나눈 값에 self.hparams.max_epochs를 곱해 총 train_step을 구한다.  
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        # self.hparams.warmup_ratio를 num_train_steps에 곱해 num_warmup_steps를 구한다.
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        # scheduler를 get_cosine_schedule_with_warmup 함수를 사용해 선언
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_train_steps)
        # lr_scheduler 선언
        lr_scheduler = {'scheduler' : scheduler, 'monitor' : 'loss', 'interval' : 'step', 'frequency' : 1}

        return [optimizer], [lr_scheduler]

class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
        # 저장된 model의 경로를 받아 BartForConditionalGeneration model을 선언한다.
        self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_path)
        self.model.train()
        # token 설정
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        # tokenizer를 PreTrainedTokenizerFast 함수를 통해 선언한다.
        # self.tokenizer = PreTrainedTokenizerFast(tokenizer_file = os.path.join(self.hparams.tokenizer_path, 'model.json'),
        #     bos_token = self.bos_token, eos_token = self.eos_token, unk_token = '<unk>', pad_token = '<pad>', mask_token = '<mask>')

        # tokenizer를 BartTokenizer 함수를 통해 선언한다.
        self.tokenizer = BartTokenizer(vocab_file = self.hparams.vocab_path,
                                        merges_file = self.hparams.merges_file, 
                                        bos_token = self.bos_token,
                                        eos_token = self.eos_token)

    def forward(self, inputs):
        return self.model(input_ids = inputs['input_ids'],
                            attention_mask = inputs['attention_mask'],
                            decoder_input_ids = inputs['decoder_input_ids'],
                            decoder_attention_mask = inputs['decoder_attention_mask'],
                            labels = inputs['labels'], return_dict = True)
    
    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar = True, on_step = True, on_epoch = True)
        return loss

    def validataion_step(self, batch, bath_idx):
        outs = self(batch)
        loss = outs['loss']
        self.log('val_loss', loss, on_step = True, on_epoch = True, prog_bar = True)

    def chat(self, text):
        # text 문장의 token을 id로 바꾸고 bos_token, eos_token을 붙여 input_ids를 만든다.
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        # 문장 ids를 생성
        res_ids = self.model.generate(torch.tensor([input_ids]),
                                        max_length = self.hparams.max_seq_len,
                                        num_beanms = 5,
                                        eos_token_id = self.tokenizer.eos_token_id,
                                        bad_words_ids = [[self.tokenizer.unk_token_id]])
        # ids를 문장으로 변환
        a = self.tokenizer.batch_decode(res_ids.tolist())[0]
        # token들을 공백으로 치환하고 return
        return a.replace('<s>', '').replace('</s>', '')
        
if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_level_specific_args(parser)
    parser = ChatDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    # parser의 값을 args 변수로 받아온다.
    args = parser.parse_args()
    logging.info(args)
    # KoBARTConditionalGeneration에 args parameter들을 넣어 model을 선언
    model = KoBARTConditionalGeneration(args)

    dm = ChatDataModule(args.train_file,
                        args.test_file,
                        args.vocab_path,
                        args.merges_file,
                        max_seq_len = args.max_seq_len,
                        batch_size=32,
                        num_workers = args.num_workers)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor = 'val_loss',
                                                        dirpath = args.default_root_dir,
                                                        filename = 'model_chp/{epoch:02d}-{val_loss:.3f}',
                                                        verbose = True,
                                                        save_last = True,
                                                        mode = 'min',
                                                        save_top_k = 4,
                                                        prefix = 'kobart_chitchat')

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose = False,
        mode='min'
    )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger = tb_logger, callbacks = [checkpoint_callback, lr_logger, early_stopping_callback])

    # model과 data를 넣어 학습
    trainer.fit(model, dm)
    if args.chat:
        # evaluation mode로 전환
        model.model.eval()
        while 1:
            # 문장 입력
            q = input('user > ').strip()
            # 입력 받은 문장이 'quit'일 시 break
            if q == 'quit':
                break
            # 생성된 문장 출력
            print("Simsimi > {}".format(model.chat(q)))