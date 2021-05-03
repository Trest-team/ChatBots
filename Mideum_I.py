import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from transformers import (BartForConditionalGeneration, PreTrainedTokenizerFast)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser(description = "PIGO's Mideom_I")
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
                            default = 'datasets/ChatbotData.csv',
                            help = 'train file')

        parser.add_argument('--test_file',
                            type = str,
                            default = '',
                            help = 'test file')

        parser.add_argument('--tokenizer_path',
                            type = str,
                            default = 'tokenizer',
                            help = 'tokenizer')

        parser.add_argument('--batch_size',
                            type = int,
                            default = 14,
                            help = '')

        parser.add_argument('--max_seq_len',
                            type = int,
                            default = 36,
                            help = 'max seq len')

        return parser

class ChatDataset(Dataset):
    def __init__(self, filepath, tok_vocab, max_seq_len = 128) -> None:
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file = tok_vocab,
            bos_token = self.bos_token,
            eos_token = self.eos_token,
            unk_token = '<unk>',
            pad_token = '<pad>',
            mask_token = '<mask>'
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
            labels += [-100]

        return {'input_ids' : np.array(encoder_input_id, dtype = np.int_),
                'attention_mask' : np.array(encoder_attention_mask, dtype = np.float_),
                'decoder_input_ids' : np.array(decoder_input_id, dtype = np.int_),
                'decoder_attention_mask' : np.array(decoder_attention_mask, dtype = np.float_),
                'labels' : np.array(labels, dtype = np.int_)}

class ChatDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, tok_vocab, max_seq_len = 128, batch_size = 32, num_workers = 5):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok_vocab = tok_vocabs
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        pass