import argparse
from os import read
import transformers
from transformers import TrainingArguments, Trainer
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers import EarlyStoppingCallback
import torch
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--chat',
                    action = 'store_true',
                    default = False,
                    help = 'response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='./kogpt2_repo/model_-epoch=03-train_loss=23.42.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

train_data = pd.read_csv('datasets/ChatbotData_shuf_train.csv')
val_data = pd.read_csv('datasets/ChatbotData_shuf_valid.csv')
test_data = pd.read_csv('datasets/ChatbotData_shuf_test.csv')

