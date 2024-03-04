# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import json
import numpy as np
sys.path.append("/home/zqhu/home/adapter-transformers-tst/src/")

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import BartTokenizer
from transformers import AutoTokenizer

sys.path.append("/home/zqhu/home/adapter-transformers-tst/")
from utils.helper import evaluate_sc
from utils.dataset import SCIterator
from utils.dataset import load_embedding
from utils.optim import ScheduledOptim

filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]
device = 'cuda' if cuda.is_available() else 'cpu'
special_tokens = [{'bos_token': '<bos>'},
                  {'eos_token': '<eos>'}, {'sep_token': '<sep>'},
                  {'pad_token': '<pad>'}, {'unk_token': '<unk>'}]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# AutoTokenizer['TOKENIZERS_PARALLELISM'] = True

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, embeding):
        super(EmbeddingLayer, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embed_dim)
        if embeding is not None:
            self.embeding.weight.data = torch.FloatTensor(embeding)

    def forward(self, x):
        if len(x.size()) == 2:
            y = self.embeding(x)
        else:
            y = torch.matmul(x, self.embeding.weight)
        return y


class TextCNN(nn.Module):
    '''A style classifier TextCNN'''

    def __init__(self, embed_dim, vocab_size, filter_sizes, 
                 num_filters, num_label, embedding=None, dropout=0.0):
        super(TextCNN, self).__init__()

        self.feature_dim = sum(num_filters)
        self.embeder = EmbeddingLayer(vocab_size, embed_dim, embedding)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim))
            for (n, f) in zip(num_filters, filter_sizes)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(self.feature_dim, int(self.feature_dim / 2)), nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), num_label)
        )

    def forward(self, inp):
        inp = self.embeder(inp).unsqueeze(1)
        convs = [F.relu(conv(inp)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        out = torch.cat(pools, 1)
        logit = self.fc(out)

        return logit

    def build_embeder(self, vocab_size, embed_dim, embedding=None):
        embeder = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(embeder.weight, mean=0, std=embed_dim ** -0.5)
        if embedding is not None:
            embeder.weight.data = torch.FloatTensor(embedding)

        return embeder
    
def data_reader(opt, tokenizer, max_len, test_only=False, test_path=None):
    # specify data path according to dataset name
    dataset = opt.dataset
    multi_attr = False
    if dataset == "GYAFC":
        train_path = "data/datasets/GYAFC/train/formality_transfer_unsup.json"
        valid_path = "data/datasets/GYAFC/test/formality_transfer_unsup.json"
    elif dataset == "yelp":
        train_path = "data/datasets/yelpbaseline/train/sentiment_transfer_unsup.json"
        valid_path = "data/datasets/yelpbaseline/test/sentiment_transfer_unsup.json"
    elif dataset == "tense_adjadv_removal":
        assert opt.style != None, "Please indicate style when using multi-attribute datasets with --style !"
        train_path = "data/datasets/StylePTB/adapterTST/tense_adjadv_removal/train/style_transfer_unsup.json"
        valid_path = "data/datasets/StylePTB/adapterTST/tense_adjadv_removal/test/style_transfer_unsup.json"
        if opt.style == "tense":
            style_position = 0
        elif opt.style == "adjadv_removal":
            style_position = 1
    elif dataset == "tense_pp_front_back":
        assert opt.style != None, "Please indicate style when using multi-attribute datasets with --style !"
        train_path = "data/datasets/StylePTB/adapterTST/tense_pp_front_back/train/style_transfer_unsup.json"
        valid_path = "data/datasets/StylePTB/adapterTST/tense_pp_front_back/test/style_transfer_unsup.json"
        if opt.style == "tense":
            style_position = 0
        elif opt.style == "pp":
            style_position = 1
    elif dataset == "tense_pp_removal":
        assert opt.style != None, "Please indicate style when using multi-attribute datasets with --style !"
        train_path = "data/datasets/StylePTB/adapterTST/tense_pp_removal/train/style_transfer_unsup.json"
        valid_path = "data/datasets/StylePTB/adapterTST/tense_pp_removal/test/style_transfer_unsup.json"
        if opt.style == "tense":
            style_position = 0
        elif opt.style == "pp":
            style_position = 1
    elif dataset == "tense_voice":
        assert opt.style != None, "Please indicate style when using multi-attribute datasets with --style !"
        train_path = "data/datasets/StylePTB/adapterTST/tense_voice/train/style_transfer_unsup.json"
        valid_path = "data/datasets/StylePTB/adapterTST/tense_voice/test/style_transfer_unsup.json"
        if opt.style == "tense":
            style_position = 0
        elif opt.style == "voice":
            style_position = 1


    if test_only:
        test_sent, test_label = [], []
        with open(test_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                if opt.style == None:
                    test_sent.append(tokenizer.encode(sample["sentence"])[:max_len])
                    test_label.append(sample["style_label"])
                else:
                    test_sent.append(tokenizer.encode(sample["sentence"])[:max_len])
                    test_label.append(sample["style_label"])
        return test_sent, test_label

    train_sent, train_label, valid_sent, valid_label = [], [], [], []
    # load training data
    with open(train_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if opt.style == None:
                train_sent.append(tokenizer.encode(sample["sentence"])[:max_len])
                train_label.append(sample["style_label"])
            else:
                train_sent.append(tokenizer.encode(sample["sentence"])[:max_len])
                train_label.append(sample["style_label"][style_position])
        
    # load valid data
    with open(valid_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if opt.style == None:
                valid_sent.append(tokenizer.encode(sample["sentence"])[:max_len])
                valid_label.append(sample["style_label"])
            else:
                valid_sent.append(tokenizer.encode(sample["sentence"])[:max_len])
                valid_label.append(sample["style_label"][style_position])
    return train_sent, train_label, valid_sent, valid_label


def main():
    parser = argparse.ArgumentParser('Style Classifier TextCNN')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-dataset', default='GYAFC', type=str, help='the name of dataset')
    parser.add_argument('-style', default=None, type=str, help='the name of style')
    parser.add_argument('-num_label', default=2, type=int, help='the number of categories')
    parser.add_argument('-tokenizer', default='facebook/bart-large', type=str, help='the name of the tokenizer')
    parser.add_argument('-embed_dim', default=300, type=int, help='the embedding size')
    parser.add_argument('-seed', default=3407, type=int, help='pseudo random number seed')
    parser.add_argument("-dropout", default=0.5, type=float, help="Keep prob in dropout.")
    parser.add_argument('-max_len', default=50, type=int, help='maximum tokens in a batch')
    parser.add_argument('-log_step', default=100, type=int, help='print log every x steps')
    parser.add_argument('-eval_step', default=500, type=int, help='early stopping training')
    parser.add_argument('-batch_size', default=8, type=int, help='maximum sents in a batch')
    parser.add_argument('-epoch', default=50, type=int, help='force stop at specified epoch')
    parser.add_argument('-test_only', default=False, type=bool, help='run test only for evaluation')
    parser.add_argument('-gen_path', default=None, type=str, help='the path of generated sentence for evaluation')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    print(f'[Info] Loading tokenizer {opt.tokenizer}...')
    tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer)
    # for token in ['<E>', '<F>']:
    #     tokenizer.add_tokens(token)

    train_sent, train_label, valid_sent, valid_label = data_reader(opt, tokenizer, opt.max_len)
  
    print('[Info] {} instances from train set'.format(len(train_sent)))
    print('[Info] {} instances from valid set'.format(len(valid_sent)))
    train_loader = SCIterator(train_sent, train_label, opt)
    valid_loader = SCIterator(valid_sent, valid_label, opt)

    if "t5" in opt.tokenizer:
        if os.path.exists('checkpoints_cls/embedding.pt'):
            embedding = torch.load('checkpoints_cls/embedding_t5.pt')
        else:
            embed_path = 'checkpoints_cls/glove.840B.300d.txt'
            embedding = load_embedding(tokenizer, 300, embed_path)
            torch.save(embedding, 'checkpoints_cls/embedding_t5.pt')
    elif "bart" in opt.tokenizer:
        if os.path.exists('checkpoints_cls/embedding.pt'):
            embedding = torch.load('checkpoints_cls/embedding_bart.pt')
        else:
            embed_path = 'checkpoints_cls/glove.840B.300d.txt'
            embedding = load_embedding(tokenizer, 300, embed_path)
            torch.save(embedding, 'checkpoints_cls/embedding_bart.pt')

    model = TextCNN(opt.embed_dim, len(tokenizer), filter_sizes, 
                    num_filters, opt.num_label,embedding=embedding, dropout=opt.dropout)

    loss_fn = nn.CrossEntropyLoss()

    if opt.test_only:
        if opt.dataset == "yelp":
            model.load_state_dict(torch.load(f'checkpoints_cls/textcnn_yelp_bart_large.chkpt'))
        elif opt.dataset == "GYAFC":
            model.load_state_dict(torch.load('checkpoints_cls/textcnn_GYAFC_bart_large.chkpt'))
        elif opt.dataset == "tense_adjadv_removal":
            model.load_state_dict(torch.load('checkpoints_cls/textcnn_tense_adjadv_removal_{}_bart_large.chkpt'.format(opt.style)))
        elif opt.dataset == "tense_pp_front_back":
            model.load_state_dict(torch.load('checkpoints_cls/textcnn_tense_pp_front_back_{}_bart_large.chkpt'.format(opt.style)))
        elif opt.dataset == "tense_pp_removal":
            model.load_state_dict(torch.load('checkpoints_cls/textcnn_tense_pp_removal_{}_bart_large.chkpt'.format(opt.style)))
        elif opt.dataset == "tense_voice":
            model.load_state_dict(torch.load('checkpoints_cls/textcnn_tense_voice_{}_bart_large.chkpt'.format(opt.style)))
        model.to(device).eval()
        test_sent, test_label = data_reader(opt, tokenizer, opt.max_len, test_only=True, test_path=opt.gen_path)
        test_loader = SCIterator(test_sent, test_label, opt)

        test_acc, test_loss = evaluate_sc(model, test_loader, loss_fn, 0)
        print(f"The test accuracy is {test_acc}, and the test loss is {test_loss}")
        exit()

        
    model.to(device).train()

    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), opt.lr)


    print('[Info] Built a model with {} parameters'.format(
           sum(p.numel() for p in model.parameters())))
    print('[Info]', opt)


    tab = 0
    avg_acc = 0
    total_acc = 0.
    total_num = 0.
    loss_list = []
    start = time.time()
    for e in range(opt.epoch):

        model.train()
        for idx, batch in enumerate(train_loader):
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            y_hat = logits.argmax(dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)

            if optimizer.steps % opt.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] Epoch {}-{}: | average acc: {}% | average loss: {} | lr: {} | second: {}'.format(e, optimizer.steps, total_acc / total_num * 100, np.mean(loss_list), lr, time.time() - start))
                start = time.time()

            if optimizer.steps % opt.eval_step == 0:
                valid_acc, valid_loss = evaluate_sc(model, valid_loader, loss_fn, e)
                if avg_acc < valid_acc:
                    avg_acc = valid_acc
                    if "t5" in opt.tokenizer:
                        if opt.style:
                            save_path = 'checkpoints_cls/textcnn_{}_{}_t5_large.chkpt'.format(opt.dataset, opt.style)
                        else:
                            save_path = 'checkpoints_cls/textcnn_{}_t5_large.chkpt'.format(opt.dataset)
                    elif "bart" in opt.tokenizer:
                        if opt.style:
                            save_path = 'checkpoints_cls/textcnn_{}_{}_bart_large.chkpt'.format(opt.dataset, opt.style)
                        else:
                            save_path = 'checkpoints_cls/textcnn_{}_bart_large.chkpt'.format(opt.dataset)
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == 10:
                        exit()

if __name__ == '__main__':
    main()
