# coding: UTF-8
import numpy as np
import torch
from sklearn.utils import shuffle
from tqdm import tqdm
import time
from datetime import timedelta
import random

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(x_train, y_train, x_dev, y_dev, x_test, y_test, config):
    def load_dataset(x, y, pad_size=32):
        contents = []
        list1 = np.array(x)
        list2 = np.array(y)
        for i in range(list1.shape[0]):
            content, label = list1[i][0], list2[i]  # content放文本，label放0/1
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(x_train, y_train, config.pad_size)
    dev = load_dataset(x_dev, y_dev, config.pad_size)
    test = load_dataset(x_test, y_test, config.pad_size)
    return train, dev, test


# class DatasetIterater(object):
#     def __init__(self, batches, batch_size, device):
#         self.batch_size = batch_size
#         self.batches = batches
#         self.n_batches = len(batches) // batch_size
#         self.residue = False  # 记录batch数量是否为整数
#         if len(batches) % self.n_batches != 0:
#             self.residue = True
#         self.index = 0
#         self.device = device
#
#     def _to_tensor(self, datas):
#         x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
#         y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
#
#         # pad前的长度(超过pad_size的设为pad_size)
#         seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
#         mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
#         return (x, seq_len, mask), y
#
#     def __next__(self):
#         if self.residue and self.index == self.n_batches:
#             batches = self.batches[self.index * self.batch_size: len(self.batches)]
#             self.index += 1
#             batches = self._to_tensor(batches)
#             return batches
#
#         elif self.index >= self.n_batches:
#             self.index = 0
#             raise StopIteration
#         else:
#             batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
#             self.index += 1
#             batches = self._to_tensor(batches)
#             return batches
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         if self.residue:
#             return self.n_batches + 1
#         else:
#             return self.n_batches
#


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, shuffle=False):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # Record if the batch count is an integer
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.shuffle = shuffle

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # Sequence length before padding (set to pad_size if exceeding)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        # Shuffle batches if enabled
        if self.shuffle:
            random.shuffle(self.batches)
        self.index = 0  # Reset index for new iteration
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, shuffle=True)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
