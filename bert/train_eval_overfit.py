# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
import pandas as pd  # 用于保存错误样本


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


# 自定义损失函数，处理不均衡数据时可以用
def loss_fn(outputs, targets, class_weights):
    return F.cross_entropy(outputs, targets, weight=class_weights)


# Learning rate scheduler
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()  # 开启参数更新
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    model.train()

    incorrect_samples = []  # List to store incorrect samples

    for epoch in range(config.num_epochs):
        print(f'Epoch [{epoch + 1}/{config.num_epochs}]')
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            tensor1, tensor2, tensor3 = trains
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)

                # Collect incorrect samples for overfitting
                incorrect = (predic != true)
                incorrect_samples.extend([
                    ((tensor1[idx], tensor2[idx], tensor3[idx]), labels[idx])
                    for idx in range(len(true)) if incorrect[idx]
                ])

                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    last_improve = total_batch

                time_dif = get_time_dif(start_time)
                print(
                    f'Iter: {total_batch},  Train Loss: {loss.item():.2f},  Train Acc: {train_acc:.2%},  Val Loss: {dev_loss:.2f},  Val Acc: {dev_acc:.2%},  Time: {time_dif}')

                model.train()

            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    # # Overfit on incorrect samples
    # if incorrect_samples:
    #     print("Overfitting on incorrectly predicted samples...")
    #     for epoch in range(2):  # Can reduce num_epochs for overfitting
    #         for trains, labels in incorrect_samples:
    #             outputs = model(trains)  # Add batch dimension
    #             loss = F.cross_entropy(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
    # test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
