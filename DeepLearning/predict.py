# coding: UTF-8
import time
import torch
import numpy as np
import pandas as pd
from train_eval import train, init_network
from utils import build_dataset, build_iterator, get_time_dif
from sklearn.model_selection import train_test_split
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default="TextRCNN")
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')
parser.add_argument('--num_epochs', default=50, type=int, help='epochs')
parser.add_argument('--learning_rate', default=0.01, type=int, help='learning_rate')
args = parser.parse_args()


def fun1(x):
    return " ".join(x)


if __name__ == '__main__':
    # 数据导入 特征筛选
    data = pd.read_csv("./Dataset/data/train.csv")
    data["Report Content"] = data["Report Content"].apply(lambda x: x.split("##"))
    data["Report Content"] = data["Report Content"].apply(fun1)

    x_train = data.iloc[:, [ 1, 2, 5]]
    y_train = data.iloc[:, 6]

    t = pd.DataFrame(x_train.astype(str))
    x_train["new"] = t["Title"]  + ' ' + t["Report Content"]
    x_train = x_train.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)



    # 参数传入“设置类”
    dataset = '.\Dataset'  # 数据集
    embedding = 'embedding_SougouNews_plus.npz'
    # embedding = 'sgns.sogounews.bigram-char'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(43)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


    # 构建训练集 验证集 测试集数据（文字转数据）及其迭代器；构建词表
    vocab, train_data, dev_data, test_data = build_dataset(x_train, y_train, x_train, y_train, x_train, y_train, config,
                                                           args.word)
    train_iter = build_iterator(train_data, config)
    # 神经网络初始化和训练模型
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    print("config.save_path", config.save_path)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():  # 只是想看一下训练的效果，并不需要更新网络时
        for texts, labels in train_iter:
            outputs = model(texts)
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
        test = data[["id"]]
        test["label"] = predict_all
        result_save_path = "test.csv"
        # 结果输出到result_save_path
        test.to_csv(result_save_path, index=None)