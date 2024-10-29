# coding: UTF-8
import time
import torch
import numpy as np
import pandas as pd
# from train_eval import train, init_network
from train_eval import train, init_network
from sklearn.model_selection import train_test_split
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default="bert_RNN", help='choose a model: bert, ERNIE')
args = parser.parse_args()



import re

def shorten_url(url):
    # 使用正则表达式提取中间的8位字符
    match = re.search(r"(?<=biz=|jpg/)(\w{10})", url)
    if match:
        # return ""
        return match.group(1)
    return " "  # 如果匹配不到，返回原始URL


def fun1(x):
    return " ".join(x)
if __name__ == '__main__':
    data = pd.read_csv("./Dataset/data/matched_urls_output_all.csv")
    data2 = pd.read_csv("./Dataset/data/matched_urls_output_all.csv")
    data["Report Content"] = data["Report Content"].apply(lambda x: x.split("##"))
    data["Report Content"] = data["Report Content"].apply(fun1)
    data2["Report Content"] = data2["Report Content"].apply(lambda x: x.split("##"))
    data2["Report Content"] = data2["Report Content"].apply(fun1)
    #
    # train_set, dev_set = train_test_split(data, test_size=0.1, random_state=43)
    # x_train = train_set.iloc[:, [0, 1, 4]]
    # y_train = train_set.iloc[:, 5]
    # x_dev = dev_set.iloc[:, [0, 1, 4]]
    # y_dev = dev_set.iloc[:, 5]
    # x_test = data2.iloc[:, [0, 1, 4]]
    # y_test = data2.iloc[:, 5]

    train_set, dev_set = train_test_split(data2, test_size=0.9, random_state=43)
    x_train = data.iloc[:, [0, 1,2,3, 4]]
    y_train = data.iloc[:, 5]
    x_dev = data2.iloc[:, [0, 1, 2,3,4]]
    y_dev = data2.iloc[:, 5]
    x_test = data2.iloc[:, [0, 1,2,3, 4]]
    y_test = data2.iloc[:, 5]
    # train_set, dev_set = train_test_split(data2, test_size=0.9, random_state=43)
    # x_train = data.iloc[:, [0, 1, 4]]
    # y_train = data.iloc[:, 5]
    # x_dev = dev_set.iloc[:, [0, 1, 4]]
    # y_dev = dev_set.iloc[:, 5]
    # x_test = data2.iloc[:, [0, 1, 4]]
    # y_test = data2.iloc[:, 5]

    t = pd.DataFrame(x_train.astype(str))


    t["Short News Url"] = t["News Url"].apply(shorten_url)
    # print(t["Short News Url"])
    t["Short Image Url"] = t["Image Url"].apply(shorten_url)
    x_train["new"] = t["Title"] +" "+ t["Ofiicial Account Name"] +" "+ t["Report Content"] + " "+t["Short News Url"]+t["Short Image Url"]
    x_train = x_train.drop(["Title", "Ofiicial Account Name", "Report Content","News Url","Image Url"], axis=1)

    t = pd.DataFrame(x_dev.astype(str))
    t["Short News Url"] = t["News Url"].apply(shorten_url)
    t["Short Image Url"] = t["Image Url"].apply(shorten_url)
    # print(t["Short News Url"])
    x_dev["new"] = t["Title"] + " "+t["Ofiicial Account Name"] +" "+ t["Report Content"] + " "+t["Short News Url"]+t["Short Image Url"]
    x_dev = x_dev.drop(["Title", "Ofiicial Account Name", "Report Content","News Url","Image Url"], axis=1)
    print(x_dev["new"][10])
    t = pd.DataFrame(x_test.astype(str))
    t["Short News Url"] = t["News Url"].apply(shorten_url)
    t["Short Image Url"] = t["Image Url"].apply(shorten_url)
    # print(t["Short News Url"])
    x_test["new"] = t["Title"] +" "+ t["Ofiicial Account Name"] + " "+t["Report Content"] +" "+ t["Short News Url"]+t["Short Image Url"]
    x_test = x_test.drop(["Title", "Ofiicial Account Name", "Report Content","News Url","Image Url"], axis=1)
    # print(x_train["new"][1])
    # train_set, dev_set = train_test_split(data2, test_size=0.5, random_state=43)
    # x_train = data.iloc[:, [0, 1, 4]]
    # y_train = data.iloc[:, 5]
    # x_dev = dev_set.iloc[:, [0, 1, 4]]
    # y_dev = dev_set.iloc[:, 5]
    # x_test = data2.iloc[:, [0, 1, 4]]
    # y_test = data2.iloc[:, 5]

    # t = pd.DataFrame(x_train.astype(str))
    # x_train["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    # x_train = x_train.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    #
    # t = pd.DataFrame(x_dev.astype(str))
    # x_dev["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    # x_dev = x_dev.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    #
    # t = pd.DataFrame(x_test.astype(str))
    # x_test["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    # x_test = x_test.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    dataset = 'Dataset'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(43)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(x_train,y_train,x_dev,y_dev,x_test,y_test,config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    #模型加载
    model.load_state_dict(torch.load(config.save_path))
    train(config, model, train_iter, dev_iter, test_iter)
