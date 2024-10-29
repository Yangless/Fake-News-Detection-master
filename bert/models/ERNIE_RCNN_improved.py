# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'ERNIE_RCNN_improved'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 100                                             # epoch数
        self.batch_size = 128
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5                                       # 学习率
        self.bert_path = './ernie-1.0-base-zh'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.5
        self.rnn_hidden = 256
        self.num_layers = 2
        self.class_weights = torch.tensor([0.2, 0.8], device=self.device)  # 类别权重处理不均衡数据


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attention_weights = torch.tanh(self.attention(lstm_output)).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_output = lstm_output * attention_weights.unsqueeze(-1)
        return weighted_output.sum(dim=1)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True  # 微调BERT参数
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.self_attention = SelfAttention(config.rnn_hidden * 2)
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子同样大小，padding部分用0表示
        encoder_out, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        lstm_out, _ = self.lstm(encoder_out)  # 使用LSTM处理BERT的输出
        attn_out = self.self_attention(lstm_out)  # 添加自注意力机制
        out = self.dropout(attn_out)
        out = self.fc(out)
        return out

# Loss Function with Class Weights
def loss_fn(outputs, targets, class_weights):
    return F.cross_entropy(outputs, targets, weight=class_weights)

# Learning rate scheduler
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
