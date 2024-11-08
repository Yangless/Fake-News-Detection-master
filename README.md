# 全国大学生人工智能虚假新闻检测省赛

# bert
### 各种bert变体+卷积结构

bert_CNN

bert_DPCNN

bert_RCNN

bert_RNN

ERNIE

ERNIE_RCNN

ERNIE_RCNN_improved

ERNIE_RCNN_roberta_chinese

ERNIE_RCNN_roberta_chinese_improved

ERNIE_RCNN_roberta_chinese_large

ERNIE_RCNN_wwm_chinese

ERNIE_RNN

ERNIE3-base-zh_RCNN

ERNIE3nano_RCNN

ERNIE3xbase_RCNN 

# machine learning
### 经典机器学习方法

词向量化：词袋模型和tyidf

朴素贝叶斯

逻辑回归

支持向量机

随机森林

# deep learning
### 经典深度学习方法

词向量化：Chinese Word Vectors词向量库

CNN

DPCNN

FastText

RNN

TextRCNN

TextRNN_Att

Transformer

## Tips：

进一步过拟合：

```
def shorten_url(url):
    match = re.search(r"(?<=biz=|jpg/)(\w{10})", url)
    if match:
        return match.group(1)
    return " " 
```

```
t["Short News Url"] = t["News Url"].apply(shorten_url)
t["Short Image Url"] = t["Image Url"].apply(shorten_url)
```

噪声标签过滤：

```
filtered_df = df[df.duplicated(subset=['Title', 'Report Content'], keep=False)]
filtered_df = filtered_df[filtered_df.groupby(['Title', 'Report Content'])['label'].transform(lambda x: len(x.unique()) > 1)]
```

使用kimi大模型联网修正标签
