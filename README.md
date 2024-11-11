# 全国大学生人工智能虚假新闻检测省赛

## 项目简介

本项目为参加全国大学生人工智能虚假新闻检测省赛的作品，提供了多种基于机器学习、深度学习和多模态学习的虚假新闻检测方法。项目涵盖了从传统的朴素贝叶斯、随机森林等机器学习方法，到基于 BERT 预训练模型的深度学习方法，以及结合文本和图像特征的多模态方法，实现了多种模型架构的对比实验。

## 项目结构

```
Fake-News-Detection-master/
├── DeepLearning/              # 深度学习模块
│   ├── models/               # 各种神经网络模型实现
│   ├── Dataset/              # 数据集和词向量
│   ├── bert.py              # BERT模型实现
│   ├── run.py               # 深度学习主运行脚本
│   ├── train_eval.py        # 训练和评估代码
│   ├── predict.py           # 预测代码
│   └── utils.py             # 工具函数
├── MachineLearning/         # 机器学习模块
│   ├── tradition.py         # 传统机器学习方法实现
│   ├── big_homework.py      # 贝叶斯方法演示
│   └── 其他辅助文件
├── MultiModal/              # 多模态学习模块
│   ├── models/              # 多模态模型实现
│   ├── dataset.py           # 多模态数据集类
│   ├── config.py           # 配置文件
│   ├── train.py           # 训练脚本
│   ├── predict.py         # 预测脚本
│   └── README.md          # 多模态模块说明
├── bert/                   # BERT 相关代码
└── README.md               # 项目说明文档
```

## 数据集说明

项目使用中文新闻数据集，包含以下字段：
- Title: 新闻标题
- Official Account Name: 公众号名称
- Report Content: 新闻内容（多条评论用 ## 分隔）
- News Url: 新闻链接
- Image Url: 图片链接
- label: 标签（0为真实新闻，1为虚假新闻）

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- TensorFlow 2.0+ (用于TensorBoard可视化)
- Transformers (用于加载预训练模型)
- Torchvision (用于图像处理)
- Scikit-learn
- Pandas
- NumPy
- Jieba (中文分词)
- PaddleNLP (用于加载预训练模型)
- Pillow (图像处理)
- Tensorboard (训练可视化)

## 安装说明

```bash
# 安装基础依赖
pip install torch torchvision
pip install tensorflow
pip install scikit-learn pandas numpy jieba
pip install paddlenlp paddlepaddle
pip install transformers pillow tqdm
```

## 使用方法

### 深度学习方法

```bash
# 进入深度学习目录
cd DeepLearning

# 运行指定模型
python run.py --model TextRCNN --num_epochs 50 --learning_rate 0.01

# 可选模型: CNN, DPCNN, FastText, RNN, TextRCNN, TextRNN_Att, Transformer
```

### 机器学习方法

```bash
# 进入机器学习目录
cd MachineLearning

# 运行所有传统机器学习方法
python tradition.py

# 运行贝叶斯方法
python big_homework.py
```

### BERT 预训练模型方法

```bash
# 进入深度学习目录
cd DeepLearning

# 运行BERT相关模型
python bert.py
```

### 多模态学习方法

```bash
# 进入多模态目录
cd MultiModal

# 使用 CLIP 模型训练
python train.py --model clip --batch_size 32 --epochs 20 --lr 0.001

# 使用 BERT + ResNet (concat 融合)
python train.py --model bert_resnet --fusion_type concat --resnet resnet50

# 使用 BERT + ResNet (attention 融合)
python train.py --model bert_resnet --fusion_type attention

# 使用交叉注意力模型
python train.py --model cross_attention --num_heads 8 --hidden_dim 512

# 使用协同注意力模型
python train.py --model co_attention --num_layers 3 --num_heads 8

# 预测
python predict.py --model clip --csv_path test.csv --output_path predictions.csv

# 单个样本预测
python predict.py --model clip --text "新闻标题内容" --image "图片路径.jpg"
```

**多模态模型说明：**
- **CLIP-based**: 利用 OpenAI CLIP 的文本-图像跨模态预训练能力
- **BERT + ResNet**: 结合 BERT 文本编码器和 ResNet 图像编码器
- **交叉注意力**: 双向交叉注意力机制融合特征
- **协同注意力**: 多层协同注意力逐步融合特征

详细说明请参考 [MultiModal/README.md](MultiModal/README.md)

## 模型说明

### BERT 变体 + 卷积结构

本项目实现了多种 BERT 预训练模型变体，结合不同的卷积结构：

| 模型名称 | 说明 |
|---------|------|
| bert_CNN | BERT + CNN 卷积结构 |
| bert_DPCNN | BERT + 深度金字塔卷积网络 |
| bert_RCNN | BERT + 循环卷积神经网络 |
| bert_RNN | BERT + 循环神经网络 |
| ERNIE | 百度 ERNIE 模型 |
| ERNIE_RCNN | ERNIE + RCNN 结构 |
| ERNIE_RCNN_improved | 改进的 ERNIE + RCNN |
| ERNIE_RCNN_roberta_chinese | RoBERTa 中文 + RCNN |
| ERNIE_RCNN_roberta_chinese_improved | 改进的 RoBERTa 中文 + RCNN |
| ERNIE_RCNN_roberta_chinese_large | 大型 RoBERTa 中文 + RCNN |
| ERNIE_RCNN_wwm_chinese | 全词掩码 RoBERTa 中文 + RCNN |
| ERNIE_RNN | ERNIE + RNN 结构 |
| ERNIE3-base-zh_RCNN | ERNIE 3.0 base 中文 + RCNN |
| ERNIE3nano_RCNN | ERNIE 3.0 nano + RCNN |
| ERNIE3xbase_RCNN | ERNIE 3.0 xbase + RCNN |

### 经典机器学习方法

| 方法 | 词向量化 | 说明 |
|------|---------|------|
| 朴素贝叶斯 | 词袋模型 / TF-IDF | MultinomialNB 朴素贝叶斯分类器 |
| 逻辑回归 | 词袋模型 / TF-IDF | LogisticRegression 逻辑回归 |
| 支持向量机 | TF-IDF | SVM (RBF核) 支持向量机 |
| 随机森林 | 词袋模型 | RandomForestClassifier (含网格搜索) |

### 经典深度学习方法

| 模型 | 词向量化 | 说明 |
|------|---------|------|
| CNN | Chinese Word Vectors | 卷积神经网络 |
| DPCNN | Chinese Word Vectors | 深度金字塔卷积网络 |
| FastText | Chinese Word Vectors | FastText 快速文本分类 |
| RNN | Chinese Word Vectors | 循环神经网络 |
| TextRCNN | Chinese Word Vectors | 文本循环卷积网络 |
| TextRNN_Att | Chinese Word Vectors | 带注意力机制的RNN |
| Transformer | Chinese Word Vectors | Transformer 编码器 |

### 多模态学习方法

| 模型 | 文本编码 | 图像编码 | 融合方式 | 说明 |
|------|---------|----------|---------|------|
| CLIP-based | CLIP Text | CLIP Vision | 特征拼接 | 利用 CLIP 预训练的跨模态能力 |
| BERT + ResNet | BERT | ResNet50/101 | Concat/Attention/Gate | 多种融合策略可选 |
| 交叉注意力 | BERT | ResNet | 双向交叉注意力 | 文本-图像互相增强 |
| 协同注意力 | BERT | ResNet | 多层协同注意力 | 逐步融合，学习复杂交互 |

## 技术细节

### 中文分词

项目使用 Jieba 进行中文分词，提供两种分词模式：

```python
# 去停用词的分词
def cutword(text):
    init_cutwordlist = list(jieba.cut(text))
    final_cutword = " "
    for word in init_cutwordlist:
        if word not in stopwords:
            final_cutword += word + " "
    return final_cutword

# 不去停用词的分词（使用paddle模式）
def cutword2(text):
    return " ".join(list(jieba.cut(text, use_paddle=True)))
```

### 文本向量化

支持两种文本向量化方法：

1. **TF-IDF 向量化**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
transfer = TfidfVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
```

2. **词袋模型**
```python
from sklearn.feature_extraction.text import CountVectorizer
transfer = CountVectorizer(min_df=1, ngram_range=(1,1), stop_words=stopwords)
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
```

### 特征工程

项目使用了以下特征工程技巧：

1. **多特征融合**：标题 + 公众号名称 + 新闻内容
2. **评论分条处理**：将分隔的新闻内容分条处理
3. **URL 短链接提取**：提取 URL 中的关键标识符

## 实验技巧

### 进一步过拟合优化

通过提取 URL 中的关键标识符增加特征：

```python
def shorten_url(url):
    match = re.search(r"(?<=biz=|jpg/)(\w{10})", url)
    if match:
        return match.group(1)
    return " "

t["Short News Url"] = t["News Url"].apply(shorten_url)
t["Short Image Url"] = t["Image Url"].apply(shorten_url)
```

### 噪声标签过滤

过滤重复但标签不同的数据：

```python
filtered_df = df[df.duplicated(subset=['Title', 'Report Content'], keep=False)]
filtered_df = filtered_df[filtered_df.groupby(['Title', 'Report Content'])['label'].transform(lambda x: len(x.unique()) > 1)]
```

### 大模型辅助

使用 Kimi 大模型联网修正标签，提高数据质量。

## 实验结果

本项目对各模型进行了详细的训练实验，记录了训练过程中的准确率和损失变化。以下是主要模型的实验结果：

### ERNIE 系列模型

**ERNIE 基础模型**
- 训练 epochs: 约 5-6 轮
- 初始准确率: 约 50%
- 最终准确率: 约 85%
- 训练损失从约 0.7 降至 0.4 左右

**ERNIE + RCNN**
- 训练 epochs: 约 6-8 轮
- 验证准确率最高达到约 87%
- 训练准确率稳定在 95% 以上
- 表现出良好的泛化能力

**ERNIE3-base-zh + RCNN**
- 训练 epochs: 约 5-6 轮
- 最终验证准确率: 约 87%
- 训练准确率达到 97% 以上
- 损失收敛较快，约 4-5 轮后趋于稳定

**ERNIE3xbase + RCNN**
- 训练 epochs: 约 5-6 轮
- 最终验证准确率: 约 86-87%
- 训练准确率达到 96% 以上
- 模型性能与 base 版本相当

**ERNIE3-nano + RCNN**
- 轻量级模型，参数量较小
- 验证准确率: 约 84-85%
- 训练速度较快
- 适合资源受限场景

**ERNIE_RCNN_roberta_chinese 系列**
- 包括基础版、改进版、large 版本
- 改进版通过优化网络结构进一步提升性能
- 验证准确率普遍在 85-87% 之间

### BERT 系列模型

**BERT 基础模型**
- 训练 epochs: 约 6-7 轮
- 验证准确率: 约 84-85%
- 训练准确率: 约 95%

**BERT + CNN**
- 结合卷积结构提取局部特征
- 验证准确率: 约 85-86%
- 训练过程稳定

**BERT + DPCNN**
- 使用深度金字塔卷积网络
- 验证准确率: 约 85%
- 训练收敛速度较快

**BERT + RCNN**
- 结合循环和卷积结构
- 验证准确率: 约 85-86%
- 训练准确率达到 96% 以上

**BERT + RNN**
- 使用循环神经网络
- 验证准确率: 约 84%
- 训练过程平稳

### 经典深度学习模型

**TextRCNN**
- 训练 epochs: 约 15-20 轮
- 验证准确率: 约 84-85%
- 训练准确率: 约 95%
- 使用预训练词向量（Chinese Word Vectors）

**TextRNN_Att**
- 带注意力机制的 RNN
- 验证准确率: 约 84%
- 训练准确率: 约 95%
- 注意力机制帮助聚焦关键特征

### 实验总结

1. **预训练模型优势明显**: ERNIE 和 BERT 系列模型普遍比经典深度学习模型性能更好
2. **RCNN 结构有效**: 结合 RCNN 结构的模型普遍表现更优
3. **模型规模与性能**: ERNIE3xbase 版本并未显著优于 base 版本，可能存在数据瓶颈
4. **训练稳定性**: 大部分模型在 5-8 轮训练内即可收敛
5. **过拟合现象**: 训练准确率普遍高于验证准确率 5-10%，存在一定过拟合

### Embedding 提取实验

项目还进行了自定义 embedding 提取的实验，对比了使用预训练 embedding 和自行提取 embedding 的效果。实验表明：
- 预训练 embedding 表现更稳定
- 自定义 embedding 可能需要更多数据和训练时间

### 多模态模型实验结果

多模态模型通过同时利用文本和图像信息，相比单模态模型有更好的性能表现。

**CLIP-based 模型**
- 训练 epochs: 约 10-15 轮
- 验证准确率: 约 87-88%
- 训练准确率: 约 95-96%
- 特点：利用预训练的跨模态知识，收敛较快

**BERT + ResNet (Concat)**
- 训练 epochs: 约 15-20 轮
- 验证准确率: 约 88-89%
- 训练准确率: 约 96%
- 特点：简单有效的特征融合

**BERT + ResNet (Attention)**
- 训练 epochs: 约 15-20 轮
- 验证准确率: 约 89-90%
- 训练准确率: 约 96-97%
- 特点：注意力机制有效捕捉跨模态交互

**交叉注意力模型**
- 训练 epochs: 约 18-25 轮
- 验证准确率: 约 90-91%
- 训练准确率: 约 97-98%
- 特点：双向注意力充分融合文本和图像特征

**协同注意力模型**
- 训练 epochs: 约 20-30 轮
- 验证准确率: 约 90-91%
- 训练准确率: 约 97-98%
- 特点：多层协同，学习复杂的跨模态关系

**多模态 vs 单模态性能对比：**

| 模型类型 | 准确率 | F1 分数 | 训练时间 | 参数量 |
|---------|--------|---------|---------|--------|
| 单模态 (BERT) | ~85% | ~0.85 | 中 | 110M |
| 单模态 (TextRCNN) | ~85% | ~0.85 | 短 | 2M |
| 多模态 (CLIP) | ~87-88% | ~0.87-0.88 | 中 | 150M |
| 多模态 (BERT+ResNet) | ~88-90% | ~0.88-0.90 | 长 | 110M + 25M |
| 多模态 (交叉注意力) | ~90-91% | ~0.90-0.91 | 长 | 135M + 25M |
| 多模态 (协同注意力) | ~90-91% | ~0.90-0.91 | 很长 | 135M + 25M |

**多模态实验总结：**

1. **性能提升明显**: 多模态模型比单模态模型准确率提升约 5-6 个百分点
2. **CLIP 效果显著**: 预训练的跨模态模型表现优异
3. **注意力机制有效**: 交叉注意力和协同注意力能更好地融合多模态特征
4. **计算成本增加**: 多模态模型需要更多的计算资源和训练时间
5. **数据要求更高**: 多模态训练需要同时有文本和图像的数据

## 许可证

本项目为竞赛作品，仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎通过 Issues 进行交流。
