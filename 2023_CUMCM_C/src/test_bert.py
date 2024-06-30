import torch
from transformers import BertTokenizer, BertModel

# 指定权重路径
weight_root = '/Users/drew/Documents/DeepLearning/weights'
weight_path = f'{weight_root}/bert-base-uncased'

# 加载分词器和模型
bert_tokenizer = BertTokenizer.from_pretrained(weight_path)
bert_model = BertModel.from_pretrained(weight_path)

# 准备文本
text = 'My name is Drew'

# 分词并转换为模型输入
inputs = bert_tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    padding='max_length',
    max_length=10,
    truncation=True,  # 截断文本以适应max_length
    return_tensors="pt"
)

# 设置模型为评估模式
bert_model.eval()

# 不计算梯度
with torch.no_grad():
    # 进行推理
    outputs = bert_model(**inputs)

# 获取最后一层的隐藏状态
last_hidden_states = outputs.last_hidden_state

# 打印输出
print(last_hidden_states)
