import torch
import torch.nn as nn
from transformers import BertModel


class BaseBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = outputs[:, 0, :]
        logits = self.dense(pooled_output)
        # logits = self.softmax(self.dense(pooled_output))
        return logits


class VanillaBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = outputs[:, 0, :]
        # logits = self.dense(pooled_output)
        logits = self.softmax(self.dense(pooled_output))
        return logits


class BertLayerKL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        # tuple of size [bs, length, 768]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state
        layer_cls = [hidden_state[:, 0, :] for hidden_state in hidden_states[:-1]]
        pooled_output = last_hidden_state[:, 0, :]
        # logits = self.dense(pooled_output)
        logits = self.softmax(self.dense(pooled_output))
        return logits, layer_cls
        

# 层级权重系数无监督学习，全0/全1初始化
class AdaptiveLayerBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Sigmoid()
        self.layer_weights = nn.Parameter(torch.ones(self.encoder.config.num_hidden_layers))
    
    def forward(self, input_ids, attention_mask):
        # tuple of size [bs, length, 768]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state
        layer_cls = [hidden_state[:, 0, :] for hidden_state in hidden_states[:-1]]
        pooled_output = last_hidden_state[:, 0, :]
        # logits = self.dense(pooled_output)
        logits = self.softmax(self.dense(pooled_output))
        return logits, layer_cls


# 层级权重系数无监督学习，线性初始化
# 将 layer_weights 初始化为一个从 0 到 1 线性变化的张量，其中每个元素代表对应隐藏层的权重
class LinearLayerBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Sigmoid()
        # 线性初始化 layer_weights
        num_layers = self.encoder.config.num_hidden_layers
        linear_weights = torch.arange(num_layers).float() / (num_layers - 1)
        self.layer_weights = nn.Parameter(linear_weights)
    
    def forward(self, input_ids, attention_mask):
        # tuple of size [bs, length, 768]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state
        layer_cls = [hidden_state[:, 0, :] for hidden_state in hidden_states[:-1]]
        pooled_output = last_hidden_state[:, 0, :]
        # logits = self.dense(pooled_output)
        logits = self.softmax(self.dense(pooled_output))
        return logits, layer_cls


# 层级权重系数无监督学习，指数初始化
# 首先生成一个从 0 到 num_hidden_layers - 1 的线性分布张量 linear_scale，
# 然后通过 torch.exp(linear_scale - num_layers) 将其转换为指数分布的权重。
# 这样，较低的层（接近输入）将获得较小的权重，而较高的层（接近输出）将获得较大的权重。
class ExponentialLayerBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Sigmoid()
        # 按照指数形式初始化 layer_weights
        num_layers = self.encoder.config.num_hidden_layers
        linear_scale = torch.arange(num_layers).float()
        exponential_weights = torch.exp(linear_scale - num_layers)
        self.layer_weights = nn.Parameter(exponential_weights)
    
    def forward(self, input_ids, attention_mask):
        # tuple of size [bs, length, 768]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state
        layer_cls = [hidden_state[:, 0, :] for hidden_state in hidden_states[:-1]]
        pooled_output = last_hidden_state[:, 0, :]
        # logits = self.dense(pooled_output)
        logits = self.softmax(self.dense(pooled_output))
        return logits, layer_cls


# 层级权重系数无监督学习，对数初始化
# 从 1 到 num_hidden_layers 的张量，并对它应用了自然对数函数 torch.log。
# 这样生成的 layer_weights 将按对数方式变化，较低层的权重相对较高，而较高层的权重增长速度减缓
class LogarithmicLayerBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Sigmoid()
        # 按照对数形式初始化 layer_weights
        num_layers = self.encoder.config.num_hidden_layers
        logarithmic_scale = torch.log(torch.arange(1, num_layers + 1).float())
        self.layer_weights = nn.Parameter(logarithmic_scale)
    
    def forward(self, input_ids, attention_mask):
        # tuple of size [bs, length, 768]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state
        layer_cls = [hidden_state[:, 0, :] for hidden_state in hidden_states[:-1]]
        pooled_output = last_hidden_state[:, 0, :]
        # logits = self.dense(pooled_output)
        logits = self.softmax(self.dense(pooled_output))
        return logits, layer_cls

# 随机初始化
class RandomLayerBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Sigmoid()
        # 按照对数形式初始化 layer_weights
        num_layers = self.encoder.config.num_hidden_layers
        self.layer_weights = nn.Parameter(torch.randn(num_layers))
    
    def forward(self, input_ids, attention_mask):
        # tuple of size [bs, length, 768]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state
        layer_cls = [hidden_state[:, 0, :] for hidden_state in hidden_states[:-1]]
        pooled_output = last_hidden_state[:, 0, :]
        # logits = self.dense(pooled_output)
        logits = self.softmax(self.dense(pooled_output))
        return logits, layer_cls