import torch
import random
import numpy as np
import transformers
import pandas as pd
from math import log
from Bio import SeqIO
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader,Dataset,random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set deterministic CUDA ops
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}.'.format('GPU' if device == 'cuda' else 'CPU (this may be much slower)'))

####################################计算相对位置relative_position方法#######################################

"""
位置信息的角度值（positional thetas）在头维度（head_dim）的每个值上都是不同的，遵循论文中规定的方法。这些角度值用于在保留的并行和循环形式中更新位置嵌入。
实际的角度值在论文中没有具体指定，因此从官方实现中复制了这些值。
"""
def positionThetas(head_dim, scale = 10000, device = "cuda"):
    x = torch.linspace(0, 1, steps=head_dim // 2, device=device)
    thetas = 1 / (scale**x)
    return repeat(thetas, "d -> (d n)", n=2)

def multiplyByi(x):
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(start_dim=-2)

def thetaShift(x, sin, cos):
    return (x * cos) + (multiplyByi(x) * sin)
###########################################################################################################

####################################计算并行retention_parallel#############################################
"""
在 RetNet（保留网络）中，每个保留头部（retention head）的衰减值是不同的，这是按照论文中规定的方法进行的。这里的“衰减值”通常指的是用于计算衰减系数（decay coefficients）的值。
在概念上，作者认为每个头部都有一个不同的“保留窗口”，这是头部可以回顾的过去时间步数的有效数量。这个“保留窗口”表示头部可以关注的时间跨度，而这个时间跨度实际上由衰减系数来决定。
每个头部有一个衰减系数，这个系数决定了该头部可以在过去的时间范围内关注的步数。较大的衰减系数将导致较短的保留窗口，头部能够关注的时间跨度较小；较小的衰减系数则会导致更长的保留窗口，头部可以关注更远的时间跨度。
衰减系数的调整是为了控制不同头部的关注范围，以适应不同的任务需求和数据特性。
"""
def calDecayGammas(num_heads, device):
    xmin, xmax = log(1 / 32), log(1 / 512)
    x = torch.linspace(xmin, xmax, steps=num_heads, device=device)
    return 1 - torch.exp(x)

"""
“衰减掩码是使得并行保留等效于循环保留的关键组成部分之一。衰减系数会被预先计算，并一次性应用于相似性矩阵，而不是像在循环保留的形式中逐个元素地应用。
"""
def calDecayMask(q_len, k_len, decay_gammas, device):
    q_pos = torch.arange(q_len, device=device)
    k_pos = torch.arange(k_len, device=device)
    
    distance = torch.abs(q_pos.unsqueeze(-1) - k_pos.unsqueeze(0)).float()

    # 将上三角距离设置为无穷大，这样只有过去的键才能影响当前的查询。 （将距离设置为无穷大确保在这些位置上衰减矩阵为0，因为在 -1 < x < 1 时，x^(inf) = 0。
    distance_mask = torch.ones_like(distance, dtype=torch.bool).triu_(diagonal=1)
    distance = distance.masked_fill(distance_mask, float("inf"))
    
    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(decay_gammas, "h -> h () ()")
    return decay_gammas**distance

"""
计算并行retention
"""
def retention_parallel(q,k,v):
    decay_gammas = calDecayGammas(num_heads=q.shape[1], device=q.device)
    decay_mask = calDecayMask(q_len=q.shape[2], k_len=k.shape[2], decay_gammas=decay_gammas, device=q.device)
    
    scale = k.size(-1) ** 0.5
    k = k / scale
    
    similarity = einsum(q, k, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, v, "b h n s, b h s d -> b h n d")
    
    return retention, None

###########################################################################################################

class MultiScaleRetention(torch.nn.Module):
    def __init__(self,embedding_dim = 320, num_heads = 4, dropout = 0.1 , activation = "swish", group_norm_eps = 1e-6, device = "cuda", bias = True):
        super(MultiScaleRetention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.activation = torch.nn.functional.silu
        self.head_dim = embedding_dim // num_heads
        
        self.q_projection = torch.nn.Linear(embedding_dim, embedding_dim, bias=bias, device=device)
        self.k_projection = torch.nn.Linear(embedding_dim, embedding_dim, bias=bias, device=device)
        self.v_projection = torch.nn.Linear(embedding_dim, embedding_dim, bias=bias, device=device)
        self.group_norm = torch.nn.GroupNorm(num_groups=num_heads, num_channels=num_heads, affine=False, eps=group_norm_eps, device=device)
        self.g_projection = torch.nn.Linear(embedding_dim, embedding_dim, bias=bias, device=device)
        self.out_projection = torch.nn.Linear(embedding_dim, embedding_dim, bias=bias, device=device)
        
        thetas = positionThetas(head_dim = self.head_dim, device=device)
        self.register_buffer("thetas", thetas)
        self.init_parameters()
    
    def init_parameters(self):
        torch.nn.init.xavier_normal_(self.q_projection.weight)
        if self.q_projection.bias is not None:
            torch.nn.init.constant_(self.q_projection.bias, 0)
        torch.nn.init.xavier_normal_(self.k_projection.weight)
        if self.k_projection.bias is not None:
            torch.nn.init.constant_(self.k_projection.bias, 0)
        torch.nn.init.xavier_normal_(self.v_projection.weight)
        if self.v_projection.bias is not None:
            torch.nn.init.constant_(self.v_projection.bias, 0)
        torch.nn.init.xavier_normal_(self.g_projection.weight)
        if self.g_projection.bias is not None:
            torch.nn.init.constant_(self.g_projection.bias, 0)
        torch.nn.init.xavier_normal_(self.out_projection.weight)
        if self.out_projection.bias is not None:
            torch.nn.init.constant_(self.out_projection.bias, 0)
    
    # parallel并行训练
    def forward(self, query, k, v):
        q = self.q_projection(query)
        k = self.k_projection(k)
        v = self.v_projection(v)
        
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        
        # 计算相对位置 relative_position
        indices = torch.arange(q.size(2), device=q.device)
        indices = rearrange(indices, "n -> () () n ()")
        thetas = rearrange(self.thetas, "d -> () () () d")
        angles = indices * thetas
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        q = thetaShift(q, sin, cos)
        k = thetaShift(k, sin, cos)
        
        retention, weights = retention_parallel(q, k, v)
        
        # 为了以与循环形式等效的方式应用分组归一化，我们将序列维度折叠到批次维度中。否则，归一化将在整个输入序列上应用。
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) h d")
        retention = torch.nn.functional.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        retention = rearrange(retention, "(b n) h d -> b n (h d)", b=batch_size)
        
        # 与多头注意力不同，保留机制论文应用了 "swish" 门，以增加模型的非线性容量。（在我看来，这很可能是为了弥补保留机制中缺少 "softmax" 激活的不足。）
        gate = self.activation(self.g_projection(query))
        retention = self.out_projection(retention * gate)
        
        return retention, weights


"""
主要来自于 'torch.nn.TransformerDecoderLayer'，但有所变化：
    使用 MultiScaleRetention 替代 MultiheadAttention
    没有交叉注意力层，因为保留机制与其不兼容
"""

class RetNetLayer(torch.nn.Module):
    def __init__(self, embedding_dim = 320, num_heads = 4, dim_feedforward = 1024, dropout = 0.1, layer_norm_eps = 1e-6, device = "cuda"):
        super(RetNetLayer,self).__init__()
        self.activation = torch.nn.functional.silu
        self.dropout = torch.nn.Dropout(dropout)
        self.layernorm1 = torch.nn.LayerNorm(embedding_dim, eps=layer_norm_eps, device=device)
        self.layernorm2 = torch.nn.LayerNorm(embedding_dim, eps=layer_norm_eps, device=device)
        self.retention = MultiScaleRetention(embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout, device=device)
        self.linear1 = torch.nn.Linear(embedding_dim, dim_feedforward, device=device)
        self.linear2 = torch.nn.Linear(dim_feedforward, embedding_dim, device=device)
        self.init_parameters()
    
    def init_parameters(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.xavier_normal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0)
    
    def forward(self, x):
        x_tmp = self.layernorm1(x)
        x_tmp, _ = self.retention(x_tmp, x_tmp, x_tmp)
        x_tmp = self.dropout(x_tmp)
        x = x + x_tmp
        
        x_tmp = self.layernorm2(x)
        x_tmp = self.activation(self.linear1(x_tmp))
        x_tmp = self.dropout(x_tmp)
        x_tmp = self.linear2(x_tmp)
        x_tmp = self.dropout(x_tmp)
        x = x + x_tmp
        
        return x
    
class RetNetBlock(torch.nn.Module):
    def __init__(self, retnetLayers, num_layers):
        super(RetNetBlock,self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([deepcopy(retnetLayers) for _ in range(num_layers)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class RetNet(torch.nn.Module):
    def __init__(self,vocbal_size = 33 ,seq_len = 1024, embedding_dim = 320, num_heads = 4, num_layers = 3, device = "cuda", dtype = None, 
                 dropout = 0.1, activation = "swish", dim_feedforward = 1024, norm_first = True,  layer_norm_eps = 1e-6):
        super(RetNet,self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(seq_len, embedding_dim)
        
        retnetLayer = RetNetLayer()
        self.block = RetNetBlock(retnetLayer, num_layers)
        
        self.output = torch.nn.Linear(embedding_dim, vocbal_size, device=device)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.block(x)
        #x = self.output(x)
        return x

class ProjHead(torch.nn.Module):
    def __init__(self,in_dim = 640, hid_dim = 320, out_dim = 33, droupout = 0.1):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(in_dim,hid_dim,bias=True),
            torch.nn.Dropout(droupout,inplace=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim,out_dim)
        )
    
    def forward(self,combined_embedding):
        outputs = self.layer(combined_embedding)
        return outputs

class  SARSCoV2ESM2(torch.nn.Module):
    def __init__(self,esm2,retnet):
        super(SARSCoV2ESM2, self).__init__()
        self.esm2 = esm2
        self.retnet = retnet
        self.head = ProjHead()
    def forward(self,x):
        x_dict = {'input_ids': x.to(device), 'attention_mask': torch.ones(len(x), 225).to(device)}
        esm_outputs = self.esm2(**x_dict).last_hidden_state
        retnet_outputs = self.retnet(x.to(device))
        combined_embedding = torch.cat((esm_outputs,retnet_outputs),dim=2)
        outputs = self.head(combined_embedding)
        return outputs
