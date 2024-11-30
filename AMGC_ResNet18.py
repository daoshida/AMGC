import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SKConv(nn.Module):
    def __init__(self, channel, M=3, r=16, stride=1, kernel_size=7):
        super(SKConv, self).__init__()
        d = int(channel/r)       #压缩后的通道数
        pad = kernel_size // 2
        self.M = M       #尺度分支的个数
        self.channel = channel     #输入通道数
        
        # channel attention
        self.gap = nn.AdaptiveAvgPool2d((1,1))  #全局平均池化（4，c，1，1）
        self.fc = nn.Sequential(nn.Conv2d(channel, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))    #进行压缩（4，c，1，1）->（4，d，1，1）
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, channel, kernel_size=1, stride=1)
            )     #将通道数还原，同时生成3各特征层
            
        # spatial attention
        self.cons = nn.ModuleList([])
        for i in range(M):
            self.cons.append(
                 nn.Conv2d(2, 1, kernel_size=7, padding=pad, bias=False)
            )     #将最大化及平均化后的特征图通道数变为1，同时生成3各特征层
        
        self.softmax = nn.Softmax(dim=1)  #按不同的特征层方向进行softmax
        
    def forward(self, x_1,x_2,x_3):
        
        batch_size = x_1.shape[0]   #4
           
        channel = torch.cat((x_1,x_2,x_3), dim=1)
        channel = channel.view(batch_size, self.M, self.channel, channel.shape[2], channel.shape[3])  #(4,3,c=x_1的通道数,w,h)
        
        channel_U = torch.sum(channel, dim=1)#按照不同尺度的维度进行求和（4，c，w，h）
        # channel attention
        channel_S = self.gap(channel_U)#（4，c，1，1）
        channel_Z = self.fc(channel_S)#（4，d，1，1）
 
        attention_vectors = [fc(channel_Z) for fc in self.fcs]#生成三个（4，c，1，1）
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.channel, 1, 1)#（4，3，c，1，1）
        attention_vectors = self.softmax(attention_vectors)#（4，3，c，1，1）
        channel_V = torch.sum(channel*attention_vectors, dim=1)#（4，c，w，h）
        
        # spatial attention
        level_weight_max = torch.max(channel_U, dim=1, keepdim=True).values
        level_weight_avg = torch.mean(channel_U, dim=1, keepdim=True)
        level_weight_Z = torch.cat([level_weight_max,level_weight_avg],dim=1)
        
        level_weight = [con(level_weight_Z) for con in self.cons]
        level_weight = torch.cat(level_weight, dim=1)
        level_weight = level_weight.view(batch_size, self.M, 1, level_weight.shape[2], level_weight.shape[3])
        level_weight = self.softmax(level_weight)
        spatial_V = torch.sum(channel*level_weight, dim=1)
       
        return channel_V + spatial_V
    
    
class Attention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0.1):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float))

    def forward(self, query, key, value, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x
    

class Conv2d_bn(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation_rate=1, relu=True, bn=True, bias=False):
        super(Conv2d_bn, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation_rate,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self,x):
        x = self.conv2d(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    
class RFB(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1,scale=0.1):
        super(RFB, self).__init__()
        inter_planes = in_planes // 8

        self.branch_0 = nn.Sequential(
            Conv2d_bn(in_planes, inter_planes * 2, kernel_size=1, stride=stride),
            Conv2d_bn(inter_planes * 2,  inter_planes * 2, kernel_size=3, stride=1, padding=1,dilation_rate=1, relu=False)
        )

        self.branch_1 = nn.Sequential(
            Conv2d_bn(in_planes, inter_planes, kernel_size=1, stride=1),
            Conv2d_bn(inter_planes, inter_planes * 2, kernel_size=3, stride=stride, padding=1),
            Conv2d_bn(inter_planes * 2, inter_planes * 2, kernel_size=3, stride=1, padding=3, dilation_rate=3, relu=False)
        )

        self.branch_2 = nn.Sequential(
            Conv2d_bn(in_planes, inter_planes, kernel_size=1, stride=1),
            Conv2d_bn(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            Conv2d_bn((inter_planes // 2) * 3, inter_planes * 2, kernel_size=3, stride=stride, padding=1),
            Conv2d_bn(inter_planes * 2, inter_planes * 2, kernel_size=3, stride=1, padding=5, dilation_rate=5, relu=False)
        )

        self.ConvLinear = Conv2d_bn(2 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.sk = SKConv(2 * inter_planes)
        self.selfattention = Attention(hid_dim=in_planes, n_heads=4, dropout=0.1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        
        out = self.sk(x0, x1, x2)
        out = self.ConvLinear(out)

#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         out = self.relu(out)
#         out = torch.reshape(out, (out.size(0), -1 , out.size(1))) 

        y = torch.reshape(x, (x.size(0), -1 , x.size(1))) 
        y = self.selfattention(y, y, y)
        y = torch.reshape(y, (y.size(0), y.size(-1), 40, 40)) 
#         y = y.view(out.size(0), -1)
        out = out * y

        return out
    
    
#定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=6):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(3,2,1)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.rfb = RFB(512,512)
        self.fc = nn.Linear(819200, num_classes)
        
    #这个函数主要是用来，重复同一个残差块    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.rfb(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out