import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class Swish(nn.Module):#激活函数
    def forward(self, x):
        return x * torch.sigmoid(x)
class AttnBlock(nn.Module):#注意力
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)#三部分的线性变换
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)#最后再经过线性变换

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)#先组标准化
        q = self.proj_q(h)#q做线性变换
        k = self.proj_k(h)#k做线性变换
        v = self.proj_v(h)#v做线性变换

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)#q调整维度，成三维
        k = k.view(B, C, H * W)#k的转置
        w = torch.bmm(q, k) * (int(C) ** (-0.5))#第一次矩阵乘法q*k^T/根号下d
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)#

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)#v调整维度，成三维
        h = torch.bmm(w, v)#第二次矩阵乘法
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)#最终结果调整维度B*H*W*C
        h = self.proj(h)#最后再经过线性变换

        return x + h#做个残差连接
class TimeEmbedding(nn.Module):#时间编码
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)#编码维度：128，这里是64
        pos = torch.arange(T).float()#这里指出多少个时间进行编码
        emb = pos[:, None] * emb[None, :]#这里矩阵乘法得到T行，emb列的矩阵
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)#emb通过sin和cos操作，得到时间编码，128维的编码，这个编码是embedding的初始权重，但是权重不更新，因此就是时间的编码
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb
class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels,tdim=None, pre_Batch_Norm=True,attn=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tdim=tdim
        if(tdim!=None):
            self.time_proj=nn.Sequential(
                #使用全连接调整时间编码的维度
                Swish(),
                nn.Linear(tdim,out_channels)
            )

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU())
        if(attn):
            self.attn=AttnBlock(out_channels)
        else:
            self.attn=nn.Identity()

    def forward(self, x,temb=None):
        x = self.Conv_forward(x)
        if(temb!=None):
            x=x+self.time_proj(temb)[:,:,None,None]
        x=self.attn(x)
        return x


class UnetPlusPlus(nn.Module):
    def __init__(self,T=0, num_classes=3,ch=64,ch_mult=[1,2,4,8,16], deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        tdim=ch*4
        self.time_embedding=TimeEmbedding(T,d_model=ch,dim=tdim)

        self.filters =[ch*i for i in ch_mult]# [64, 128, 256, 512, 1024]

        self.CONV3_1 = ContinusParalleConv(self.filters[3] * 2, self.filters[3], pre_Batch_Norm=True)

        self.CONV2_2 = ContinusParalleConv(self.filters[2] * 3, self.filters[2], pre_Batch_Norm=True)
        self.CONV2_1 = ContinusParalleConv(self.filters[2] * 2, self.filters[2], pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(self.filters[1] * 2, self.filters[1], pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(self.filters[1] * 3, self.filters[1], pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(self.filters[1] * 4, self.filters[1], pre_Batch_Norm=True)

        self.CONV0_1 = ContinusParalleConv(self.filters[0] * 2, self.filters[0], pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(self.filters[0] * 3, self.filters[0], pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(self.filters[0] * 4, self.filters[0], pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(self.filters[0] * 5, self.filters[0], pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(self.num_classes*2, self.filters[0], pre_Batch_Norm=False,tdim=tdim,attn=False)
        self.stage_1 = ContinusParalleConv(self.filters[0], self.filters[1], pre_Batch_Norm=False,tdim=tdim,attn=False)
        self.stage_2 = ContinusParalleConv(self.filters[1], self.filters[2], pre_Batch_Norm=False,tdim=tdim,attn=False)
        self.stage_3 = ContinusParalleConv(self.filters[2], self.filters[3], pre_Batch_Norm=False,tdim=tdim)
        self.stage_4 = ContinusParalleConv(self.filters[3], self.filters[4], pre_Batch_Norm=False,tdim=tdim)

        self.pool = nn.MaxPool2d(2)

        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=self.filters[4], out_channels=self.filters[3], kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=self.filters[3], out_channels=self.filters[2], kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=self.filters[3], out_channels=self.filters[2], kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=self.filters[2], out_channels=self.filters[1], kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=self.filters[2], out_channels=self.filters[1], kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=self.filters[2], out_channels=self.filters[1], kernel_size=4, stride=2, padding=1)

        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=self.filters[1], out_channels=self.filters[0], kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=self.filters[1], out_channels=self.filters[0], kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=self.filters[1], out_channels=self.filters[0], kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=self.filters[1], out_channels=self.filters[0], kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
            nn.BatchNorm2d(self.filters[0]),
            nn.ReLU(),
            nn.Conv2d(self.filters[0], self.num_classes, 3, padding=1),
        )
        self.final_super_0_2 = nn.Sequential(
            nn.BatchNorm2d(self.filters[0]),
            nn.ReLU(),
            nn.Conv2d(self.filters[0], self.num_classes, 3, padding=1),
        )
        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(self.filters[0]),
            nn.ReLU(),
            nn.Conv2d(self.filters[0], self.num_classes, 3, padding=1),
        )
        self.final_super_0_4 = nn.Sequential(
            nn.BatchNorm2d(self.filters[0]),
            nn.ReLU(),
            nn.Conv2d(self.filters[0], self.num_classes, 3, padding=1),
        )

    def forward(self, x,t,labels=None):

        temb=self.time_embedding(t)
        x=torch.cat((x,labels),dim=1)
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0),temb)
        x_2_0 = self.stage_2(self.pool(x_1_0),temb)
        x_3_0 = self.stage_3(self.pool(x_2_0),temb)
        x_4_0 = self.stage_4(self.pool(x_3_0),temb)

        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)

        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)

        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            out_put=(out_put1+out_put2+out_put3+out_put4)/4
            return out_put#[out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)


if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 1, 224, 224)).to(device)
    T = 1000
    t = torch.randint(T, size=(1,))
    model = UnetPlusPlus(num_classes=1,T=T, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs,t)
    print(outputs.shape)

    print("deep_supervision: True")
    deep_supervision = True
    model = UnetPlusPlus(num_classes=1, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    for out in outputs:
        print(out.shape)