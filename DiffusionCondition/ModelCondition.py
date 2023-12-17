'''

网络结构还未修改
'''
   
import math
from telnetlib import PRAGMA_HEARTBEAT
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

class Swish(nn.Module):#激活函数
    def forward(self, x):
        return x * torch.sigmoid(x)


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


class ConditionalEmbedding(nn.Module):#条件编码
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(#对条件进行编码，有10个条件，先编码成128维，再通过全连接编码成512维，这个维度和时间编码一致
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),#输入条件是低分图像，所以第一个参数是11
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb


class DownSample(nn.Module):#下采样
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb=None):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):#上采样
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x, temb, cemb=None):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x


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



class ResBlock(nn.Module):#残差块
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True,downstep=0):#输入维度，输出维度，时间维度，失活比例，是否使用注意力
        super().__init__()
        self.block1 = nn.Sequential(#残差块有两层，这是第一层
            nn.GroupNorm(32, in_ch),#上一个残差块的标准化
            Swish(),#上一个残差块的激活
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),#3*3卷积，本残差块的卷积
        )
        self.temb_proj = nn.Sequential(#使用全连接调整时间编码的维度
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        # self.cond_proj = nn.Sequential(#使用全连接调整条件编码的维度
        #     Swish(),
        #     nn.Linear(tdim, out_ch),#条件编码的维度和时间编码的维度相同
        # )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:#输入和输出通道数不同，使用1*1卷积调整维度
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:#输入和输出通道数相同，使用空层占位
            self.shortcut = nn.Identity()
        if attn:#使用注意力
            self.attn = AttnBlock(out_ch)
        else:#使用空层占位
            self.attn = nn.Identity()
        self.down_step=nn.Sequential()
        if downstep>0:
            for i in range(downstep):
                # self.down_step
                self.down_step.append(nn.Conv2d(1,1,1,2))
        else:
            self.down_step=nn.Identity()


    def forward(self, x, temb, labels=None):
        h = self.block1(x)#经过第一层后，与时间编码，条件编码进行叠加
        h += self.temb_proj(temb)[:, :, None, None]
        if labels is not None:
            l=self.down_step(labels)
            h +=l
        h = self.block2(h)#残差块的第二层

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
        
class ESPCN(nn.Module):
    def __init__(self,upscale_factor=2):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self,x):
        x=F.tanh(self.conv1(x))
        x=F.tanh(self.conv2(x))
        x=self.conv3(x)
        x=self.pixel_shuffle(x)
        x=F.sigmoid(x)
        return x

class EncoderTDI(nn.Module):
    def __init__(self,ch,in_ch=1,out_ch=1):
        super(EncoderTDI, self).__init__()

        self.fibre_density_Conv1 = nn.Sequential(
            nn.Conv2d(1, ch, (3, 3), (1, 1), padding=(1, 1)),
            nn.GroupNorm(32, ch),
            Swish(),
            # nn.Conv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            # nn.GroupNorm(32, ch),
        )
        # self.atten = AttnBlock(ch)
        self.fibre_density_Conv2 = nn.Sequential(
            nn.Conv2d(ch,out_ch,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.GroupNorm(1,out_ch)
        )
        self.active = Swish()
        self.inchange = nn.Conv2d(in_ch,out_ch,1,1)
    def forward(self,input):
        x=self.fibre_density_Conv1(input)
        # x=self.atten(x)
        x = self.fibre_density_Conv2(x)
        y = self.inchange(input)
        out = x+y
        out=self.active(out)
        return out

class UNet(nn.Module):
    def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.espcn=ESPCN()
        self.fibre_density_map_Conv=EncoderTDI(ch=ch)
        # self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)#开始一个卷积
        self.downblocks = nn.ModuleList()
        chs = []  # record output channel when dowmsample for upsample
        now_ch = ch
        # down_step=0
        for i, mult in enumerate(ch_mult):#下采样部分
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout,))#downstep=down_step))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                # down_step+=1
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([#中间部分
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),#,downstep=down_step),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False)#,downstep=down_step),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):#上采样部分
            out_ch = ch * mult

            for _ in range(num_res_blocks):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
                chs.pop()

        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),#倒数第二层的标准化和激活
            Swish(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1),#最后一层只有卷积
            #将最后输出压缩到0-1范围内
        )
 

    def forward(self, x, t, labels,fibre_density):
        # Timestep embedding，不论是下采样，中间，上采样部分，都会与时间、条件编码进行叠加
        temb = self.time_embedding(t)
        fibre_density=self.fibre_density_map_Conv(fibre_density)
        labels=self.espcn(labels)
        x=torch.cat((x,labels,fibre_density),dim=1)#拼接条件,条件的比重更大，占75%
        # print(x.shape,temb.shape)
        # Downsampling
        h = self.head(x)
        hs = []#暂存下采样结果

        for layer in self.downblocks:
            h = layer(h, temb)#,labels)
            if isinstance(layer,ResBlock):
                hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)#,labels)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


# if __name__ == '__main__':
#     batch_size = 8
#     model = UNet(
#         T=1000, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
#         num_res_blocks=2, dropout=0.1)
#     x = torch.randn(batch_size, 3, 32, 32)
#     t = torch.randint(1000, size=[batch_size])
#     labels = torch.randint(10, size=[batch_size])
#     # resB = ResBlock(128, 256, 64, 0.1)
#     # x = torch.randn(batch_size, 128, 32, 32)
#     # t = torch.randn(batch_size, 64)
#     # labels = torch.randn(batch_size, 64)
#     # y = resB(x, t, labels)
#     y = model(x, t, labels)
#     print(y.shape)

