import h5py
import numpy as np
from torch.utils.data import Dataset
'''
自定义数据集：训练集和验证集
'''

class TrainDataset(Dataset):#训练集是30通道的数据，但是30通道打乱，所以是单通道数据
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file#指出数据集是哪一个h5文件

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:#均值0.07135591，标准差：0.1137401
            # sr=(f['lr'][idx][...,None] -np.array([0.07135591],dtype=np.float32))/np.array([0.113740],dtype=np.float32)#数据进行标准化，Z=(X-均值)/标准差
            # hr=(f['hr'][idx][...,None] -np.array([0.07135591],dtype=np.float32))/np.array([0.113740],dtype=np.float32)
            lr=f['lr'][idx][None,...]
            hr=f['hr'][idx][None,...]
            # fibre=f['fibre'][idx][...,None]
            fibre_density=f['fibre_density'][idx][None,...]
            # sr=np.array(sr,dtype=np.float32)
            # hr=np.array(hr,dtype=np.float32)
            return lr, hr,fibre_density

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr'])


class EvalDataset(Dataset):#测试集是30通道数据，没有打乱，所以仍是30通道数据
    def __init__(self, h5_file,number):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.number = number

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:#标准化：Z=(x-均值)/标准差


            lr = f[self.number]['lr'][idx]#[..., None]
            hr = f[self.number]['hr'][idx]#[..., None]
            fibre_density=f[self.number]['fibre_density'][idx]
            return lr,  hr, fibre_density

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f[self.number]['hr'])
