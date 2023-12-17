import argparse

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from dipy.io.image import load_nifti,save_nifti
from dipy.align.reslice import reslice
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import Resize
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train(args):
    #训练集只有一个梯度方向的切片
    print('开始训练集')
    h5_file_x = h5py.File(args.output_train_path_x, 'w')
    h5_file_y = h5py.File(args.output_train_path_y, 'w')
    h5_file_z = h5py.File(args.output_train_path_z, 'w')

    lr_patches_x = []
    hr_patches_x = []

    fibreDensity_patches_x=[]
    lr_patches_y = []
    hr_patches_y = []
    fibre_patches_y=[]
    fibreDensity_patches_y=[]
    lr_patches_z = []
    hr_patches_z = []

    fibreDensity_patches_z=[]


    num_list=["685058","748662","751348","756055","837560","845458","896778","901442","979984","984472"]

    for i, number in enumerate(num_list):#sorted(glob.glob('{}/*'.format(args.images_dir))):
        print(f'--------------------------------------------开始{number}------------------------------------------------------')
        image_path=os.path.join(args.images_train_dir,f'{number}_hr.nii.gz')#高清的图像的路径
        mask_path=os.path.join(args.images_train_dir,f'{number}_mask.nii.gz')#mask图像的路径
        b0_path=os.path.join(args.images_train_dir,f'{number}_b0.nii.gz')

        fibre_density_path=os.path.join(args.fibre_density_dir,f'{number}_tdi.nii.gz')
        #读取数据
        b0,affine_b0,voxel_size_b0=load_nifti(b0_path,return_voxsize=True)#读取b0数据
        data,affine,voxel_size_hr=load_nifti(image_path,return_voxsize=True)#读取高清图像
        mask,affine_mask,voxel_size_mask=load_nifti(mask_path,return_voxsize=True)#读取mask
        mask = mask.astype(np.int32)

        data_fibre_density,_ = load_nifti(fibre_density_path)
        data_fibre_density = data_fibre_density.astype(np.float32)

        np.savetxt(f'{args.affine_dir}/{number}_data_affine',affine)
        np.savetxt(f'{args.affine_dir}/{number}_b0_affine',affine_b0)
        np.savetxt(f'{args.affine_dir}/{number}_mask_affine',affine_mask)

        #调整高分图像的尺寸:对数据进行裁剪
        src_x,src_y,src_z,src_b=data.shape#原图的大小
        # data=data[src_x//2-args.img_src_x//2:src_x//2+args.img_src_x//2,src_y//2-args.img_src_y//2:src_y//2+args.img_src_y//2,src_z//2-args.img_src_z//2:src_z//2+args.img_src_z//2,:]
        # mask=mask[src_x//2-args.img_src_x//2:src_x//2+args.img_src_x//2,src_y//2-args.img_src_y//2:src_y//2+args.img_src_y//2,src_z//2-args.img_src_z//2:src_z//2+args.img_src_z//2]
        # data_b0 = b0[src_x // 2 - args.img_src_x // 2:src_x // 2 + args.img_src_x // 2,
        #           src_y // 2 - args.img_src_y // 2:src_y // 2 + args.img_src_y // 2,
        #           src_z // 2 - args.img_src_z // 2:src_z // 2 + args.img_src_z // 2, :]
        # data_fibre=data_fibre[src_x//2-args.img_src_x//2:src_x//2+args.img_src_x//2,
        #            src_y//2-args.img_src_y//2:src_y//2+args.img_src_y//2,src_z//2-args.img_src_z//2:src_z//2+args.img_src_z//2,:]

        data_fibre_density=data_fibre_density[src_x//2-args.img_src_x//2:src_x//2+args.img_src_x//2,
                            src_y//2-args.img_src_y//2:src_y//2+args.img_src_y//2,
                             src_z//2-args.img_src_z//2:src_z//2+args.img_src_z//2][...,None]

        print('归一化前，密度图最大值：', np.max(data_fibre_density), 'data最大：', np.max(data),'b0最大：',np.max(b0))
        '''
        对数据进行归一化
        '''
        for ci in range(data.shape[3]):
            # data[:, :, :, ci] = np.squeeze(data[:, :, :, ci]) / np.squeeze(data_b0)
            data[:, :, :, ci] = data[:, :, :, ci] * mask
        #对纤维密度进行归一化
        data_fibre_density[...,0]=np.squeeze(data_fibre_density[...,0])/np.squeeze(b0)
        data_fibre_density[...,0]=data_fibre_density[...,0]*mask

        # 归一化后的异常值的处理：
        data[np.isinf(data)] = 0.0
        data[np.isnan(data)] = 0.0
        data[data > 1.0] = 1.0
        data[data < 0.0] = 0.0

        data_fibre_density[np.isnan(data_fibre_density)]=0.0
        data_fibre_density[np.isinf(data_fibre_density)]=0.0
        data_fibre_density[data_fibre_density>1.0]=1.0
        data_fibre_density[data_fibre_density<0.0]=0.0
        print( '密度图平均值：', np.mean(data_fibre_density),'data平均：',np.mean(data))
        print('密度图最大值：', np.max(data_fibre_density), 'data最大：', np.max(data))

        #获取低分图像，通过平均池化
        data_hr=torch.tensor(data.copy(),dtype=torch.float32).permute(3,0,1,2)
        data_fibre_density=torch.tensor(data_fibre_density.copy(),dtype=torch.float32).permute(3,0,1,2)

        X,Y,Z=mask.shape
        '''
        先开始z方向
        '''
        # ii=0
        for zi in range(Z):#切片的索引
            #这30通道就取开头30个通道
            for bi in range(0,30):#一个subject30个通道，但是要打乱它，所以就一个一个通道一个通道喂入
                hr=data_hr[bi,:,:,zi]
                lr=Resize((X//args.scale,Y//args.scale),Image.BICUBIC)(hr[None,...])[0]
                # fibre=data_fibre[:,:,zi,0]
                fibre_density=data_fibre_density[0,:,:,zi]

                fibreDensity_patches_z.append(fibre_density)
                hr_patches_z.append(hr)
                lr_patches_z.append(lr)
                # ii+=1

        '''
        :y方向
        '''
        # ii=0
        for yi in range(Y):#切片的索引

            #这30通道就取开头30个通道
            for bi in range(0,30):#一个subject30个通道，但是要打乱它，所以就一个一个通道一个通道喂入
                hr=data_hr[bi,:,yi,:]
                lr=Resize((X//args.scale,Z//args.scale),Image.BICUBIC)(hr[None,...])[0]

                fibre_density=data_fibre_density[0,:,yi,:]
                #下面把切片放到列表中
                fibreDensity_patches_y.append(fibre_density)
                # fibre_patches_y.append(fibre)
                hr_patches_y.append(hr)
                lr_patches_y.append(lr)

        '''
        x方向
        '''
        for xi in range(X):  # 切片的索引

            # 这30通道就取开头30个通道
            for bi in range(0, 30):  # 一个subject30个通道，但是要打乱它，所以就一个一个通道一个通道喂入
                hr = data_hr[bi,xi, :, :]
                lr = Resize((Y//args.scale,Z//args.scale),Image.BICUBIC)(hr[None,...])[0]
                fibre_density=data_fibre_density[0,xi,:,:]

                # fibre_patches_x.append(fibre)
                hr_patches_x.append(hr)
                lr_patches_x.append(lr)
                fibreDensity_patches_x.append(fibre_density)

    print('x方向开始转为ndarray')
    lr_patches_x = torch.tensor([ x.detach().numpy() for x in lr_patches_x])
    hr_patches_x = torch.tensor([x.detach().numpy() for x in hr_patches_x])
    # fibre_patches_x=np.array(fibre_patches_x,dtype=np.float32)
    fibreDensity_patches_x=torch.tensor([x.detach().numpy() for x in fibreDensity_patches_x])
    print('x方向低分形状：',lr_patches_x.shape,'对应的高分：',hr_patches_x.shape,'对应的纤维密度：',fibreDensity_patches_x.shape)
    print('y方向开始转为ndarray')
    fibreDensity_patches_y=torch.tensor([x.detach().numpy() for x in fibreDensity_patches_y])
    lr_patches_y = torch.tensor([x.detach().numpy() for x in lr_patches_y])
    hr_patches_y = torch.tensor([x.detach().numpy() for x in hr_patches_y])
    print('y方向低分形状：',lr_patches_y.shape,'对应的高分形状：',hr_patches_y.shape,'对应的纤维密度：',fibreDensity_patches_y.shape)
    print('z方向开始转为ndarray')
    lr_patches_z = torch.tensor([x.detach().numpy() for x in lr_patches_z])
    hr_patches_z = torch.tensor([x.detach().numpy() for x in hr_patches_z])
    fibreDensity_patches_z=torch.tensor([x.detach().numpy() for x in fibreDensity_patches_z])
    print('z方向低分形状：',lr_patches_z.shape,'对应的高分形状：',hr_patches_z.shape,'对应的纤维密度：',fibreDensity_patches_z.shape)
    print('开始存入h5file')
    h5_file_x.create_dataset('lr', data=lr_patches_x)
    h5_file_x.create_dataset('hr', data=hr_patches_x)
    h5_file_x.create_dataset('fibre_density',data=fibreDensity_patches_x)

    h5_file_y.create_dataset('lr', data=lr_patches_y)
    h5_file_y.create_dataset('hr', data=hr_patches_y)
    h5_file_y.create_dataset('fibre_density',data=fibreDensity_patches_y)

    h5_file_z.create_dataset('lr', data=lr_patches_z)
    h5_file_z.create_dataset('hr', data=hr_patches_z)
    h5_file_z.create_dataset('fibre_density',data=fibreDensity_patches_z)


    h5_file_x.close()
    h5_file_y.close()
    h5_file_z.close()
    print('训练集h5文件完成')

    # eval(args, image_list[-1000:-500])
    # test(args, image_list[-500:])


def test(args,image_list=None):
    #测试集要三个方向的切片
    print('开始测试集h5文件')

    num_list = ["991267","898176","761957","901038","833148"]
    ii = 0

    h5_file_x = h5py.File(args.output_test_path_x, 'w')
    h5_file_y = h5py.File(args.output_test_path_y, 'w')
    h5_file_z = h5py.File(args.output_test_path_z, 'w')

    for index, number in enumerate(num_list):  # sorted(glob.glob('{}/*'.format(args.images_dir))):
        gx = h5_file_x.create_group(number)
        gy = h5_file_y.create_group(number)
        gz = h5_file_z.create_group(number)

        print('测试集：',number)
        image_path = os.path.join(args.images_test_dir, f'{number}_hr.nii.gz')  # 高清的图像的路径
        mask_path = os.path.join(args.images_test_dir, f'{number}_mask.nii.gz')  # mask图像的路径
        b0_path = os.path.join(args.images_test_dir, f'{number}_b0.nii.gz')
        fibre_density_path = os.path.join(args.fibre_density_dir, f'{number}_tdi.nii.gz')

        # 读取数据

        data, affine, voxel_size_hr = load_nifti(image_path, return_voxsize=True)  # 读取高清图像
        mask, affine_mask, voxel_size_mask = load_nifti(mask_path, return_voxsize=True)  # 读取mask
        mask = mask.astype(np.int32)
        b0, affine_b0, voxel_size_b0 = load_nifti(b0_path, return_voxsize=True)  # 读取b0数据
        # data_fibre, _ = load_nifti(fibre_data_path)#读取纤维图
        data_fibre_density, _ = load_nifti(fibre_density_path)#读取纤维密度
        data_fibre_density=data_fibre_density.astype(np.float32)

        np.savetxt(f'{args.affine_dir}/{number}_data_affine', affine)
        np.savetxt(f'{args.affine_dir}/{number}_b0_affine', affine_b0)
        np.savetxt(f'{args.affine_dir}/{number}_mask_affine', affine_mask)


        # 调整高分图像的尺寸:对数据进行裁剪
        src_x, src_y, src_z, src_b = data.shape  # 原图的大小
        # data = data[src_x // 2 - args.img_src_x // 2:src_x // 2 + args.img_src_x // 2,
        #           src_y // 2 - args.img_src_y // 2:src_y // 2 + args.img_src_y // 2,
        #           src_z // 2 - args.img_src_z // 2:src_z // 2 + args.img_src_z // 2, :]
        # mask = mask[src_x // 2 - args.img_src_x // 2:src_x // 2 + args.img_src_x // 2,
        #        src_y // 2 - args.img_src_y // 2:src_y // 2 + args.img_src_y // 2,
        #        src_z // 2 - args.img_src_z // 2:src_z // 2 + args.img_src_z // 2]

        # data_b0=b0[src_x // 2 - args.img_src_x // 2:src_x // 2 + args.img_src_x // 2,
        #           src_y // 2 - args.img_src_y // 2:src_y // 2 + args.img_src_y // 2,
        #           src_z // 2 - args.img_src_z // 2:src_z // 2 + args.img_src_z // 2, :]
        # data_fibre = data_fibre[src_x // 2 - args.img_src_x // 2:src_x // 2 + args.img_src_x // 2,
        #              src_y // 2 - args.img_src_y // 2:src_y // 2 + args.img_src_y // 2,
        #              src_z // 2 - args.img_src_z // 2:src_z // 2 + args.img_src_z // 2, :]
        data_fibre_density = data_fibre_density[src_x // 2 - args.img_src_x // 2:src_x // 2 + args.img_src_x // 2,
                             src_y // 2 - args.img_src_y // 2:src_y // 2 + args.img_src_y // 2,
                             src_z // 2 - args.img_src_z // 2:src_z // 2 + args.img_src_z // 2][..., None]
        print('密度图最大值：', np.max(data_fibre_density), 'data最大：', np.max(data), 'b0最大：', np.max(b0))

        '''
               对数据进行归一化
               '''
        for i in range(data.shape[3]):
            # data[:, :, :, i] = np.squeeze(data[:, :, :, i]) / np.squeeze(data_b0)
            data[:, :, :, i] = data[:, :, :, i] * mask
        data_fibre_density[..., 0] = np.squeeze(data_fibre_density[..., 0]) / np.squeeze(b0)
        data_fibre_density[..., 0] = data_fibre_density[..., 0] * mask

        # 归一化后的异常值的处理：
        data[np.isinf(data)] = 0.0
        data[np.isnan(data)] = 0.0
        data[data > 1.0] = 1.0
        data[data < 0.0] = 0.0

        data_fibre_density[np.isnan(data_fibre_density)] = 0.0
        data_fibre_density[np.isinf(data_fibre_density)] = 0.0
        data_fibre_density[data_fibre_density > 1.0] = 1.0
        data_fibre_density[data_fibre_density < 0.0] = 0.0
        print( '密度图平均值：', np.mean(data_fibre_density), 'data平均：', np.mean(data))
        print( '密度图最大值：', np.max(data_fibre_density), 'data最大：', np.max(data))

        # 获取低分图像，通过平均池化
        data_hr=torch.tensor(data.copy(),dtype=torch.float32).permute(3,0,1,2)
        data_fibre_density=torch.tensor(data_fibre_density,dtype=torch.float32).permute(3,0,1,2)

        X, Y, Z = mask.shape

        '''
        下面取三个方向的切片
        '''
        '''
        先z方向
        '''
        lr_patches_z = []
        hr_patches_z = []
        # fibre_patch_z=[]
        fibre_density_patch_z=[]
        ii=0
        for zi in range(Z):

            bi = 45
            hr = data_hr[0:30,: ,: , zi]
            lr = Resize((X//args.scale,Y//args.scale),Image.BICUBIC)(hr[None,...])[0]

            fibre_density=data_fibre_density[:,:,:,zi]

            hr_patches_z.append(hr)
            lr_patches_z.append(lr)
            # fibre_patch_z.append(fibre)
            fibre_density_patch_z.append(fibre_density)


            ii += 1
            # if (ii % 20 == 0):
            #     print(ii, '张数据')
            # # print(ii)
            a = 0
        print('z方向切片数量：',ii)
        print('z方向开始转为ndarray')
        lr_patches_z = torch.tensor([x.detach().numpy() for x in lr_patches_z])
        hr_patches_z = torch.tensor([x.detach().numpy() for x in hr_patches_z])
        # fibre_patch_z=np.array(fibre_patch_z)
        fibre_density_patch_z=torch.tensor([x.detach().numpy() for x in fibre_density_patch_z])
        print('z hr方向形状：',hr_patches_z.shape,'低分形状',lr_patches_z.shape,'density形状：',fibre_density_patch_z.shape)
        print('z方向开始存入h5file')
        # h5_file_z.create_dataset('lr', data=lr_patches_z)
        gz.create_dataset('lr',data=lr_patches_z)
        # h5_file_z.create_dataset('hr', data=hr_patches_z)
        gz.create_dataset('hr',data=hr_patches_z)
        # h5_file_z.create_dataset('fibre_density',data=fibre_density_patch_z)
        gz.create_dataset('fibre_density',data=fibre_density_patch_z)


        '''
        y方向
        '''


        lr_patches_y = []
        hr_patches_y = []
        # fibre_patch_y=[]
        fibre_density_patch_y=[]
        ii=0#计数归零
        for yi in range(Y):
            bi = 45
            hr = data_hr[0:30,:, yi, :]
            lr = Resize((X//args.scale,Z//args.scale),Image.BICUBIC)(hr[None,...])[0]
            # fibre=data_fibre[:,yi,:,:]
            fibre_density=data_fibre_density[:,:,yi,:]

            hr_patches_y.append(hr)
            lr_patches_y.append(lr)
            # fibre_patch_y.append(fibre)
            fibre_density_patch_y.append(fibre_density)

            ii += 1
            # if (ii % 20 == 0):
            #     print(ii, '张数据')
            # print(ii)
            a = 0
        print('y方向切片数：',ii)
        print('y方向开始转为ndarray')
        lr_patches_y = torch.tensor([x.detach().numpy() for x in lr_patches_y])
        hr_patches_y = torch.tensor([x.detach().numpy() for x in hr_patches_y])
        # fibre_patch_y=np.array(fibre_patch_y)
        fibre_density_patch_y=torch.tensor([x.detach().numpy() for x in fibre_density_patch_y])
        print('y hr方向形状：', hr_patches_y.shape,'低分形状',lr_patches_y.shape,'density形状：',fibre_density_patch_y.shape)
        print('y方向开始存入h5file')
        # h5_file_y.create_dataset('lr', data=lr_patches_y)
        gy.create_dataset('lr',data=lr_patches_y)
        gy.create_dataset('hr',data=hr_patches_y)
        gy.create_dataset('fibre_density',data=fibre_density_patch_y)
        # h5_file_y.create_dataset('hr', data=hr_patches_y)

        # h5_file_y.create_dataset('fibre_density',data=fibre_density_patch_y)

        # h5_file_y.close()

        '''
        x方向
        '''


        lr_patches_x = []
        hr_patches_x = []
        # fibre_patch_x=[]
        fibre_density_patch_x=[]
        ii=0
        for xi in range(X):
            # print(zi)
            # if (mask[:, :, zi].sum() > 1):#测试集，30个通道，不打乱，这30个通道取开头30通道
            # for bi in range(0,args.bgrad,):

            hr = data_hr[0:30,xi, :, :]
            lr = Resize((Y//args.scale,Z//args.scale),Image.BICUBIC)(hr[None,...])[0]
            # fibre=data_fibre[xi,:,:,:]
            fibre_density=data_fibre_density[:,xi,:,:]
            hr_patches_x.append(hr)
            lr_patches_x.append(lr)
            # fibre_patch_x.append(fibre)
            fibre_density_patch_x.append(fibre_density)
            ii += 1
            # if (ii % 20 == 0):
            #     print(ii, '张数据')
            # print(ii)
            a = 0
        print('x方向切片数：',ii)
        print('x方向开始转为ndarray')
        lr_patches_x = torch.tensor([x.detach().numpy() for x in lr_patches_x])
        hr_patches_x = torch.tensor([x.detach().numpy() for x in hr_patches_x])
        # fibre_patch_x=np.array(fibre_patch_x)
        fibre_density_patch_x=torch.tensor([x.detach().numpy() for x in fibre_density_patch_x])
        print('x方向形状：', hr_patches_x.shape,'低分形状：',lr_patches_x.shape,'density形状：',fibre_density_patch_x.shape)
        print('x方向开始存入h5file')
        # h5_file_x.create_dataset('lr', data=lr_patches_x)
        # h5_file_x.create_dataset('hr', data=hr_patches_x)
        # h5_file_x.create_dataset('fibre_density',data=fibre_density_patch_x)
        gx.create_dataset('lr',data=lr_patches_x)
        gx.create_dataset('hr',data=hr_patches_x)
        gx.create_dataset('fibre_density',data=fibre_density_patch_x)


    h5_file_z.close()
    h5_file_y.close()
    h5_file_x.close()
    print('测试集h5文件完成')

if __name__ == '__main__':
    #patch的尺寸：16*16，更小一点，试试效果
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='/data_new/tyj/data')
    parser.add_argument('--images-train_dir', type=str, default='/data_new/tyj/low_data')
    parser.add_argument('--images_eval_dir',type=str,default='/data_new/tyj/low_data')
    parser.add_argument('--images_test_dir',type=str,default='/data_new/tyj/low_data')

    parser.add_argument('--fibre_density_dir',type=str,default='/data_new/tyj/low_data/result/tdi/0.5')#先用网格1.0的纤维密度尺寸看看
    parser.add_argument('--output-train_path_x', type=str, default='h5data/train_data_x')
    parser.add_argument('--output-train_path_y', type=str, default='h5data/train_data_y')
    parser.add_argument('--output-train_path_z', type=str, default='h5data/train_data_z')
    parser.add_argument('--output_eval_path',type=str,default='h5data/eval_data')
    parser.add_argument('--output_test_path_x',type=str,default='h5data/test_data_x')
    parser.add_argument('--output_test_path_y', type=str, default='h5data/test_data_y')
    parser.add_argument('--output_test_path_z', type=str, default='h5data/test_data_z')
    parser.add_argument('--bgrad',type=int,default=90)#方向数
    parser.add_argument('--img_src_x', type=int, default=112)#要裁剪的x的维度
    parser.add_argument('--img_src_y', type=int, default=144)#要裁剪的y的维度
    parser.add_argument('--img_src_z', type=int, default=112)#要裁剪的z的维度
    # parser.add_argument('--patch_size',default=16,type=int)#patch的尺寸
    # parser.add_argument('--patch_stride',default=16,type=int)#取patch的步长
    parser.add_argument('--affine_dir',type=str,default='affine')
    parser.add_argument('--scale', type=int, default=2)
    # parser.add_argument('--eval', action='store_true',default=False)
    args = parser.parse_args()

    # eval(args)
    train(args)
    test(args)
