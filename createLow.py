import numpy as np
import os
import torch
from torch.nn import functional as F
from dipy.align.reslice import reslice
from dipy.io.image import load_nifti,save_nifti
from dipy.segment.mask import median_otsu
import argparse
from shutil import copyfile

def create(args):
    ids=["685058", "748662", "751348", "756055", "761957", "833148", "837560", "845458", "896778", "898176", "901038", "901442", "979984", "984472", "991267"]
    # ids=["748662", "751348", "756055", "761957", "833148", "837560", "845458", "896778", "898176", "901038", "901442", "979984", "984472", "991267"]
    for id in ids:
        print('开始处理：',id)
        hr_data_file = os.path.join(args.data_path,f'{id}_hr.nii.gz')
        mask_file = os.path.join(args.data_path,f'{id}_mask.nii.gz')
        b0_file = os.path.join(args.data_path,f'{id}_b0.nii.gz')
        bvals_file = os.path.join(args.data_path,f'{id}_bvals.txt')
        bvecs_file = os.path.join(args.data_path,f'{id}_bvecs.txt')
        bvals_b0_file = os.path.join(args.data_path, f'{id}_bvals_b0.txt')
        bvecs_b0_file = os.path.join(args.data_path, f'{id}_bvecs_b0.txt')

        data_hr, affine_hr, vox_hr = load_nifti(hr_data_file, return_voxsize=True)
        mask, affine_mask, vox_mask = load_nifti(mask_file, return_voxsize=True)
        b0_data,affine_b0 = load_nifti(b0_file)

        bvals=np.loadtxt(bvals_file)
        bvecs=np.loadtxt(bvecs_file)
        bvals_b0 = np.loadtxt(bvals_b0_file)
        bvecs_b0 = np.loadtxt(bvecs_b0_file)
        bvals = bvals[:30]
        bvecs = bvecs[:,:30]


        data_hr = data_hr[...,:30]

        # 如有未裁剪的进行裁剪
        X,Y,Z,b=data_hr.shape
        # if X>112:
        #     data_hr = data_hr[X//2-args.img_x//2:X//2+args.img_x//2,Y//2-args.img_y//2:Y//2+args.img_y//2,
        #               Z//2-args.img_z//2:Z//2+args.img_z//2,:]

        # 数据进行去脑壳
        for bi in range(data_hr.shape[3]):
            data_hr[...,bi] = data_hr[...,bi] * mask
        for bi in range(b0_data.shape[3]):
            b0_data[...,bi] = b0_data[...,bi] * mask
        # data_hr,mask = median_otsu(data_hr,vol_idx=range(30))
        # mask = np.array(mask,dtype=np.int32)
        # b0_data,mask_b0 = median_otsu(b0_data,vol_idx=range(18))
        b0_data_mean = np.mean(b0_data, axis=3, dtype=np.float32, keepdims=True)



        #进行下采样
        data_hr = torch.tensor(data_hr, dtype=torch.float32)
        data_lr0=torch.tensor(data_hr,dtype=torch.float32).permute(3,0,1,2)[None,...]
        data_lr=F.avg_pool3d(data_lr0,kernel_size=2).permute(0,2,3,4,1)[0].cpu().data.numpy()
        vox_hr_new=(0.625,0.625,0.625)
        data_interpolation_lr, affine_interpolation_lr = reslice(data_lr, affine_hr, vox_hr, vox_hr_new)
        data_hr = data_hr.cpu().data.numpy()
        print('高分形状：',data_hr.shape,'低分形状：',data_lr.shape,'插值形状：',data_interpolation_lr.shape)
        print('高分最大值最小值：', np.max(data_hr),np.min(data_hr),
              '低分最大值最小值：',np.max(data_lr),np.min(data_lr), '插值最大值最小值：', np.max(data_interpolation_lr),np.min(data_interpolation_lr))

        #保存数据
        hr_save_file = os.path.join(args.save_path,f'{id}_hr.nii.gz')
        lr_save_file = os.path.join(args.save_path,f'{id}_lr.nii.gz')
        lr_interpolation_file = os.path.join(args.save_path,f'{id}_interpolation_lr.nii.gz')
        b0_save_file = os.path.join(args.save_path,f'{id}_b0.nii.gz')
        b0_src_save_file = os.path.join(args.save_path, f'{id}_b0_src.nii.gz')
        mask_save_file = os.path.join(args.save_path,f'{id}_mask.nii.gz')
        bvals_save_file = os.path.join(args.save_path,f'{id}_bvals.txt')
        bvecs_save_file = os.path.join(args.save_path,f'{id}_bvecs.txt')
        bvals_save_b0_file = os.path.join(args.save_path,f'{id}_bvals_b0.txt')
        bvecs_save_b0_file = os.path.join(args.save_path,f'{id}_bvecs_b0.txt')

        save_nifti(hr_save_file,data_hr,affine_hr)
        save_nifti(lr_save_file,data_lr,affine_hr)
        save_nifti(lr_interpolation_file,data_interpolation_lr,affine_interpolation_lr)
        save_nifti(b0_save_file,b0_data_mean,affine_b0)
        save_nifti(b0_src_save_file,b0_data,affine_b0)
        save_nifti(mask_save_file,mask,affine_hr)
        np.savetxt(bvals_save_file,bvals)
        copyfile(bvals_b0_file,bvals_save_b0_file)
        np.savetxt(bvecs_save_file,bvecs)
        copyfile(bvecs_b0_file,bvecs_save_b0_file)
        # np.savetxt(bvals_save_file,bvals)
        # np.savetxt(bvecs_save_file,bvecs)
        print('完成：',id)
        # break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str, default='/data_new/tyj/denoised_hcp_data')
    parser.add_argument('--save_path',type=str,default='/data_new/tyj/low_data')
    parser.add_argument('--img_x',type=int,default=112)
    parser.add_argument('--img_y',type=int,default=144)
    parser.add_argument('--img_z',type=int,default=112)

    args = parser.parse_args()
    create(args)
