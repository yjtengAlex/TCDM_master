import argparse

import matplotlib.pyplot as plt
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.denoise.localpca import localpca
from dipy.denoise.patch2self import patch2self
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.io.image import load_nifti,save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
import os
from time import time


def denoise(args):
    ids = ["685058","748662","751348","756055","761957","833148","837560","845458","896778","898176","901038","901442","979984","984472","991267"]
    # ids = ["748662","751348","756055","761957","833148","837560","845458","896778","898176","901038","901442","979984","984472","991267"]
    for id in ids:
        # 文件名
        print('开始：',id)
        dwi_fileName = os.path.join(args.data_file,id,"Diffusion/data.nii.gz")
        bval_fileName = os.path.join(args.data_file,id,'Diffusion/data.bvals')
        bvecs_fileName = os.path.join(args.data_file,id,'Diffusion/data.bvecs')
        mask_fileName = os.path.join(args.data_file,id,'Diffusion/nodif_brain_mask.nii.gz')

        dwi,affine_dwi = load_nifti(dwi_fileName)
        mask,affine_mask = load_nifti(mask_fileName)
        bvals = np.loadtxt(bval_fileName)
        bvecs = np.loadtxt(bvecs_fileName)
        # bvals,bvecs = read_bvals_bvecs(bval_fileName,bvecs_fileName)
        # x=np.sum(mask,axis=0)
        # y=np.sum(mask,axis=1)
        # z=np.sum(mask,axis=2)

        #挑选b0和b1000
        b0_index = []
        b1000_index = []
        for index,bval in enumerate(bvals):
            if bval < 50:
                b0_index.append(index)
            if 700 < bval <1300:
                b1000_index.append(index)



        # _, mask = median_otsu(dwi,b0_index,2,1)
        # mask = np.array(mask,dtype=np.int32)
        # 进行去脑壳
        X,Y,Z = mask.shape
        for xx in range(X):
            for yy in range(Y):
                for zz in range(Z):
                    if mask[xx,yy,zz]<0.5:
                        for bi in range(dwi.shape[3]):
                            dwi[xx,yy,zz,bi] = 0
        # for bi in range(dwi.shape[3]):
        #     dwi[...,bi] = dwi[...,bi]*mask

        b0_dwi = dwi[:, :, :, b0_index]
        bvecs_b0 = bvecs[:,b0_index]
        bvecs_b1000 = bvecs[:,b1000_index]
        bvals_b0 = bvals[b0_index]
        bvals_b1000 = bvals[b1000_index]
        b0_b1000_index = b0_index+b1000_index
        bvals_b0_b1000 = bvals[b0_b1000_index]
        bvecs_b0_b1000 = bvecs[:,b0_b1000_index]
        b0_b1000_dwi = dwi[:,:,:,b0_b1000_index]

        #进行裁剪
        # X,Y,Z = mask.shape
        b0_b1000_dwi=b0_b1000_dwi[X//2-args.x//2:X//2+args.x//2,Y//2-args.y//2:Y//2+args.y//2,Z//2-args.z//2:Z//2+args.z//2,:]
        # b0_dwi = np.mean(b0_dwi,axis=3,keepdims=True)
        b0_dwi = b0_dwi[X//2-args.x//2:X//2+args.x//2,Y//2-args.y//2:Y//2+args.y//2,Z//2-args.z//2:Z//2+args.z//2,:]
        mask = mask[X//2-args.x//2:X//2+args.x//2,Y//2-args.y//2:Y//2+args.y//2,Z//2-args.z//2:Z//2+args.z//2]

        save_mask_file = os.path.join(args.save_path, f'{id}_mask.nii.gz')
        save_nifti(save_mask_file, mask, affine_dwi)


        #进行降噪
        print('进行降噪')
        start_time=time()
        gtab = gradient_table(bvals_b0_b1000,bvecs_b0_b1000)
        # sigma = pca_noise_estimate(b0_b1000_dwi,gtab,correct_bias=True, smooth=3)
        # denoised_arr = localpca(b0_b1000_dwi, sigma, tau_factor=2.3, patch_radius=2)

        denoised_arr = patch2self(b0_b1000_dwi,bvals_b0_b1000)
        # denoised_arr = b0_b1000_dwi.copy()
        print('降噪耗时：',time()-start_time)
        b1000_dwi = denoised_arr[...,-1*len(b1000_index):]
        print('降噪后b1000的形状：',b1000_dwi.shape,'denoised形状：',denoised_arr.shape)

        # 进行归一化
        b0_dwi_means = np.mean(b0_dwi,axis=3,keepdims=True)
        for bi in range(b1000_dwi.shape[3]):
            b1000_dwi[...,bi] = np.squeeze(b1000_dwi[...,bi])/np.squeeze(b0_dwi_means)
            b1000_dwi[...,bi] = b1000_dwi[...,bi] * mask

        b1000_dwi[np.isinf(b1000_dwi)] = 0.0
        b1000_dwi[np.isnan(b1000_dwi)] = 0.0
        b1000_dwi[b1000_dwi>1.0] = 1.0
        b1000_dwi[b1000_dwi<0.0] = 0.0

        # # 可视化降噪数据
        # sli = b0_b1000_dwi.shape[2]//2
        # grad = b0_b1000_dwi.shape[3]//2
        # orig = b0_b1000_dwi[:,:,sli,grad]
        # den = denoised_arr[:,:,sli,grad]
        # rms_diff = np.sqrt((orig - den)**2)# errorMap
        #
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(orig,cmap = 'gray', origin='lower', interpolation='none')
        # ax[0].set_title('original')
        # ax[0].set_axis_off()
        # ax[1].imshow(den,cmap='gray', origin='lower', interpolation='none')
        # ax[1].set_title('denoised Output')
        # ax[1].set_axis_off()
        # ax[2].imshow(rms_diff, cmap='gray', origin='lower', interpolation='none')
        # ax[2].set_title('Residual')
        # ax[2].set_axis_off()
        # plt.savefig(f'denoised_patch2self_{id}.png')

        #保存数据集
        print('保存数据')

        save_dwi_file = os.path.join(args.save_path,f'{id}_hr.nii.gz')
        save_b0_file = os.path.join(args.save_path,f'{id}_b0.nii.gz')
        save_bvals_file = os.path.join(args.save_path,f'{id}_bvals.txt')
        save_bvecs_file = os.path.join(args.save_path,f'{id}_bvecs.txt')
        save_bvecs_file_b0 = os.path.join(args.save_path,f'{id}_bvecs_b0.txt')
        save_bvals_file_b0 = os.path.join(args.save_path,f'{id}_bvals_b0.txt')

        save_nifti(save_dwi_file,b1000_dwi,affine_dwi)
        save_nifti(save_b0_file,b0_dwi,affine_dwi)

        np.savetxt(save_bvals_file,bvals_b1000)
        np.savetxt(save_bvecs_file,bvecs_b1000)
        np.savetxt(save_bvals_file_b0,bvals_b0)
        np.savetxt(save_bvecs_file_b0,bvecs_b0)
        print('完成：',id)
        # break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',type=str,default="/data_new/tyj/hcp_data")
    parser.add_argument('--x',type=int,default=112)
    parser.add_argument('--y',type=int,default=144)
    parser.add_argument('--z',type=int,default=112)
    parser.add_argument('--save_path',type=str,default='/data_new/tyj/denoised_hcp_data')
    args = parser.parse_args()
    denoise(args)