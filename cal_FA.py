import argparse
import os.path

import numpy as np
from dipy.core.gradients import gradient_table#用于梯度表
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti,save_nifti#读取数据
from dipy.reconst import dti
from dipy.reconst.dti import fractional_anisotropy,color_fa
from dipy.segment.mask import median_otsu
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def cal_rmse(id,img1,img2,is_sr=False,b0=None, rmse_list:list = None):

    # for i in range(img2.shape[3]):
    #     img1[:,:,:,i]=np.squeeze(img1[:,:,:,i])*np.squeeze(b0)
    #     img2[:, :, :, i] = np.squeeze(img2[:, :, :, i]) * np.squeeze(b0)
    img1*=255.0
    img2*=255.0
    # img1[np.isnan(img1)]=0.0
    # img1[np.isinf(img1)]=0.0
    # img2[np.isnan(img2)]=0.0
    # img2[np.isinf(img2)]=0.0
    m=mse(img1,img2)
    sqmse=np.sqrt(m)
    if(is_sr):
        print(f'{id},超分结果和高分的rmse：',sqmse)
        rmse_list.append(sqmse)
    else:
        print(f'{id},低分和高分的rmse：',sqmse)
        rmse_list.append(sqmse)
def cal_psnr(id,img1,img2,is_sr=False,is_FA=False, psnr_list: list = None, psnr_FA_list: list = None):
    p=psnr(img1,img2,data_range=1)
    if is_FA:
        if is_sr:
            print(f'{id},超分结果和高分FA的psnr：',p)
            psnr_FA_list.append(p)
        else:
            print(f'{id},低分和高分的FA的psnr:',p)
            psnr_FA_list.append(p)
        return
    if(is_sr):
        print(f'{id},超分结果和高分的psnr:',p)
        psnr_list.append(p)
    else:
        print(f'{id},低分和高分的psnr=',p)
        psnr_list.append(p)
def cal_ssim(id,img1,img2,is_sr=False,is_FA=False, ssim_list: list = None, ssim_FA_list: list = None):

    sim=ssim(img1,img2,channel_axis=3,data_range=1)
    if is_FA:
        if is_sr:
            print(f'{id},超分结果FA的ssim：',sim)
            ssim_FA_list.append(sim)
        else:
            print(f'{id},低分FA的ssim：',sim)
            ssim_FA_list.append(sim)
        return
    if(is_sr):
        print(f'{id},超分结果的ssim:',sim)
        ssim_list.append(sim)
    else:
        print(f'{id},低分和高分的ssim=',sim)
        ssim_list.append(sim)
def cal(args):
    ids = ["991267","898176","761957","901038","833148"]
    psnr_list = []
    psnr_FA_list = []
    ssim_list = []
    ssim_FA_list = []
    rmse_list = []
    psnr_lr_list = []
    ssim_lr_list = []
    psnr_FA_lr_list = []
    ssim_FA_lr_list = []
    rmse_lr_list =[]
    for id in ids:
    # id=args.id#'108323'
        print(f'----------------------------{id}开始------------------------------------------')

        lr_file=os.path.join(args.data_dir,f'{id}_interpolation_lr.nii.gz')
        hr_file=os.path.join(args.data_dir,f'{id}_hr.nii.gz')
        sr_file=os.path.join('out_file',f'{id}_sr_.nii.gz')
        b0_file=os.path.join(args.data_dir,f'{id}_b0.nii.gz')
        mask_file=os.path.join(args.data_dir,f'{id}_mask.nii.gz')
        # b0_file=os.path.join('data','test_data',f'{id}_padding_b0_avg.nii.gz')
        bval_file=os.path.join(args.data_dir,f'{id}_bvals.txt')
        bvec_file=os.path.join(args.data_dir,f'{id}_bvecs.txt')
        bval_b0_file = os.path.join(args.data_dir,f'{id}_bvals_b0.txt')
        bvec_b0_file = os.path.join(args.data_dir,f'{id}_bvecs_b0.txt')
        #读取数据
        hr_data,affine_hr=load_nifti(hr_file)
        # hr_data=hr_data_src[:,:,:,:30]
        lr_data,affine_lr=load_nifti(lr_file)
        # lr_data=lr_data_src[:,:,:,:30]
        sr_data,affine_sr=load_nifti(sr_file)
        mask,mask_affine=load_nifti(mask_file)
        mask = mask.astype(np.int32)
        b0_data, affine_b0 = load_nifti(b0_file)
        src_x,src_y,src_z=mask.shape
        # mask=mask[src_x // 2 - args.img_src_x // 2:src_x // 2 + args.img_src_x // 2,
        #                src_y // 2 - args.img_src_y // 2:src_y // 2 + args.img_src_y // 2,
        #                src_z // 2 - args.img_src_z // 2:src_z // 2 + args.img_src_z // 2]

        for i in range(sr_data.shape[3]):
            sr_data[:,:,:,i]=sr_data[:,:,:,i]*mask
        # 进行归一化
        # for ci in range(hr_data.shape[3]):
        #     hr_data[:, :, :, ci] = np.squeeze(hr_data[:, :, :, ci]) / np.squeeze(b0_data)
        #     lr_data[:, :, :, ci] = np.squeeze(lr_data[:, :, :, ci]) / np.squeeze(b0_data)
        #     hr_data[:, :, :, ci] = hr_data[:, :, :, ci] * mask
        #     lr_data[:, :, :, ci] = lr_data[:, :, :, ci] * mask
        sr_data[np.isnan(sr_data)] = 0.0
        sr_data[np.isinf(sr_data)] = 0.0
        # lr_data[np.isnan(lr_data)] = 0.0
        # lr_data[np.isinf(lr_data)] = 0.0
        sr_data[sr_data > 1] = 1.0
        sr_data[sr_data < 0] = 0.0
        # lr_data[lr_data > 1] = 1.0
        # lr_data[lr_data < 0] = 0.0
        #先计算几个小指标

        cal_psnr(id,lr_data,hr_data,psnr_list=psnr_lr_list)
        cal_psnr(id,sr_data,hr_data,True,psnr_list=psnr_list)
        cal_rmse(id,lr_data.copy(),hr_data.copy(),b0=b0_data,rmse_list=rmse_lr_list)
        cal_rmse(id,sr_data.copy(),hr_data.copy(),True,b0=b0_data,rmse_list=rmse_list)
        cal_ssim(id,lr_data,hr_data,ssim_list=ssim_lr_list)
        cal_ssim(id,sr_data,hr_data,True,ssim_list=ssim_list)
        #后面计算FA

        hr_data=np.concatenate((b0_data,hr_data),axis=3)
        lr_data=np.concatenate((b0_data,lr_data),axis=3)
        sr_data=np.concatenate((b0_data,sr_data),axis=3)



        hr_maskdata, m = median_otsu(hr_data, vol_idx=range(1, 31), median_radius=3,
                                  numpass=1, autocrop=False, dilate=2)
        lr_maskdata, m = median_otsu(lr_data, vol_idx=range(1, 31), median_radius=3,
                                  numpass=1, autocrop=False, dilate=2)
        sr_maskdata, m = median_otsu(sr_data, vol_idx=range(1, 31), median_radius=3,
                                     numpass=1, autocrop=False, dilate=2)

        bvals,bvecs=read_bvals_bvecs(bval_file,bvec_file)#读取bvals,bvecs
        bvals_b0,bvecs_b0 = read_bvals_bvecs(bval_b0_file,bvec_b0_file)
        # index=[0]
        # # b0_index=[]
        # for i in range(len(bvals)):
        #     if 700<bvals[i]<1300:
        #         index.append(i)
        #     if(len(index)==31):
        #         break
        bvals=np.concatenate((bvals_b0[0:1],bvals),axis=0)
        bvecs=np.concatenate((bvecs_b0[0:1],bvecs),axis=0)

        print(f'{id},创建梯度表')
        gtab=gradient_table(bvals,bvecs)#创建梯度表
        print(f'{id},创建张量模型')
        tenmodel=dti.TensorModel(gtab)#创建张量模型
        #低分辨率的
        tenfit_lr=tenmodel.fit(lr_maskdata)
        print(f'{id},计算FA')
        FA_lr=fractional_anisotropy(tenfit_lr.evals)
        #高分辨率的
        tenfir_hr=tenmodel.fit(hr_maskdata)
        FA_hr=fractional_anisotropy(tenfir_hr.evals)
        #超分结果
        tenfir_sr=tenmodel.fit(sr_maskdata)
        FA_sr=fractional_anisotropy(tenfir_sr.evals)


        FA_sr[np.isnan(FA_sr)]=0
        FA_lr[np.isnan(FA_lr)]=0
        FA_hr[np.isnan(FA_hr)]=0

        FA_hr=FA_hr[...,None]
        FA_sr=FA_sr[...,None]
        FA_lr=FA_lr[...,None]
        cal_psnr(id,FA_sr,FA_hr,is_sr=True,is_FA=True,psnr_FA_list=psnr_FA_list)
        cal_psnr(id,FA_lr,FA_hr,is_sr=False,is_FA=True,psnr_FA_list=psnr_FA_lr_list)
        cal_ssim(id,FA_sr,FA_hr,is_sr=True,is_FA=True,ssim_FA_list=ssim_FA_list)
        cal_ssim(id,FA_lr,FA_hr,is_sr=False,is_FA=True,ssim_FA_list=ssim_FA_lr_list)

        #计算error Map
        sr_err_map=np.abs(FA_hr-FA_sr)
        lr_err_map=np.abs(FA_hr-FA_lr)


        #保存
        hr_FA_file=os.path.join('out_file',f'{id}_FA_hr.nii.gz')
        lr_FA_file=os.path.join('out_file',f'{id}_FA_lr.nii.gz')
        sr_FA_file=os.path.join('out_file',f'{id}_FA_sr.nii.gz')
        lr_err_map_file=os.path.join('out_file',f'{id}_FA_err_lr_map.nii.gz')
        sr_err_map_file=os.path.join('out_file',f'{id}_FA_err_sr_map.nii.gz')
        save_nifti(hr_FA_file,FA_hr.astype(np.float32),affine_hr)
        save_nifti(lr_FA_file,FA_lr.astype(np.float32),affine_lr)
        save_nifti(sr_FA_file,FA_sr.astype(np.float32),affine_sr)
        save_nifti(lr_err_map_file,lr_err_map.astype(np.float32),affine_lr)
        save_nifti(sr_err_map_file,sr_err_map.astype(np.float32),affine_sr)

        mFA_sr=np.mean(FA_sr)
        mFA_hr=np.mean(FA_hr)
        mFA_lr=np.mean(FA_lr)
        print(f'{id},hr平均：',mFA_hr)
        # print('lr平均：',mFA_lr)
        print(f'{id},sr平均：',mFA_sr)
        mFA_hr=np.max(FA_hr)
        mFA_lr=np.max(FA_lr)
        mFA_sr=np.max(FA_sr)
        print(f'{id},hr最大：',mFA_hr)
        # print('lr最大：',mFA_lr)
        print(f'{id},sr最大：',mFA_sr)

        print(f'----------------------------{id}完成------------------------------------------')
    print('-------------所有sub都计算完成，下面列出所有内容-------------------------')
    print('超分psnr值：',psnr_list)
    print('超分FA的psnr值：',psnr_FA_list)
    print('超分ssim值：',ssim_list)
    print('超分FA的ssim值：',ssim_FA_list)
    print('超分rmse值：',rmse_list)

    print('线性插值psnr值：', psnr_lr_list)
    print('线性插值FA的psnr值：', psnr_FA_lr_list)
    print('线性插值ssim值：', ssim_lr_list)
    print('线性插值FA的ssim值：', ssim_FA_lr_list)
    print('线性插值rmse值：', rmse_lr_list)
    print('----------------下面计算平均值----------------------------------------')
    psnr_list = np.array(psnr_list)
    psnr_FA_list = np.array(psnr_FA_list)
    ssim_list = np.array(ssim_list)
    ssim_FA_list = np.array(ssim_FA_list)
    rmse_list = np.array(rmse_list)

    psnr_lr_list = np.array(psnr_lr_list)
    psnr_FA_lr_list = np.array(psnr_FA_lr_list)
    ssim_lr_list = np.array(ssim_lr_list)
    ssim_FA_lr_list = np.array(ssim_FA_lr_list)
    rmse_lr_list = np.array(rmse_lr_list)
    print('psnr:',psnr_list.mean(),'\t ssim:',ssim_list.mean(), '\t rmse:',rmse_list.mean(),
          '\t psnr_FA:',psnr_FA_list.mean(),'\t ssim_FA:',ssim_FA_list.mean())
    print('psnr:', psnr_lr_list.mean(), '\t ssim:', ssim_lr_list.mean(), '\t rmse:', rmse_lr_list.mean(),
          '\t psnr_FA:', psnr_FA_lr_list.mean(), '\t ssim_FA:', ssim_FA_lr_list.mean())

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--img_src_x',type=int,default=112)
    parser.add_argument('--img_src_y',type=int,default=144)
    parser.add_argument('--img_src_z',type=int,default=112)

    parser.add_argument('--data_dir',default='/data_new/tyj/low_data')
    args=parser.parse_args()
    cal(args)
