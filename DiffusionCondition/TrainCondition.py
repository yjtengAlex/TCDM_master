

import os
from typing import Dict
import numpy as np
from matplotlib import pyplot as plt
import torch
from time import time
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datasetsDDPM import TrainDataset,EvalDataset
from DiffusionCondition.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer,GaussianDiffusion_ddim
from DiffusionCondition.ModelCondition import UNet
# from DiffusionCondition.ModelCondition0 import UnetPlusPlus
from dipy.io.image import load_nifti,save_nifti#读取数据，保存数据
import PIL.Image as pil_image
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16384"

def train(modelConfig: Dict):
    gpus=modelConfig['gpus']
    torch.cuda.set_device(modelConfig['output_device'])

    device = torch.device(modelConfig["device"])
    print(device)
    # dataset，第二步：数据集
    dataset_x = TrainDataset('h5data/train_data_x')
    loader_x = DataLoader(
        dataset_x, batch_size=modelConfig["batch_size"], shuffle=True,  drop_last=True)

    dataset_y = TrainDataset('h5data/train_data_y')
    loader_y = DataLoader(
        dataset_y, batch_size=modelConfig["batch_size"], shuffle=True, drop_last=True)

    dataset_z = TrainDataset('h5data/train_data_z')
    loader_z = DataLoader(
        dataset_z, batch_size=modelConfig["batch_size"], shuffle=True, drop_last=True)

    # eval_set=EvalDataset('h5data/eval_data')
    # loader=DataLoader(eval_set,batch_size=16)
    # print('x方向：',len(dataset_x),'y方向：',len(dataset_y),'z方向：',len(dataset_z))
    # for lr,hr,fibre_density in loader_x:
    #     print(lr.shape,hr.shape,fibre_density.shape)
    #     break
    # for lr,hr,fibre_density in loader_y:
    #     print(lr.shape,hr.shape,fibre_density.shape)
    #     break
    # for lr,hr,fibre_density in loader_z:
    #     print(lr.shape,hr.shape,fibre_density.shape)
    #     break
    #     save_image(torch.cat((lr,hr),dim=0),'a.png')

    # model setup 第三步：网络结构,num_labels:标签的种类
    net_model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    #num_classes:通道数
    # net_model=UnetPlusPlus(T=modelConfig['T'],num_classes=modelConfig['num_classes'],ch=modelConfig['channel'],
    #                        ch_mult=modelConfig['channel_mult'],deep_supervision=modelConfig['deep_supervision'])

    net_model=nn.DataParallel(net_model.cuda(),device_ids=gpus,output_device=gpus[0])
    net_model.train()
    start_epoch=0
    optimizer = torch.optim.AdamW(  # 优化器
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    if modelConfig["training_load_weight"] is not None:#加载预训练模型
        # m=
        pre_train=torch.load(os.path.join(modelConfig["save_dir"], modelConfig["training_load_weight"]),map_location=device)
        net_model.load_state_dict(pre_train['model'], strict=False)
        # optimizer.load_state_dict(pre_train['optim'])
        start_epoch += pre_train['epoch']
        start_epoch += 1
        print("Model weight load down:",modelConfig["training_load_weight"],'epoch数：',pre_train['epoch'])


    # cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    # warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],#预热学习率
    #                                          warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    #下面自定义学习率变化情况
    def lam(epoch):
        # if(epoch<5):
        #     return 1+epoch*0.01
        if (epoch<5):
            return 1.0
        elif(epoch<10):
            return 0.5
        else:
            return 0.5*0.9999**(epoch-10)
    schduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lam)

    # start training，第五步：训练
    best_loss=100000000
    best_model={}
    loss_curver=[]
    net_model.train()
    for e in range(start_epoch,modelConfig["epoch"]+start_epoch):
        loss_value=0
        ii=0

        '''
        x方向：
        '''
        with tqdm(loader_x, dynamic_ncols=False) as tqdmDataLoader:
            tqdmDataLoader.set_description('x方向切片-epoch:{}/{}'.format(e,modelConfig['epoch']+start_epoch))
            for index,(labels,images,fibre_density) in enumerate(tqdmDataLoader):#labels:低分图像，images：高分图像
                ii+=1

                b = images.shape[0]#batch_size
                optimizer.zero_grad()#梯度归零
                x_0 = images.clamp(0,1).to(device)#原始图像，即高分图像
                labels = labels.clamp(0,1).to(device)#条件，即低分图像
                # fibre=fibre.to(device)
                fibre_density=fibre_density.to(device)# 条件，即纤维密度
                # if np.random.rand() < 0.1:#10%的概率标签是0
                #     labels = torch.zeros_like(labels).to(device)

                loss = trainer(x_0, labels,fibre_density)#.sum() #/ b ** 2.
                loss.backward()
                loss_value+=loss.item()
                loss_curver.append(loss.item())
                torch.nn.utils.clip_grad_norm_(#防止梯度爆炸
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                    # "epoch": e,

                    "img shape: ": x_0.shape,

                })

        '''
        y方向
        '''
        with tqdm(loader_y, dynamic_ncols=False) as tqdmDataLoader:
            tqdmDataLoader.set_description('y方向切片-epoch:{}/{}'.format(e,modelConfig['epoch']+start_epoch))
            for index,(labels,images,fibre_density) in enumerate(tqdmDataLoader):#labels:低分图像，images：高分图像
                ii+=1

                b = images.shape[0]#batch_size
                optimizer.zero_grad()#梯度归零
                x_0 = images.clamp(0,1).to(device)#原始图像，即高分图像
                labels = labels.clamp(0,1).to(device)#条件，即低分图像
                # fibre=fibre.to(device)
                fibre_density=fibre_density.to(device)
                # if np.random.rand() < 0.1:#10%的概率标签是0
                #     labels = torch.zeros_like(labels).to(device)

                loss = trainer(x_0, labels,fibre_density)#.sum() #/ b ** 2.
                loss.backward()
                loss_value+=loss.item()
                loss_curver.append(loss.item())
                torch.nn.utils.clip_grad_norm_(#防止梯度爆炸
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                    # "epoch": e,

                    "img shape: ": x_0.shape,

                })


        '''
        z方向
        '''
        with tqdm(loader_z, dynamic_ncols=False) as tqdmDataLoader:
            tqdmDataLoader.set_description('z方向切片-epoch:{}/{}'.format(e,modelConfig['epoch']+start_epoch))
            for index,(labels,images,fibre_density) in enumerate(tqdmDataLoader):#labels:低分图像，images：高分图像
                ii+=1

                b = images.shape[0]#batch_size
                optimizer.zero_grad()#梯度归零
                x_0 = images.clamp(0,1).to(device)#原始图像，即高分图像
                labels = labels.clamp(0,1).to(device)#条件，即低分图像
                fibre_density=fibre_density.to(device)
                # fibre=fibre.to(device)#条件，纤维图
                # if np.random.rand() < 0.1:#10%的概率标签是0
                #     labels = torch.zeros_like(labels).to(device)

                loss = trainer(x_0, labels,fibre_density)#.sum() #/ b ** 2.
                loss.backward()
                loss_value+=loss.item()
                loss_curver.append(loss.item())
                torch.nn.utils.clip_grad_norm_(#防止梯度爆炸
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                    # "epoch": e,

                    "img shape: ": x_0.shape,

                })

        '''
        各方向完成
        '''

        loss_value/=ii
        print('当前平均损失：',loss_value)
        schduler.step()
        if(loss_value<best_loss):
            best_loss=loss_value
            best_model['model']=net_model.state_dict()
            best_model['epoch']=e

        if((e)%1==0):
            content={}
            content['model'] = net_model.state_dict()
            content['optim'] = optimizer.state_dict()
            content['epoch'] = e
            torch.save(content, os.path.join(
                modelConfig["save_dir"], f'ckpt_{e:0>4d}.pth'))
    #保存最佳模型
    net_model.eval()
    torch.save(best_model,os.path.join(modelConfig['save_dir'],'best.pth'))
    print('最佳轮数：',best_model['epoch'],'最佳损失：',best_loss)
    plt.plot(loss_curver)
    plt.title('loss_curver')
    plt.savefig('loss_curver.png')
    plt.show()

def eval(modelConfig: Dict):

    torch.set_printoptions(threshold=np.inf)
    torch.cuda.set_device(modelConfig['output_device'])
    device = torch.device(modelConfig["device"])#设备，即GPU
    print(device)
    # mean = np.array([0.07135591])
    # std = np.array([0.113740])
    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)
        eval_set=TrainDataset('h5data/train_data')
        loader=DataLoader(eval_set,batch_size=modelConfig['batch_size'])

        hr=None
        lr=None
        for i, (lr0,hr0) in enumerate(loader):
            if(i<6):
                continue
            hr=hr0.to(device)#clamp(-1,1).to(device)
            lr=lr0#.clamp(-1,1)
            break

        lr_num=lr.cpu().data.numpy()
        hr_num=hr.cpu().data.numpy()
        print('hr最大值：',np.max(hr_num))
        plt.subplot(1,2,1)
        plt.title('lr')
        plt.imshow(lr[0].permute(1,2,0).cpu().data.numpy()[...,15],cmap='gray')
        plt.subplot(1,2,2)
        show_hr=hr[0].permute(1,2,0).cpu().data.numpy()
        print('展示的hr的最大值：',np.max(show_hr))
        plt.imshow(show_hr[...,15],cmap='gray')
        plt.title('hr')
        plt.show()


        labels=lr.to(device)
        img = torch.cat([labels, hr], dim=0)
        # save_image(img, 'out_pic/img.png')

        #第三步：网络模型
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

        model=nn.DataParallel(model.cuda(),device_ids=modelConfig['gpus'],output_device=modelConfig['gpus'][0])

        #加载预训练权重
        ckpt = torch.load(os.path.join(#加载训练的参数
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)

        model.load_state_dict(ckpt['model'])#将训练的网络参数载入网络
        print("model load weight done：",modelConfig['test_load_weight'])
        model.eval()
        sampler = GaussianDiffusionSampler(#创建Gaussian采样器
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(#从高斯噪声开始
            size=[modelConfig["batch_size"], 30, modelConfig["img_size_x"], modelConfig["img_size_y"]], device=device)

        saveNoisy = noisyImage
        # save_image(saveNoisy, os.path.join(#保存高斯噪声（处理前的图片）
        #     modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage, labels)#进行采样，进行采样的图片还是均值0，方差1
        sampledImgs=torch.cat((sampledImgs,hr),dim=0)
        # print('采样后hr最大值：',np.max(hr.cpu().data.numpy()))
        # for i in range(sampledImgs.shape[0]):#图片遍历
        #     for j in range(1):#通道遍历
        #         sampledImgs[i,j,:,:] = sampledImgs[i,j,:,:] *std[j]+mean[j]#  # 把生成的图片恢复成原均值和方差
        # print(sampledImgs)
        # print('去除标准化后hr最大值：',np.max(sampledImgs[-1].cpu().data.numpy()))
        show_hr1=sampledImgs[-2].permute(1,2,0).cpu().data.numpy()
        show_hr2=sampledImgs[-1].permute(1,2,0).cpu().data.numpy()
        show_lr=labels[0].permute(1,2,0).cpu().data.numpy()
        plt.subplot(1,3,1)
        plt.imshow(show_lr[...,15],cmap='gray')
        plt.title('condition')
        plt.subplot(1,3,2)
        plt.imshow(show_hr1[...,15],cmap='gray')
        plt.title('gen')
        plt.subplot(1,3,3)
        plt.imshow(show_hr2[...,15],cmap='gray')
        plt.title('hr')
        plt.show()
        # save_image(sampledImgs, os.path.join(
        #     modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        # psnr=cal_psnr(sampledImgs[-2].permute(1,2,0),sampledImgs[-1].permute(1,2,0))
        # print('psnr=',psnr)#37.2797

def eval_ddim(modelConfig: Dict):

    torch.set_printoptions(threshold=np.inf)
    torch.cuda.set_device(modelConfig['output_device'])#设置使用哪一个GPU，一个GPU即可
    device = torch.device(modelConfig["device"])#设备，即GPU
    print(device)
    with torch.no_grad():
        # 第三步：网络模型
        # model = UnetPlusPlus(T=modelConfig['T'],num_classes=modelConfig['num_classes'],ch=modelConfig['channel'],
        #                    ch_mult=modelConfig['channel_mult'],deep_supervision=modelConfig['deep_supervision'])
        model = UNet(T=modelConfig["T"], num_labels=10, ch=modelConfig["channel"],
                         ch_mult=modelConfig["channel_mult"],
                         num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

        model = nn.DataParallel(model.cuda(), device_ids=modelConfig['gpus'], output_device=modelConfig['gpus'][0])

        # 加载预训练权重
        if modelConfig["test_load_weight"] != None:
            ckpt = torch.load(os.path.join(  # 加载训练的参数
                modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)

            model.load_state_dict(ckpt['model'])  # 将训练的网络参数载入网络
            print("model load weight done：", modelConfig['test_load_weight'], 'epoch数：', ckpt['epoch'])
        model.eval()
        # 高斯采样器
        sampler = GaussianDiffusion_ddim(  # 创建Gaussian采样器
            model, modelConfig["beta_1"], modelConfig["beta_T"], image_size_x=modelConfig['img_size_x']
            , image_size_y=modelConfig['img_size_y'], batch_size=modelConfig['batch_size'],
            channels=modelConfig['channel']
            , timesteps=modelConfig['T'], ddim_timesteps=modelConfig['ddim_timesteps'], device=device,U=modelConfig['U']).to(device)
        # 第二步：数据
        num_list = ["991267","898176","761957","901038","833148"]

        for index in range(len(num_list)):

            eval_set_x=EvalDataset(modelConfig['test_file_x'],num_list[index])
            loader_x=DataLoader(eval_set_x,batch_size=modelConfig['batch_size'])
            eval_set_y = EvalDataset(modelConfig['test_file_y'],num_list[index])
            loader_y = DataLoader(eval_set_y, batch_size=modelConfig['batch_size'])
            eval_set_z = EvalDataset(modelConfig['test_file_z'],num_list[index])
            loader_z = DataLoader(eval_set_z, batch_size=modelConfig['batch_size'])

            print('--------------------开始：',num_list[index],'-----------------------')
            # for lr,hr,fibre_density in loader_x:
            #     print(lr.shape,hr.shape,fibre_density.shape)
            #     break
            # for lr,hr,fibre_density in loader_y:
            #     print(lr.shape,hr.shape,fibre_density.shape)
            #     break
            # for lr,hr,fibre_density in loader_z:
            #     print(lr.shape,hr.shape,fibre_density.shape)
            #     break

            hr=None
            lr=None
            #三个方向的切片
            full_img_z=[]
            full_img_y=[]
            full_img_x=[]
            # inter_slice=[]#保存中间切片

            start_time=time()
            #-----------------------z方向开始
            for i, (lr0,hr0,fibre_density) in enumerate(loader_z):#一个切片一个切片地取,z切片

                # if(i==48):
                #     inter_slice.append(lr0[0])#30*112*144
                #     inter_slice.append(hr0[0])#30*112*144
                hr=hr0.to(device)#clamp(-1,1).to(device)
                lr=lr0#.clamp(-1,1)

                labels=lr.to(device)
                # fibre=fibre.to(device)
                fibre_density=fibre_density.to(device)

                # Sampled from standard normal distribution
                noisyImage = torch.randn(#从高斯噪声开始,bs*c*x*y
                    size=[modelConfig["batch_size"], 30, modelConfig["img_size_x"], modelConfig["img_size_y"]], device=device)


                sampledImgs=[]
                for ii in range(30):
                    print(f'-{num_list[index]}--z方向---第{i}切片,第{ii}通道--------')
                    noisyImage_single=noisyImage[:,ii,:,:][None,...]#1*1*x*y
                    labels_single=labels[:,ii,:,:][None,...]#1*1*x*y
                    sampledImgs.append(sampler(noisyImage_single, labels_single,fibre_density))#进行采样，进行采样的图片还是均值0，方差1
                sampledImgs=torch.cat((sampledImgs),dim=1)#把一个切片的30个梯度方向拼接成一张图1*30*x*y
                full_img_z.append(sampledImgs[...,None])#每一个元素的尺寸：1*30*x*y*1
                print('z方向一个切片的梯度方向拼接完成后的切片尺寸：', sampledImgs.shape)#1*30*x*y
            full_img_z=torch.cat((full_img_z),dim=-1)[0]#[0]去掉batch_size
            print('z切片完全拼接完成后的尺寸：',full_img_z.shape)#30*112*144*112
            full_img_z=full_img_z.permute(1,2,3,0)#z方向把通道维度移动到最后一维:112*144*112*30
            #--------------------------z方向结束，y方向开始-------------------------
            for i, (lr0,hr0,fibre_density) in enumerate(loader_y):#一个切片一个切片地取,z切片

                hr=hr0.to(device)#clamp(-1,1).to(device)
                lr=lr0#.clamp(-1,1)

                labels=lr.to(device)
                # fibre=fibre.to(device)
                fibre_density=fibre_density.to(device)

                # Sampled from standard normal distribution
                noisyImage = torch.randn(#从高斯噪声开始,bs*c*x*z
                    size=[modelConfig["batch_size"], 30, modelConfig["img_size_x"], modelConfig["img_size_z"]], device=device)


                sampledImgs=[]
                for ii in range(30):
                    print(f'-{num_list[index]}--y方向---第{i}切片,第{ii}通道--------')
                    noisyImage_single=noisyImage[:,ii,:,:][None,...]#1*1*x*z
                    labels_single=labels[:,ii,:,:][None,...]#1*1*x*z
                    sampledImgs.append(sampler(noisyImage_single, labels_single,fibre_density))#进行采样，进行采样的图片还是均值0，方差1
                sampledImgs=torch.cat((sampledImgs),dim=1)#把一个切片的30个梯度方向拼接成一张图1*30*x*z
                full_img_y.append(sampledImgs[:,:,:,None,:])#每一个元素的尺寸：1*30*x*1*z
                print('y方向一个切片的梯度方向拼接完成后的切片尺寸：', sampledImgs.shape)#1*30*x*z
            full_img_y=torch.cat((full_img_y),dim=3)[0]#[0]去掉batch_size
            print('y切片完全拼接完成后的尺寸：',full_img_y.shape)#30*112*144*112
            full_img_y=full_img_y.permute(1,2,3,0)#z方向把通道维度移动到最后一维:112*144*112*30


            #--------------------------y方向结束，x方向开始-------------------------
            for i, (lr0,hr0,fibre_density) in enumerate(loader_x):#一个切片一个切片地取,z切片

                hr=hr0.to(device)#clamp(-1,1).to(device)
                lr=lr0#.clamp(-1,1)

                labels=lr.to(device)
                # fibre=fibre.to(device)
                fibre_density=fibre_density.to(device)

                # Sampled from standard normal distribution
                noisyImage = torch.randn(#从高斯噪声开始,bs*c*x*z
                    size=[modelConfig["batch_size"], 30, modelConfig["img_size_y"], modelConfig["img_size_z"]], device=device)


                sampledImgs=[]
                for ii in range(30):
                    print(f'-{num_list[index]}--x方向---第{i}切片,第{ii}通道--------')
                    noisyImage_single=noisyImage[:,ii,:,:][None,...]#1*1*y*z
                    labels_single=labels[:,ii,:,:][None,...]#1*1*y*z
                    sampledImgs.append(sampler(noisyImage_single, labels_single,fibre_density))#进行采样，进行采样的图片还是均值0，方差1
                sampledImgs=torch.cat((sampledImgs),dim=1)#把一个切片的30个梯度方向拼接成一张图1*30*y*z
                full_img_x.append(sampledImgs[:,:,None,:,:])#每一个元素的尺寸：1*30*1*y*z
                print('x方向一个切片的梯度方向拼接完成后的切片尺寸：', sampledImgs.shape)#1*30*y*z
            full_img_x=torch.cat((full_img_x),dim=2)[0]#[0]去掉batch_size
            print('x切片完全拼接完成后的尺寸：',full_img_y.shape)#30*112*144*112
            full_img_x=full_img_x.permute(1,2,3,0)#z方向把通道维度移动到最后一维:112*144*112*30

            #--------------------------x方向结束--------------------------------------
            full_img=(full_img_x+full_img_y+full_img_z)/3

            endtime = time()
            print('总耗时：', endtime - start_time)
            save_img = full_img.cpu().data.numpy()  # 用来保存的数据，转换成ndarray
            '''
            保存30通道数据
            '''

            data_affine=np.loadtxt(f'affine/{num_list[index]}_data_affine',dtype=np.float32)
            save_nifti(f'out_file/{num_list[index]}_sr_.nii.gz',save_img,data_affine)
            print('保存id:',num_list[index])

        #可视化展示









if __name__ == '__main__':
    torch.manual_seed(23)
    a=torch.randn(4,1,96,128)
    torch.manual_seed(24)
    b=torch.randn(4,1,96,128)
