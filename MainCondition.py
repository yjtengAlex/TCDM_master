from DiffusionCondition.TrainCondition import train, eval
import torch
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)


def main(model_config=None):
    modelConfig = {
        "state": "train", # or evalno
        "epoch": 51,#训练轮数
        "batch_size": 1,#
        "T": 2000,#扩散次数
        "channel": 64,#网络通道数
        "channel_mult": [1, 2],#网络通道数的倍数,[1,2,2,2]
        "num_res_blocks": 2,#多少个残差块一个下采样2
        'num_classes':1,#数据通道数
        "dropout": 0.15,#失活比例
        "lr": 1e-4,#学习率
        "multiplier": 2.5,#学习率预热倍数
        "beta_1": 1e-4,#β初始值
        "beta_T": 0.028,#β终值
        "img_size_x": 112,#图片尺寸，训练时不使用，图片尺寸96*128q
        "img_size_y":144,
        'img_size_z':112,
        'deep_supervision':True,#是否考虑中间层的结果
        "grad_clip": 1.,#防止梯度爆炸的系数
        "gpus":[2],#多卡训练的GPU列表
        "output_device":'cuda:2',#汇总梯度的gpu编号,即列表中第一个gpu
        "device": "cuda",#GPU,多卡训练不指定哪一个GPU
        "w": 1.8,#降低波动的权重
        "save_dir": "./CheckpointsCondition/",#模型保存位置
        "training_load_weight":None,#'best_pre.pth',# ,#是否有预训练
        "test_load_weight": "ckpt_0048.pth",#测试时训练的模型
        "sampled_dir": "./SampledImgs/",#图片保存位置
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)
if __name__ == '__main__':
    main()
