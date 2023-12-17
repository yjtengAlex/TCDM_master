from DiffusionCondition.TrainCondition import train, eval,eval_ddim

def main(model_config=None):
    modelConfig = {
        "state": "eval_ddim", # or eval_ddim
        "epoch": 50,
        "batch_size": 1,
        "T": 2000,
        "channel": 64,
        "channel_mult": [1, 2],#原[1,2,2,2]
        "num_res_blocks": 2,
        'num_classes':1,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size_x": 112,
        'img_size_y':144,
        'img_size_z':112,
        'deep_supervision':True,
        "grad_clip": 1.,
        "device": "cuda",
        'ddim_timesteps':100,#ddim的采样步数
        'gpus':[0],
        'output_device':'cuda:0',
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": 'best.pth',#"ckpt_0050.pth",
        "sampled_dir": "./out_pic/",
        "sampledNoisyImgName": "NoisyGuidenceImgs-001.png",
        "sampledImgName": "Standard_Generature_ddpm.png",
        "nrow": 8,
        'test_file_x': './h5data/test_data_x',  # 测试的图片,x方向
        'test_file_y':'./h5data/test_data_y',#y方向
        'test_file_z':'./h5data/test_data_z',#z方向
        'scale': 4,  # 放大倍数
        'U':1,#重采样次数
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig['state']=='eval':
        eval(modelConfig)
    else:
        eval_ddim(modelConfig)

if __name__ == '__main__':
    main()