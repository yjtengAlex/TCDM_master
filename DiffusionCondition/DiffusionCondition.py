import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)#从v中抽取数据，抽取索引是t,此时维度是[bs,]
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))#调整维度[bs,1,1,1]

class GaussianDiffusionTrainer(nn.Module):#用于训练
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels,fibre_density):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)#随机选择扩散时间
        noise = torch.randn_like(x_0)#噪声
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise#计算xt
        loss = F.smooth_l1_loss(self.model(x_t, t, labels,fibre_density), noise, reduction='sum',beta=1.0)#前向传播，计算损失
        return loss


class GaussianDiffusionSampler(nn.Module):#用于反向采样
    def __init__(self, model, beta_1, beta_T, T, w = 0.):#网络模型，起始β，终止β，扩散时间步数，条件分布的比重
        super().__init__()

        self.model = model#模型
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.#w=0，并且label=0，意味着没有条件
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.#w越大，条件的作用越大
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())#β，线性提高
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))#计算方差

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations,计算方差，就是betas
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)#抽取第t步的方差
        eps = self.model(x_t, t, labels)#网络预测的条件分布
        # nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))#网络预测的非条件分布
        # eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        out=torch.tensor(labels)
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            if(time_step%50==0):
                out=torch.cat((out,x_t),dim=0)
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        out=torch.cat((out,x_0),dim=0)
        return torch.clip(out,0,1)#torch.clip(out, -1, 1)


class GaussianDiffusion_ddim(nn.Module):
    def __init__(self, model,beta_1,beta_T,image_size_x,image_size_y,batch_size,channels,device,ddim_timesteps=50,ddim_discr_method='uniform',ddim_eta=1.0,clip_denoised=True,timesteps=1000, beta_schedule='linear',U=3):
        super(GaussianDiffusion_ddim, self).__init__()
        self.model=model
        self.image_size_x=image_size_x
        self.image_size_y=image_size_y
        self.batch_size=batch_size
        self.channels=channels
        self.ddim_timesteps=ddim_timesteps
        self.ddim_discr_method=ddim_discr_method
        self.ddim_eta=ddim_eta
        self.clip_denoised=clip_denoised
        self.timesteps=timesteps
        self.beta_schdule=beta_schedule
        self.U=U#重复计算的次数

        self.device=device
        betas=torch.linspace(beta_1,beta_T,timesteps,device=device)
        self.alphas=1-betas
        self.alphas_cumprod=torch.cumprod(self.alphas,dim=0)
        a=0
    # ...

    # use ddim to sample
    @torch.no_grad()

    def forward(self,x_T,label,fibre_density):
        # make ddim timestep sequence
        if self.ddim_discr_method == 'uniform':#线性采样，从0-timesteps中线性采出子序列，长度是ddim_timesteps
            c = self.timesteps // self.ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif self.ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.timesteps * .8), self.ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{self.ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1#子序列中的ddim_timesteps个数都加上1
        # previous sequence，后移序列
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = self.device
        # start from pure noise (for each example in the batch)#从纯噪声开始
        sample_img =x_T# torch.randn((self.batch_size, self.channels, self.image_size_x, self.image_size_y), device=device)
        # out=torch.tensor(label)#输出的一个列表
        for i in tqdm(reversed(range(0, self.ddim_timesteps)), desc='sampling loop time step', total=self.ddim_timesteps):
            # 第多少扩散步
            t = torch.full((self.batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            # print(t.item())
            # 扩散步前1步
            prev_t = torch.full((self.batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            xishu = self.cal_alpha_cump(prev_t, t)#计算前向的系数
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, sample_img.shape)
            for u in range(1,self.U+1):#重复计算当前时间步


                # 2. predict noise using model使用模型预测噪声
                pred_noise = self.model(sample_img, t,label,fibre_density)
                # print('pred_noise:',pred_noise)
                # 3. get the predicted x_0，根据前向过程表达式反推x0
                pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
                if self.clip_denoised:
                    pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

                # 4. compute variance: "sigma_t(η)" -> see formula (16)，σ的表达式
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = self.ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

                # 5. compute "direction pointing to x_t" of formula (12)，等式12的第2项
                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

                # 6. compute x_{t-1} of formula (12)等式12
                x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)
                if u<self.U and i>=1:#还在当前时间步
                    sample_img=xishu[0]*x_prev+xishu[1]*torch.randn_like(x_prev)
                else:#上一个时间步
                    sample_img = x_prev
            # out=torch.cat((out,sample_img),dim=0)
        # out=torch.cat((out,sample_img),dim=0)
        return sample_img

    def cal_alpha_cump(self,t_pred,t):
        '''
        计算由前一子集的x_t-1到x_t的alpha的累乘
        :param t_pred: 前一图像的扩散时间
        :param t: 当前图像的扩散时间
        :return:
        '''
        a=1
        for ii in range(t_pred+1,t+1):
            a=a*self.alphas[ii]
        return torch.sqrt(a),torch.sqrt(1-a)

