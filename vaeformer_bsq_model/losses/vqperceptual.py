from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transcoder.losses.lpips import LPIPS
from transcoder.models.discriminator.patchgan import PatchGANDiscriminator, weights_init
from transcoder.models.discriminator.stylegan import StyleGANDiscriminator
from transcoder.models.discriminator.stylegan3d import StyleGAN3DDiscriminator
from transcoder.models.stylegan_utils.ops.conv2d_gradfix import no_weight_gradients

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)


def vanilla_g_loss(logits_fake):
    return torch.mean(F.softplus(-logits_fake))

def d_r1_loss(logits_real, img_real):
    with no_weight_gradients():
        grad_real, = torch.autograd.grad(
            outputs=logits_real.sum(), inputs=img_real,
            allow_unused=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty



class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_type='patchgan',
                 disc_input_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 disc_reg_freq=0, disc_reg_r1=10,
                 reconstruct_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", skip_disc=False, use_bf16=False, 
                 use_adaptive_disc_weight=True,
                 zero_mean=False, num_frames=1,
                 codebook_rampup_multiplier=1.0, codebook_rampup_steps=0, vari_name='u',
                 **kwargs):
        """
        Inputs:
            - disc_start: int, the global step at which the discriminator starts to be trained
            - codebook_weight: float, the weight of the codebook loss
            - perceptual_weight: float, the weight of the perceptual loss
            - disc_weight: float, the weight of the discriminator loss
            - disc_factor: {0, 1} whether to mask the discriminator loss
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.codebook_rampup_multiplier = codebook_rampup_multiplier
        self.codebook_rampup_steps = codebook_rampup_steps
        self.pixel_weight = pixelloss_weight
        # self.perceptual_loss = LPIPS(zero_mean=zero_mean).eval()
        self.reconstruct_weight = reconstruct_weight
        # self.perceptual_weight = perceptual_weight
        self.adaptive_dweight = use_adaptive_disc_weight
        self.disc_type = disc_type
        self.num_frames = num_frames
        self.vari_name = vari_name

        # 用于era5 变量的反归一化
        self.total_levels= [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
        775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
        350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
        70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,
        1.]
        self.era5_single_dict = {
            'v10': 100,'u10': 200,'v100': 300, 'u100': 400, 't2m': 500,'tcc': 600, 'sp': 700, 'msl': 800
        }
        self.era5_single_dict_reverse = {v: k for k, v in self.era5_single_dict.items()}

        self.era5_tp_dict = {
            'tp1h': 100, 'tp2h': 200,'tp3h': 300, 'tp4h': 400, 'tp5h': 500, 'tp6h': 600
        }
        self.era5_tp_dict_reverse = {v: k for k, v in self.era5_tp_dict.items()}

        # 加载JSON文件
        with open('/mnt/petrelfs/zhaosijie/weather_latent_autoencoder_bsq/transcoder/data/china_data/cma_1km/era5_json_data/mean_std.json', 'r') as f:
            self.era5_mean_std = json.load(f)
        with open('/mnt/petrelfs/zhaosijie/weather_latent_autoencoder_bsq/transcoder/data/china_data/cma_1km/era5_json_data/mean_std_single.json', 'r') as f:
            self.era5_mean_std_single = json.load(f)

        if vari_name != 'single' and vari_name != 'tp':
            # 提取vari_list
            self.vari_std_list = self.era5_mean_std['std'][vari_name]
            # 创建total_level到vari_list的映射
            self.level_to_vari = {level: vari for level, vari in zip(self.total_levels, self.vari_std_list)}

        if disc_type.lower() == 'patchgan':
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm,
                ndf=disc_ndf,
            ).apply(weights_init)
        elif disc_type.lower() == 'stylegan':
            self.discriminator = StyleGANDiscriminator(
                0, disc_input_size, disc_in_channels,
                num_fp16_res=8 if use_bf16 else 0,    # 8 is sufficiently large to cover all res
                epilogue_kwargs={'mbstd_group_size': 4},
            )
        elif disc_type.lower() == 'stylegan3d':
            self.discriminator = StyleGAN3DDiscriminator(
                num_frames, disc_input_size,
                video_channels=disc_in_channels,
            )
        else:
            raise ValueError(f"Unsupported disc_type {disc_type}")
        self.discriminator_iter_start = disc_start
        self.skip_disc_before_start = skip_disc
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
            self.gen_loss = hinge_g_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
            self.gen_loss = vanilla_g_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.disc_reg_freq = disc_reg_freq
        self.disc_reg_r1 = disc_reg_r1

    # def era5_std(self, vari_metadata):

    #     # 根据tensor_a的值查找对应的vari_list中的值
    #     vari_values = [self.level_to_vari[float(val)] for val in vari_metadata]

    #     # 转换成Tensor
    #     vari_std = torch.tensor(vari_values)

    #     return vari_std

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def create_geospatial_weight_tensor(self, h, w):
        # 计算由于投影导致的像素大小差距，对应的weight_tensor
        # 创建一个从90到-90等间距排列的h行1列tensor
        row_values = torch.linspace(90, -90, h)
        # 扩展成h行w列的tensor，每一行的值都相同
        base_tensor = row_values.unsqueeze(1).repeat(1, w)

        # 计算cos函数
        cos_tensor = torch.cos(torch.deg2rad(base_tensor))  # 将角度转为弧度再计算cos

        # 计算cos_tensor一列值的总和
        sum_value = cos_tensor[:, 0].sum()

        # 计算最终的weight_tensor
        weight_tensor = cos_tensor / sum_value * h

        return weight_tensor

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", metadata=None):
        if inputs.ndim == 5:
            assert self.num_frames == inputs.shape[2], f"Number of frames does not match input "
            inputs = rearrange(inputs, 'n c t h w -> (n t) c h w')
            reconstructions = rearrange(reconstructions, 'n c t h w -> (n t) c h w')

        # 计算由于投影导致的weight
        b,c,h,w = inputs.shape
        geospatial_weight = self.create_geospatial_weight_tensor(h,w).unsqueeze(0).unsqueeze(0).to(inputs.device, inputs.dtype)
        
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = (inputs.contiguous() - reconstructions.contiguous())**2  # 尝试使用mse代替mae，去除噪点的影响
        rec_loss = rec_loss * geospatial_weight  # 乘以地理权重

        # 获取std，用来计算真实的mae和rmse
        metadata_log = metadata[0]
        if self.vari_name == 'single':
            single_vari_name = [self.era5_single_dict_reverse[int(val.item())] for val in metadata_log]
            vari_std = [self.era5_mean_std_single['std'][k] for k in single_vari_name]
            vari_std = torch.tensor(vari_std).to(device=rec_loss.device, dtype=rec_loss.dtype)
        elif self.vari_name == 'tp':
            tp_vari_name = [self.era5_tp_dict_reverse[int(val.item())] for val in metadata_log]
            vari_std = [self.era5_mean_std_single['std'][k] for k in tp_vari_name]
            vari_std = torch.tensor(vari_std).to(device=rec_loss.device, dtype=rec_loss.dtype)
        else:
            # 根据tensor_a的值查找对应的vari_list中的值
            vari_values = [self.level_to_vari[float(val)] for val in metadata_log]
            # 转换成Tensor
            vari_std = torch.tensor(vari_values).to(device=rec_loss.device, dtype=rec_loss.dtype)

        if self.vari_name in ['u', 'v', 't']:
            # 当计算高度层变量的损失时，需要针对它们的std来平衡各个高度层之间的损失，保证所有高度层都能够比较平均的取得好效果。由于使用的是mse，所以采用平方的std
            # 注意只对u,v,t变量使用高度层损失权重，因为它们不同高度层的方差差别不会太大
            vari_std_loss_weight = ((F.normalize(vari_std, p=2, dim=0)**2)*vari_std.shape[0]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # 保证权重的平方均值为1
            rec_loss = rec_loss * vari_std_loss_weight
            # print(f'vari_std_loss_weight: {vari_std_loss_weight}')


        # print(f'vari_std: {vari_std}')
        # print(f'vari_std_loss_weight: {vari_std_loss_weight}')
        rec_rmse = (inputs.contiguous() - reconstructions.contiguous())**2
        rec_rmse = rec_rmse * geospatial_weight  # 乘以地理权重
        rec_rmse = rearrange(rec_rmse, 'n c h w -> n (h w) c')
        rec_rmse = torch.sqrt(rec_rmse.mean(dim=1)).mean(dim=0)
        # print(f'normalizer rmse: {rec_rmse}')
        # print(f'vari_std: {vari_std}')
        rec_rmse = rec_rmse * vari_std
        # print(f'rec_rmse: {rec_rmse}')


        rec_mae = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_mae = rec_mae * geospatial_weight  # 乘以地理权重
        rec_mae = rearrange(rec_mae, 'n c h w -> n (h w) c')
        rec_mae = rec_mae.mean(dim=1).mean(dim=0)
        rec_mae = rec_mae * vari_std

        # if self.perceptual_weight > 0:
        #     p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        # else:
        #     p_loss = torch.zeros_like(rec_loss)
        

        # nll_loss = self.reconstruct_weight * rec_loss + self.perceptual_weight * p_loss
        rec_loss = rearrange(rec_loss, 'n c h w -> n (h w) c')
        rec_loss = rec_loss.mean(dim=1).mean(dim=1)
        nll_loss = self.reconstruct_weight * rec_loss
        nll_loss = torch.mean(nll_loss)

        if global_step < self.discriminator_iter_start and self.skip_disc_before_start:
            # before the discriminator joins the party, we only care about the reconstruction loss
            # no need to run the discriminator if it does not update anything
            loss = nll_loss + self.codebook_weight * codebook_loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                #    "{}/p_loss".format(split): p_loss.detach().mean()
                   }
            return nll_loss, log
        else:
            disc_factor = 1.0 if global_step >= self.discriminator_iter_start else 0.

        # now the GAN part
        
        if optimizer_idx == 0:
            # generator update
            if self.disc_type == 'stylegan3d':
                reconstructions = rearrange(reconstructions, '(n t) c h w -> n c t h w', t=self.num_frames)
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous(), metadata=metadata)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1), metadata=metadata)
            g_loss = self.gen_loss(logits_fake)

            try:
                if self.adaptive_dweight:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                else:
                    d_weight = torch.tensor(self.discriminator_weight)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            if self.codebook_rampup_steps > 0:
                rampup_rate =  min(self.codebook_rampup_steps, global_step) / self.codebook_rampup_steps
                cb_weight = self.codebook_weight * (1.0 * rampup_rate  + self.codebook_rampup_multiplier * (1 - rampup_rate))
            else:
                cb_weight = self.codebook_weight

            if self.vari_name == 'tp':
                # 降水重建不要使用gan相关的 loss
                loss = nll_loss + cb_weight * codebook_loss
            else:
                loss = nll_loss + d_weight * disc_factor * g_loss + cb_weight * codebook_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                #    "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            for i in range(rec_rmse.shape[0]):
                log.update({
                   f"{split}/channel_{metadata_log[i]}_rmse": rec_rmse[i].detach(),
                   f"{split}/channel_{metadata_log[i]}_mae": rec_mae[i].detach(),
                })

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if self.disc_type == 'stylegan3d':
                inputs = rearrange(inputs, '(n t) c h w -> n c t h w', t=self.num_frames)
                reconstructions = rearrange(reconstructions, '(n t) c h w -> n c t h w', t=self.num_frames)
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach(), metadata=metadata)
                logits_fake = self.discriminator(reconstructions.contiguous().detach(), metadata=metadata)
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1), metadata=metadata)
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1), metadata=metadata)

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }

            if self.disc_reg_freq > 0 and (global_step + 1) % self.disc_reg_freq == 0:
                inputs.requires_grad = True
                logits_real = self.discriminator(inputs.contiguous(), metadata=metadata)
                r1_loss = d_r1_loss(logits_real, inputs)
                r1_loss_scale = self.disc_reg_r1 / 2 * r1_loss * self.disc_reg_freq
                d_loss = d_loss + r1_loss_scale
                log.update({
                    "{}/disc_r1_loss".format(split): r1_loss.detach().mean(),
                    "{}/disc_r1_loss_scale".format(split): r1_loss_scale.detach().mean(),
                })

            return d_loss, log
