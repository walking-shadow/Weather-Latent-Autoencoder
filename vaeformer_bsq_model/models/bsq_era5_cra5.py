import torch
import torch.nn.functional as F
import lightning as L
import random
import matplotlib.pyplot as plt
import cmaps
import io
from PIL import Image
import torchvision.transforms as T
import numpy as np
from main import instantiate_from_config
from contextlib import contextmanager
from collections import OrderedDict
from einops import rearrange
import torch
from torchvision import transforms
from PIL import Image
from functools import partial

import numpy as np
import lzma
import os
from datetime import datetime

from transcoder.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay, Scheduler_LinearWarmup_CosineDecay_BSQ
from transcoder.scheduler.ema import LitEma


import torch
import torch.nn as nn
import torch.nn.functional as F

from transcoder.losses.logit_laplace_loss import LogitLaplaceLoss
from transcoder.models.quantizer.bsq import BinarySphericalQuantizer
from transcoder.models.quantizer.vq import VectorQuantizer
# from transcoder.models.transformer import TransformerDecoder, TransformerEncoder
from transcoder.models.vit_nlc import ViT_Encoder, ViT_Decoder

class BSQModel(L.LightningModule):
    def __init__(self,
                # vitconfig,
                lossconfig,
                embed_dim,
                embed_group_size=9,
                ## Quantize Related
                l2_norm=False, logit_laplace=False, ckpt_path=None, ignore_keys=[],
                dvitconfig=None, beta=0., gamma0=1.0, gamma=1.0, zeta=1.0,
                persample_entropy_compute='group',
                cb_entropy_compute='group',
                post_q_l2_norm=False,
                inv_temperature=1.,
                ### scheduler config
                resume_lr=None,
                min_learning_rate = 0,
                use_ema = False,
                stage = None,
                lr_drop_epoch = None,
                lr_drop_rate = 0.1,
                warmup_epochs = 1.0, #warmup epochs
                scheduler_type = "linear-warmup_cosine-decay-bsq",
                lr_start = 0.1,
                lr_max = 1.0,
                lr_min = 0.5,
                ):
        super().__init__()
        # CRA5 中的设置
        feature_dim = 1024
        large_default_dict =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=(15,14), patch_stride=(10,10), patch_padding=(2,2), in_chans=6, out_chans=6, embed_dim=feature_dim, depth=16,
            num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            z_dim =  None,
            learnable_pos= True,
            window= True,
            window_size = [(36, 36), (18, 72), (72, 18)],  # 专门为721,1440图像设计
            interval = 4,
            round_padding= True,
            pad_attn_mask= True , # to_do: ablation
            test_pos_mode= 'learnable_simple_interpolate', # to_do: ablation
            lms_checkpoint_train= True,
            img_size= (721, 1440)
        )
        large_default_dict_decoder =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=(15,14), patch_stride=(10,10), patch_padding=(2,2), in_chans=6, out_chans=6, embed_dim=feature_dim, depth=32,
            num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            z_dim =  None,
            learnable_pos= True,
            window= True,
            window_size = [(36, 36), (18, 72), (72, 18)],  # 专门为721,1440图像设计
            interval = 4,
            round_padding= True,
            pad_attn_mask= True , # to_do: ablation
            test_pos_mode= 'learnable_simple_interpolate', # to_do: ablation
            lms_checkpoint_train= True,
            img_size= (721, 1440)
        )

        self.latent_h = large_default_dict['img_size'][0] // large_default_dict['patch_stride'][0]
        self.encoder = ViT_Encoder(**large_default_dict)
        self.decoder = ViT_Decoder(**large_default_dict_decoder)

        self.quantize = BinarySphericalQuantizer(
            embed_dim, beta, gamma0, gamma, zeta,
            group_size=embed_group_size,
            persample_entropy_compute=persample_entropy_compute,
            cb_entropy_compute=cb_entropy_compute,
            input_format='blc',
            l2_norm=post_q_l2_norm,
            inv_temperature=inv_temperature,
        )
        self.loss = instantiate_from_config(lossconfig)

        self.n_embed = 2 ** embed_dim
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.quant_embed = nn.Linear(in_features=feature_dim, out_features=embed_dim)
        self.post_quant_embed = nn.Linear(in_features=embed_dim, out_features=feature_dim)
        self.l2_norm = l2_norm
        self.logit_laplace = logit_laplace

        # set quantizer params
        self.beta = beta      # commit loss
        self.gamma0 = gamma0  # entropy
        self.gamma = gamma    # entropy penalty
        self.zeta = zeta      # lpips
        self.embed_group_size = embed_group_size
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.post_q_l2_norm = post_q_l2_norm
        self.inv_temperature = inv_temperature

        self.use_ema = use_ema
        if self.use_ema and stage is None: #no need to construct EMA when training Transformer
            self.model_ema = LitEma(self)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)
        self.resume_lr = resume_lr
        self.lr_drop_epoch = lr_drop_epoch
        self.lr_drop_rate = lr_drop_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False

        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min

        self.strict_loading = False
        self.img2tensor = T.ToTensor()

        self.validate_metrics = {}

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        filter out the non-used keys
        '''
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}
        
    def init_from_ckpt(self, path, ignore_keys=list(), stage="transformer"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue 
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v                  
        missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False) #first stage
        print(f"Restored from {path}")

    def encode(self, x, metadata, skip_quantize=False):
        # x shape: B,C,721,1440
        # x = F.interpolate(x, (720,1440), mode='bilinear')
        # print(f'input x shape:{x.shape}')
        h = self.encoder(x, metadata)  # b,c,h,w
        # print(f'encoder x shape: {h.shape}')
        h = rearrange(h, 'b c h w -> b (h w) c')
        h = self.quant_embed(h)
        if self.l2_norm:
            h = F.normalize(h, dim=-1)
        if skip_quantize:
            assert not self.training, 'skip_quantize should be used in eval mode only.'
            return h, {}, {}
        quant, loss, info = self.quantize(h)
        return quant, loss, info

    def decode(self, quant, metadata):
        # print(f'decoder input x shape: {quant.shape}')

        h = self.post_quant_embed(quant)
        h = rearrange(h, 'b (h w) c -> b c h w', h=self.latent_h)

        x = self.decoder(h, metadata)
        # x = F.interpolate(x, (721,1440), mode='bilinear')
        # print(f'decoder output x shape: {x.shape}')

        return x

    def load_save_bit_data(self, x_compress):
        x = x_compress.astype(np.float32)
        x = x*2 - 1
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        x = x*q_scale
        x = torch.tensor(x)
        return x


    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    def forward(self, input, metadata, skip_quantize=False):
        quant, loss, info = self.encode(input, metadata, skip_quantize=skip_quantize)
        dec = self.decode(quant, metadata)
        
        return dec, loss, info

    def on_train_start(self):
        """
        change lr after resuming
        """
        if self.resume_lr is not None:
            opt_gen, opt_disc = self.optimizers()
            for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                opt_gen_param_group["lr"] = self.resume_lr
                opt_disc_param_group["lr"] = self.resume_lr

    # # fix mulitple optimizer bug
    # # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    # def training_step(self, batch, batch_idx):
    #     # x = self.get_input(batch, self.image_key)

    #     x, metadata = batch

    #     # x_log = x[0].detach().cpu().numpy()
    #     # if x_log.shape[0] == 25:
    #     #     np.save(f'/mnt/petrelfs/zhaosijie/weather_latent_autoencoder_bsq/paper_figure/{self.loss.vari_name}_{batch_idx}.npy', x_log)
    #     #     import sys
    #     #     sys.exit(0)

    #     xrec, eloss,  loss_info = self(x, metadata)
    #     # q的值域为>=0，并且大量的值分布在0附近
    #     if self.loss.vari_name == 'q':
    #         xrec = F.relu(xrec)

    #     opt_gen, opt_disc = self.optimizers()
    #     scheduler_gen, scheduler_disc = self.lr_schedulers()

    #     ####################
    #     # fix global step bug
    #     # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
    #     opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
    #     opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
    #     # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
    #     # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
    #     ####################
        

    #     # optimize generator
    #     aeloss, log_dict_ae = self.loss(eloss, x, xrec, 0, self.global_step,
    #                                     last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)
    #     opt_gen.zero_grad()
    #     self.manual_backward(aeloss)
    #     opt_gen.step()
    #     scheduler_gen.step()
        
    #     # optimize discriminator
    #     if self.loss.vari_name != 'tp':
    #         discloss, log_dict_disc = self.loss(eloss, x, xrec, 1, self.global_step,
    #                                             last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)
    #         opt_disc.zero_grad()
    #         self.manual_backward(discloss)
    #         opt_disc.step()
    #         scheduler_disc.step()

    #         self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            
    #     self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)





    # def compress_binary_array(self, binary_array):

    #     # （2）用 packbits 将二值数组按位打包
    #     # 这样会将一维的 8 个二值元素合并到一个字节当中
    #     packed_data = np.packbits(binary_array, axis=None)
    #     # 这里 axis=None 的意思是先把 data 拉平成一维，然后再打包

    #     # （3）用 LZMA 进行无损压缩
    #     # preset=9 表示尽量使用最高级别的压缩，不过速度会稍慢
    #     compressed_bytes = lzma.compress(packed_data, preset=9)
        
    #     return compressed_bytes



    
    def training_step(self, batch, batch_idx):
        # 这个training_step是为了多卡压缩数据，真正的训练函数在上面

        with torch.inference_mode(mode=True):
            x, metadata, sample_time_str = batch
            b,c,h,w = x.shape
            quant, loss, info = self.encode(x, metadata)
            vari_name = self.loss.vari_name
            x_compress = info['zq_compress']
            for b_i in range(b):
                x_tensor_compress = x_compress[b_i].detach().cpu().numpy()  # l,c
                x_tensor_compress_save = np.packbits(x_tensor_compress, axis=-1)

                sample_time_str_bi = sample_time_str[b_i]
                sample_time_bi = datetime.strptime(sample_time_str_bi, '%Y%m%dT%H')
                sample_time_month_str_bi = sample_time_bi.strftime('%m')
                if vari_name == 'single':
                    file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/train/single/{vari_name}_{c}/{sample_time_bi.year}/{sample_time_month_str_bi}'
                elif vari_name == 'tp':
                    if c == 6:
                        file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/train/single/{vari_name}_{c}/{sample_time_bi.year}/{sample_time_month_str_bi}'
                    else:
                        if int(metadata[b_i]) == 100:
                            file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/train/single/tp_1h/{sample_time_bi.year}/{sample_time_month_str_bi}'
                        else:
                            # print(f'saved in tp_6h')
                            file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/train/single/tp_6h/{sample_time_bi.year}/{sample_time_month_str_bi}'
                else:
                    file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/train/altitude/{vari_name}_{c}/{sample_time_bi.year}/{sample_time_month_str_bi}'
                os.makedirs(file_folder, exist_ok=True)
                
                filename = file_folder + f'/{vari_name}_{c}_{sample_time_str_bi}.npy'
                np.save(filename, x_tensor_compress_save)

                # x_tensor_decompress = np.unpackbits(x_tensor_compress_save, axis=-1)
                # if np.array_equal(x_tensor_decompress, x_tensor_compress):
                #     print('equal!')

                # x_compress_load = self.load_save_bit_data(x_tensor_decompress).to(device=quant.device, dtype=quant.dtype)
                
                # is_equal = torch.equal(quant[b_i], x_compress_load)
                # if is_equal:
                #     print('torch.equal(quant[b_i], x_compress_load) True!')
                # else:
                #     np.save(f'quant_{b_i}.npy', quant[b_i].detach().cpu().numpy())
                #     np.save(f'x_compress_load_{b_i}.npy', x_compress_load.detach().cpu().numpy())
                #     import sys
                #     sys.exit(0)



    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    # def on_train_epoch_end(self):
    #     ### update lr
    #     self.lr_annealing()

    # def lr_annealing(self):
    #     """
    #     Perform Lr decay
    #     """
    #     if self.lr_drop_epoch is not None:
    #         current_epoch = self.trainer.current_epoch
    #         if (current_epoch + 1) in self.lr_drop_epoch:
    #             opt_gen, opt_disc = self.optimizers()
    #             for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
    #                 opt_gen_param_group["lr"] = opt_gen_param_group["lr"] * self.lr_drop_rate
    #                 opt_disc_param_group["lr"] = opt_disc_param_group["lr"] * self.lr_drop_rate

    def cal_rmse_metrics(self, x, xrec):
        # 计算归一化之前的RMSE
        cma_mean_std = {
            "mean": {
                "u10": 0.1415459487851344,
                "v10": -0.10691198255871551,
                "t2m": 287.74907750465064,
                "r2": 70.60789577472954
            },
            "std": {
                "u10": 3.319792868863362,
                "v10": 2.9267358870964375,
                "t2m": 16.031881191310198,
                "r2": 20.169904063835272
            }
        }


        x_weather = x

        data_device = x_weather.device
        data_dtype = x_weather.dtype
        cma_mean = torch.tensor([v for v in cma_mean_std["mean"].values()]).to(device=data_device, dtype=data_dtype)
        cma_std = torch.tensor([v for v in cma_mean_std["std"].values()]).to(device=data_device, dtype=data_dtype)

        xrec_weather = xrec
        B,C,H,W = x_weather.shape
        cma_mean = cma_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(B, C, H, W)
        cma_std = cma_std.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(B, C, H, W)
        x_weather = x_weather*cma_std + cma_mean
        xrec_weather = xrec_weather*cma_std + cma_mean

        log_weather_rmse = torch.sqrt(((x_weather-xrec_weather)**2).mean()).detach()
        log_u10_rmse = torch.sqrt(((x_weather[:,0,...]-xrec_weather[:,0,...])**2).mean()).detach()
        log_v10_rmse = torch.sqrt(((x_weather[:,1,...]-xrec_weather[:,1,...])**2).mean()).detach()
        log_t2m_rmse = torch.sqrt(((x_weather[:,2,...]-xrec_weather[:,2,...])**2).mean()).detach()
        log_r2_rmse = torch.sqrt(((x_weather[:,3,...]-xrec_weather[:,3,...])**2).mean()).detach()


        log_metric_dict = {
            "weather_rmse_metric": log_weather_rmse,
            "u10_rmse_metric": log_u10_rmse,
            "v10_rmse_metric": log_v10_rmse,
            "t2m_rmse_metric": log_t2m_rmse,
            "r2_rmse_metric": log_r2_rmse,

        }
        return log_metric_dict

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)


    # def _validation_step(self, batch, batch_idx, suffix=""):

    #     x, metadata = batch

    #     xrec, eloss,  loss_info = self(x, metadata)
    #     # q的值域为>=0，并且大量的值分布在0附近
    #     if self.loss.vari_name == 'q':
    #         xrec = F.relu(xrec)
    #     # log_rmse_metric = self.cal_rmse_metrics(x, x_rec)
    #     # self.log_dict(log_rmse_metric, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    #     aeloss, log_dict_ae = self.loss(eloss, x, xrec, 0, self.global_step,
    #                                     last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)
    #     discloss, log_dict_disc = self.loss(eloss, x, xrec, 1, self.global_step,
    #                                         last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)


    #     self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    #     self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    #     return self.log_dict
            
    # def _validation_step(self, batch, batch_idx, suffix=""):
    #     # 这个_validation_step专门用来计算验证集上的模型指标
    #     x, metadata = batch

    #     xrec, eloss,  loss_info = self(x, metadata)
    #     # q的值域为>=0，并且大量的值分布在0附近
    #     if self.loss.vari_name == 'q':
    #         xrec = F.relu(xrec)
    #     # log_rmse_metric = self.cal_rmse_metrics(x, x_rec)
    #     # self.log_dict(log_rmse_metric, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    #     aeloss, log_dict_ae = self.loss(eloss, x, xrec, 0, self.global_step,
    #                                     last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)
    #     # discloss, log_dict_disc = self.loss(eloss, x, xrec, 1, self.global_step,
    #     #                                     last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)


    #     self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    #     # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    #     for k in log_dict_ae:
    #         if k in self.validate_metrics:
    #             self.validate_metrics[k] = self.validate_metrics[k] + log_dict_ae[k]
    #         else:
    #             self.validate_metrics[k] = log_dict_ae[k]

    #     return self.log_dict
    
    def _validation_step(self, batch, batch_idx, suffix=""):
        # 这个_validation_step专门用于压缩
        with torch.inference_mode(mode=True):
            x, metadata, sample_time_str = batch
            b,c,h,w = x.shape
            quant, loss, info = self.encode(x, metadata)
            vari_name = self.loss.vari_name
            x_compress = info['zq_compress']
            for b_i in range(b):
                x_tensor_compress = x_compress[b_i].detach().cpu().numpy()  # c,h,w
                x_tensor_compress_save = np.packbits(x_tensor_compress, axis=-1)

                sample_time_str_bi = sample_time_str[b_i]
                sample_time_bi = datetime.strptime(sample_time_str_bi, '%Y%m%dT%H')
                sample_time_month_str_bi = sample_time_bi.strftime('%m')

                if vari_name == 'single':
                    file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/val/single/{vari_name}_{c}/{sample_time_bi.year}/{sample_time_month_str_bi}'
                elif vari_name == 'tp':
                    if c == 6:
                        file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/val/single/{vari_name}_{c}/{sample_time_bi.year}/{sample_time_month_str_bi}'
                    else:
                        if int(metadata[b_i]) == 100:
                            file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/val/single/tp_1h/{sample_time_bi.year}/{sample_time_month_str_bi}'
                        else:
                            file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/val/single/tp_6h/{sample_time_bi.year}/{sample_time_month_str_bi}'
                else:
                    file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/val/altitude/{vari_name}_{c}/{sample_time_bi.year}/{sample_time_month_str_bi}'
                os.makedirs(file_folder, exist_ok=True)
                
                filename = file_folder + f'/{vari_name}_{c}_{sample_time_str_bi}.npy'
                np.save(filename, x_tensor_compress_save)

                # x_tensor_decompress = np.unpackbits(x_tensor_compress_save, axis=-1)
                # if np.array_equal(x_tensor_decompress, x_tensor_compress):
                #     print('equal!')
            
    def test_step(self, batch, batch_idx, suffix=""):
        # 这个_validation_step专门用于测试集的一个月pixel压缩和全部latent压缩
        with torch.inference_mode(mode=True):
            x, metadata, sample_time_str = batch
            b,c,h,w = x.shape
            quant, loss, info = self.encode(x, metadata)
            vari_name = self.loss.vari_name
            x_compress = info['zq_compress']
            for b_i in range(b):
                x_tensor_compress = x_compress[b_i].detach().cpu().numpy()  # c,h,w
                x_tensor_compress_save = np.packbits(x_tensor_compress, axis=-1)

                sample_time_str_bi = sample_time_str[b_i]
                sample_time_bi = datetime.strptime(sample_time_str_bi, '%Y%m%dT%H')
                sample_time_month_str_bi = sample_time_bi.strftime('%m')

                if vari_name == 'single':
                    file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/test_latent/single/{vari_name}_{c}/{sample_time_bi.year}/{sample_time_month_str_bi}'
                elif vari_name == 'tp':
                    if c == 6:
                        file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/test_latent/single/{vari_name}_{c}/{sample_time_bi.year}/{sample_time_month_str_bi}'
                    else:
                        if int(metadata[b_i]) == 100:
                            file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/test_latent/single/tp_1h/{sample_time_bi.year}/{sample_time_month_str_bi}'
                        else:
                            file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/test_latent/single/tp_6h/{sample_time_bi.year}/{sample_time_month_str_bi}'
                else:
                    file_folder = f'/mnt/hwfile/zhaosijie_delete_soon/compressed_era5_latent/test_latent/altitude/{vari_name}_{c}/{sample_time_bi.year}/{sample_time_month_str_bi}'
                os.makedirs(file_folder, exist_ok=True)
                
                filename = file_folder + f'/{vari_name}_{c}_{sample_time_str_bi}.npy'
                np.save(filename, x_tensor_compress_save)

                # x_tensor_decompress = np.unpackbits(x_tensor_compress_save, axis=-1)
                # if np.array_equal(x_tensor_decompress, x_tensor_compress):
                #     print('equal!')

    def on_validation_epoch_end(self):
        # 这个函数用于验证结束后，计算总体的指标，并进行记录

        epoch_validate_metrics = {}
        for k in self.validate_metrics:
            self.validate_metrics[k] = self.validate_metrics[k] / len(self.trainer.datamodule._val_dataloader())
            epoch_validate_metrics[f'epoch_{k}'] = self.validate_metrics[k]
            print(f'epoch_{k}: {self.validate_metrics[k]}')
        self.log_dict(epoch_validate_metrics, prog_bar=False, logger=True, on_epoch=True)



    def configure_optimizers(self):
        lr = self.learning_rate
        print(f'optimizer lr: {lr}')
        opt_gen = torch.optim.AdamW(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_embed.parameters())+
                                  list(self.post_quant_embed.parameters()),
                                  lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, eps=1e-8)
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, eps=1e-8)

        # if self.trainer.is_global_zero:
        #     print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))

        # step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        # warmup_steps = step_per_epoch * self.warmup_epochs
        # training_steps = step_per_epoch * self.trainer.max_epochs
        # max_decay_steps = training_steps

        # 由于数据在加载的时候会按照概率从多个dataloader中进行采样，因此多卡会独立的读取数据，导致虽然在进行模型训练的时候，batch size会切实的增大，
        # 但是在数据划分的时候，扩大的batchsize不会纳入其中，因此在进行学习率策略调整的时候，需要进行相应的改动，即step_per_epoch=len(dataloader)
        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader())))

        step_per_epoch  = len(self.trainer.datamodule._train_dataloader())
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs
        max_decay_steps = training_steps
        # max_decay_steps = training_steps * 0.7  # TODO ZSJ 注意，这里乘以0.7，是因为single变量可能由于学习率太大训练崩溃了，所以减少训练步数来降低学习率，正常训练不需要乘0.7

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen}, {"optimizer": opt_disc})
    
        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))

        elif self.scheduler_type == "linear-warmup_cosine-decay-bsq":
            multipler_min = self.min_learning_rate / self.learning_rate

            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay_BSQ(warmup_steps=warmup_steps, lr_min=self.lr_min, lr_max=self.lr_max, lr_start=self.lr_start, max_decay_steps=max_decay_steps)),
                "interval": "step",  # 设置为 step 级别更新
                "frequency": 1,      # 每个 step 更新
            }

            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay_BSQ(warmup_steps=warmup_steps, lr_min=self.lr_min, lr_max=self.lr_max, lr_start=self.lr_start, max_decay_steps=max_decay_steps)),
                "interval": "step",
                "frequency": 1,
            }


        else:
            raise NotImplementedError()
        return [
            {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, 
            {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}
            ]

    def get_last_layer(self):
        # return self.decoder.conv_out.weight
        return self.decoder.last_layer_weight

    def visualize_tensor_as_heatmap(self, tensor):
        # 创建一个新的图像
        fig, ax = plt.subplots()

        dict = cmaps.WhiteBlueGreenYellowRed

        # 将 tensor 转换为 numpy 格式
        tensor = tensor.detach().cpu().float().numpy()

        tensor = (tensor - tensor.mean())/tensor.std()

        contour = ax.contourf(tensor, levels=[-5 + 0.2 * x for x in range(50)], cmap=dict, extend='both')

        # 添加 colorbar
        fig.colorbar(contour, ax=ax)

        # 将图像保存到内存中的字节缓冲区
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # 使用 PIL.Image 打开图像
        image = Image.open(buf)

        image = self.img2tensor(image)

        image = image*2 -1  # 把范围从0-1变换到-1到1

        # 关闭当前图像，避免后续绘制时重叠
        plt.close(fig)

        # 关闭缓冲区
        buf.close()

        return image

    # def log_images(self, batch, **kwargs):
    #     log = dict()
    #     x, metadata = batch

    #     x = x[0].unsqueeze(0)  # 1,c,h,w
    #     x = x.to(self.device)
    #     metadata = metadata.to(self.device)
    #     xrec, eloss,  loss_info = self(x, metadata)
    #     # q的值域为>=0，并且大量的值分布在0附近
    #     if self.loss.vari_name == 'q':
    #         xrec = F.relu(xrec)
    #     x = x[0]
    #     xrec = xrec[0]
    #     metadata = metadata[0]

    #     # for i in range(x.shape[0]):
    #     #     log[f'input_channel_{i}'] = self.visualize_tensor_as_heatmap(x[i]).unsqueeze(0)
    #     # for i in range(xrec.shape[0]):
    #     #     log[f'recon_channel_{i}'] = self.visualize_tensor_as_heatmap(xrec[i]).unsqueeze(0)

    #     for i in range(x.shape[0]):
    #         log[f'input_channel_{metadata[i]}'] = self.visualize_tensor_as_heatmap(x[i]).unsqueeze(0)
    #     for i in range(xrec.shape[0]):
    #         log[f'recon_channel_{metadata[i]}'] = self.visualize_tensor_as_heatmap(xrec[i]).unsqueeze(0)

    #     return log

    
    def log_images(self, batch, **kwargs):
        # 这个log_images负责检查在compress过程中，是否使用了正确的模型，即图像的重建效果是否好
        log = dict()
        x, metadata, sample_time_str = batch

        x = x[0].unsqueeze(0)  # 1,c,h,w
        x = x.to(self.device)
        metadata = metadata.to(self.device)
        xrec, eloss,  loss_info = self(x, metadata)
        # q的值域为>=0，并且大量的值分布在0附近
        if self.loss.vari_name == 'q':
            xrec = F.relu(xrec)
        x = x[0]
        xrec = xrec[0]
        metadata = metadata[0]

        # for i in range(x.shape[0]):
        #     log[f'input_channel_{i}'] = self.visualize_tensor_as_heatmap(x[i]).unsqueeze(0)
        # for i in range(xrec.shape[0]):
        #     log[f'recon_channel_{i}'] = self.visualize_tensor_as_heatmap(xrec[i]).unsqueeze(0)

        for i in range(x.shape[0]):
            log[f'input_channel_{metadata[i]}'] = self.visualize_tensor_as_heatmap(x[i]).unsqueeze(0)
        for i in range(xrec.shape[0]):
            log[f'recon_channel_{metadata[i]}'] = self.visualize_tensor_as_heatmap(xrec[i]).unsqueeze(0)

        return log



    # def log_images(self, batch, **kwargs):
    #     # 下面的代码用于检验把一个全国的区域划分成多个区域的时候，它们的排列到底是什么样的
    #     log = dict()
    #     # x = self.get_input(batch, self.image_key)
    #     x = batch[self.image_key]
    #     x_log = x[0]  # c,h,w
    #     for i in range(x_log.shape[0]):
    #         log[f'national_{self.image_key}_channel_{i}'] = self.visualize_tensor_as_heatmap(x_log[i], mode=self.image_key, channle_num=i).unsqueeze(0)
    #     x_multi_region = self.unfold_national_region_data(x)
    #     for region_num in range(x_multi_region.shape[0]):
    #         x_region = x_multi_region[region_num]  # b,c,h,w
    #         x_region = x_region[0].unsqueeze(0)  # 1,c,h,w
    #         x_region = x_region.to(self.device)
    #         x_region = x_region[0]
    #         for i in range(x_region.shape[0]):
    #             log[f'input_{self.image_key}_region_{region_num}_channel_{i}'] = self.visualize_tensor_as_heatmap(x_region[i], mode=self.image_key, channle_num=i).unsqueeze(0)

    #     return log
