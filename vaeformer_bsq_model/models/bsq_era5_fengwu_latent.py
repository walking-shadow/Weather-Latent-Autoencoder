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
from omegaconf import OmegaConf
import os

from transcoder.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay, Scheduler_LinearWarmup_CosineDecay_BSQ
from transcoder.scheduler.ema import LitEma


import torch
import torch.nn as nn
import torch.nn.functional as F

from transcoder.losses.logit_laplace_loss import LogitLaplaceLoss
from transcoder.models.quantizer.bsq import BinarySphericalQuantizer
from transcoder.models.quantizer.vq import VectorQuantizer
# from transcoder.models.transformer import TransformerDecoder, TransformerEncoder
from transcoder.models.vit_nlc_fengwu_latent import ViT_Encoder, ViT_Decoder

class BSQModel(L.LightningModule):
    def __init__(self,
                # vitconfig,
                # lossconfig,
                # embed_dim,
                # embed_group_size=9,
                # ## Quantize Related
                # l2_norm=False, logit_laplace=False, ckpt_path=None, ignore_keys=[],
                # dvitconfig=None, beta=0., gamma0=1.0, gamma=1.0, zeta=1.0,
                # persample_entropy_compute='group',
                # cb_entropy_compute='group',
                # post_q_l2_norm=False,
                # inv_temperature=1.,
                feature_dim = 1152,
                enc_dim = 192,
                dec_dim = 192,
                inchans_list = [13,13,13,13,13,4],
                outchans_list = [13,13,13,13,13,4],
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
                # 加载额外的autoencoder，用于decode到pixel图像
                autoencoder_config = None
                ):
        super().__init__()

        large_default_dict =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=(1,1), patch_stride=(1,1), patch_padding=(0,0), in_chans=128, out_chans=128, embed_dim=feature_dim, depth=16,
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
            img_size= (72, 144),
            enc_dim=enc_dim,
            inchans_list = inchans_list
        )
        large_default_dict_decoder =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=(1,1), patch_stride=(1,1), patch_padding=(0,0), in_chans=128, out_chans=128, embed_dim=feature_dim, depth=32,
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
            img_size= (72, 144),
            dec_dim=dec_dim,
            outchans_list=outchans_list
        )

        self.latent_h = large_default_dict['img_size'][0] // large_default_dict['patch_stride'][0]
        self.latent_w = large_default_dict['img_size'][1] // large_default_dict['patch_stride'][1]
        self.encoder = ViT_Encoder(**large_default_dict)
        self.decoder = ViT_Decoder(**large_default_dict_decoder)
        self.geospatial_weight = self.create_geospatial_weight_tensor(self.latent_h, self.latent_w)
        self.vari_std = torch.tensor([6.140327314617203, 7.938564815618819, 8.192114701884003, 9.175805942700245, 10.340602688341654, 11.986027099582492, 14.33946056494411, 17.12045982966527, 17.977177813531593, 17.69148389950133, 16.056695212588917, 13.55491464926049, 15.318521072654127, 5.306940844651339, 6.470156667075895, 6.266700191444819, 6.877771523242563, 7.809666612099256, 9.18251663286078, 11.22276718096732, 13.334186836536356, 13.380152330387363, 11.882368756344407, 9.569863315260177, 7.481110148371953, 7.048363250814122, 17.150016898045944, 16.10382314601404, 15.618140256419158, 14.813941559446079, 13.440611169947582, 13.07674741486614, 12.706022174156281, 10.741003486659052, 8.556557941315505, 7.20549807849724, 8.919876336953504, 12.511374710344702, 10.289176307569424, 0.005914419774424922, 0.005073961520515164, 0.004117046343069407, 0.0025571967034993836, 0.0017741358110583878, 0.0010849351306180838, 0.0005117846897917297, 0.0001694515258900334, 7.491947249309529e-05, 2.2839891216096137e-05, 3.8235403884289174e-06, 5.745443423940812e-07, 3.6192002581501286e-07, 1074.9199020806243, 1232.4089579327479, 1474.2017982150333, 2141.4810516984885, 2702.2786981064755, 3360.471981740866, 4158.920994614597, 5101.660075378283, 5547.829936635138, 5833.915395494301, 5842.4527389603, 5537.9530506878555, 5910.056826569052, 4.764288381746439, 5.545988031235574, 21.28296546562788, 1334.8954331832485])
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.in_chans = large_default_dict['in_chans']

        # 获取模型在最初加载的时候的设备和dtype，用于后续把autoencoder放到和数据相同的设备和dtype上
        self.model_device = next(self.encoder.parameters()).device
        self.model_dtype = next(self.encoder.parameters()).dtype

        self.use_ema = use_ema
        if self.use_ema and stage is None: #no need to construct EMA when training Transformer
            self.model_ema = LitEma(self)

        self.resume_lr = resume_lr
        self.lr_drop_epoch = lr_drop_epoch
        self.lr_drop_rate = lr_drop_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        # self.automatic_optimization = False

        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min

        self.strict_loading = False
        self.img2tensor = T.ToTensor()

        self.validate_metrics = {}

        if autoencoder_config is not None:
            self.autoencoder = {}
            for k,v in autoencoder_config.items():
                ae_config = OmegaConf.load(v.config_path)
                autoencoder = instantiate_from_config(ae_config.model)
                model_ckpt = torch.load(v.pretrained_path)['state_dict']
                autoencoder.load_state_dict(model_ckpt)
                autoencoder.eval()
                for name, param in autoencoder.named_parameters():
                    param.requires_grad = False
                self.autoencoder[k] = autoencoder


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
        
    # def init_from_ckpt(self, path, ignore_keys=list(), stage="transformer"):
    #     sd = torch.load(path, map_location="cpu")["state_dict"]
    #     ema_mapping = {}
    #     new_params = OrderedDict()
    #     if stage == "transformer": ### directly use ema encoder and decoder parameter
    #         if self.use_ema:
    #             for k, v in sd.items(): 
    #                 if "encoder" in k:
    #                     if "model_ema" in k:
    #                         k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
    #                         new_k = ema_mapping[k]
    #                         new_params[new_k] = v   
    #                     s_name = k.replace('.', '')
    #                     ema_mapping.update({s_name: k})
    #                     continue
    #                 if "decoder" in k:
    #                     if "model_ema" in k:
    #                         k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
    #                         new_k = ema_mapping[k]
    #                         new_params[new_k] = v 
    #                     s_name = k.replace(".", "")
    #                     ema_mapping.update({s_name: k})
    #                     continue 
    #         else: #also only load the Generator
    #             for k, v in sd.items():
    #                 if "encoder" in k:
    #                     new_params[k] = v
    #                 elif "decoder" in k:
    #                     new_params[k] = v                  
    #     missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False) #first stage
    #     print(f"Restored from {path}")



    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    def forward(self, era5_latent_data, altitude_embed, single_embed, skip_quantize=False):


        h = self.encoder(era5_latent_data, altitude_embed, single_embed)  # b,c,h,w
        dec = self.decoder(h)
        
        return dec

    def on_train_start(self):
        """
        change lr after resuming
        """
        if self.resume_lr is not None:
            opt_gen, opt_disc = self.optimizers()
            for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                opt_gen_param_group["lr"] = self.resume_lr
                opt_disc_param_group["lr"] = self.resume_lr

    def create_geospatial_weight_tensor(self, h, w):
        # 计算由于投影导致的像素大小差距, 对应的weight_tensor
        # 创建一个从90到-90等间距排列的h行1列tensor
        row_values = torch.linspace(90, -90, h)
        # 扩展成h行w列的tensor, 每一行的值都相同
        base_tensor = row_values.unsqueeze(1).repeat(1, w)

        # 计算cos函数
        cos_tensor = torch.cos(torch.deg2rad(base_tensor))  # 将角度转为弧度再计算cos

        # 计算cos_tensor一列值的总和
        sum_value = cos_tensor[:, 0].sum()

        # 计算最终的weight_tensor
        weight_tensor = cos_tensor / sum_value * h

        return weight_tensor

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        # x = self.get_input(batch, self.image_key)

        era5_latent_data, future_era5_latent_data, altitude_embed, single_embed = batch
        # b,c,h,w = era5_latent_data.shape
        # print(f'era5_latent_data:{era5_latent_data}')
        # print(f'future_era5_latent_data:{future_era5_latent_data}')

        era5_tensor_predict = self(era5_latent_data, altitude_embed, single_embed)  # b,l,c*2
        era5_tensor_predict = rearrange(era5_tensor_predict, 'b l (c n) -> b l c n', n=2)

        # 由于要计算交叉熵损失，因此要调节形状
        era5_tensor_predict = era5_tensor_predict.view(-1,2).to(torch.float32)
        future_era5_latent_data = future_era5_latent_data.view(-1).to(torch.long)

        bce_loss = self.criterion(era5_tensor_predict, future_era5_latent_data)

        rmse_log_dict = {'train/bce_loss': bce_loss}
        
        self.log_dict(rmse_log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return bce_loss




    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)

            
    # def _validation_step(self, batch, batch_idx, suffix=""):
    #     era5_latent_data, future_era5_latent_data, altitude_embed, single_embed = batch
    #     # b,c,h,w = era5_latent_data.shape
    #     # print(f'era5_latent_data:{era5_latent_data}')
    #     # print(f'future_era5_latent_data:{future_era5_latent_data}')

    #     era5_tensor_predict = self(era5_latent_data, altitude_embed, single_embed)  # b,l,c*2
    #     era5_tensor_predict = rearrange(era5_tensor_predict, 'b l (c n) -> b l c n', n=2)

    #     # 由于要计算交叉熵损失，因此要调节形状
    #     era5_tensor_predict = era5_tensor_predict.view(-1,2).to(torch.float32)
    #     future_era5_latent_data = future_era5_latent_data.view(-1).to(torch.long)

    #     bce_loss = self.criterion(era5_tensor_predict, future_era5_latent_data)

    #     rmse_log_dict = {'val/bce_loss': bce_loss}
        
    #     self.log_dict(rmse_log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    #     return bce_loss

    def _validation_step(self, batch, batch_idx, suffix=""):
        # ZSJ 这个函数用于计算验证集上pixel的指标

        era5_latent_data, future_era5_latent_data, altitude_embed, single_embed = batch
        b,l,c = era5_latent_data.shape
        data_device, data_dtype = era5_latent_data.device, era5_latent_data.dtype

        if self.model_device != data_device or self.model_dtype != data_dtype:
            for k in self.autoencoder:
                self.autoencoder[k] = self.autoencoder[k].to(device=data_device, dtype=data_dtype)
            self.model_device = data_device
            self.model_dtype = data_dtype
        # print(f'era5_latent_data device, dtype:{data_device}, {data_dtype}')

        era5_tensor_predict = self(era5_latent_data, altitude_embed, single_embed)  # b,l,c

        era5_tensor_predict = rearrange(era5_tensor_predict, 'b l (c n) -> b l c n', n=2)

        # 由于要计算交叉熵损失，因此要调节形状
        era5_tensor_predict_loss = era5_tensor_predict.view(-1,2).to(torch.float32)
        future_era5_latent_data_loss = future_era5_latent_data.view(-1).to(torch.long)


        bce_loss = self.criterion(era5_tensor_predict_loss, future_era5_latent_data_loss)

        rmse_log_dict = {'val/bce_loss': bce_loss}
        
        self.log_dict(rmse_log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        altitude_pressure_level = {
            0: [1000., 925., 850., 700., 500., 200.],
            1: [1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.],
            2: [1000.,  950.,  925.,  900.,  850.,  800.,  700.,  600.,  500.,  400.,  300.,  250.,  200., 
                                                                     150., 100.,  70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,  1.
                                                                    ]
        }
        altitude_metadata = torch.tensor(altitude_pressure_level[int(altitude_embed[0])]).to(device=data_device, dtype=data_dtype).unsqueeze(0).repeat(b,1)  # b,13
        single_metadata = torch.tensor([100,200,500,800]).to(device=data_device, dtype=data_dtype).unsqueeze(0).repeat(b,1)  # b,4，这里固定是4个地表变量

        q_scale = 1./(self.in_chans**0.5)
        vari_name_list = ['u', 'v', 't', 'q', 'z', 'single']  # ZSJ 要求预测的时候固定是这6个变量和这个顺序
        # era5_tensor_predict = rearrange(era5_tensor_predict, 'b (h w) c -> b c h w', h=self.latent_h)
        era5_tensor_predict = torch.argmax(era5_tensor_predict, dim=-1)  # b,l,c
        era5_tensor_predict = (era5_tensor_predict*2-1)*q_scale
        era5_tensor_predict_list = torch.split(era5_tensor_predict, [128]*6, dim=-1)  # 专门为6个变量预测设置的
        era5_tensor_predict_dict = {
            vari_name_list[i]: era5_tensor_predict_list[i] for i in range(len(vari_name_list))
        }
        era5_tensor_predict_pixel_dict = {
            k: self.autoencoder[k].decode(era5_tensor_predict_dict[k], single_metadata) if k == 'single' else self.autoencoder[k].decode(era5_tensor_predict_dict[k], altitude_metadata) for k in vari_name_list
        }


        # future_era5_latent_data = rearrange(future_era5_latent_data, 'b (h w) c -> b c h w', h=self.latent_h)
        future_era5_latent_data = (future_era5_latent_data*2-1)*q_scale
        future_era5_latent_data_list = torch.split(future_era5_latent_data, [128]*6, dim=-1)  # 专门为6个变量预测设置的
        future_era5_latent_data_dict = {
            vari_name_list[i]: future_era5_latent_data_list[i] for i in range(len(vari_name_list))
        }
        future_era5_latent_data_pixel_dict = {
            k: self.autoencoder[k].decode(future_era5_latent_data_dict[k], single_metadata) if k == 'single' else self.autoencoder[k].decode(future_era5_latent_data_dict[k], altitude_metadata) for k in vari_name_list
        }

        rmse_dict = {
            f'{k}_rmse': torch.sqrt(((future_era5_latent_data_pixel_dict[k] - era5_tensor_predict_pixel_dict[k])**2).mean(-1).mean(-1).mean(0)) for k in vari_name_list
        }  # dict, c
        for k in rmse_dict:
            if k in self.validate_metrics:
                self.validate_metrics[k] = self.validate_metrics[k] + rmse_dict[k]
            else:
                self.validate_metrics[k] = rmse_dict[k]

        rmse_dict_log = {
            k: rmse_dict[k].mean() for k in rmse_dict.keys()
        }  # dict, c

        self.log_dict(rmse_dict_log, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # # 尝试直接记录图片
        # self.log_images(batch)
        for k in era5_tensor_predict_pixel_dict:
            tensor = era5_tensor_predict_pixel_dict[k].detach().cpu().numpy()
            np.save(f'predict_batch_{batch_idx}_{k}.npy', tensor)

        for k in future_era5_latent_data_pixel_dict:
            tensor = future_era5_latent_data_pixel_dict[k].detach().cpu().numpy()
            np.save(f'future_batch_{batch_idx}_{k}.npy', tensor)

        return bce_loss


    # def _validation_step(self, batch, batch_idx, suffix=""):
    #     # ZSJ 这个函数用于在验证集上进行10天的未来推理，并保存结果都特定的地方，用于后续计算指标

    #     era5_latent_data, altitude_embed, single_embed, sample_time_str = batch
    #     b,t,l,c = era5_latent_data.shape
    #     assert b == 1, 'b != 1'
    #     assert t == 1, 't != 1'
    #     sample_time_str = sample_time_str[0]

    #     era5_tensor_now = era5_latent_data[:, 0]

    #     # 获取未来10天的预测结果
    #     for i_time in range(40):

    #         era5_tensor_predict = self(era5_tensor_now, altitude_embed, single_embed)  # b,l,c

    #         era5_tensor_predict = rearrange(era5_tensor_predict, 'b l (c n) -> b l c n', n=2)
    #         era5_tensor_predict = torch.argmax(era5_tensor_predict, dim=-1)

    #         era5_tensor_now = era5_tensor_predict

    #         era5_tensor_predict_compress = era5_tensor_predict[0].detach().cpu().numpy()  # l,c
    #         era5_tensor_predict_compress_save = np.packbits(era5_tensor_predict_compress, axis=-1)

    #         file_folder = f'/mnt/hwfile/ai4earth/zhaosijie/compressed_data/era5_fengwu_latent_result/{sample_time_str}/'
    #         os.makedirs(file_folder, exist_ok=True)
            
    #         filename = file_folder + f'time_{i_time}.npy'
    #         np.save(filename, era5_tensor_predict_compress_save)


    # def _validation_step(self, batch, batch_idx, suffix=""):
    #     # ZSJ 这个函数用于计算真值和模型推理保存的结果之间的指标，并且用于计算[13,13,13,13,13,4]变量组合的指标

    #     era5_tensor, era5_latent_data = batch
    #     b,t,l,c = era5_latent_data.shape
    #     assert b == 1, 'b != 1'
    #     data_device, data_dtype = era5_latent_data.device, era5_latent_data.dtype

    #     if self.model_device != data_device or self.model_dtype != data_dtype:
    #         for k in self.autoencoder:
    #             self.autoencoder[k] = self.autoencoder[k].to(device=data_device, dtype=data_dtype)
    #         self.model_device = data_device
    #         self.model_dtype = data_dtype
    #     # print(f'era5_latent_data device, dtype:{data_device}, {data_dtype}')
    #     vari_std = [6.140327314617203, 7.938564815618819, 8.192114701884003, 9.175805942700245, 10.340602688341654, 11.986027099582492, 14.33946056494411, 17.12045982966527, 17.977177813531593, 17.69148389950133, 16.056695212588917, 13.55491464926049, 15.318521072654127, 5.306940844651339, 6.470156667075895, 6.266700191444819, 6.877771523242563, 7.809666612099256, 9.18251663286078, 11.22276718096732, 13.334186836536356, 13.380152330387363, 11.882368756344407, 9.569863315260177, 7.481110148371953, 7.048363250814122, 17.150016898045944, 16.10382314601404, 15.618140256419158, 14.813941559446079, 13.440611169947582, 13.07674741486614, 12.706022174156281, 10.741003486659052, 8.556557941315505, 7.20549807849724, 8.919876336953504, 12.511374710344702, 10.289176307569424, 0.005914419774424922, 0.005073961520515164, 0.004117046343069407, 0.0025571967034993836, 0.0017741358110583878, 0.0010849351306180838, 0.0005117846897917297, 0.0001694515258900334, 7.491947249309529e-05, 2.2839891216096137e-05, 3.8235403884289174e-06, 5.745443423940812e-07, 3.6192002581501286e-07, 1074.9199020806243, 1232.4089579327479, 1474.2017982150333, 2141.4810516984885, 2702.2786981064755, 3360.471981740866, 4158.920994614597, 5101.660075378283, 5547.829936635138, 5833.915395494301, 5842.4527389603, 5537.9530506878555, 5910.056826569052, 4.764288381746439, 5.545988031235574, 21.28296546562788, 1334.8954331832485]
    #     vari_std = torch.tensor(vari_std).to(era5_tensor.device, era5_tensor.dtype)


    #     altitude_metadata = torch.tensor([1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.]).to(device=data_device, dtype=data_dtype).unsqueeze(0).repeat(b,1)  # b,13
    #     single_metadata = torch.tensor([100,200,500,800]).to(device=data_device, dtype=data_dtype).unsqueeze(0).repeat(b,1)  # b,4，这里固定是4个地表变量

    #     q_scale = 1./(self.in_chans**0.5)
    #     vari_name_list = ['u', 'v', 't', 'q', 'z', 'single']  # ZSJ 要求预测的时候固定是这6个变量和这个顺序

    #     # 获取未来10天的预测结果
    #     for i_time in range(t):
    #         era5_tensor_future = era5_tensor[:,i_time]
    #         era5_tensor_future_list = torch.split(era5_tensor_future, [13,13,13,13,13,4], dim=1)  # list, b,c,h,w
    #         future_era5_latent_data_pixel_dict = {
    #             vari_name_list[i]: era5_tensor_future_list[i] for i in range(len(vari_name_list))
    #         }


    #         era5_tensor_predict = era5_latent_data[:,i_time]  # b,l,c

    #         # era5_tensor_predict = rearrange(era5_tensor_predict, 'b (h w) c -> b c h w', h=self.latent_h)
    #         era5_tensor_predict = (torch.sign(era5_tensor_predict)*2-1)*q_scale
    #         era5_tensor_predict_list = torch.split(era5_tensor_predict, [128]*6, dim=-1)  # 专门为6个变量预测设置的
    #         era5_tensor_predict_dict = {
    #             vari_name_list[i]: era5_tensor_predict_list[i] for i in range(len(vari_name_list))
    #         }
    #         era5_tensor_predict_pixel_dict = {
    #             k: self.autoencoder[k].decode(era5_tensor_predict_dict[k], single_metadata) if k == 'single' else self.autoencoder[k].decode(era5_tensor_predict_dict[k], altitude_metadata) for k in vari_name_list
    #         }


    #         rmse_dict = {
    #             f'{k}': torch.sqrt(((future_era5_latent_data_pixel_dict[k] - era5_tensor_predict_pixel_dict[k])**2).mean(-1).mean(-1).mean(0)) for k in vari_name_list
    #         }  # dict, c
    #         rmse_tensor = []
    #         for k in vari_name_list:
    #             rmse_tensor.append(rmse_dict[k])
    #         rmse_tensor = torch.cat(rmse_tensor, dim=0)*vari_std  # c

    #         rmse_dict_log = {
    #         }  # dict, c

    #         for i in range(c):
    #             rmse_dict_log[f'val/time_{i_time+1}_channel_{i}_rmse'] = rmse_tensor[i]
    #             print(f'val/time_{i_time+1}_channel_{i}_rmse: {rmse_tensor[i]}')
            

    #         for k in rmse_dict_log:
    #             if k in self.validate_metrics:
    #                 self.validate_metrics[k] = self.validate_metrics[k] + rmse_dict_log[k]
    #             else:
    #                 self.validate_metrics[k] = rmse_dict_log[k]



    # def on_validation_epoch_end(self):
    #     # 这个函数用于验证结束后, 计算总体的指标, 并进行记录

    #     epoch_validate_metrics = {}
    #     for k in self.validate_metrics:
    #         self.validate_metrics[k] = self.validate_metrics[k] / len(self.trainer.datamodule._val_dataloader()) / self.trainer.world_size
    #         epoch_validate_metrics[f'epoch_{k}'] = self.validate_metrics[k]
    #         print(f'epoch_{k}: {self.validate_metrics[k]}')
    #     self.log_dict(epoch_validate_metrics, prog_bar=False, logger=True, on_epoch=True)



    def configure_optimizers(self):
        lr = self.learning_rate
        print(f'optimizer lr: {lr}')
        opt_gen = torch.optim.AdamW(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, eps=1e-8)



        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))

        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs
        max_decay_steps = training_steps
        # max_decay_steps = training_steps * 0.7  # TODO ZSJ 注意, 这里乘以0.7, 是因为single变量可能由于学习率太大训练崩溃了, 所以减少训练步数来降低学习率, 正常训练不需要乘0.7

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen})
    
        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))

        elif self.scheduler_type == "linear-warmup_cosine-decay-bsq":
            multipler_min = self.min_learning_rate / self.learning_rate

            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay_BSQ(warmup_steps=warmup_steps, lr_min=self.lr_min, lr_max=self.lr_max, lr_start=self.lr_start, max_decay_steps=max_decay_steps)),
                "interval": "step",  # 设置为 step 级别更新
                "frequency": 1,      # 每个 step 更新
            }
        else:
            raise NotImplementedError()
        return [
            {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, 
            ]



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

        # 关闭当前图像, 避免后续绘制时重叠
        plt.close(fig)

        # 关闭缓冲区
        buf.close()

        return image

    # def log_images(self, batch, **kwargs):
    #     log = dict()
    #     era5_tensor_now, era5_tensor_future = batch
    #     b,c,h,w = era5_tensor_now.shape

    #     era5_tensor_now = era5_tensor_now[0].unsqueeze(0)
    #     era5_tensor_future = era5_tensor_future[0].unsqueeze(0)


    #     era5_tensor_predict = self(era5_tensor_now)  # b,c,h,w

    #     era5_tensor_predict = era5_tensor_predict[0]
    #     era5_tensor_future = era5_tensor_future[0]
    #     era5_tensor_predict = torch.cat([era5_tensor_predict[:65:13], era5_tensor_predict[-4:]], dim=0)
    #     era5_tensor_future = torch.cat([era5_tensor_future[:65:13], era5_tensor_future[-4:]], dim=0)


    #     # for i in range(x.shape[0]):
    #     #     log[f'input_channel_{i}'] = self.visualize_tensor_as_heatmap(x[i]).unsqueeze(0)
    #     # for i in range(xrec.shape[0]):
    #     #     log[f'recon_channel_{i}'] = self.visualize_tensor_as_heatmap(xrec[i]).unsqueeze(0)

    #     for i in range(era5_tensor_predict.shape[0]):
    #         log[f'predict_channel_{i}'] = self.visualize_tensor_as_heatmap(era5_tensor_predict[i]).unsqueeze(0)
    #     for i in range(era5_tensor_future.shape[0]):
    #         log[f'future_channel_{i}'] = self.visualize_tensor_as_heatmap(era5_tensor_future[i]).unsqueeze(0)

    #     return log

    def log_images(self, batch, **kwargs):
        # 由于latent空间中难以可视化图片, 所以干脆不进行可视化
        log = dict()

        return log
    
    # def log_images(self, batch, **kwargs):
    #     # 下面的代码用于在计算pixel指标的时候，可视化一些图像
    #     log = dict()
    #     # ZSJ 这个函数用于计算验证集上pixel的指标

    #     era5_latent_data, future_era5_latent_data, altitude_embed, single_embed = batch
    #     era5_latent_data = era5_latent_data[:1]
    #     future_era5_latent_data = future_era5_latent_data[:1]
    #     altitude_embed = altitude_embed[:1]
    #     single_embed = single_embed[:1]

    #     b,l,c = era5_latent_data.shape
    #     data_device, data_dtype = era5_latent_data.device, era5_latent_data.dtype

    #     if self.model_device != data_device or self.model_dtype != data_dtype:
    #         for k in self.autoencoder:
    #             self.autoencoder[k] = self.autoencoder[k].to(device=data_device, dtype=data_dtype)
    #         self.model_device = data_device
    #         self.model_dtype = data_dtype
    #     # print(f'era5_latent_data device, dtype:{data_device}, {data_dtype}')

    #     era5_tensor_predict = self(era5_latent_data, altitude_embed, single_embed)  # b,l,c


    #     altitude_pressure_level = {
    #         0: [1000., 925., 850., 700., 500., 200.],
    #         1: [1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.],
    #         2: [1000.,  950.,  925.,  900.,  850.,  800.,  700.,  600.,  500.,  400.,  300.,  250.,  200., 
    #                                                                  150., 100.,  70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,  1.
    #                                                                 ]
    #     }
    #     altitude_metadata = torch.tensor(altitude_pressure_level[int(altitude_embed[0])]).to(device=data_device, dtype=data_dtype).unsqueeze(0).repeat(b,1)  # b,13
    #     single_metadata = torch.tensor([100,200,500,800]).to(device=data_device, dtype=data_dtype).unsqueeze(0).repeat(b,1)  # b,4，这里固定是4个地表变量

    #     q_scale = 1./(self.in_chans**0.5)
    #     vari_name_list = ['u', 'v', 't', 'q', 'z', 'single']  # ZSJ 要求预测的时候固定是这6个变量和这个顺序
    #     # era5_tensor_predict = rearrange(era5_tensor_predict, 'b (h w) c -> b c h w', h=self.latent_h)
    #     era5_tensor_predict = (torch.sign(era5_tensor_predict)*2-1)*q_scale
    #     era5_tensor_predict_list = torch.split(era5_tensor_predict, [128]*6, dim=-1)  # 专门为6个变量预测设置的
    #     era5_tensor_predict_dict = {
    #         vari_name_list[i]: era5_tensor_predict_list[i] for i in range(len(vari_name_list))
    #     }
    #     era5_tensor_predict_pixel_dict = {
    #         k: self.autoencoder[k].decode(era5_tensor_predict_dict[k], single_metadata) if k == 'single' else self.autoencoder[k].decode(era5_tensor_predict_dict[k], altitude_metadata) for k in vari_name_list
    #     }


    #     # future_era5_latent_data = rearrange(future_era5_latent_data, 'b (h w) c -> b c h w', h=self.latent_h)
    #     future_era5_latent_data = (future_era5_latent_data*2-1)*q_scale
    #     future_era5_latent_data_list = torch.split(future_era5_latent_data, [128]*6, dim=-1)  # 专门为6个变量预测设置的
    #     future_era5_latent_data_dict = {
    #         vari_name_list[i]: future_era5_latent_data_list[i] for i in range(len(vari_name_list))
    #     }
    #     future_era5_latent_data_pixel_dict = {
    #         k: self.autoencoder[k].decode(future_era5_latent_data_dict[k], single_metadata) if k == 'single' else self.autoencoder[k].decode(future_era5_latent_data_dict[k], altitude_metadata) for k in vari_name_list
    #     }

    #     for k in era5_tensor_predict_pixel_dict:
    #         era5_tensor_predict_pixel_tensor = era5_tensor_predict_pixel_dict[k][0]  # c,h,w
    #         for i in range(era5_tensor_predict_pixel_tensor.shape[0]):
    #             log[f'{k}_predict_channel_{i}'] = self.visualize_tensor_as_heatmap(era5_tensor_predict_pixel_tensor[i]).unsqueeze(0)

    #     for k in future_era5_latent_data_pixel_dict:
    #         future_era5_latent_data_pixel_tensor = future_era5_latent_data_pixel_dict[k][0]  # c,h,w
    #         for i in range(future_era5_latent_data_pixel_tensor.shape[0]):
    #             log[f'{k}_future_channel_{i}'] = self.visualize_tensor_as_heatmap(future_era5_latent_data_pixel_tensor[i]).unsqueeze(0)

    #     return log




    # def log_images(self, batch, **kwargs):
    #     # 下面的代码用于检验把一个全国的区域划分成多个区域的时候, 它们的排列到底是什么样的
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
