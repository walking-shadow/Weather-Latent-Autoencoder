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

from transcoder.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay, Scheduler_LinearWarmup_CosineDecay_BSQ
from transcoder.scheduler.ema import LitEma


import torch
import torch.nn as nn
import torch.nn.functional as F

from transcoder.losses.logit_laplace_loss import LogitLaplaceLoss
from transcoder.models.quantizer.bsq import BinarySphericalQuantizer
from transcoder.models.quantizer.vq import VectorQuantizer
from transcoder.models.transformer import TransformerDecoder, TransformerEncoder


class BSQModel(L.LightningModule):
    def __init__(self,
                vitconfig,
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
        self.encoder = TransformerEncoder(**vitconfig)
        dvitconfig = vitconfig if dvitconfig is None else dvitconfig
        self.decoder = TransformerDecoder(**dvitconfig, logit_laplace=logit_laplace)
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
        self.quant_embed = nn.Linear(in_features=vitconfig['width'], out_features=embed_dim)
        self.post_quant_embed = nn.Linear(in_features=embed_dim, out_features=dvitconfig['width'])
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
        x = F.interpolate(x, (720,1440), mode='bilinear')

        h = self.encoder(x, metadata)
        h = self.quant_embed(h)
        if self.l2_norm:
            h = F.normalize(h, dim=-1)
        if skip_quantize:
            assert not self.training, 'skip_quantize should be used in eval mode only.'
            return h, {}, {}
        quant, loss, info = self.quantize(h)
        return quant, loss, info

    def decode(self, quant, metadata):
        h = self.post_quant_embed(quant)
        x = self.decoder(h, metadata)
        x = F.interpolate(x, (721,1440), mode='bilinear')

        return x


    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

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

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        # x = self.get_input(batch, self.image_key)

        x, metadata = batch

        xrec, eloss,  loss_info = self(x, metadata)

        opt_gen, opt_disc = self.optimizers()
        scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        

        # optimize generator
        aeloss, log_dict_ae = self.loss(eloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)
        opt_gen.zero_grad()
        self.manual_backward(aeloss)
        opt_gen.step()
        scheduler_gen.step()
        

        # optimize discriminator
        discloss, log_dict_disc = self.loss(eloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()
        scheduler_disc.step()

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    
    # def training_step(self, batch, batch_idx):
    #     # 这个training_step是为了多卡压缩数据，真正的训练函数在上面

    #     with torch.inference_mode(mode=True):
    #         x = batch[self.image_key]
    #         time = batch['time']
    #         batch_num = x.shape[0]
    #         for b_i in range(batch_num):
    #             x_tensor = x[b_i].unsqueeze(0)
    #             time_tensor = time[b_i]
    #             x_multi_region = self.unfold_national_region_data(x_tensor)
    #             for region_num in range(x_multi_region.shape[0]):
    #                 x_region = x_multi_region[region_num]  # 1,c,h,w
    #                 file_path = f'/mnt/hwfile/ai4earth/zhaosijie/compressed_data/weather_china_nation_1km_lfq/train/weather_bin/region_{region_num}_{time_tensor}.bin'
    #                 quant, diff, indices, loss_break = self.encode(x_region)
                    
    #                 quant = quant > 0
    #                 # print(f'quant shape:{quant.shape}, dtype: {quant.dtype}')
    #                 quant = np.array(quant.squeeze(0).detach().cpu())  # c,h,w
    #                 # print(f'quant array shape:{quant.shape}, dtype: {quant.dtype}')
    #                 # 压缩布尔值到位级别
    #                 compressed_quant = np.packbits(quant)

    #                 # 保存压缩后的文件
    #                 compressed_quant.tofile(file_path)
    #     self.log_dict({'save_num':1}, prog_bar=False, logger=True, on_step=True, on_epoch=True)

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

    # def validation_step(self, batch, batch_idx): 
    #     # 这个validation_step是为了多卡压缩数据，真正的训练函数在上面

    #     if self.use_ema:
    #         with self.ema_scope():
    #             with torch.inference_mode(mode=True):
    #                 x = batch[self.image_key]
    #                 time = batch['time']
    #                 batch_num = x.shape[0]
    #                 for b_i in range(batch_num):
    #                     x_tensor = x[b_i].unsqueeze(0)
    #                     time_tensor = time[b_i]
    #                     x_multi_region = self.unfold_national_region_data(x_tensor)
    #                     for region_num in range(x_multi_region.shape[0]):
    #                         x_region = x_multi_region[region_num]  # 1,c,h,w
    #                         file_path = f'/mnt/hwfile/ai4earth/zhaosijie/compressed_data/weather_china_nation_1km_lfq/val/weather_bin/region_{region_num}_{time_tensor}.bin'
    #                         quant, diff, indices, loss_break = self.encode(x_region)
                            
    #                         quant = quant > 0
    #                         # print(f'quant shape:{quant.shape}, dtype: {quant.dtype}')
    #                         quant = np.array(quant.squeeze(0).detach().cpu())  # c,h,w
    #                         # print(f'quant array shape:{quant.shape}, dtype: {quant.dtype}')
    #                         # 压缩布尔值到位级别
    #                         compressed_quant = np.packbits(quant)

    #                         # 保存压缩后的文件
    #                         compressed_quant.tofile(file_path)
    #     else:
    #         log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):

        x, metadata = batch

        xrec, eloss,  loss_info = self(x, metadata)

        # log_rmse_metric = self.cal_rmse_metrics(x, x_rec)
        # self.log_dict(log_rmse_metric, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss, log_dict_ae = self.loss(eloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)
        discloss, log_dict_disc = self.loss(eloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), cond=None, split="train", metadata=metadata)


        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict

    # ZSJ TODO 注意还需要修改optimizer
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_gen = torch.optim.AdamW(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_embed.parameters())+
                                  list(self.post_quant_embed.parameters()),
                                  lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, eps=1e-8)
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, eps=1e-8)

        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))

        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs
        max_decay_steps = training_steps

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

            # scheduler_disc = {
            #     "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay_BSQ(warmup_steps=warmup_steps, lr_min=self.lr_min, lr_max=self.lr_max, lr_start=self.lr_start, max_decay_steps=max_decay_steps)),
            #     "interval": "step",
            #     "frequency": 1,
            # }

            # ZSJ 好像出现了鉴别器过强的问题，尝试调低鉴别器的学习率，看看能不能控制这个问题
            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay_BSQ(warmup_steps=warmup_steps, lr_min=self.lr_min*0.5, lr_max=self.lr_max*0.5, lr_start=self.lr_start, max_decay_steps=max_decay_steps)),
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

    def log_images(self, batch, **kwargs):
        log = dict()
        x, metadata = batch

        x = x[0].unsqueeze(0)  # 1,c,h,w
        x = x.to(self.device)
        metadata = metadata.to(self.device)
        xrec, eloss,  loss_info = self(x, metadata)
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
    #     # 这个log_images负责检查在compress过程中，压缩之后的文件的重建效果如何
    #     log = dict()
    #     x, metadata = batch

    #     x = x[0].unsqueeze(0)  # 1,c,h,w
    #     x = x.to(self.device)
    #     quant, diff, indices, loss_break = self.encode(x)

    #     # print(f'indices shape:{indices.shape}, indices min max:{indices.min()},{indices.max()}')
    #     quant = quant > 0

    #     # print(f'quant shape:{quant.shape}, dtype: {quant.dtype}')
    #     quant = np.array(quant.squeeze(0).detach().cpu())  # c,h,w
    #     C,H,W = quant.shape
    #     # print(f'quant array shape:{quant.shape}, dtype: {quant.dtype}')
    #     # 压缩布尔值到位级别
    #     compressed_quant = np.packbits(quant)

    #     decompressed_array = np.unpackbits(compressed_quant).astype(bool)

    #     # 调整形状为原始形状
    #     decompressed_array = decompressed_array.reshape((C,H,W))

    #     print(np.array_equal(quant, decompressed_array))  # 验证解压结果

    #     decompressed_array = torch.tensor(decompressed_array).unsqueeze(0).to(device=x.device)  # 1,c,h,w
        
    #     # codebook_value = torch.Tensor([1.0]).to(device=decompressed_array.device, dtype=decompressed_array.dtype)
    #     decompressed_array = torch.where(decompressed_array, 1, -1) # higher than 0 filled 
    #     decompressed_array = decompressed_array.to(dtype=x.dtype)

    #     xrec = self.decode(decompressed_array)

    #     x = x[0]
    #     xrec = xrec[0]
    #     for i in range(x.shape[0]):
    #         log[f'input_{self.image_key}_channel_{i}'] = self.visualize_tensor_as_heatmap(x[i], mode=self.image_key, channle_num=i).unsqueeze(0)
    #     for i in range(xrec.shape[0]):
    #         log[f'recon_{self.image_key}_channel_{i}'] = self.visualize_tensor_as_heatmap(xrec[i], mode=self.image_key, channle_num=i).unsqueeze(0)

    #     return log



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
