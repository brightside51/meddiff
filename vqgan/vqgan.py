# ============================================================================================

# Package Import
import sys
import math
import argparse
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl

# --------------------------------------------------------------------------------------------

# Functionality Import | Fundamentals
from utils import shift_dim, adopt_weight, comp_getattr
from lpips import LPIPS
from codebook import Codebook

# ============================================================================================
# BUILDING FUNCTIONS
# ============================================================================================

def silu(x):
    return x*torch.sigmoid(x)

def Normalise(in_channel, norm_type = 'group', num_groups = 32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups = num_groups,
            num_channels = in_channel, eps = 1e-6, affine = True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channel)
    
# --------------------------------------------------------------------------------------------

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

# ============================================================================================
# BUILDING BLOCKS
# ============================================================================================

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel = None,
        conv_shortcut = False,
        kernel_size = 3,
        dropout = 0.0,
        norm_type = 'group',
        padding_type = 'replicate',
        num_groups = 32
    ):
        super().__init__()
        self.in_channel = in_channel
        out_channel = in_channel if out_channel is None else out_channel
        self.out_channel = out_channel
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalise(in_channel, norm_type, num_groups = num_groups)
        self.conv1 = SamePadConv3d(in_channel, out_channel,
            kernel_size = kernel_size, padding_type = padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalise(in_channel, norm_type, num_groups = num_groups)
        self.conv2 = SamePadConv3d(out_channel, out_channel,
            kernel_size = kernel_size, padding_type = padding_type)
        if self.in_channel != self.out_channel:
            self.conv_shortcut = SamePadConv3d(in_channel, out_channel,
                kernel_size = kernel_size, padding_type = padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channel != self.out_channel:
            x = self.conv_shortcut(x)

        return x + h
    
# --------------------------------------------------------------------------------------------

class SamePadConv3d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride = 1,
        bias = True,
        padding_type = 'replicate'
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size,
                            stride = stride, padding = 0, bias = bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode = self.padding_type))

# --------------------------------------------------------------------------------------------

class SamePadConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride = 1,
        bias = True,
        padding_type = 'replicate'
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type
        self.convt = nn.ConvTranspose3d(in_channel, out_channel, kernel_size,
                                        stride = stride, bias = bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode = self.padding_type))

# --------------------------------------------------------------------------------------------

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _

# --------------------------------------------------------------------------------------------

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _

# ============================================================================================
# VQGAN MODEL COMPONENTS
# ============================================================================================

# Encoder Class
class Encoder(nn.Module):
    def __init__(
        self,
        num_hidden,
        downsample,
        img_channel = 3,
        norm_type = 'group',
        padding_type = 'replicate',
        num_groups = 32
    ):

        # Encoder Architecture | Initial Convolution
        super().__init__()
        num_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.block_list = nn.ModuleList()
        self.layer_in = SamePadConv3d(  img_channel, num_hidden,
                                        kernel_size = 3, padding_type = padding_type)

        # Encoder Architecture | Downsampling Blocks
        for i in range(num_downsample.max()):
            block = nn.Module()
            in_channel = num_hidden * 2 ** i
            out_channel = num_hidden * 2 ** (i + 1)
            stride = tuple([2 if d > 0 else 1 for d in num_downsample])
            block.down = SamePadConv3d( in_channel, out_channel, 4,
                                        stride = stride, padding_type = padding_type)
            block.res = ResBlock(       out_channel, out_channel,
                                        norm_type=norm_type, num_groups=num_groups)
            self.block_list.append(block)
            num_downsample -= 1
        
        # Encoder Architecture | Final Activation
        self.layer_out = nn.Sequential(Normalise(out_channel, norm_type, num_groups), nn.SiLU())
        self.out_channel = out_channel

    # --------------------------------------------------------------------------------------------

    # Encoder Forward Pass
    def forward(self, x):
        h = self.layer_in(x)
        for block in self.block_list:
            h = block.down(h)
            h = block.res(h)
        h = self.layer_out(h)
        return h

# ============================================================================================

# Decoder Class
class Decoder(nn.Module):
    def __init__(
        self,
        num_hidden,
        upsample,
        img_channel = 1,
        norm_type = 'group',
        num_groups = 32
    ):

        # Decoder Architecture | Initial Activation
        super().__init__()
        num_upsample = np.array([int(math.log2(u)) for u in upsample])
        self.block_list = nn.ModuleList(); in_channel = num_hidden * 2 ** num_upsample.max()
        self.layer_in = nn.Sequential(Normalise(in_channel, norm_type, num_groups), nn.SiLU())

        # Decoder Architecture | Upsampling Blocks
        for i in range(num_upsample.max()):
            block = nn.Module()
            in_channel = in_channel if i == 0 else num_hidden * 2 ** (num_upsample.max() - i + 1)
            out_channel = num_hidden * 2 ** (num_upsample.max() - i)
            stride = tuple([2 if u > 0 else 1 for u in num_upsample])
            block.up = SamePadConvTranspose3d(  in_channel, out_channel, 4, stride = stride)
            block.res1 = ResBlock(  out_channel, out_channel,
                                    norm_type = norm_type, num_groups = num_groups)
            block.res2 = ResBlock(  out_channel, out_channel,
                                    norm_type = norm_type, num_groups = num_groups)
            self.block_list.append(block)
            num_upsample -= 1
        
        # Decoder Architecture | Final Convolution
        self.layer_out = SamePadConv3d( out_channel, img_channel, kernel_size = 3)
    
    # --------------------------------------------------------------------------------------------

    # Decoder Forward Pass
    def forward(self, x):
        h = self.layer_in(x)
        for block in self.block_list:
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.layer_out(h)
        return h

# ============================================================================================
# VQGAN MODEL
# ============================================================================================

# VQGAN Class
class VQGAN(pl.LightningModule):

    # Constructor / Initialiser
    def __init__(
        self,
        data_args,
        run_args,
        run_logger = None
    ):
        super().__init__()
        self.save_hyperparameters({
            "data": vars(data_args),
            "run": vars(run_args)},
            ignore = ["run_logger"])
        self.data_args = data_args
        self.run_args = run_args
        self.run_logger = run_logger
        self.automatic_optimization = False
        self.optimizer_idx = 1

        # Encoder Initialisation
        self.encoder = Encoder(
            num_hidden = self.run_args.vqgan.num_hidden,
            downsample = self.run_args.vqgan.downsample,
            img_channel = self.data_args.img_channel,
            norm_type = self.run_args.vqgan.norm_type,
            padding_type = self.run_args.vqgan.pad_type,
            num_groups = self.run_args.vqgan.num_groups)
        
        # Decoder Initialisation
        self.decoder = Decoder(
            num_hidden = self.run_args.vqgan.num_hidden,
            upsample = self.run_args.vqgan.downsample,
            img_channel = self.data_args.img_channel,
            norm_type = self.run_args.vqgan.norm_type,
            num_groups = self.run_args.vqgan.num_groups)
        
        # Additional Block Initialisation
        self.encoder_out_channel = self.encoder.out_channel
        self.pre_conv = SamePadConv3d(
            self.encoder_out_channel,
            self.run_args.vqgan.dim_latent, 1,
            padding_type = self.run_args.vqgan.pad_type)
        self.post_conv = SamePadConv3d(
            self.run_args.vqgan.dim_latent,
            self.encoder_out_channel, 1)
        
        # Codebook Initialisation
        self.codebook = Codebook(   self.run_args.vqgan.num_codes,
                                    self.run_args.vqgan.dim_latent,
                                    no_random_restart = self.run_args.vqgan.rand_restart,
                                    restart_thres = self.run_args.vqgan.restart_thres)
        
        # Discriminator Initialisation
        self.img_disc = NLayerDiscriminator(    self.data_args.img_channel,
                                                self.run_args.vqgan.disc_channel,
                                                self.run_args.vqgan.disc_layer,
                                                norm_layer = nn.BatchNorm2d)
        self.vid_disc = NLayerDiscriminator3D(  self.data_args.img_channel,
                                                self.run_args.vqgan.disc_channel,
                                                self.run_args.vqgan.disc_layer,
                                                norm_layer = nn.BatchNorm3d)

        # Loss Function Initialisation
        if self.run_args.vqgan.disc_loss == 'vanilla': self.disc_loss = vanilla_d_loss
        elif self.run_args.vqgan.disc_loss == 'hinge': self.disc_loss = hinge_d_loss
        self.percept_model = LPIPS().eval()
        
    # --------------------------------------------------------------------------------------------

    # Encoding Functionality
    def encode(
        self, x,
        include_embeddings = False,
        quantise = True
    ):
        
        h = self.pre_conv(self.encoder(x))
        if quantise:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h
    
    # Decoding Functionality
    def decode(
        self,
        latent,
        quantise = False
    ):
        
        if quantise:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_conv(shift_dim(h, -1, 1))
        return self.decoder(h)
    
    # ============================================================================================

    # Forward Pass
    def forward(self, x, optimiser_idx = None):
        
        # Forward Pass
        B, C, T, H, W = x.shape
        z = self.pre_conv(self.encoder(x))
        vq_out = self.codebook(z)
        x_recon = self.decoder(self.post_conv(vq_out['embeddings']))
        if self.run_args.verbose: print(f"VQGAN | Forward Pass | x: {x.shape} | z: {z.shape} | vq_out: {vq_out['embeddings'].shape} | x_recon: {x_recon.shape}")

        # Loss Calculation | Reconstruction Loss
        recon_loss = F.l1_loss(x_recon, x) * self.run_args.vqgan.l1_weight
        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        slice_real = torch.gather(x, 2, frame_idx).squeeze(2)
        slice_recon = torch.gather(x_recon, 2, frame_idx).squeeze(2)
        print(f"Real Slice: {slice_real.shape}")
        print(f"Recon Slice: {slice_recon.shape}")

        # --------------------------------------------------------------------------------------------

        # Generator Training
        if optimiser_idx == 0:

            # Loss Calculation | SSIM Indexes
            ssim_loss = SSIM(x_recon, x)
            msssim_loss = MS_SSIM(x_recon, x)

            # Loss Calculation | Perceptual LPIPS
            percept_loss = 0
            if self.run_args.vqgan.percept_weight > 0:
                percept_loss = self.percept_model (slice_real, slice_recon).mean() * self.run_args.vqgan.percept_weight

            # Loss Calculation | CMMD

            # Loss Calculation | Precision & Rec

            # Loss Calculation | Medical FID

            # Loss Calculation | Discriminator Loss
            logits_img_fake, pred_img_fake = self.img_disc(slice_recon)
            logits_vid_fake, pred_vid_fake = self.vid_disc(x_recon)
            gen_img_loss = -torch.mean(logits_img_fake)
            gen_vid_loss = -torch.mean(logits_vid_fake)
            gen_loss = (gen_img_loss * self.run_args.vqgan.img_weight) + (gen_vid_loss * self.run_args.vqgan.vid_weight)
            disc_factor = adopt_weight(self.global_step, threshold = self.run_args.vqgan.disc_start)
            ae_loss = disc_factor * gen_loss

            # Loss Calculation | GAN Feature Matching
            ganfeat_img_loss, ganfeat_vid_loss = 0, 0
            if self.run_args.vqgan.img_weight > 0:
                logits_img_real, pred_img_real = self.img_disc(slice_real); del logits_img_real
                for i in range(len(pred_img_fake) - 1):
                    ganfeat_img_loss += F.l1_loss(pred_img_fake[i], pred_img_real[i].detach()) * self.run_args.vqgan.img_weight
            if self.run_args.vqgan.vid_weight > 0:
                logits_vid_real, pred_vid_real = self.vid_disc(x); del logits_vid_real
                for i in range(len(pred_vid_fake) - 1):
                    ganfeat_vid_loss += F.l1_loss(pred_vid_fake[i], pred_vid_real[i].detach()) * self.run_args.vqgan.vid_weight
            ganfeat_loss = disc_factor * self.run_args.vqgan.ganfeat_weight * (ganfeat_img_loss + ganfeat_vid_loss)

            # Value Logging | WandB
            if self.run_args.log_method == 'wandb' and self.run_logger is not None:
                self.run_logger.log({   "train/vqgan/gen/step": self.global_step,
                                        "train/vqgan/gen/recon_loss": recon_loss.item(),
                                        "train/vqgan/gen/ssim_loss": ssim_loss.item(),
                                        "train/vqgan/gen/msssim_loss": msssim_loss.item(),
                                        "train/vqgan/gen/percept_loss": percept_loss.item(),
                                        "train/vqgan/gen/gen_loss": gen_loss.item(),
                                        "train/vqgan/gen/ganfeat_loss": ganfeat_loss.item(),
                                        "train/vqgan/gen/ae_loss": ae_loss.item()})

            # Value Logging | Tensorboard
            #else:
                
            return recon_loss, x_recon, vq_out, ae_loss, percept_loss, ganfeat_loss
                
        # --------------------------------------------------------------------------------------------

        # Discriminator Training
        if optimiser_idx == 1:

            # Loss Calculation | Discriminator Loss
            logits_img_real, pred_img_real = self.img_disc(slice_real.detach())
            logits_vid_real, pred_vid_real = self.vid_disc(x.detach())
            logits_img_fake, pred_img_fake = self.img_disc(slice_recon.detach())
            logits_vid_fake, pred_vid_fake = self.vid_disc(x_recon.detach())
            del pred_img_real, pred_vid_real, pred_img_fake, pred_vid_fake

            # 
            disc_img_loss = self.disc_loss(logits_img_real, logits_img_fake)
            disc_vid_loss = self.disc_loss(logits_vid_real, logits_vid_fake)
            disc_factor = adopt_weight(self.global_step, threshold = self.run_args.vqgan.disc_start)
            disc_loss = disc_factor *  ((disc_img_loss * self.run_args.vqgan.img_weight) + \
                                        (disc_vid_loss * self.run_args.vqgan.vid_weight))

            # Value Logging | WandB
            if self.run_args.log_method == 'wandb' and self.run_logger is not None:
                self.run_logger.log({   "train/vqgan/disc/step": self.global_step,
                                        "train/vqgan/disc/logits_img_real": logits_img_real.mean().detach().item(),
                                        "train/vqgan/disc/logits_vid_real": logits_vid_real.mean().detach().item(),
                                        "train/vqgan/disc/logits_img_fake": logits_img_fake.mean().detach().item(),
                                        "train/vqgan/disc/logits_vid_fake": logits_vid_fake.mean().detach().item(),
                                        "train/vqgan/disc/disc_img_loss": disc_img_loss.item(),
                                        "train/vqgan/disc/disc_vid_loss": disc_vid_loss.item(),
                                        "train/vqgan/disc/disc_loss": disc_loss.item(),
                                        "train/vqgan/disc/disc_loss": disc_loss.item()})

            # Value Logging | Tensorboard
            #else:

            return disc_loss

        percept_loss = self.percept_model (slice_real, slice_recon) * self.run_args.vqgan.percept_weight
        return recon_loss, x_recon, vq_out, percept_loss

    # ============================================================================================

    # Training Setup
    def training_step(self, batch, batch_idx):

        # Data Extraction
        print(self.device)
        x = batch.to(self.device)
        gen_opt, disc_opt = self.optimizers()

        # Generator Training
        if self.optimizer_idx == 0:
            recon_loss, x_recon, vq_out, ae_loss, percept_loss, ganfeat_loss = self.forward(x, optimiser_idx = 0)
            commit_loss = vq_out['commitment_loss']
            loss = recon_loss + commit_loss + ae_loss + percept_loss + ganfeat_loss
            gen_opt.zero_grad(); self.manual_backward(loss); gen_opt.step()

            # Value Logging | WandB
            if self.run_args.log_method == 'wandb' and self.run_logger is not None:
                self.run_logger.log({   "train/vqgan/step": self.global_step,
                                        "train/vqgan/gen/commit_loss": commit_loss.item(),
                                        "train/vqgan/gen/loss": loss.item()})
        
        # Discriminator Training
        if self.optimizer_idx == 1:
            disc_loss = self.forward(x, optimiser_idx = 1)
            loss = disc_loss
            disc_opt.zero_grad(); self.manual_backward(loss); disc_opt.step()
        return loss
    
    # --------------------------------------------------------------------------------------------

    # Validation Setup
    def validation_step(self, batch, batch_idx):
        
        # Data Extraction
        x = batch
        recon_loss, x_recon, vq_out, percept_loss = self.forward(x)        

        # Value Logging | WandB
        if self.run_args.log_method == 'wandb' and self.run_logger is not None:
            self.run_logger.log({   "val/vqgan/step": self.global_step,
                                    "val/vqgan/recon_loss": recon_loss.item(),
                                    "val/vqgan/percept_loss": percept_loss.item(),
                                    "val/vqgan/commit_loss": vq_out['commitment_loss'].item(),
                                    "val/vqgan/perplexity": vq_out['perplexity'].item()})
        
        # Value Logging | Tensorboard
        #else:

        return recon_loss
        
    # --------------------------------------------------------------------------------------------

    # Optimiser Configuration
    def configure_optimizers(self):
        lr = self.run_args.vqgan.lr_base
        gen_opt = torch.optim.Adam( list(self.encoder.parameters()) +
                                    list(self.decoder.parameters()) +
                                    list(self.pre_conv.parameters()) +
                                    list(self.post_conv.parameters()) +
                                    list(self.codebook.parameters()),
                                    lr = lr, betas = (0.5, 0.9))
        disc_opt = torch.optim.Adam(list(self.img_disc.parameters()) +
                                    list(self.vid_disc.parameters()),
                                    lr = lr, betas = (0.5, 0.9))
        return [gen_opt, disc_opt], []
