# ============================================================================================

# Package Import
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
import torch
import wandb
import pytorch_lightning as pl
import hydra
import torch.distributed as dist
import SimpleITK as sitk
#sys.path.append('/usr/local/lib/python3.12/dist-packages')
sys.path.append("/usr/local/miniconda3/lib/python3.13/site-packages")
import cv2
import monai
#import huggingface_hub
#import frd_score
#import clip_mmd

# --------------------------------------------------------------------------------------------

# Functionality Import | Fundamentals
from pathlib import Path
from math import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, dataset
from datetime import datetime
from omegaconf import DictConfig, open_dict, OmegaConf

# Functionality Import | Torch
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# --------------------------------------------------------------------------------------------

# Functionality Import | Custom
sys.path.append('/nas-ctm01/homes/pfsousa/data')
from data_parser import data_parser
from __init__ import get_ds
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff')
from run_parser import run_parser
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff/ddpm')
from diffusion import UNet3D, GaussianDiffusion, Trainer

# Functionality Import | Evaluation
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff/eval/frd_score/src/frd_score')
#import frd as frd_score
#from frd_score import compute_frd
from monai.networks.nets import resnet10
#sys.path.append('/nas-ctm01/homes/pfsousa/meddiff/eval/cmmd')
#from main import compute_cmmd

import torch; print(torch.__version__); print(torch.cuda.is_available())

# ============================================================================================

def cycle(dl):
    while True:
        for data in dl:
            yield data

def extract_fid_features(extractor, volume_batch):
    """
    volume_batch: torch.Tensor of shape (B, C, D, H, W)
    Returns: features of shape (B, feat_dim)
    """
    with torch.no_grad():
        features = extractor(volume_batch)
    return features

# --------------------------------------------------------------------------------------------

def extract_frd_features(extractor, volume_tensor):
    """
    volume_tensor: torch.Tensor of shape (C, D, H, W) or (1, C, D, H, W)
    Returns: numpy array of radiomic features (length depends on settings)
    """
    # Remove batch dimension if present
    if volume_tensor.dim() == 5:
        volume_tensor = volume_tensor[0]          # (C, D, H, W)
    
    # Convert to numpy and ensure correct axis order.
    # The FRD library typically expects (H, W, D, C) or (H, W, D) for grayscale.
    # Since C is usually 1 (grayscale), we can squeeze channel and transpose.
    # Adjust this based on your actual data.
    vol_np = volume_tensor.cpu().numpy()          # (C, D, H, W)
    if vol_np.shape[0] == 1:
        vol_np = vol_np[0]                       # (D, H, W)
    # FRD may expect (H, W, D) – reorder axes as needed
    vol_np = np.transpose(vol_np, (1, 2, 0))     # (H, W, D)
    
    features = extractor.extract(vol_np)
    return features

# ============================================================================================

def infer_ddpm(
    data_args = None,
    run_args = None,
    run_logger = None
):

    # Training Dataset & Dataloader Initialisation
    if type(data_args) == list:
        train_ds = []
        for data_arg in data_args:
            train_ds.append(get_ds(data_arg, mode = 'train'))
        data_args = data_args[0]
        train_ds = ConcatDataset(train_ds)
    else: train_ds = get_ds(data_args, mode = 'train')
    #train_dl = DataLoader(  dataset = train_ds, pin_memory = False,
    #                        batch_size = 1,#data_args.batch_size,
    #                        num_workers = data_args.num_workers,
    #                        shuffle = False)#data_args.shuffle)
    #train_dl = cycle(train_dl)
    
    # --------------------------------------------------------------------------------------------

    # 3D U-Net Backbone Initialisation
    model = UNet3D(dim = data_args.img_size, dim_mult = run_args.ddpm.dim_mult,
                    num_channel = run_args.ddpm.num_channel).to(run_args.device)
    
    # Diffusion Process Initialisation
    diff = GaussianDiffusion(model,
        data_args = data_args, run_args = run_args).to(run_args.device)
    
    # Diffusion Process Trainer Initialisation & Loading
    trainer = Trainer(diff, train_ds, data_args, run_args, run_logger)
    trainer.load(run_args.ddpm.resume_ckpt, map_location = run_args.device)
    trainer.ema_model.eval()

    # --------------------------------------------------------------------------------------------

    # Evaluation Metrics Initialisation
    """
    fid_ext = resnet10(
        spatial_dims = 3,          # B*C*D*H*W
        n_input_channels = 1,
        num_classes = 1000,        # Keep classification layer for now
        pretrained = True,         # This will load weights pre-trained on Med3D (a large 3D medical imaging dataset)[reference:1]
        feed_forward = False,      # Use the full network as a feature extractor
        bias_downsample = False,
    )
    fid_ext = torch.nn.Sequential(*list(fid_ext.children())[:-1])
    fid_ext.to(run_args.device).eval()
    #fid_ext = resnet10_3d(pretrained = True)        # FID Feature Extractor | ResNet10 3D Pretrained on Kinetics-400
    #fid_ext.to(run_args.device).eval()
    real_fid_feats, fake_fid_feats = [], []
    frd_ext = frd_score.get_feature_extractor()
    real_frd_feats, fake_frd_feats = [], []
    """

    # --------------------------------------------------------------------------------------------

    # Dataset Evaluation Loop
    """
    for i_real in range(0, len(train_ds)):
        real_sample = next(train_dl).to(run_args.device)

        # Sample Evaluation | Real FID & FRD
        real_fid_feats.append(extract_fid_features(fid_ext, real_sample).cpu())
        real_frd_feats.append(extract_frd_features(frd_ext, real_sample.squeeze(0)))

        # Sample Evaluation | Metric Logging
        run_logger.log({"real/sample_no": i,
                        "real/fid_mean": real_fid_feats[-1].mean(dim = 0).item(),
                        "real/frd_mean": real_frd_feats[-1].mean(dim = 0).item(),
                        })
    """

    # --------------------------------------------------------------------------------------------
    
    # Sample Inference Loop
    if not Path(run_args.infer.infer_fp).exists(): os.makedirs(run_args.infer.infer_fp)
    for i in range(0, run_args.infer.num_sample):
        with torch.no_grad():

            # Sample Inference
            fake_sample = trainer.ema_model.sample(batch_size = 1).to(run_args.device)
            if run_args.verbose: print(fake_sample.shape)
            if data_args.data_format == 'pt':
                torch.save(fake_sample, f"{run_args.infer.infer_fp}/sample_{i}.pt")
            elif data_args.data_format == 'npy':
                np.save(f"{run_args.infer.infer_fp}/sample_{i}.npy", fake_sample.cpu().numpy())
            else: pass

            """
            # Sample Evaluation | Synthetic FID & FRD
            fake_fid_feats.append(extract_fid_features(fid_ext, fake_sample).cpu())
            print(fake_fid_feats[-1].mean(dim = 0))
            fake_frd_feats.append(extract_frd_features(frd_ext, fake_sample.squeeze(0)))

            # Sample Evaluation | Metric Logging
            run_logger.log({"fake/sample_no": i,
                            "fake/fid_mean": fake_fid_feats[-1].mean(dim = 0).item(),
                            "fake/frd_mean": fake_frd_feats[-1].mean(dim = 0).item(),
                            })
            """
    
    # --------------------------------------------------------------------------------------------

    """
    # Dataset Evaluation | FID
    real_fid_feats = torch.cat(real_fid_feats, dim = 0)
    print(real_fid_feats.shape)
    real_fid_mean = real_fid_feats.mean(dim = 0); real_fid_cov = torch.cov(real_fid_feats.T)
    fake_fid_feats = torch.cat(fake_fid_feats, dim = 0)
    print(fake_fid_feats.shape)
    fake_fid_mean = fake_fid_feats.mean(dim = 0); fake_fid_cov = torch.cov(fake_fid_feats.T)
    fid_loss = monai.metrics.compute_fid(real_fid_mean, real_fid_cov, fake_fid_mean, fake_fid_cov)
    print(fid_loss)

    # Dataset Evaluation | FRD
    real_frd_feats = real_frd_feats.numpy(); fake_frd_feats = fake_frd_feats.numpy()
    real_frd_mean = np.mean(real_frd_feats, axis = 0); real_frd_cov = np.cov(real_frd_feats, rowvar = False)
    fake_frd_mean = np.mean(fake_frd_feats, axis = 0); fake_frd_cov = np.cov(fake_frd_feats, rowvar = False)
    frd_loss = frd_score.calculate_frechet_distance(real_frd_mean, real_frd_cov, fake_frd_mean, fake_frd_cov)
    print(frd_loss)

    # Dataset Evaluation | CMMD
    cmmd_prep = logic.CMMD(data_parallel = True, device = run_args.device)
    real_cmmd_feats = cmmd_prep.extract_features(
        data_path = f"{data_args.data_fp}",
        save_file = f"{run_args.logs_fp}/real_cmmd_feats.pth")
    fake_cmmd_feats = cmmd_prep.extract_features(
        data_path = f"{run_args.infer_fp}",
        save_file = f"{run_args.logs_fp}/fake_cmmd_feats.pth")
    cmmd_loss = cmmd_prep.calculate(real_cmmd_feats, fake_cmmd_feats)
    print(cmmd_loss)

    # Dataset Evaluation | CMMD
    cmmd_loss = compute_cmmd(
        ref_dir = f"{data_args.data_fp}",
        eval_dir = f"{run_args.infer_fp}",
        batch_size = 1, max_count = 1000)

    # Sample Evaluation | Metric Logging
    run_logger.log({"infer/fid_loss": fid_loss.item(),
                    "infer/frd_loss": frd_loss.item(),
                    #"infer/cmmd_loss": cmmd_loss.item(),
                    })
    """

    #MMD
    #Earth Mover's Distance/Wasserstein Distance
    #Kernel Inception Distance
    #NIQE
    #BRISQUE
    #KL Divergence
    #MS-SSIM

    # Dataset Evaluation | Metric Logging
    #run_logger.log({"infer/fid": fid_loss.item(),
    #                        })

# ============================================================================================

if __name__ == '__main__':
    infer_ddpm()