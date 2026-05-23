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

# --------------------------------------------------------------------------------------------

# Functionality Import | Fundamentals
from pathlib import Path
from math import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, dataset
from datetime import datetime
from omegaconf import DictConfig, open_dict

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

# ============================================================================================

def train_ddpm(
    data_args = None,
    run_args = None,
    run_logger = None
):

    # Training Dataset Initialisation
    if type(data_args) == list:
        train_ds = []
        for data_arg in data_args:
            train_ds.append(get_ds(data_arg, mode = 'train'))
        data_args = data_args[0]
        train_ds = ConcatDataset(train_ds)
    else: train_ds = get_ds(data_args, mode = 'train')

    # 3D U-Net Backbone Initialisation
    model = UNet3D(dim = data_args.img_size, dim_mult = run_args.ddpm.dim_mult,
                    num_channel = run_args.ddpm.num_channel).to(run_args.device)

    # Diffusion Process Initialisation
    diff = GaussianDiffusion(model,
        data_args = data_args, run_args = run_args).to(run_args.device)
    
    # Diffusion Process Trainer Initialisation
    trainer = Trainer(diff, train_ds, data_args, run_args, run_logger)
    if run_args.ddpm.resume: trainer.load(run_args.ddpm.resume_ckpt)
    trainer.train()

# ============================================================================================

if __name__ == '__main__':
    train_ddpm()