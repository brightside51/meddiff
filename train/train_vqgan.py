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
from pytorch_lightning.callbacks import ModelCheckpoint

# --------------------------------------------------------------------------------------------

# Functionality Import | Custom
sys.path.append('/nas-ctm01/homes/pfsousa/data')
from data_parser import data_parser
from __init__ import get_ds
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff')
from run_parser import run_parser
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff/vqgan')
from vqgan import VQGAN
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff/train')
from callbacks import ImageLogger, VideoLogger

# Node Definition
#def ddp_setup():
#    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
#    init_process_group(backend = "nccl")

# ============================================================================================

# TRAINING SCRIPT
#@hydra.main(config_path = '/nas-ctm01/homes/pfsousa/meddiff/runs', config_name = 'base_cfg', version_base = None)
def train_vqgan(
    #cfg = None
    data_args = None,
    run_args = None,
    run_logger = None
):

    # Node Setup
    pl.seed_everything(run_args.seed)
    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #global_rank = int(os.environ.get("WORLD_SIZE", 1))
    #run_args.device = torch.device("cuda")

    # --------------------------------------------------------------------------------------------

    # Training Dataset Initialisation
    if type(data_args) == list:
        train_ds = []
        for data_arg in data_args:
            train_ds.append(get_ds(data_arg, mode = 'train'))
        #data_args_rep = data_args
        data_args = data_args[0]
        train_ds = ConcatDataset(train_ds)
    else: train_ds = get_ds(data_args, mode = 'train')

    # Training Dataloader Initialisation
    train_dl = DataLoader(  dataset = train_ds, pin_memory = True,
                            batch_size = data_args.batch_size,
                            num_workers = data_args.num_workers,
                            shuffle = data_args.shuffle)
                            #sampler = DistributedSampler(train_ds))
    
    # --------------------------------------------------------------------------------------------

    # VQGAN Initialisation
    vqgan_model = VQGAN(data_args, run_args, run_logger)
    #vqgan_model = DDP(vqgan_model, device_ids = [local_rank])
    if run_args.verbose:
            print(f"\nVQGAN Model | Allocated: {torch.cuda.memory_allocated() / 1024**2} MB")
            print(f"VQGAN Model | Reserved: {torch.cuda.memory_reserved() / 1024**2} MB\n")

    # Model Saving Callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                    save_top_k = 3, mode = 'min', filename = 'latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps = run_args.save_interval,
                    save_top_k = -1, filename='{epoch}-{step}'))
    callbacks.append(ImageLogger(batch_frequency = run_args.log_interval,
                            max_images = run_args.save_img, clamp = True))
    callbacks.append(VideoLogger(batch_frequency = run_args.log_interval,
                            max_videos = run_args.save_img, clamp = True))

    # Checkpoint Loading
    base_dir = os.path.join(run_args.logs_fp, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn),
                              os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                resume_fp = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      resume_fp)

    # Accelerator Definition
    #accel = None
    #if run_args.num_gpu > 1: accel = 'ddp'
    accel = 'cuda' if torch.cuda.is_available() else 'cpu'

    # VQGAN Training
    trainer = pl.Trainer(
        devices = run_args.num_gpu, precision = 32,
        accumulate_grad_batches = run_args.vqgan.grad_accum,
        default_root_dir = f"{run_args.logs_fp}/vqgan",
        #ckpt_path = run_args.vqgan.resume_ckpt if run_args.vqgan.resume else None,
        callbacks = callbacks, enable_progress_bar = True,
        max_steps = run_args.vqgan.num_steps,
        max_epochs = run_args.vqgan.num_epochs,
        #gradient_clip_val = run_args.vqgan.grad_clip,
        accelerator = accel, num_sanity_val_steps = 0,
        strategy = "ddp" if run_args.num_gpu > 1 else "auto",
    )
    trainer.fit(vqgan_model, train_dl, train_dl)
    run_logger.finish()

# ============================================================================================

if __name__ == '__main__':
    train_vqgan()