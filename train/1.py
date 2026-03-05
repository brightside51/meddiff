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
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff/vqgan')
from vqgan import VQGAN

# Node Definition
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

# ============================================================================================

# TRAINING SCRIPT

@hydra.main(config_path='/nas-ctm01/homes/pfsousa/meddiff/runs',
            config_name='base_cfg', version_base=None)
def train_vqgan(data_args, run_args):

    # Node Setup
    pl.seed_everything(run_args.seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    # --------------------------------------------------------------------------------------------

    # Training Dataset Initialisation
    if data_args is list:
        train_ds = []
        for data_arg in data_args:
            train_ds.append(get_ds(data_arg, mode = 'train'))
        data_args_rep = data_args ; data_args = data_args[0]
        train_ds = ConcatDataset(train_ds)
    else: train_ds = get_ds(data_args, mode = 'train')

    # Training Dataloader Initialisation
    train_dl = DataLoader(  dataset = train_ds, pin_memory = True,
                            batch_size = data_args.batch_size,
                            num_workers = data_args.num_workers,
                            shuffle = data_args.shuffle,
                            sampler = DistributedSampler(train_ds))
    
    # --------------------------------------------------------------------------------------------

    # WandB Setup
    wandb.login()
    run_args = run_parser(model = 'meddiff', runV = 'V0', save = True)
    run_logger = wandb.init(entity = "brightside51", project = run_args.model,
                            name = f"{run_args.runV}_{datetime.now().strftime('%H:%M_%d/%m/%Y')}",
                            config = {  "dataV": data_args.dataV, "runV": run_args.runV,})
    #os.environ['WANDB_API_KEY'] = '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'

    # --------------------------------------------------------------------------------------------

    # VQGAN Initialisation
    vqgan_model = VQGAN(data_args, run_args, run_logger)
    vqgan_model = DDP(vqgan_model, device_ids = [local_rank])
    accelerator = None if run_args.num_gpu <= 1 else 'ddp'

    # VQGAN Training
    trainer = pl.Trainer(
        gpus = run_args.num_gpu, precision = 16,
        accumulate_grad_batches = run_args.vqgan.grad_accum,
        default_root_dir = f"{run_args.logs_fp}/vqgan",
        resume_from_checkpoint = run_args.vqgan.resume_ckpt if run_args.vqgan.resume else None,
        #callbacks = callbacks,
        max_steps = run_args.vqgan.num_steps,
        max_epochs = run_args.vqgan.num_epochs,
        gradient_clip_val = run_args.vqgan.grad_clip,
        accelerator = accelerator,
    )
    trainer.fit(vqgan_model, train_dl)

    run_logger.finish()

if __name__ == '__main__': train_vqgan()