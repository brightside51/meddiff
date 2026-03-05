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

# --------------------------------------------------------------------------------------------

# Functionality Import | Fundamentals
from pathlib import Path
from math import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, dataset
from datetime import datetime

# Functionality Import | Torch
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# --------------------------------------------------------------------------------------------

# Functionality Import | Custom
sys.path.append('/nas-ctm01/homes/pfsousa/data')
from data_parser import data_parser
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff')
from run_parser import run_parser
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff/vqgan')
from vqgan import VQGAN

# Node Definition
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


# ============================================================================================

# DATASET ACCESS

# Dataset Initialisation | Duke Breast Cancer
metabreast_args = data_args = data_parser(dataset = 'metabreast', dataV = 'V0', save = True)
sys.path.append(f"{dukebreast_args.reader_fp}")
from dukebreast_reader import NCDataset as DukebreastDataset
dukebreast_ds = DukebreastDataset(dukebreast_args, mode = 'train')
dukebreast_img = dukebreast_ds.__getitem__(0)
print(dukebreast_img.shape)

# --------------------------------------------------------------------------------------------

# Dataset Initialisation | Metabreast
dukebreast_args = data_parser(dataset = 'dukebreast', dataV = 'V0', save = True)
sys.path.append(f"{metabreast_args.reader_fp}")
print(metabreast_args.reader_fp)
from metabreast_reader import NCDataset as MetabreastDataset
metabreast_ds = MetabreastDataset(metabreast_args, mode = 'train')
metabreast_img = metabreast_ds.__getitem__(0)
print(metabreast_img.shape)

# --------------------------------------------------------------------------------------------

# Dataloader Initialisation
train_ds = ConcatDataset([dukebreast_ds, metabreast_ds])
train_dl = DataLoader(  dataset = train_ds, pin_memory = True,
                        batch_size = data_args.batch_size,
                        num_workers = data_args.num_workers,
                        shuffle = data_args.shuffle,
                        sampler = DistributedSampler(train_ds))

# ============================================================================================

# WandB Setup
wandb.login()
run_args = run_parser(model = 'meddiff', runV = 'V0', save = True)
run_logger = wandb.init(entity = "brightside51", project = run_args.model,
                        name = f"{run_args.runV}_{datetime.now().strftime('%H:%M_%d/%m/%Y')}",
                        config = {  "dataV": data_args.dataV, "runV": run_args.runV,})
#os.environ['WANDB_API_KEY'] = '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'

# --------------------------------------------------------------------------------------------

# VQGAN TRAINING
vqgan_model = VQGAN(data_args, run_args, run_logger)
vqgan_model = DDP(vqgan_model, device_ids = [local_rank])
accelerator = None
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