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
import torch.distributed as torch_dist

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
#from torch.distributed import init_process_group, destroy_process_group
torch.set_float32_matmul_precision('high')

# --------------------------------------------------------------------------------------------

# Functionality Import | Custom
sys.path.append('/projects/F202509580CPCAA1/pedrofsousa/data')
from data_parser import data_parser
sys.path.append('/projects/F202509580CPCAA1/pedrofsousa/meddiff')
from run_parser import run_parser
#sys.path.append('/projects/F202509580CPCAA1/pedrofsousa/meddiff/vqgan')
#from vqgan import VQGAN
#sys.path.append('/projects/F202509580CPCAA1/pedrofsousa/meddiff/train')
#from train_vqgan import train_vqgan
#from train_ddpm import train_ddpm
#from infer_ddpm import infer_ddpm

# # ============================================================================================

# Argument Initialisation
dukebreast_args = data_parser(dataset = 'dukebreast', dataV = 'V0_deucalion', save = True)
run_args = run_parser(model = 'meddiff', runV = 'VI_deucalion', save = True)

# DDP Initialisation
torch_dist.init_process_group(backend="nccl")
rank = torch_dist.get_rank()
world_size = torch_dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")

print(f"[Rank {rank}/{world_size}] GPU: {torch.cuda.get_device_name(rank)}")
print(f"[Rank {rank}/{world_size}] CUDA available: {torch.cuda.is_available()}")

# Simple tensor op to verify GPU works
x = torch.randn(1000, 1000).to(device)
y = torch.mm(x, x)
print(f"[Rank {rank}/{world_size}] Tensor op OK: {y.shape}")

# W&B (rank 0 only)
os.environ['WANDB_API_KEY'] = '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'
if rank == 0:
    wandb.login()
    wandb.init(entity = "brightside51", project = run_args.model,
                        name = f"{run_args.runV} ({datetime.now().strftime('%H:%M %d/%m/%Y')})",
                        config = {"dataV": dukebreast_args.dataV, "runV": run_args.runV,})
    for i in range(5):
        wandb.log({"step": i, "loss": 1.0 / (i + 1)})
        print(f"[Rank 0] Logged step {i} to W&B")
    wandb.finish()
    print("W&B offline run saved successfully")

torch_dist.barrier()
torch_dist.destroy_process_group()
