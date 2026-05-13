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
torch.set_float32_matmul_precision('high')

# --------------------------------------------------------------------------------------------

# Functionality Import | Custom
sys.path.append('/nas-ctm01/homes/pfsousa/data')
from data_parser import data_parser
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff')
from run_parser import run_parser
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff/vqgan')
from vqgan import VQGAN
sys.path.append('/nas-ctm01/homes/pfsousa/meddiff/train')
from train_vqgan import train_vqgan
from train_ddpm import train_ddpm

# # ============================================================================================

# Argument Initialisation
metabreast_args = data_parser(dataset = 'metabreast', dataV = 'V0', save = True)
dukebreast_args = data_parser(dataset = 'dukebreast', dataV = 'V0', save = True)
run_args = run_parser(model = 'meddiff', runV = 'V0', save = True)
print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list())

# WandB Setup
wandb.login()
run_logger = wandb.init(entity = "brightside51", project = run_args.model,
                        name = f"{run_args.runV} ({datetime.now().strftime('%H:%M %d/%m/%Y')})",
                        config = {"dataV": metabreast_args.dataV, "runV": run_args.runV,})
#os.environ['WANDB_API_KEY'] = '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'

# --------------------------------------------------------------------------------------------

# VQGAN Training Script
#train_vqgan([metabreast_args, dukebreast_args], run_args, run_logger)

# DDPM Training Script
train_ddpm([metabreast_args, dukebreast_args], run_args, run_logger)
