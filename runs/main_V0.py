# ============================================================================================

# Package Import
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
import torch

# --------------------------------------------------------------------------------------------

# Functionality Import
from pathlib import Path
from math import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms


# Dataset Reader Import
sys.path.append('/nas-ctm01/homes/pfsousa/data')
from data_parser import data_parser
sys.path.append('/nas-ctm01/homes/pfsousa')
from run_parser import run_parser

# ============================================================================================

# DATASET ACCESS

# Dataset Initialisation | Duke Breast Cancer
dukebreast_args = data_parser(dataset = 'dukebreast', dataV = 'V0', save = False)
sys.path.append(f'{dukebreast_args.reader_fp}')
from reader import NCDataset as DukebreastDataset
dukebreast_ds = DukebreastDataset(dukebreast_args, mode = 'train')
dukebreast_img = dukebreast_ds.__getitem__(0)
print(dukebreast_img.shape)

# Dataset Initialisation | Metabreast
metabreast_args = data_parser(dataset = 'metabreast', dataV = 'V0', save = False)
sys.path.append(f'{metabreast_args.reader_fp}')
from reader import NCDataset as MetabreastDataset
metabreast_ds = MetabreastDataset(metabreast_args, mode = 'train')
metabreast_img = metabreast_ds.__getitem__(0)
print(metabreast_img.shape)

# Dataloader Initialisation
#train_ds = ConcatDataset([dukebreast_ds, metabreast_ds])

# ============================================================================================

# VQGAN TRAINING
meddiff_args = run_parser(dataset = 'meddiff', dataV = 'V0', save = True)
