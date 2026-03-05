# Library Imports
import os
import random
import json
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

# Function Imports
from pathlib import Path

# ============================================================================================

def nest_args(flat_dict):
    nested = {}
    for key, value in flat_dict.items():
        if "." in key:
            group, subkey = key.split(".", 1)
            nested.setdefault(group, {})[subkey] = value
        else:
            nested[key] = value
    return nested

def dict_to_namespace(d):
    from argparse import Namespace
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return Namespace(**d)

# --------------------------------------------------------------------------------------------

# Run Arguments Initialisation
def run_parser(
    model: str = 'meddiff',
    runV: str = 'V0',
    save: bool = False,
):  
    
    # Run Fundamentals
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = model)
    parser.add_argument('--runV', type = str, default = runV)
    parser.add_argument('--verbose', type = bool, default = True)
    parser.add_argument('--base_fp', type = str,
                        default = f"/nas-ctm01/homes/pfsousa")
    args = parser.parse_args("")

    # --------------------------------------------------------------------------------------------

    # Load Existing Arguments if Available
    save_fp = Path(f"{args.base_fp}/{args.model}/runs/args_{args.runV}.yaml")
    #save_fp = Path(f"{args.base_fp}/{args.model}/runs/args_{args.runV}.json")
    if save_fp.exists():
        if args.verbose: print(f"Loading ARGUMENT PARSER | {save_fp}")
        with open(save_fp, "r") as f: args = dict_to_namespace(yaml.safe_load(f))
        #with open(save_fp, "r") as f: args = dict_to_namespace(json.load(f))
        #args = argparse.Namespace(args)
    else:

    # ============================================================================================

        # Directory Arguments
        parser.add_argument('--script_fp', type = str,
                            default = f"{args.base_fp}/{args.model}")
        parser.add_argument('--logs_fp', type = str,
                            default = f"{args.base_fp}/{args.model}/logs/run_{args.runV}")
        
        # --------------------------------------------------------------------------------------------

        # Result Logging Arguments 
        parser.add_argument('--num_gpu', type = int, default = 1)
        parser.add_argument('--save_interval', type = int, default = 500)
        parser.add_argument('--log_interval', type = int, default = 1)
        parser.add_argument('--save_img', type = int, default = 2)
        parser.add_argument('--log_method', type = str,
                            choices = {'wandb', 'tensorboard', None},
                            default = 'wandb')
        
        # ============================================================================================

        # Architecture Fundamentals Arguments
        parser.add_argument('--seed', type = int, default = 1234)
        #parser.add_argument('--dim', type = int, default = 64)
        #parser.add_argument('--num_channel', type = int, default = 1)

        # ============================================================================================

        # VQGAN Architecture Arguments | Run Basics
        parser.add_argument('--vqgan.resume', type = bool, default = False)
        parser.add_argument('--vqgan.resume_ckpt', type = str,
            default = f"{args.base_fp}/{args.model}/logs/run_{args.runV}/vqgan/latest.ckpt")
        parser.add_argument('--vqgan.num_steps', type = int, default = -1)
        parser.add_argument('--vqgan.num_epochs', type = int, default = -1)

        # VQGAN Architecture Arguments | Block Info
        parser.add_argument('--vqgan.i3d_feat', type = bool, default = False)
        parser.add_argument('--vqgan.restart_thres', type = float, default = 1.0)
        parser.add_argument('--vqgan.rand_restart', type = bool, default = False)
        parser.add_argument('--vqgan.norm_type', type = str, default = 'group')
        parser.add_argument('--vqgan.pad_type', type = str, default = 'replicate')
        parser.add_argument('--vqgan.num_groups', type = int, default = 32)

        # --------------------------------------------------------------------------------------------

        # VQGAN Architecture Arguments | Arch Basics
        parser.add_argument('--vqgan.dim_latent', type = int, default = 256)
        parser.add_argument('--vqgan.num_codes', type = int, default = 2048)
        parser.add_argument('--vqgan.num_hidden', type = int, default = 240)
        parser.add_argument('--vqgan.downsample', type = list, default = [4, 4, 4])
        parser.add_argument('--vqgan.grad_clip', type = float, default = 1.0)
        parser.add_argument('--vqgan.grad_accum', type = int, default = 1)

        # VQGAN Architecture Arguments | Learning Rate
        parser.add_argument('--vqgan.lr_base', type = float, default = 3e-4)
        parser.add_argument('--vqgan.lr_decay', type = float, default = 0.999)
        parser.add_argument('--vqgan.lr_step', type = int, default = 250)
        parser.add_argument('--vqgan.lr_min', type = float, default = 1e-6)
    
        # --------------------------------------------------------------------------------------------

        # VQGAN Architecture Arguments | Discriminator
        parser.add_argument('--vqgan.disc_channel', type = int, default = 64)
        parser.add_argument('--vqgan.disc_layer', type = int, default = 3)
        parser.add_argument('--vqgan.disc_start', type = int, default = 50000)
        parser.add_argument('--vqgan.disc_loss', type = str,
                            choices = {'hinge', 'lsgan', 'vanilla'},
                            default = 'hinge')

        # VQGAN Architecture Arguments | Loss Functions
        parser.add_argument('--vqgan.img_weight', type = float, default = 1.0)
        parser.add_argument('--vqgan.vid_weight', type = float, default = 1.0)
        parser.add_argument('--vqgan.l1_weight', type = float, default = 4.0)
        parser.add_argument('--vqgan.ganfeat_weight', type = float, default = 1.0)
        parser.add_argument('--vqgan.percept_weight', type = float, default = 0.0)

        # ============================================================================================

        """
        # Training & Diffusion Arguments
        parser.add_argument('--noise_type', type = str, default = 'gaussian')
        parser.add_argument('--num_epochs', type = int, default = 30)
        parser.add_argument('--num_ts', type = int, default = 500)
        parser.add_argument('--num_steps', type = int, default = 150000)
        parser.add_argument('--lr_base', type = float, default = 1e-4)
        parser.add_argument('--lr_decay', type = float, default = 0.999)
        parser.add_argument('--lr_step', type = float, default = 250)
        parser.add_argument('--lr_min', type = float, default = 1e-6)
        """

        # ============================================================================================

        # Argument File Saving
        args = parser.parse_args("")
        if save:
            if args.verbose: print(f"Saving ARGUMENT PARSER | {save_fp}")
            if not save_fp.parent.exists(): os.makedirs(save_fp.parent)
            #with open(save_fp, "w") as f: json.dump(nest_args(vars(args)), f)
            with open(save_fp, "w") as f: yaml.safe_dump(nest_args(vars(args)), f)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    return args

# ============================================================================================

