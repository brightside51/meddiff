#!/bin/bash
#
#SBATCH --partition=gpu_min12gb
#SBATCH --output=/nas-ctm01/homes/pfsousa/meddiff/logs/run_V0/output.out
#SBATCH --error=/nas-ctm01/homes/pfsousa/meddiff/logs/run_V0/error.err
#SBATCH --job-name=meddif
#SBATCH --time=1-00:00
#SBATCH --qos=gpu_min12GB

PL_TORCH_DISTRIBUTED_BACKEND=gloo
source activate base
conda init --all
conda activate meddiff
#python -m pip install -r /nas-ctm01/homes/pfsousa/meddiff/requirements.txt
python main_V0.py

#python train/train_vqgan.py dataset=default dataset.root_dir=../../../../nas-ctm01/datasets/private/METABREST/T1W_Breast/video_data model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='own_dataset' model.precision=16 model.embedding_dim=8 model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=8 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384 model.accumulate_grad_batches=1 
#python train/train_ddpm.py model=ddpm dataset=default model.results_folder_postfix='own_dataset' model.vqgan_ckpt=../../../../nas-ctm01/homes/pfsousa/MedDiff/DEFAULT/own_dataset/lightning_logs/version_3562/checkpoints/latest_checkpoint.ckpt model.diffusion_img_size=64 model.diffusion_depth_size=64 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=2 model.gpus=1
