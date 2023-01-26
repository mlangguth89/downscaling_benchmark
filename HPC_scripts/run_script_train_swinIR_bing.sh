#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-swinIR-out.%j
#SBATCH --error=train-swinIR-err.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
#ml SciPy-bundle/2021.10
source ../env_setup/venv_booster/bin/activate


train_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/train
val_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/val
save_dir=../results/exp_20230123
# save_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/results/swinUnet_exp1017_origin_booster_3years
#train_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/preprocessing/preprocessed_ifs_radklim_full_disk/train
#val_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/preprocessing/preprocessed_ifs_radklim_full_disk/val
#save_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/results/swinSR_exp1110_origin_booster_3years_x2_5x4

epochs=4
#model_type=vitSR
model_type=swinIR
#model_type=unet
#model_type=swinUnet

patch_size=4
window_size=8
upscale_swinIR=4
srun --overlap python ../main_scripts/main_train.py --train_dir ${train_dir} --val_dir ${val_dir} --save_dir ${save_dir} --epochs ${epochs} --model_type ${model_type} --patch_size ${patch_size} --window_size ${window_size} --upscale_swinIR ${upscale_swinIR}
