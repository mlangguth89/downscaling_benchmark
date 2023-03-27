#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-test.%j
#SBATCH --error=train-test.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

#source ../venv_booster/bin/activate


#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision/0.12.0-CUDA-11.5
module load xarray/0.20.1
module load matplotlib/3.4.3
#ml SciPy-bundle/2021.10
source ../env_setup/venv_hdfml/bin/activate

train_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/train
test_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/test
checkpoint_dir=../results/exp_20230315_unet/32000_G.pth
save_dir=../results/exp_20230315_unet_test/
#model_type=vitSR
#model_type=swinSR
model_type=unet
#model_type=swinUnet
python ../main_scripts/main_test.py --test_dir ${test_dir} --stat_dir ${train_dir} --checkpoint_dir ${checkpoint_dir}  --save_dir ${save_dir} --model_type ${model_type} > vis.output
       #python ../main_scripts/dataset_prep.py
#python ../main_scripts/main_train_precip.py  
