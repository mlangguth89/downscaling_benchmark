#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-out.%j
#SBATCH --error=train-err.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

#source ../venv_booster/bin/activate

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
#ml SciPy-bundle/2021.10
source ../venv_booster/bin/activate

#train_dir=/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train
test_dir=/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/test
checkpoint_dir=../results/exp_test/1800_G.pth
save_dir=../results/exp_vis
#model_type=vitSR
#model_type=swinSR
model_type=unet
python ../main_scripts/main_test.py --test_dir ${test_dir} --checkpoint_dir ${checkpoint_dir}  --save_dir ${save_dir} / 
       --model_type ${model_type} > vis.output
       #python ../main_scripts/dataset_prep.py
#python ../main_scripts/main_train_precip.py  
