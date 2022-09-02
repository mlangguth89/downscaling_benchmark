#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-out.%j
#SBATCH --error=train-err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maximbr@post.bgu.ac.il

#source ../venv_booster/bin/activate

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
#ml SciPy-bundle/2021.10
source ../venv_booster/bin/activate

train_dir=/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train
test_dir=/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/test
save_dir=../results/exp_test
epochs=2
#model_type=vitSR
model_type=swinSR
#model_type=unet
#python ../main_scripts/main_train.py --train_dir ${train_dir} --test_dir ${test_dir} --save_dir ${save_dir} --epochs ${epochs} --model_type ${model_type} > output.test #python ../main_scripts/dataset_prep.py
#python ../main_scripts/main_train_precip.py
python ../main_scripts/dataset_temp.py
