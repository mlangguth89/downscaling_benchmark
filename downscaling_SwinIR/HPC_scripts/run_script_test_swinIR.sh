#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=test-swinIR-out.%j
#SBATCH --error=test-swinIR-err.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.ji@fz-juelich.de

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
#ml SciPy-bundle/2021.10
source ../venv_booster/bin/activate

train_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/preprocessing/preprocessed_ifs_radklim_full_disk/train
test_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/preprocessing/preprocessed_ifs_radklim_full_disk/test
checkpoint_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/results/swinSR_exp1110_origin_booster_3years_x2_5x4/21400_G.pth
save_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/results/swinSR_exp1110_origin_booster_3years_x2_5x4/prediction_21400

#model_type=vitSR
model_type=swinSR
#model_type=unet
#model_type=swinUnet

patch_size=4
window_size=8
upscale_swinIR=4

srun python ../main_scripts/main_test.py --test_dir ${test_dir} --checkpoint_dir ${checkpoint_dir}  --save_dir ${save_dir} --model_type ${model_type} --stat_dir ${train_dir}  --patch_size ${patch_size} --window_size ${window_size} --upscale_swinIR ${upscale_swinIR} # > vis.output


