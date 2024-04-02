#! /bin/bash -x
#SBATCH --account=deepacf
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=train_samos-model-out.%j
#SBATCH --error=train_samos-model-err.%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad.mayer@geosphere.at

# loading modules
module load R/4.3.2

# link R library for packages not included in the module
export R_LIBS_USER=/p/project/deepacf/maelstrom/mayer3/R:$R_LIBS_USER


# declare directory-variables
export source_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/dataset
export destination_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/results/samos_benchmark_t2m


srun Rscript $(pwd)/models/samos_climatology.R -in ${source_dir} -out ${destination_dir} --dataset "COSMO-REA6" # TODO: dataset is currently hard coded for testing as calculating both in one go may exceed walltime
# srun Rscript $(pwd)/models/samos_model.R -in ${source_dir} -out ${destination_dir} 
