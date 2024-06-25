#! /bin/bash -x
#SBATCH --account=deepacf
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=1:30:00
#SBATCH --output=train_samos-model-out.%j
#SBATCH --error=train_samos-model-err.%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad.mayer@geosphere.at
#SBATCH --array=1-64

# loading modules
module load R/4.3.2

# link R library for packages not included in the module
export R_LIBS_USER=/p/project/deepacf/maelstrom/mayer3/R:$R_LIBS_USER


# declare directory-variables
export source_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/dataset/with_snow
export destination_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/results/samos_benchmark_t2m
export models_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/trained_models/samos_benchmark_t2m

srun Rscript $(pwd)/models/samos_climatology.R --in ${source_dir} --out ${destination_dir} --models ${models_dir} --tile ${SLURM_ARRAY_TASK_ID}
