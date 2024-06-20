#!/usr/bin/env Rscript

##################################################################
#Description    : Specify tiles for climatology modelling as
#                 needed for SAMOS
#Author         : Konrad Mayer
##################################################################

library(stars)
library(dplyr)
library(R.utils)

args <- commandArgs(trailingOnly = TRUE, asValues = TRUE,
    defaults = c(
        "in" = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/dataset/with_snow/",
        "out" = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/results/samos_benchmark_t2m"
    ))

in_fls <- dir(args[["in"]], ".*train.*\\.nc$", full.names = TRUE)
dat <- read_stars(in_fls[[1]], proxy = TRUE, sub = "t2m_in") # loaded as proxy
tiles <- st_tile(nrow(dat), ncol(dat), 16, 18)

# for some reason some lines with 0 are included (bug in st_tile?) - manually remove them:
tiles <- tiles[rowSums(tiles) != 0, ]

# write tile plan to disk
saveRDS(tiles, file.path(args[["out"]], "climatology", "tiles.rds"))

# print how many slurm array indizes are needed - to be used in HPC_batch_scripts/train_samos_model.sh
nrow(tiles)
