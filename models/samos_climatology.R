#!/usr/bin/env Rscript

##################################################################
#Description    : Model climatology for location and scale with 
#                 trend and seasonality components, then calculate 
#                 residuals
#Author         : Konrad Mayer based on code from Markus Dabernig
##################################################################

library(stars)
library(dplyr)
library(tidyr)
library(lubridate)
library(hms)
library(crch)
library(stringr)
library(glue)
library(here)
library(purrr)
library(stars.ncdf) 
library(logger)
library(R.utils)

args <- commandArgs(trailingOnly = TRUE, asValues = TRUE,
    defaults = c(
        "in" = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/dataset",
        "out" = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/benchmark_t2m/results/samos_benchmark_t2m"
    ))

# TODO: this can easily be parallelized if RAM allows using furrr by replacing calls to `map` with `future_map` and uncommenting the lines below
library(future)
library(furrr)
plan(multicore, workers = 20)

# write selected datasets provided as commandline arguments to object, calculate for both ERA5 and COSMO-REA6 if none are provided
datasets <- args[["dataset"]]
if (length(datasets) == 0) {
  datasets <- c("ERA5", "COSMO-REA6")
}

# helpers

striptease <- function(fit) {
    # reduce object size of stored models
    attr(fit$terms$location, '.Environment') <- attr(fit$terms$scale, '.Environment') <- attr(fit$terms$full, '.Environment') <- NULL
    fit$residuals <- fit$fitted.values <- fit$model <- fit$link$scale$dmu.deta <- fit$control$start <- fit$formula <- NULL
    fit
}

reshape_results <- function(x, dat) {
    aperm(array(unlist(x), dim = dim(dat)[c(3, 2, 1)]), c(3, 2, 1))
}

# main

dothis <- function(lead_time, dataset, variable = "t2m") {
    lead_time <- hms(hours = lead_time)
    tryCatch(
    {

        log_info("Start calculation for lead time {lead_time}.")
        suffix <- switch(dataset,
            "ERA5" = "_in",
            "COSMO-REA6" = "_tar"
        )

        # load data
        in_fls <- dir(args[["in"]], ".*train.*\\.nc$", full.names = TRUE)
        dat <- read_stars(in_fls, proxy = TRUE, sub = paste0(variable, suffix)) # loaded as proxy

        dat <- dat %>%
            .[ , , , which(as_hms(time) == lead_time)] %>% # would be more elegant with dplyr::filter, but there lead_time is not found
            st_as_stars() %>% # load to memory
            units::drop_units()
        st_crs(dat) <- 4326

        log_info("Data loaded.")

        # extract time coordinate and derive objects
        timestamps <- st_get_dimension_values(dat, "time")
        year <- year(timestamps)
        yday <- yday(timestamps)

        # model components
        predictors <- tibble(
            sin1 = sin(yday * 2 * pi / 365),
            cos1 = cos(yday * 2 * pi / 365),
            sin2 = sin(yday * 4 * pi / 365),
            cos2 = cos(yday * 4 * pi / 365),
            trend = year - min(year) + yday / max(yday)
        )

        # dataframe with row and column indizes
        mdls <- expand_grid(
            i = seq_len(nrow(dat)),
            j = seq_len(ncol(dat))
        )

        # fit model per pixel
        log_info("Start fit of climatology models.")
        mdls <- mdls |> 
            mutate(mdl = future_map2(i, j, ~striptease(crch(dat[[1]][.x, .y, ] ~ sin1 + cos1 + sin2 + cos2 + trend | 
                            sin1 + cos1 + sin2 + cos2 + trend, data = predictors,
                            dist = 'gaussian')), .progress = interactive()))

        saveRDS(mdls, file.path(args[["out"]], glue("climatology/t2m_{tolower(dataset)}_{lead_time}_climatology-models.rds"))) # TODO: this takes quite some time (and space on disk), probably its better to only store coefficients instead of whole models
        log_info("Models fittet and saved to disk.")

        # fill predicted mu and sd to stars object
        log_info("Predict mu and sd from climatology models.")
        prediction <- dat[0] #initialize empty stars object with same coordinates as dat
        prediction$mu_modeled <- reshape_results(map(mdls$mdl, ~predict(.x, newdata = predictors, type = "location"), .progress = interactive()), dat)
        prediction$sd_modeled <- reshape_results(map(mdls$mdl, ~predict(.x, newdata = predictors, type = "scale"), .progress = interactive()), dat)

        st_crs(prediction) <- 4326
        write_stars_ncdf(prediction[1], file.path(args[["out"]], glue("climatology/t2m_{tolower(dataset)}_{lead_time}_mu-prediction.nc")))
        write_stars_ncdf(prediction[2], file.path(args[["out"]], glue("climatology/t2m_{tolower(dataset)}_{lead_time}_sd-prediction.nc")))
        log_info("Predicted mu and sd, based on climatology model, saved to disk.")

        
        # calculate residuals
        residuals <- (dat - prediction["mu_modeled"]) / prediction["sd_modeled"]
        write_stars_ncdf(residuals, file.path(args[["out"]], glue("climatology/t2m_{tolower(dataset)}_{lead_time}_residuals.nc")))
        log_info("Residuals saved to disk.")

        log_info("Calculation for lead time {lead_time} was successful.")
    },
        error = function(e) {log_error("Calculation of climatology and residuals for lead time {lead_time} failed: "); print(e)}
    )}

doall_dataset <- function(dataset) {
    log_info("START iterations to calculate climatologies and residuals from {toupper(dataset)} data.")
    lead_times <- seq(0, 21, by = 3)
    walk(lead_times, ~dothis(lead_time = .x, dataset = dataset, variable = switch(dataset, "ERA5" = "t2m", "COSMO-REA6" = "t_2m")))
    log_info("END iterations to calculate climatologies and residuals from {toupper(dataset)} data.")
}

walk(datasets, doall_dataset)