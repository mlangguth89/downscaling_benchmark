# Benchmark dataset for statistical downscaling with deep neural networks

This repository aims to provide a benchmark dataset for statistical downscaling of meteorological fields with deep neural networks.
The work is pursued in scope of the [MAELSTROM project](https://www.maelstrom-eurohpc.eu/) which aims to develop new machine learning applications for weather and climate under the cooridniantion of ECMWF. <br>
The benchmark dataset is based on two reanalysis datasets that are well established and quality-controlled in meteorology: The ERA5-reanalysis data serves as the input dataset, whereas the COSMO-REA6 datasets provides the target data for the downscaling. With a grid spacing of 0.275° of the input data compared to 0.0625° of the target data, a downscaling factor equals to 4 in the benchmark dataset. Furthermore, different specific downscaling tasks adapted from the literature are provided. These pertain the following meteorlogical quantities:
- 2m temerature 
- 10m wind 
- solar irradiance 
