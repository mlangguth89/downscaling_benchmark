# Stochastic, Recurrent Super-Resolution GAN for Downscaling Time-Evolving Atmospheric Fields

This is a reference implementation of a stochastic, recurrent super-resolution GAN for downscaling time-evolving fields, intended for use in the weather and climate sciences domain. This code supports a paper to be submitted to IEEE Transactions in Geoscience and Remote Sensing.

## Important note

The above mentioned work may serve as a starting point for statistical downscaling in scope of WP 5 in the the ()[MAELSTROM project]. However, we started off with a simple U-net ot downscale the 2m temperature (see `downscaling_gan`-director of this branch). Nonetheless, the original code provided by Leinonen et al., 2020 can be run on the JSC's HPC-system as detailed below. The original description for running the code (provided [here]()) has been complemented for this purpose.

## Get the repository
Simply clone this repository to your desired directory by 
```
# Either use...
git clone git@gitlab.jsc.fz-juelich.de:esde/machine-learning/downscaling_maelstrom.git        # ...with SSH 
# ...or...
git clone https://gitlab.jsc.fz-juelich.de/esde/machine-learning/downscaling_maelstrom.git    # with HTTPS
```
Ensure that your directory provides a reasonable amount of space (i.e. a feq GB). On JSC's HPC-system, it is recommended to choose a directory under `/p/project/` and to avoid cloning in your home-directory!

## Obtaining the data

The radar precipitation dataset (MCH-RZC in the paper) can be downloaded at https://doi.org/10.7910/DVN/ZDWWMG by following the instructions there. The GOES cloud optical thickness dataset (GOES-COT) can be found [in this data repository](https://doi.org/10.5281/zenodo.3835849) as "goes-samples-2019-128x128.nc". <br>
On JUST, the data is already made available under `/p/project/deepacf/maelstrom/data/downscaling_gan_leinonen/`.

## Obtaining the trained network

The trained generator weights selected for use in the paper are included in the `models` directory. The weights for the other time steps can be found [here](https://doi.org/10.5281/zenodo.3835849).

## Running the code on JSC's HPC-system
To train a Leinonen's GAN-model on the provided dataset by yourself, a batch-script based job submission on JSC's HPC-system is made available. The environment as well as the provided template runscript under the `batch_scripts`-directory may also serve for testing on other datasets. In the following, the installation of the virtual environment and the required sytsem-side software preparations are described.

### Getting a TensorFlow v1.15-container with Nvidia support 
The provided source-code requires TensorFlow v1.15 for training. To allow training on the HPC-system (Juwels, Juwels Booster and HDF-ML) while exploiting the system's GPU capacities, a singularity container of this rather old TensorFlow version with Nvidia support is mandatory. Such a containers are distributed [here](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/overview.html#overview). Currently (2021-11-18), the [TensorFlow release 2021.09](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_21-09.html#rel_21-09) can be used since the container is shipped with an OFED- and MPI-version and  that fits the versions of JSC's HPC-system. <br><br>
This container is already available under `/p/project/deepacf/deeprain/video_prediction_shared_folder/containers_juwels_booster/`. <br>
To avoid duplicates of big singularity containers, it is recommended to link the respective file to the `env_setup`-directory of this repository:
```
cd env_setup
ln -s /p/project/deepacf/deeprain/video_prediction_shared_folder/containers_juwels_booster/nvidia_tensorflow_21.09-tf1-py3.sif tensorflow_21.09-tf1-py3.sif
```
Note that the (linked) container-file under `create_env` is named approriately!

### Setting up the virtual environment
The singularity container lacks the netCDF-package for Python which is required to read the input data.
Thus, we need to set-up a virtual environment within the singularity container. The helper script `create_env.sh` accomplishs this together with `install_venv_container.sh`:
```
# in env_setup-directory
source create_env.sh <venv_name>       # please set <venv_name> 
```
This will create a virtual environment under the `virtual_envs` which is created under the base-directory.

### Running the code on JSC's HPC-system
To run the code, please create a copy of the batch-script template `train_downscaling_GAN_template.sh` and edit this script to your personal needs, i.e. set your e-mail adress in the SLURM-commands, remove the template header, set the name of the virtual environment according the previous step, choose an application (e.g. "mchrzc" or "goescod") and adjust the paths to in- and output.<br>
Afterwards, the job can be submitted.
```
cp train_downscaling_GAN_template.sh train_downscaling_GAN_<some_string>.sh     # please set <some_string>
# Modify the created batch-script as mentioned above
sbatch train_downscaling_GAN_<some_string>.sh 
```  
Please choose again an output-directory that 
1. provides suifficient space for the output (> some GBs) and
2. is accessible for the computing nodes (e.g. under `/p/project/[...]`)

### Producing plots
For evaluating the trained model and for creating the plots, the batch-script `train_downscaling_GAN_template.sh` is provided. Similarly to the training step, the template should be duplicated and edited. 
Additionally, the `h5`-files from the training or the pre-trained model must be linked to the `models`-directory.
Important note: If you want to evaluate your own model, set the variables `mchrzc_gen_weights_fn` and `goes_gen_weights_fn` in the `plot_all`-function of `dsrnngan/plots.py` accordingly!
```
# Change to models-directory and link the (pre-)trained models
cd models
ln -s <training_outdir>/<trained_model_for_mchrzc>.h5 ./<trained_model_for_mchrzc>.h5
ln -s <training_outdir>/<trained_model_for_goes>.h5 ./<trained_model_for_goes>.h5
# Changge to batch_scripts-directory
cd ../batch_scripts
cp plot_downscaling_GAN_template.sh plot_downscaling_GAN_<some_string>.sh     # please set <some_string>
# Modify the created batch-script analogous to above
sbatch plot_downscaling_GAN_<some_string>.sh 
```  

## Running the code somewhere else (original description)

For training, you'll want a machine with a GPU and around 32 GB of memory (the training procedure for the radar dataset loads the entire dataset into memory). Running the pre-trained model should work just fine on a CPU.

You may want to work with the code interactively; in this case, just start a Python shell in the `dsrnngan` directory.

If you want the simplest way to run the code, the following two options are available. You may also want to look at what `main.py` does in order to get an idea of how the training/plotting flow works.

### Producing plots

You can replicate the plots in the paper (except for Fig. 7 for which we unfortunately cannot release the source data) by going to the `dsrnngan` directory and using
```
python main.py plot --mchrzc_data_file=<mchrzc_data_file> --goescod_data_file=<<mchrzc_data_file>>
```
where `<mchrzc_data_file>` is the path to the radar precipitation dataset and `<<mchrzc_data_file>` is the path to the GOES cloud optical thickness dataset. For more control over the plotting process, see the function `plot_all` in `plots.py`.

### Training the model

Run the following to start the training:
```
python main.py train --application=<application> --data_file=<data_file> --save_weights_root=<save_weights_root> --log_path=<log_path>
```
where `<application>` is either `mchrzc` (for the MCH-RZC dataset) or `goescod` (for the GOES-COT dataset), `<data_file>` is the training data file appropriate for the application, `<save_weights_root>` is the directory and file name root to which you want to save all model weights, and `<log_path>` is a path to a log directory where the logs and generator weights over time will be saved. 

The above command will run the training loop for 400000 generator training sequences and save the weights after each 3200 sequences.

