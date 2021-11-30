# Building a customized Jupyter kernel for running the downscaling application of AP5

### Preamble
To run the downscaling application published with deliverable 1.2 in the scope of the MAELSTROM project, a customized Jupyter Notebook kernel must be set up. In addition to TensorFlow 2.5.0, this kernel includes the package `climetlab-maelstrom-downscaling` for data retrieval and the package `cartopy` to plot the data. The latter two packages are not available with the default kernels. However, some modules of the HPC-system ease their integration.

### Set-up the customized Jupyter kernel

- Log in on a login node of Juwels at Jupyter-JSC and naviagte to `jupyter4jsc/j4j_notebooks/001-Jupyter/` to open the Notebook `Create_JupyterKernel_general.ipynb`
- follow the instructions to prepare a new Jupyter kernel
- Load the follwoing modules under `1. Create/Pimp new virtual Python environment` (also change the basic modules):<br>
    ```
    module -q purge 
    module -q use $OTHERSTAGES
    ml Stages/2020
    ml GCCcore/.10.3.0
    ml GCC/10.3.0
    ml ParaStationMPI/5.4.9-1
    ml netCDF/4.7.4
    ml Cartopy/0.18.0-Python-3.8.5
    ml SciPy-Stack/2021-Python-3.8.5
    ml TensorFlow/2.5.0-Python-3.8.5
    ```
- install the following Python packages into the virtual environment under step `1.5`: <br>
    ``` 
    pip install --no-cache-dir numpy==1.19.4 xarray==0.18.2
    pip install --no-cache-dir climetlab==0.8.16 climetlab-maelstrom-downscaling
    ```
- **Note**: 
    - `numpy` v1.19.4 is explictly added to `pip install` even though it is available from the modules loaded above. However, not fixing this version would trigger an installation of more recent numpy versions (>v1.20) which are incompatible with TensorFlow
    - pip throws a few errors when installing the Python packages such as claiming an incompatability of the `six`-package. However, these errors can be ignored and mostly relate to a bug of `pip` (You may verify in a Python shell that `six` v0.15.0 is indeed loaded instead of v0.16.0).
    
- some details on the pip-instructions above:
    - fixing the `numpy` version avoids that a more recent numpy version is installed to `site-packages`. This is mandatory since TensorFlow is incompatible with `numpy`-versions newer than 1.20.
    - We must also fix the `climetlab`-version since the latest versions are also incompatible with `climetlab-maelstrom-downscaling` (to be fixed!)
- finalize the kernel creation by running all cells
    - **Note**: Ensure that you load the same modules under step `2.1` as listed above
- The created kernel called `${KERNEL_NAME}` (see Notebook) should now appear as available kernel when you open a Jupyter Notebook
    - check if the new kernel can be activated successfully
    - check if you can import `xarray` and `climetlab` to ensure that basic data handling can be undertaken