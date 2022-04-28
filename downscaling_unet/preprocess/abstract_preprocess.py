__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-03-16"
__update__ = "2022-03-16"

import os
from abc import ABC
import numpy as np
from other_utils import get_func_kwargs


class AbstractPreprocessing(ABC):
    """
    Abstract class for preprocessing
    """
    def __init__(self, name_preprocess: str, source_dir: str, target_dir: str):
        """
        Basic initialization.
        :param name_preprocess: name of preprocessing chain for easy identification
        """
        method = AbstractPreprocessing.__init__.__name__
        # sanity check
        assert isinstance(name_preprocess, str), "%{0}: name_preprocess must be a string.".format(method)
        assert os.path.isdir(source_dir), "%{0}: Parsed source_dir '{1}' does not exist.".format(method, source_dir)

        self.name_preprocess = name_preprocess
        self.source_dir = source_dir
        self.target_dir = AbstractPreprocessing.check_target_dir(target_dir)

    def __call__(self, *args, **kwargs):
        """
        To be defined in child class.
        """
        method = AbstractPreprocessing.__call__.__name__

        # get keyword-arguments to initialize PyStager
        prepare_kwargs = get_func_kwargs(self.prepare_worker, kwargs)
        pystager_instance, run_dict = self.prepare_worker(*args, **prepare_kwargs)

        assert pystager_instance.is_setup, "%{0}: PyStager was not set up by prepare_worker-method. Cannot continue."\
                                           .format(method)

        # get keyword-arguments to run PyStager and run it
        run_kwargs = get_func_kwargs(self.preprocess_worker, kwargs)
        pystager_instance.run(*run_dict["args"], **run_dict["kwargs"], **run_kwargs)

    def prepare_worker(self, worker, *args, **kwargs):
        """
        Method to prepare worker, i.e. required work to run parallelized preprocessing (see also __call__-method).
        :return: Initialized PyStager-instance as well as arguments and keyword arguments to set-up PyStager
        """
        raise NotImplementedError(self.print_implement_err(AbstractPreprocessing.prepare_worker.__name__))

    def preprocess_worker(self):
        """
        Worker task to perform (parallelized) preprocessing.
        :return: -
        """
        raise NotImplementedError(self.print_implement_err(AbstractPreprocessing.prepare_worker.__name__))

    @staticmethod
    def check_target_dir(target_dir):
        method = AbstractPreprocessing.check_target_dir.__name__

        if not os.path.isdir(target_dir):
            try:
                print("%{0}: Create target directory '{1}'.".format(method, target_dir))
                os.makedirs(target_dir)
            except Exception as err:
                print("%{0}: Problem creating target directory '{1}'. Inspect raised error.".format(method, target_dir))
                raise err
        else:
            pass

        return target_dir

    @classmethod
    def print_implement_err(cls, method):
        """
        Return error sring for required functions that are not implemented yet.
        :param method: Name of method
        :return: error-string
        """
        err_str = "%{0}: Method {1} not implemented yet. Cannot continue.".format(cls.__name__, method)

        return err_str


class CDOGridDes(ABC):
    """
    Abstract class to handle grid description files for CDO
    """
    # valid CDO grd description keys, see section 1.5.2.4 on CDO grids in documentation
    # (https://code.mpimet.mpg.de/projects/cdo/embedded/index.html#x1-220001.5.2)
    valid_keys = ["gridtype", "gridsize", "xsize", "ysize", "xvals", "yvals", "nvertx", "xbounds", "ybounds", "xfirst",
                  "xinc", "yfirst", "yinc", "xunits", "yunits"]

    def __init__(self, gdes_file: str = None, gdes_dict: dict = None):
        """
        Initialize grid description either from existing file or from dictionary
        :param gdes_file: grid description file to read
        :param gdes_dict: dictionary to convert to grid description
        """
        method = CDOGridDes.__init__.__name__

        # checks
        if gdes_file and gdes_dict:
            print("%{0}: Path to grid decription file and grid description dictionary passed. "
                  "The file from the data is ignored.")
        elif gdes_file:
            self.grid_des_dict = CDOGridDes.read_grid_des(gdes_file)
        elif gdes_dict:
            self.grid_des_dict = gdes_dict
        else:
            raise ValueError("%{0}: Either pass gdes_fiel (path to grid descrition file)".format(method) +
                             " or gdes_dict (grid description dictionary).")

        CDOGridDes.check_gdes_dict(self.grid_des_dict, lbreak=True)

    def write_grid_des_from_dict(self, filename: str, other_dict: dict = None):
        """
        Write CDO grid description dictionary to file.
        :param filename: name of grid description file to be created
        :param other_dict: Other grid description dictionary to write to file (instead of self.grid_des_dict)
        """
        method = CDOGridDes.write_grid_des_from_dict.__name__

        # sanity checks
        if os.path.isfile(filename):
            "%{0}: Grid description file '{1}' already exists. Please remove it first.".format(method, filename)

        if other_dict:                      # overwrite grid description dictionary to write to file
            CDOGridDes.check_gdes_dict(other_dict)          # check if parsed dictionary is proper
            dict2write = other_dict
        else:
            dict2write = self.grid_des_dict

        with open(filename, "w") as grid_des_file:
            for key, value in dict2write.items():
                grid_des_file.write("{0} = {1} \n".format(key, value))

        print("%{0}: Grid description file '{1}' was created successfully.".format(method, filename))

    def create_coarsened_grid_des(self, target_dir: str, downscaling_fac: int, rank: int = None,
                                  lextrapolate: bool = True, name_base: str = ""):
        """
        Create grid description for coarsening data (to be used for remapping).
        :param target_dir: Directory to save the grid description file
        :param downscaling_fac: Downscaling factor between input and coarsened grid (dx_coarse = downscaling_fac*dx_in)
        :param rank: Rank of process which calls this function to avoid duplicated writing when parallelizing (optional)
        :param lextrapolate: Flag if extrapolation takes place at later remapping (results into creation of auxiliary
                             grid description)
        :param name_base: Suffix to control naming of resulting grid description files
        :return: dictionary of coarsened grid, for lextrapolate=True also returns dictionary of auxiliary grid
        """
        method = CDOGridDes.create_coarsened_grid_des.__name__

        required_keys = ["xfirst", "yfirst", "xsize", "ysize", "xinc", "yinc", "gridtype"]

        if not all(key in required_keys for key in self.grid_des_dict):
            raise ValueError("%{0}: Not all required keys ({1}) found in grid description dictionary."
                             .format(method, ", ".join(required_keys)))
        else:
            nxy_in = (self.grid_des_dict["xsize"], self.grid_des_dict["ysize"])
            dx_in = (self.grid_des_dict["xinc"], self.grid_des_dict["yinc"])
            xyf_in = (self.grid_des_dict["xfirst"], self.grid_des_dict["yfirst"])
            gtype = self.grid_des_dict["gridtype"]

        downscaling_fac = int(downscaling_fac)
        nxy_coarse = [np.divmod(n, downscaling_fac) for n in nxy_in]
        # sanity check
        if not all([n_c[1] == 0 for n_c in nxy_coarse]):
            raise ValueError("%{0}: Element of passed nxy ({1}) must be dividable by {2:d}."
                             .format(method, ",".join(nxy_in), downscaling_fac))

        # get parameters for auxiliary grid description files
        if lextrapolate:       # enlarge coarsened grid to allow for bilinear interpolation without extrapolation later
            add_n = 2
        else:
            add_n = 0
        dx_coarse = [d * int(downscaling_fac) for d in dx_in]
        nxy_coarse = [n[0] + add_n for n in nxy_coarse]

        # create data for auxiliary grid description file
        coarse_grid_des_dict = {"gridtype": gtype, "xsize": nxy_coarse[0], "ysize": nxy_coarse[1],
                                "xfirst": xyf_in[0] - (downscaling_fac+1) / 2 * dx_in[0], "xinc": dx_coarse[0],
                                "yfirst": xyf_in[1] - (downscaling_fac+1) / 2 * dx_in[1], "yinc": dx_coarse[1]}

        # construct filename
        coarse_grid_des = os.path.join(target_dir, "{0}coarsened_grid".format(name_base))
        # write data to CDO's grid description files
        if rank == 0 or rank is None:
            self.write_grid_des_from_dict(coarse_grid_des, coarse_grid_des_dict)
        # add grid description filenames to dictionary
        coarse_grid_des_dict["file"] = coarse_grid_des

        if not lextrapolate:
            nxy_base = [n + 2 * downscaling_fac for n in nxy_in]

            # create data for auxiliary grid description file
            base_grid_des_dict = {"gridtype": gtype, "xsize": nxy_base[0], "ysize": nxy_base[1],
                                  "xfirst": xyf_in[0] - dx_coarse[0], "xinc": dx_in[0],
                                  "yfirst": xyf_in[1] - dx_coarse[1], "yinc": dx_in[1]}
            # construct filename and ...
            base_grid_des = os.path.join(target_dir, "{0}grid_base".format(name_base))
            if rank == 0 or rank is None:
                self.write_grid_des_from_dict(base_grid_des, base_grid_des_dict)
            # add grid description filenames to dictionary
            base_grid_des_dict["file"] = base_grid_des

            return base_grid_des_dict, coarse_grid_des_dict

        return coarse_grid_des

    @staticmethod
    def check_gdes_dict(grid_des_dict, lbreak=False):
        """
        Check if grid description dictionary only contains valid keys.
        :param grid_des_dict: the grid description dictionary to be checked.
        :param lbreak: Flag if error should be raised. Alternatively, a warning is printed.
        :return: -
        """

        method = CDOGridDes.check_gdes_dict.__name__

        assert isinstance(grid_des_dict, dict), "%{0}: Grid description must be a dictionary.".format(method)

        gdes_keys = list(grid_des_dict.keys())
        gdes_keys_stat = [key_req in CDOGridDes.valid_keys for key_req in gdes_keys]

        if not all(gdes_keys_stat):
            invalid_keys = [gdes_keys[i] for i in range(len(gdes_keys)) if gdes_keys_stat[i]]
            err_str = "%{0}: The following keys are not valid: {1}".format(method, ",".join(invalid_keys))
            if lbreak:
                raise ValueError(err_str)
            else:
                print("WARNING: " + err_str)

    @staticmethod
    def read_grid_des(grid_des_file):
        """
        Read CDO grid description file and put data into dictionary.
        :param grid_des_file: the grid description file to be read
        :return: dictionary with key-values from grid description parameters
                 (e.g. gridtype = lonlat -> {"gridtype": "lonlat"}).
        """
        method = AbstractPreprocessing.read_grid_des.__name__

        if not os.path.isfile(grid_des_file):
            raise FileNotFoundError("%{0}: Cannot find grid description file '{1}'.".format(method, grid_des_file))

        # read the file ...
        with open(grid_des_file, "r") as fgdes:
            lines = fgdes.readlines()

        # and put data into dictionary
        grid_des_dict = AbstracrPreprocessing.griddes_lines_to_dict(lines)

        if not grid_des_file:
            raise ValueError("%{0}: Dictionary from grid description file '{1}' is empty. Please check input."
                             .format(method, grid_des_file))
        else:
            grid_des_dict["file"] = grid_des_file

        return grid_des_dict

    @staticmethod
    def griddes_lines_to_dict(lines):
        """
        Converts lines read from CDO grid description files into dictionary.
        :param lines: the lines-list obtained from readlines-method on grid description file.
        :return: dictionary with data from file
        """
        dict_out = {}
        for line in lines:
            splitted = line.replace("\n", "").split("=")
            if len(splitted) == 2:
                dict_out[splitted[0].strip()] = splitted[1].strip()

        return dict_out

