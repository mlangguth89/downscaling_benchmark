__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-03-16"
__update__ = "2022-03-16"

import os
from abc import ABC
from other_utils import get_func_kwargs

class Abstract_Preprocessing(ABC):
    """
    Abstract class for preprocessing
    """
    def __init__(self, name_preprocess: str, source_dir: str, target_dir: str):
        """
        Basic initialization.
        :param name_preprocess: name of preprocessing chain for easy identification
        """
        method = Abstract_Preprocessing.__init__.__name__
        # sanity check
        assert isinstance(name_preprocess, str), "%{0}: name_preprocess must be a string.".format(method)
        assert os.path.isdir(source_dir), "%{0}: Parsed source_dir '{1}' does not exist.".format(method, source_dir)

        self.name_preprocess = name_preprocess
        self.source_dir = source_dir
        self.target_dir = Abstract_Preprocessing.check_target_dir(target_dir)

    def __call__(self, *args, **kwargs):
        """
        To be defined in child class.
        """
        method = Abstract_Preprocessing.__call__.__name__

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
        raise NotImplementedError(self.print_implement_err(Abstract_Preprocessing.prepare_worker.__name__))

    def preprocess_worker(self):
        """
        Worker task to perform (parallelized) preprocessing.
        :return: -
        """
        raise NotImplementedError(self.print_implement_err(Abstract_Preprocessing.prepare_worker.__name__))

    @staticmethod
    def check_target_dir(target_dir):
        method = Abstract_Preprocessing.check_target_dir.__name__

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

    @staticmethod
    def read_grid_des(grid_des_file):
        """
        Read CDO grid description file and put data into dictionary.
        :param grid_des_file: the grid description file to be read
        :return: dictionary with key-values from grid description parameters
                 (e.g. gridtype = lonlat -> {"gridtype": "lonlat"}).
        """
        method = Abstract_Preprocessing.read_grid_des.__name__

        if not os.path.isfile(grid_des_file):
            raise FileNotFoundError("%{0}: Cannot find grid description file '{1}'.".format(method, grid_des_file))

        # read the file ...
        with open(grid_des_file, "r") as fgdes:
            lines = fgdes.readlines()

        # and put data into dictionary
        grid_des_dict = Abstract_Preprocessing.griddes_lines_to_dict(lines)

        if not grid_des_file:
            raise ValueError("%{0}: Dictionary from grid description file '{1}' is empty. Please check input."
                             .format(method, grid_des_file))
        else:
            grid_des_dict["file"] = grid_des_file

        return grid_des_dict

    @staticmethod
    def write_grid_des_from_dict(grid_des_dict: dict, filename: str):
        """
        Write CDO grid description file from a dictionary.
        :param grid_des_dict: dictionary whose keys and values are to be written into CDO's grid description file.
        :param filename: name of grid description file to be created
        """
        method = Abstract_Preprocessing.write_grid_des_from_dict.__name__

        # sanity checks
        assert isinstance(grid_des_dict, dict), "%{0}: grid_des_dict must be a dictionary.".format(method)
        assert isinstance(filename, str), "%{0}: filename must be a string.".format(method)

        with open(filename, "w") as grid_des_file:
            for key, value in grid_des_dict.items():
                grid_des_file.write("{0} = {1} \n".format(key, value))

        print("%{0}: Grid description file '{1}' was created successfully.".format(method, filename))

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

    @classmethod
    def print_implement_err(cls, method):
        """
        Return error sring for required functions that are not implemented yet.
        :param method: Name of method
        :return: error-string
        """
        err_str = "%{0}: Method {1} not implemented yet. Cannot continue.".format(cls.__name__, method)

        return err_str