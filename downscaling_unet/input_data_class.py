import os, sys
from timeit import default_timer as timer
import climetlab as cml
import numpy as np
import xarray as xr

class InputDataClass(object):

    def __init__(self, datadir: str, application: str, fname_base: str = "") -> None:
        """
        Initialize Input data object by reading data from netCDF-files
        :param datadir: the directory from where netCDF-files are located (or should be located if downloaded)
        :param application: name of application (must coincide with name in s3-bucket)
        :param fname_base: prefix for naming the netCDF-files on disk
        """
        method = InputDataClass.__init__.__name__

        self.host = os.getenv("HOSTNAME") if os.getenv("HSOTNAME") is not None else "unknown"
        self.fname_base = fname_base
        self.application = application
        if os.path.isdir(datadir):
            self.datadir = datadir
        else:
            raise NotADirectoryError("%{0}: input data directory '{1}' does not exist.")

        self.ldownload, self.data_dict = self.set_download_flag()
        ds_train, ds_val, ds_test, self.loading_time = self.get_data()
        self.data = {"train": ds_train, "val": ds_val, "test": ds_test}

    def set_download_flag(self):
        """
        Depending on the hosting system and on the availability of the dataset on the filesystem
        (stored under self.datadir), the download flag is set to False or True. Also returns a dictionary for the
        respective netCDF-filenames.
        :return: Boolean flag for downloading and dictionary of data-filenames
        """
        method = InputDataClass.set_download_flag.__name__

        ldownload = True if self.has_internet() else False

        ncf_train, ncf_val, ncf_test = os.path.join(self.datadir, self.fname_base+"_train.nc"), \
                                       os.path.join(self.datadir, self.fname_base+"_val.nc"), \
                                       os.path.join(self.datadir, self.fname_base+"_test.nc")

        stat_files = all(list(map(os.path.isfile, [ncf_train, ncf_val, ncf_test])))

        if stat_files and ldownload:
            print("%{0}: Datafiles are already available under '{1}'".format(method, self.datadir))
            ldownload = False
        elif not stat_files and not ldownload:
            raise ValueError("%{0}: Datafiles are not complete under '{1}',".format(method, self.datadir) +
                             "but downloading on computing node '{0}' is not possible.".format(self.host))

        return ldownload, {"train_datafile": ncf_train, "val_datafile": ncf_val, "test_datafile": ncf_test}

    def get_data(self):
        """
        Depending on the flag ldownload, data is either downloaded from the s3-bucket or read from the file system.
        The time for the latter process is measured.
        :return: xarray-Datasets for training, validation and testing (loaded to memory) and elapsed time
        """

        method = InputDataClass.get_data.__name__

        ncf_train, ncf_val, ncf_test = self.data_dict["train_datafile"], self.data_dict["val_datafile"], \
                                       self.data_dict["test_datafile"]

        if self.ldownload:
            try:
                print("%{0}: Start downloading the data...".format(method))
                # download the data from ECMWF's s3-bucket
                cmlds_train = cml.load_dataset(self.application, dataset="training")
                cmlds_val = cml.load_dataset(self.application, dataset="validation")
                cmlds_test = cml.load_dataset(self.application, dataset="testing")
                # convert to xarray datasets and...
                ds_train, ds_val, ds_test = cmlds_train.to_xarray(), cmlds_val.to_xarray(), cmlds_test.to_xarray()
                # ...save to disk
                _, _, _ = self.ds_to_netcdf(ds_train, ncf_train), self.ds_to_netcdf(ds_val, ncf_val),\
                          self.ds_to_netcdf(ds_test, ncf_test)

                t_load = None
            except Exception as err:
                print("%{0}: Failed to download data files '{1}' for application {2} from s3-bucket."
                      .format(method, ",".join(self.data_dict.keys()), self.application))
                raise err
        else:
            t0 = timer()
            try:
                print("%{0}: Start reading the data from '{1}'...".format(method, self.datadir))
                ds_train, ds_val, ds_test = xr.open_dataset(ncf_train), xr.open_dataset(ncf_val),\
                                            xr.open_dataset(ncf_test)

                ds_train, ds_val, ds_test = ds_train.load(), ds_val.load(), ds_test.load()
            except Exception as err:
                raise RuntimeError("%{0}: Failed to read all required files '{1}' from '{2}'"
                                   .format(method, ",".join(self.data_dict.keys()), self.datadir))

            t_load = timer() - t0

        return ds_train, ds_val, ds_test, t_load

    @staticmethod
    def ds_to_netcdf(ds: xr.Dataset, fname: str, comp_lvl=5):
        """
        Create dictionary for compressing all variables of dataset in netCDF-files
        :param ds: the xarray-dataset
        :param fname: name of the target netCDF-file
        :param comp_lvl: the compression level
        :return: True in case of success
        """
        method = InputDataClass.ds_to_netcdf.__name__

        comp = dict(zlib=True, complevel=comp_lvl)
        try:
            encoding_ds = {var: comp for var in ds.data_vars}
            print("%{0}: Save dataset to netCDF-file '{1}'".format(method, fname))
            ds.to_netcdf(path=fname, encoding=encoding_ds)  # , engine="scipy")
        except Exception as err:
            print("%{0}: Failed to handle and save input dataset.".format(method))
            raise err

        return True

    @staticmethod
    def has_internet():
        """
        Checks if Internet connection is available.
        :return: True if connected, False else.
        """
        try:
            # connect to the host -- tells us if the host is actually
            # reachable
            socket.create_connection(("1.1.1.1", 53))
            return True
        except OSError:
            pass
        return False



