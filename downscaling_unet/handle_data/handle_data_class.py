__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2022-01-22"

import os, sys
import socket
from timeit import default_timer as timer
import xarray as xr


class HandleDataClass(object):

    def __init__(self, datadir: str, application: str, query: str, purpose: str = None) -> None:
        """
        Initialize Input data object by reading data from netCDF-files
        :param datadir: the directory from where netCDF-files are located (or should be located if downloaded)
        :param application: name of application (must coincide with name in s3-bucket)
        :param fname_base: prefix for naming the netCDF-files on disk
        """
        method = HandleDataClass.__init__.__name__

        self.host = os.getenv("HOSTNAME") if os.getenv("HOSTNAME") is not None else "unknown"
        purpose = query if purpose is None else purpose
        self.application = application
        if not os.path.isdir(datadir):
            os.makedirs(datadir)

        ds = self.handle_data_req(query, purpose)
        self.ldownload = self.set_download_flag()
        self.datafile = os.path.join(datadir, "{0}_{1}.nc".format(application, self.purpose))
        t0_load = timer()
        ds = self.get_data()
        load_time = t0_load - timer()
        if self.ldownload:
            print("%{0}: Downloading the data took {0:.2f}s.".format(method, load_time))
            _ = HandleDataClass.ds_to_netcdf(ds, self.datafile)
        else:
            self.timing = {"loading_times": purpose: load_time}
            self.data = {purpose: ds}
        self.data_info = {"memory_datasets": {self.purpose: ds.nbytes}}

    def handle_data_req(self, query: str, purpose):
        datafile = os.path.join(self.datadir, "{0}_{1}.nc".format(self.application))
        self.ldownload_last = HandleDataClass.set_download_flag(datafile)
        # time data retrieval
        t0_load = timer()
        ds = self.get_data(query, datafile)
        load_time = timer() - t0_load
        if self.ldownload_last:
            print("%{0}: Downloading took {1:.2f}s.".format(method, load_time))
            _ = HandleDataClass.ds_to_netcdf(ds, self.datafile)


        return ds

            self.timing = {"loading_times": purpose: load_time}
            self.data = {purpose: ds}
        self.data_info = {"memory_datasets": {self.purpose: ds.nbytes}}

    def append_data(self, query: str, purpose: str = None):

        self.purpose = query if purpose is None else purpose
        self.ldownload = self.set_download_flag()
        self.datafile = os.path.join(datadir, "{0}_{1}.nc".format(application, self.purpose))
        t0_load = timer()
        ds = self.get_data()


    def set_download_flag(self, datafile):
        """
        Depending on the hosting system and on the availability of the dataset on the filesystem
        (stored under self.datadir), the download flag is set to False or True. Also returns a dictionary for the
        respective netCDF-filenames.
        :return: Boolean flag for downloading and dictionary of data-filenames
        """
        method = HandleDataClass.set_download_flag.__name__

        ldownload = True if "login" in self.host else False

        stat_file = os.path.isfile(datafile)

        if stat_file and ldownload:
            print("%{0}: Datafiles are already available under '{1}'".format(method, self.datadir))
            ldownload = False
        elif not stat_file and not ldownload:
            raise ValueError("%{0}: Data is not available under '{1}',".format(method, self.datadir) +
                             "but downloading on computing node '{0}' is not possible.".format(self.host))

        return ldownload

    def get_data(self):
        """
        Function to either downlaod data from the s3-bucket or to read from file.
        """
        raise NotImplementedError("Please set-up a customized get_data-function.")

    @staticmethod
    def ds_to_netcdf(ds: xr.Dataset, fname: str, comp_lvl=5):
        """
        Create dictionary for compressing all variables of dataset in netCDF-files
        :param ds: the xarray-dataset
        :param fname: name of the target netCDF-file
        :param comp_lvl: the compression level
        :return: True in case of success
        """
        method = HandleDataClass.ds_to_netcdf.__name__

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
            socket.create_connection(("1.1.1.1", 53), timeout=5)
            return True
        except OSError:
            pass
        return False



