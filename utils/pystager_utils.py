# ********** Info **********
# @Creation: 2020-12-10
# @Update: 2021-07-26
# @Author: Amirpasha Mozaffari, Michael Langguth
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: pystager_utils.py
# ********** Info **********

import os
import multiprocessing
import subprocess
import numpy as np
import pandas as pd
import datetime as dt
import platform
 
# ======================= List of functions ====================================== #


class PyStager(object):
    class_name = "PyStager"

    def __init__(self, num_proc: int):
        method = PyStager.__init__.__name__

        self.num_processes = num_proc
        self.cpu_name = platform.processor()
        self.num_cpus_max = multiprocessing.cpu_count()

        # sanity check
        if self.num_processes > self.num_cpus_max:
            raise ValueError("%{0}: Number of selected processes exceeds number of available CPUs (incl. threads)."
                             .format(method))

        if self.num_processes <= 1:
            raise ValueError("%{0}: PyStager requires at least two workers.".format(method))

    def load_distributor_date(self, date_start, date_end):
        """
        Creates a transfer dictionary whose elements are lists for individual start and end dates for each processor
        param date_start: first date to convert
        param date_end: last date to convert
        return: transfer_dictionary allowing date-based parallelization
        """
        method = PyStager.load_distributor_date.__name__

        # sanity checks
        if not (isinstance(date_start, dt.datetime) and isinstance(date_end, dt.datetime)):
            raise ValueError("date_start and date_end have to datetime objects!")

        if not (date_start.strftime("%H") == "00" and date_end.strftime("%H") == "00"):
            raise ValueError("date_start and date_end must be valid at 00 UTC.")

        if not int((date_end - date_start).days) >= 1:
            raise ValueError("date_end must be at least one day after date_start.")

        # init transfer dictionary
        transfer_dict = dict.fromkeys(list(range(1, self.num_processes)))

        dates_req_all = pd.date_range(date_start, date_end, freq='1D')
        ndates = len(dates_req_all)
        days_per_node = int(np.ceil(np.float(ndates)/(self.num_processes-1)))

        for node in np.arange(self.num_processes):
            ind_max = np.minimum((node+1)*days_per_node-1, ndates -1)
            transfer_dict[node+1] = [dates_req_all[node*days_per_node],
                                     dates_req_all[ind_max]]
            if ndates-1 == ind_max:
                break

        return transfer_dict

    @staticmethod
    def directory_scanner(source_path, lprint=True):
        """
        Scans through directory and returns a couple of information.
        NOTE: Subdirectories under source_path are not recursively scanned
        :param source_path: Input idrectory to scan
        :param lprint: Boolean if info should be printed (default: True)
        :return dir_info: dictionary containing info on scanned directory with the following keys
                          "dir_detail_list": overview on number of files and required memory
                          "sub_dir_list": list of subsirectories
                          "total_size_source": total meory under source_path
                          "total_num_files": total number of files under source_path
                          "total_num_directories": total number of directories under source_path
        """

        method = PyStager.directory_scanner.__name__

        dir_detail_list = []  # directories details
        sub_dir_list = []
        total_size_source = 0
        total_num_files = 0

        if not os.path.isdir(source_path):
            raise NotADirectoryError("%{0}: The directory '' does not exist.".format(method, source_path))

        list_directories = os.listdir(source_path)

        for d in list_directories:
            path = os.path.join(source_path, d)
            if os.path.isdir(path):
                sub_dir_list.append(d)
                sub_dir_list.sort()
                # size of the files and subdirectories
                size_dir = subprocess.check_output(['du', '-sc', path])
                splitted = size_dir.split()  # fist item is the size of the folder
                size = (splitted[0])
                num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                dir_detail_list.extend([d, size, num_files])
                total_num_files = total_num_files + int(num_files)
                total_size_source = total_size_source + int(size)
            else:
                raise NotADirectoryError("%{0}: {1} does not exist".format(method, path))

        total_num_directories = int(len(list_directories))
        total_size_source = float(total_size_source / 1000000)

        if lprint:
            print("===== Info from %{0}: =====".format(method))
            print("* Total memory size of the source directory: {0:.2f}Gb.".format(total_size_source))
            print("Total number of the files in the source directory: {0:d} ".format(num_files))
            print("Total number of the directories in the source directory: {0:d} ".format(total_num_directories))

        dir_info = {"dir_detail_list": dir_detail_list, "sub_dir_list": sub_dir_list,
                    "total_size_source": total_size_source, "total_num_files": total_num_files,
                    "total_num_directories": total_num_directories}

        return dir_info


