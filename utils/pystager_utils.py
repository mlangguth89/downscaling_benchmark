# ********** Info **********
# @Creation: 2021-07-25
# @Update: 2021-07-27
# @Author: Michael Langguth, based on work by Amirpasha Mozaffari
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: pystager_utils.py
# ********** Info **********

import sys, os
# The script must be executed with the mpi4py-module to ensure that the job gets aborted when an error is risen
# see https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
if "mpi4py" not in sys.modules:
    raise ModuleNotFoundError("Python-script must be called with the mpi4py module, i.e. 'python -m mpi4py <...>.py")
import multiprocessing
import subprocess
import inspect
from typing import Union, List
from mpi4py import MPI
import logging
import numpy as np
import pandas as pd
import datetime as dt
import platform


class Distributor(object):
    """
    Class for defining (customized) distributors. The distributor selected by the distributor_engine must provide
    the dynamical arguments for a parallelized job run by PyStager (see below) which inherits from this class.
    """
    class_name = "Distributor"

    def __init__(self, distributor_name):
        self.distributor_name = distributor_name

    def distributor_engine(self, distributor_name: str):
        """
        Sets up distributor for organinzing parallelization.
        :param distributor_name: Name of distributor
        :return distributor: selected callable distributor
        """
        method = "{0}->{1}".format(Distributor.class_name, Distributor.distributor_engine.__name__)

        if distributor_name.lower() == "date":
            distributor = self.distributor_date
        elif distributor_name.lower() == "year_month_list":
            distributor = self.distributor_year_month
        else:
            raise ValueError("%{0}: The distributor named {1} is not implemented yet.".format(method, distributor_name))

        return distributor

    def distributor_date(self, date_start: dt.datetime, date_end: dt.datetime, freq: str = "1D"):
        """
        Creates a transfer dictionary whose elements are lists for individual start and end dates for each processor
        param date_start: first date to convert
        param date_end: last date to convert
        return: transfer_dictionary allowing date-based parallelization
        """
        method = "{0}->{1}".format(Distributor.class_name, Distributor.distributor_date.__name__)

        # sanity checks
        if not (isinstance(date_start, dt.datetime) and isinstance(date_end, dt.datetime)):
            raise ValueError("%{0}: date_start and date_end have to datetime objects!".format(method))

        if not (date_start.strftime("%H") == "00" and date_end.strftime("%H") == "00"):
            raise ValueError("%{0}: date_start and date_end must be valid at 00 UTC.".format(method))

        if not int((date_end - date_start).days) >= 1:
            raise ValueError("%{0}: date_end must be at least one day after date_start.".format(method))

        if not hasattr(self, "num_processes"):
            print("%{0}: WARNING: Attribute num_processes is not set and thus no parallelization will take place.")
            num_processes = 2
        else:
            num_processes = self.num_processes

        # init transfer dictionary (function create_transfer_dict_from_list does not work here since transfer dictionary
        #                           consists of tuple with start and end-date instead of a number of elements)
        transfer_dict = dict.fromkeys(list(range(1, num_processes)))

        dates_req_all = pd.date_range(date_start, date_end, freq=freq)
        ndates = len(dates_req_all)
        ndates_per_node = int(np.ceil(float(ndates)/(num_processes-1)))

        for node in np.arange(num_processes):
            ind_max = np.minimum((node+1)*ndates_per_node-1, ndates-1)
            transfer_dict[node+1] = [dates_req_all[node*ndates_per_node],
                                     dates_req_all[ind_max]]
            if ndates-1 == ind_max:
                break

        return transfer_dict

    def distributor_year_month(self, years, months):

        method = "{0}->{1}".format(Distributor.class_name, Distributor.distributor_year_month.__name__)

        list_or_int = Union[List, int]

        assert isinstance(years, list_or_int.__args__), "%{0}: Input years must be list of years or an integer."\
                                                        .format(method)
        assert isinstance(months, list_or_int.__args__), "%{0}: Input months must be list of months or an integer."\
                                                         .format(method)

        if not hasattr(self, "num_processes"):
            print("%{0}: WARNING: Attribute num_processes is not set and thus no parallelization will take place.")
            num_processes = 2
        else:
            num_processes = self.num_processes

        if isinstance(years, int): years = [years]
        if isinstance(months, int): months = [months]

        all_years_months = []
        for year in years:
            for month in months:
                all_years_months.append(dt.datetime.strptime("{0:d}-{1:02d}".format(int(year), int(month)), "%Y-%m"))

        transfer_dict = Distributor.create_transfer_dict_from_list(all_years_months, num_processes)

        return transfer_dict

    @staticmethod
    def create_transfer_dict_from_list(in_list: List, num_procs: int):
        """
        Splits up list to transfer dictionary for PyStager.
        :param in_list: list of elements that can be distributed to workers
        :param num_procs: Number of workers that are avaialable to operate on elements of list
        :return: transfer dictionary for PyStager
        """

        method = Distributor.create_transfer_dict_from_list.__name__

        assert isinstance(in_list, list), "%{0} Input argument in_list must be a list, but is of type '{1}'."\
                                          .format(method, type(in_list))

        assert int(num_procs) >= 2, "%{0}: Number of processes must be at least two.".format(method)

        nelements = len(in_list)
        nelements_per_node = int(np.ceil(float(nelements)/(num_procs-1)))

        transfer_dict = dict.fromkeys(list(range(1, num_procs)))

        for i, element in enumerate(in_list):
            ind = i % (num_procs-1) + 1
            if i < num_procs:
                transfer_dict[ind] = [element]
            else:
                transfer_dict[ind].append(element)

        print(transfer_dict)

        return transfer_dict


class PyStager(Distributor):
    """
    Organizes parallelized execution of a job.
    The job must be wrapped into a function-object that can be fed with dynamical arguments provided by an available
    distributor and static arguments (see below).
    Running PyStager constitutes a three-step approach. First PyStager must be instanciated, then it must be set-up by
    calling the setup-method and finally, the job gets executed in a parallelized manner.
    Example: Let the function 'process_data' be capable to process hourly data files between date_start and date_end.
             Thus, parallelization can be organized with distributor_date which only must be fed with a start and end
             date (the freq-argument is optional and defaults to "1D" -> daily frequency (see pandas)).
             With the data being stored under <data_dir>, PyStager can be called in a Python-script by:
             pystager_obj = PyStager(process_data, "date")
             pystager_obj.setup(<start_datetime_obj>, <end_datetime_obj>, freq="1H")
             pystager_obj.run(<static_arguments>)
    By splitting up the setup-method from the execution, multiple job executions becomes possible.
    """

    class_name = "PyStager"

    def __init__(self, job_func: callable, distributor_name: str, nmax_warn: int = 3, logdir: str = None):
        """
        Initialize PyStager.
        :param job_func: Function whose execution is meant to be parallelized. This function must accept arguments
                         dynamical arguments provided by the distributor (see distributo_engine-method) and
                         static arguments (see run-method) in the order mentioned here. Additionally, it must accept
                         a logger instance. The argument 'nmax_warn' is optional.
        :param distributor_name: Name of distributor which takes care for the paralelization (see distributo_engine
                                 -method)
        :param nmax_warn: Maximal number of accepted warnings during job execution (default: 3)
        :param logdir: directory where logfile are stored (current working directory becomes the default if not set)
        """
        super().__init__(distributor_name)
        method = PyStager.__init__.__name__

        self.cpu_name = platform.processor()
        self.num_cpus_max = multiprocessing.cpu_count()
        self.distributor = self.distributor_engine(distributor_name)
        self.logdir = PyStager.set_and_check_logdir(logdir, distributor_name)
        self.nmax_warn = int(nmax_warn)
        self.job = job_func
        self.transfer_dict = None
        self.comm = MPI.COMM_WORLD
        self.my_rank = self.comm.Get_rank()
        self.num_processes = self.comm.Get_size()

        if not callable(self.job):
            raise ValueError("%{0}: Passed function to be parallelized must be a callable function for {1}."
                             .format(method, PyStager.class_name))

        if self.nmax_warn <= 0:
            raise ValueError("%{0}: nmax_warn must be larger than zero for {1}, but is set to {2:d}"
                             .format(method, PyStager.class_name, self.nmax_warn))

        if self.num_processes < 2:
            raise ValueError("%{0}: Number of assigned MPI processes must be at least two for {1}."
                             .format(method, PyStager.class_name))

    def setup(self, *args):
        """
        Simply passes arguments to initialized distributor.
        *args : Tuple of arguments suitable for distributor (self.distributor)
        """
        method = PyStager.setup.__name__

        if self.my_rank == 0:
            try:
                self.transfer_dict = self.distributor(*args)
            except Exception as err:
                print("%{0}: Failed to set up transfer dictionary of PyStager (see raised error below)".format(method))
                raise err
        else:
            pass

    # def run(self, data_dir, *args, job_name="dummy"):
    def run(self, *args, job_name="dummy"):
        """
        Run PyStager.
        """
        method = "{0}->{1}".format(PyStager.class_name, PyStager.run.__name__)

        if self.my_rank == 0 and self.transfer_dict is None:
            raise AttributeError("%{0}: transfer_dict is still None. Call setup beforehand!".format(method))

        # if not os.path.isdir(data_dir):
        #    raise NotADirectoryError("%{0}: The passed data directory '{1}' does not exist.".format(method, data_dir))

        if self.my_rank == 0:
            logger_main = os.path.join(self.logdir, "{0}_job_main.log".format(job_name))
            if os.path.exists(logger_main):
                print("%{0}: Main logger file '{1}' already existed and was deleted.".format(method, logger_main))
                os.remove(logger_main)

            logging.basicConfig(filename=logger_main, level=logging.DEBUG,
                                format="%(asctime)s:%(levelname)s:%(message)s")
            logger = logging.getLogger(__file__)
            logger.addHandler(logging.StreamHandler(sys.stdout))

            logger.info("PyStager is started at {0}".format(dt.datetime.now().strftime("%Y-%m%-d %H:%M:%S UTC")))

            # distribute work to worker processes
            for proc in range(1, self.num_processes):
                broadcast_list = self.transfer_dict[proc]
                self.comm.send(broadcast_list, dest=proc)

            stat_mpi = self.manage_recv_mess(logger)

            if stat_mpi:
                logger.info("Job has been executed successfully on {0:d} worker processes. Job exists normally at {1}"
                            .format(self.num_processes, dt.datetime.now().strftime("%Y-%m%-d %H:%M:%S UTC")))
        else:
            # worker logger file
            logger_worker = os.path.join(self.logdir, "{0}_job_worker_{1}.log".format(job_name, self.my_rank))
            if os.path.exists(logger_worker):
                os.remove(logger_worker)

            logging.basicConfig(filename=logger_worker, level=logging.DEBUG,
                                format='%(asctime)s:%(levelname)s:%(message)s')
            logger = logging.getLogger(__file__)
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.info("==============Worker logger is activated ==============")
            logger.info("Start receiving message from master...")

            stat_worker = self.manage_worker_jobs(logger, *args)

        MPI.Finalize()

    def manage_recv_mess(self, logger):
        """
        Manages received messages from worker processes. Also accumulates warnings and aborts job if maximum number is
        exceeded
        :param logger: logger instance to add logs according to received message from worker
        :return stat: True if ok, else False
        """
        method = "{0}->{1}".format(PyStager.class_name, PyStager.manage_recv_mess.__name__)

        assert isinstance(self.comm, MPI.Intracomm), "%{0}: comm must be a MPI Intracomm-instance, but is type '{1}'"\
                                                     .format(method, type(self.comm))

        assert isinstance(logger, logging.Logger), "%{0}: logger must be a Logger-instance, but is of type '{1}'"\
                                                   .format(method, type(logger))

        message_counter = 1
        warn_counter = 0
        lexit = False
        while message_counter < self.num_processes:
            mess_in = self.comm.recv()
            worker_stat = mess_in[0][:5]
            worker_num = mess_in[0][5:7]
            worker_str = "Worker with ID {0}".format(worker_num)
            # check worker status
            if worker_stat == "IDLEE":
                logger.info("{0} is idle. Nothing to do.".format(worker_str))
            elif worker_stat == "PASSS":
                logger.info("{0} has finished the job successfully".format(worker_str))
            elif worker_stat == "WARNN":
                warn_counter += int(mess_in[1])
                logger.warning("{0} has seen a non-fatal error/warning. Number of marnings is now {1:d}"
                               .format(worker_str, warn_counter))
            elif worker_stat == "ERROR":
                logger.critical("{0} met a fatal error. System will be terminated".format(worker_str))
                lexit = True
            else:
                logger.critical("{0} has sent an unknown message: '{1}'. System will be terminated."
                                .format(method, worker_stat))
                lexit = True
            # sum of warnings exceeds allowed maximum
            if warn_counter > self.nmax_warn:
                logger.critical("Number of allowed warnings exceeded. Job will be terminated...")
                lexit = True

            if lexit:
                logger.critical("Job is shut down now.")
                raise RuntimeError("%{0}: Distributed jobs could not be run properly.".format(method))

            message_counter += 1

        return True

    def manage_worker_jobs(self, logger, *args):
        """
        Manages worker processes and runs job with passed arguments.
        Receives from master process and returns a tuple of a return-message and a worker status.
        :param logger: logger instance to add logs according to received message from master and from parallelized job
        :param args: the arguments passed to parallelized job (see self.job in __init__)
        :return stat: True if ok, else False
        """
        method = "{0}->{1}".format(PyStager.class_name, PyStager.manage_worker_jobs.__name__)

        worker_stat_fail = 9999

        # sanity checks
        assert isinstance(self.comm, MPI.Intracomm), "%{0}: comm must be a MPI Intracomm-instance, but is type '{1}'"\
                                                     .format(method, type(self.comm))

        assert isinstance(logger, logging.Logger), "%{0}: logger must be a Logger-instance, but is of type '{1}'"\
                                                   .format(method, type(logger))

        mess_in = self.comm.recv()

        if mess_in is None:
            mess_out = ("IDLEE{0}: Worker {1} is idle".format(self.my_rank, self.my_rank), 0)
            logger.info(mess_out)
            logger.info("Thus, nothing to do. Job is terminated locally on rank {0}".format(self.my_rank))
            self.comm.send(mess_out, dest=0)
            return True
        else:
            logger.info("Worker {0} received input message: {1}".format(self.my_rank, mess_in[0]))
            if "nmax_warn" in inspect.getfullargspec(self.job).args:
                worker_stat = self.job(mess_in, *args, logger, nmax_warn=self.nmax_warn)
            else:
                worker_stat = self.job(mess_in, *args, logger)


        err_mess = None
        if worker_stat == -1:
            mess_out = ("ERROR{0}: Failure was triggered.".format(self.my_rank), worker_stat_fail)
            logger.critical("Progress was unsuccessful due to a fatal error observed." +
                            " Worker{0} triggers termination of all jobs.".format(self.my_rank))
            err_mess = "Worker{0} met a fatal error. Cannot continue...".format(self.my_rank)
        elif worker_stat == 0:
            logger.debug('Progress was successful')
            mess_out = ("PASSS{0}:was finished".format(self.my_rank), worker_stat)
            logger.info('Worker {0} finished a task'.format(self.my_rank))
        elif worker_stat > 0:
            logger.debug("Progress faced {0:d} warnings which is still acceptable,".format(int(worker_stat)) +
                         " but requires investigation of the processed dataset.")
            mess_out = ("WARNN{0}: Several warnings ({1:d}) have been triggered "
                        .format(self.my_rank, worker_stat), worker_stat)
            logger.warning("Worker {0} has met relevant warnings, but still could continue...".format(self.my_rank))
        else:
            mess_out = ("ERROR{0}: Unknown worker status ({1:d}) received ".format(self.my_rank, worker_stat),
                        worker_stat_fail)
            err_mess = "Worker {0} has produced unknown worker status and triggers termination of all jobs."\
                       .format(self.my_rank)
            logger.critical(err_mess)
        # communicate to master process
        self.comm.send(mess_out, dest=0)

        if err_mess:
            return False
        else:
            return True

    @staticmethod
    def set_and_check_logdir(logdir, distributor_name):
        """
        Sets and checks logging directory.
        :param logdir: parent directory where log-files will be stored
        :param distributor_name: name of distributor-method (used for naming actual log-directory)
        :return logdir: adjusted log-directory
        """
        method = PyStager.set_and_check_logdir.__name__

        if logdir is None:
            logdir = os.path.join(os.getcwd(), "pystager_log_{0}".format(distributor_name))
            os.makedirs(logdir, exist_ok=True)
            print("%{0}: Default log directory '{1}' is used.".format(method, logdir))
        else:
            if not os.path.isdir(logdir):
                try:
                    os.mkdir(logdir)
                    print("%{0}: Created log directory '{1}'".format(method, logdir))
                except Exception as err:
                    print("%{0}: Failed to create desired log directory '{1}'".format(method, logdir))
                    raise Exception
            else:
                print("%{0}: Directory '{1}' is used as log directory.".format(method, logdir))

        return logdir

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
            raise NotADirectoryError("%{0}: The directory '{1}' does not exist.".format(method, source_path))

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
                raise NotADirectoryError("%{0}: '{1}' does not exist".format(method, path))

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
