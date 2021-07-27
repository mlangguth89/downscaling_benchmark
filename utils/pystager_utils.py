# ********** Info **********
# @Creation: 2020-12-10
# @Update: 2021-07-26
# @Author: Amirpasha Mozaffari, Michael Langguth
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: pystager_utils.py
# ********** Info **********

import sys, os
import multiprocessing
import subprocess
import inspect
from mpi4py import MPI
import logging
import numpy as np
import pandas as pd
import datetime as dt
import platform
 
# ======================= List of functions ====================================== #


class PyStager(object):
    class_name = "PyStager"

    def __init__(self, job_func: callable, distributor_name: str, num_proc: int, nmax_warn: int = 3,
                 logdir: str = None):
        """
        Initialize PyStager.
        :param job_func: Function whose execution is meant to be parallelized. This function must accept arguments
                         dynamical arguments provided by the distributor (see distributo_engine-method) and
                         static arguments (see run-method) in the order mentioned here. Additionally, it must accept
                         a logger instance. The argument 'nmax_warn' is optional.
        :param distributor_name: Name of distributor which takes care for the paralelization (see distributo_engine
                                 -method)
        :param num_proc: total number of processes available for parallelization
        :param nmax_warn: Maximal number of accepted warnings during job execution (default: 3)
        :param logdir: directory where logfile are stored (current working directory becomes the default if not set)
        """
        method = PyStager.__init__.__name__

        self.num_processes = num_proc
        self.cpu_name = platform.processor()
        self.num_cpus_max = multiprocessing.cpu_count()
        self.distributor = self.distributor_engine(distributor_name)
        self.logdir = PyStager.set_and_check_logdir(logdir, distributor_name)
        self.nmax_warn = int(nmax_warn)
        self.job = job_func
        self.transfer_dict = None
        self.comm = None
        self.my_rank = None

        # (further) sanity check
        if self.num_processes > self.num_cpus_max:
            raise ValueError("%{0}: Number of selected processes exceeds number of available CPUs (incl. threads)."
                             .format(method))

        if self.num_processes <= 1:
            raise ValueError("%{0}: PyStager requires at least two workers.".format(method))

        if not callable(self.job):
            raise ValueError("%{0}: Passed method to be parallelized must be a callable function.".format(method))

        if self.nmax_warn <= 0:
            raise ValueError("%{0}: nmax_warn must be larger than zero, but is set to {1:d}"
                             .format(method, self.nmax_warn))

    def setup(self, *args):
        """
        Simply passes arguments to initialized distributor.
        *args : Tuple of arguments suitable for distributor (self.distributor)
        """

        method = PyStager.setup.__name__
        try:
            self.transfer_dict = self.distributor(*args)
        except Exception as err:
            print("%{0}: Failed to set up transfer dictionary of PyStager (see raised error below)".format(method))
            raise err

    def run(self, data_dir, *args, job_name= "dummy"):
        """
        Run PyStager.
        """
        method = PyStager.run.__name__

        if self.transfer_dict is None:
            raise AttributeError("%{0}: transfer_dict is still None. Call setup beforehand!".format(method))

        if not os.path.isdir(data_dir):
            raise NotADirectoryError("%{0}: The passed data directory '{1}' does not exist.".format(method, data_dir))

        # initialize MPI
        self.comm = MPI.COMM_WORLD
        self.my_rank = self.comm.Get_rank()       # rank of the current process
        p = self.comm.Get_size()             # number of assigned processes

        if p != self.num_processes:
            raise ValueError("%{0}: Number of assigned MPI processes ({1:d}) and passed processes ({2}) differs."
                             .format(method, p, self.num_processes))

        if self.my_rank == 0:
            logger_main = os.path.join(self.logdir, "{0}_job_main.log".format(job_name))
            if os.path.exists(logger_main):
                print("%{0}: Main logger file '{1}' already existed and was deleted.".format(method, logger_main))
                os.remove(logger_main)

            logging.basicConfig(filename=logger_main, level=logging.DEBUG,
                                format="%(asctime)s:%(levelname)s:%(message)s")
            logger = logging.getLogger(__file__)
            logger.addHandler(logging.StreamHandle(sys.stdout))

            logger.info("PyStager is started at {0}".format(dt.datetime.now().strftime("%Y-%m%-d %H:%M:%S UTC")))

            # distribute work to worker processes
            for proc in range(1, p):
                broadcast_list = self.transfer_dict[proc]
                self.comm.send(broadcast_list, dest=proc)

            stat_mpi = self.manage_recv_mess(logger, allow_exit=False)

            if stat_mpi:
                logger.info("Job has been executed successfully on {0:d} worker processes. Job exists normally at {1}"
                            .format(p, dt.datetime.now().strftime("%Y-%m%-d %H:%M:%S UTC")))
                sys.exit(0)
            else:
                logger.critical("Job failed is shut down now.")
                sys.exit(1)
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

            lexit = self.manage_worker_jobs(logger, *args, allow_exit=False)

            if lexit:
                sys.exit(1)
            else:
                sys.exit(0)

    def distributor_engine(self, distributor_name: str):
        """
        Sets up distributor for organinzing parallelization.
        :param distributor_name: Name of distributor
        :return distributor: selected callable distributor
        """

        method = PyStager.distributor_engine.__name__

        if distributor_name.lower() == "date":
            distributor = self.load_distributor_date
        else:
            raise ValueError("%{0}: The distributor named {1} is not implemented yet.".format(method, distributor_name))

        return distributor

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
            raise ValueError("%{0}: date_start and date_end have to datetime objects!".format(method))

        if not (date_start.strftime("%H") == "00" and date_end.strftime("%H") == "00"):
            raise ValueError("%{0}: date_start and date_end must be valid at 00 UTC.".format(method))

        if not int((date_end - date_start).days) >= 1:
            raise ValueError("%{0}: date_end must be at least one day after date_start.".format(method))

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

    def manage_recv_mess(self, logger, allow_exit=True):
        """
        Manages received messages from worker processes. Also accumulates warnings and aborts job if maximum number is
        exceeded
        :param logger: logger instance to add logs according to received message from worker
        :param allow_exit: Boolean to allow job to exit on system-level (with sys.exit)
        :return stat: True if ok, else False
        """
        method = PyStager.manage_recv_mess.__name__

        assert isinstance(self.comm, MPI.Intracomm), "%{0}: comm must be a MPI Intracomm-instance, but is of type '{1}'"\
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
            elif worker_stat == "FATAL":
                logger.critical("{0} met a fatal error. System will be terminated".format(worker_str))
                lexit = True
            else:
                logger.critical("{0} has sent an unknown message: '{1}'. System will be terminated."
                                .format(method, worker_stat))
                lexit = True
            # sum of warnings exceeds allowed maximum
            if warn_counter > self.nmax_warn:
                lexit = True

            if lexit:
                if allow_exit:
                    sys.exit(1)
                return False

            message_counter += 1

        return True

    def manage_worker_jobs(self, logger, *args, allow_exit=True):
        """
        Manages worker processes and runs job with passed arguments.
        :param logger: logger instance to add logs according to received message from master and from parallelized job
        :param args: the arguments passed to parallelized job (see self.job in __init__)
        :param allow_exit: Boolean to allow job to exit on system-level (with sys.exit)
        """
        method = PyStager.manage_recv_mess.__name__

        worker_stat_fail = 9999

        # sanity checks
        assert isinstance(self.comm, MPI.Intracomm), "%{0}: comm must be a MPI Intracomm-instance, but is of type '{1}'"\
                                                .format(method, type(self.comm))

        assert isinstance(logger, logging.Logger), "%{0}: logger must be a Logger-instance, but is of type '{1}'"\
                                                   .format(method, type(logger))

        mess_in = self.comm.recv()
        if mess_in is None:
            mess_out = ("IDLEE{0}: Worker {1} is idle".format(self.my_rank, self.my_rank), 0)
            logger.info(mess_out)
            logger.info("Thus, nothing to do. Job is terminated locally on rank {0}".format(self.my_rank))
            self.comm.send(mess_out, dest=0)
            if allow_exit:
                sys.exit(0)
            else:
                return True
        else:
            if "nmax_warn" in inspect.getfullargspec(self.job).args:
                worker_stat = self.job(zip(*(mess_in, args)), logger, nmax_warn=self.nmax_warn)
            else:
                worker_stat = self.job(zip(*(mess_in, args)), logger)

        sys_stat = 0
        if worker_stat == -1:
            mess_out = ("ERROR{0}: Failure was triggered.".format(self.my_rank), worker_stat_fail)
            logger.critical("Progress was unsuccessful due to a fatal error observed." +
                            " Worker is terminating and communicating the termination of the job to main.")
            sys_stat = 1
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
            logger.critical("Unknown worker status received: {0:d}".format(worker_stat))
            mess_out = ("WARNN{0}: Unknown worker status ({1:d}) received ".format(self.my_rank, worker_stat),
                        worker_stat_fail)
            logger.warning("Worker {0} has produced unknown worker status.".format(self.my_rank))
            sys_stat = 1
        # communicate to master process
        self.comm.send(mess_out, dest=0)

        if allow_exit:
            sys.exit(sys_stat)
        else:
            return not bool(sys_stat)

    @staticmethod
    def set_and_check_logdir(logdir, distributor_name):
        """
        Sets and checks logging directory
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


