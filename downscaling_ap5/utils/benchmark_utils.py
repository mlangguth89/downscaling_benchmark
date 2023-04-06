__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-24"
__update__ = "2022-01-24"

import os
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
from handle_data_class import HandleDataClass


class BenchmarkCSV(object):
    """
    This objects helps to track the required parameters for benchmarking.
    The parameters are stored in csv-files which can be easily integrated to the Google spreadshit.
    New data can be added by parsing a dictionary whose keys should be named as the elements of the class variable
    'expected_cols' together with the respective diagnosed values.
    """

    expected_cols = ["Experiment number", "Job ID", "#Nodes", "#GPUs", "#MPI tasks", "#CPUs",
                     "Data loading time", "Total runtime", "Total training time",
                     "Avg. training time per epoch", "First epoch training time",
                     "Min. training time per epoch", "Max. training time per epoch",
                     "Avg. training time per iteration", "Final training loss", "Final validation loss",
                     "Saving model time", "Node ID", "Max. GPU power", "GPU energy consumption"]

    def __init__(self, csvfile: str):
        """
        Initialize object with some csv-file. If the csv-file already exists, its data is also made available.
        :param csvfile:
        """
        self.csv_file, self.mode, self.data = BenchmarkCSV.check_csvfile(csvfile)

        if self.mode == "a":
            self.exp_number = self.get_exp_number()
        elif self.mode == "w":
            self.exp_number = 1

    def get_exp_number(self):
        """
        Get the experiment number based on the data read from the existing csv-data.
        :return: the current experiment number (latest experiment number incremented by 1)
        """
        all_exps = sorted(self.data["Experiment number"].values)

        return all_exps[-1] + 1

    def populate_csv_from_dict(self, benchmark_dict: dict):
        """
        Write benchmark dictionary to csv-file. Either append an existing one or create it.
        :param benchmark_dict: the dictionary whose keys must provide all elements listed in 'expected_cols'.
        :return: Updated/created csv-file on disk
        """
        
        benchmark_dict[BenchmarkCSV.expected_cols[0]] = self.exp_number
        dict_keys = benchmark_dict.keys()

        _ = BenchmarkCSV.check_collist(dict_keys, ignore_case=True)

        # to allow for generic key-value queries, lowercase all keys
        benchmark_dict_l = {k.lower(): v for k, v in benchmark_dict.items()}

        benchmark_tuples = [(key, [benchmark_dict_l[key.lower()]]) for key in BenchmarkCSV.expected_cols]
        benchmark_dict_ordered = OrderedDict(benchmark_tuples)

        df_benchmark = pd.DataFrame.from_dict(benchmark_dict_ordered)

        df_benchmark.to_csv(self.csv_file, mode="a", header=not os.path.exists(self.csv_file), index=False)

    @staticmethod
    def check_csvfile(csvfile: str):
        """
        Check if CSV-file exists. In case of yes, the data from the existing file is read in.
        :param csvfile: the path to the CSV-file
        :return: tuple of path to the CSV-file, mode for further file operation and DataFrame (if file is present).
        """

        method = BenchmarkCSV.check_csvfile.__name__

        # sanity check
        assert csvfile.endswith(".csv"), "%{0}: Parsed file must be a csv-file".format(method)

        # either read data from existing file or change mode to "writing" and set data to None.
        if os.path.isfile(csvfile):
            try:
                data = pd.read_csv(csvfile)
            except BaseException as err:
                print("%{0}: Unable to open existing csv-file ''. Inspect error-message.".format(method, csvfile))
                raise err

            # check if expected columns are present
            columns = list(data.columns)
            _ = BenchmarkCSV.check_collist(columns)

            mode = "a"
        else:
            mode = "w"
            data = None

        return csvfile, mode, data

    @staticmethod
    def check_collist(column_list: list, ignore_case: bool = False):
        """
        Checks if all elements in class vairable 'expected_cals' can be found in column_list.
        :param column_list: The column list (i.e. the column names from a Pandas DataFrame).
        :param ignore_case: flag if elements are checked in a case-insensitive manner.
        :return:
        """

        method = BenchmarkCSV.check_collist.__name__

        if ignore_case:
            column_list_l = [col.lower() for col in column_list]
            stat = [True if expected_col.lower() in column_list_l else False
                    for expected_col in BenchmarkCSV.expected_cols]
        else:
            stat = [True if expected_col in column_list else False for expected_col in BenchmarkCSV.expected_cols]

        if np.all(stat):
            return True
        else:
            misses = [BenchmarkCSV.expected_cols[i] for i in range(len(stat)) if not stat[i]]
            raise ValueError("%{0}: The following keys/columns are missing: {1}".format(method, ", ".join(misses)))


def get_training_time_dict(epoch_times: list, steps):

    tot_time = np.sum(epoch_times)

    training_times = {"Total training time": np.sum(epoch_times), "Avg. training time per epoch": np.mean(epoch_times),
                      "Min. training time per epoch": np.amin(epoch_times),
                      "Max. training time per epoch": np.amax(epoch_times[1:]),
                      "First epoch training time": epoch_times[0], "Avg. training time per iteration": tot_time/steps}

    return training_times


def write_dataset_info(data_obj: HandleDataClass, nsamples: int, shape_sample: tuple):
    """
    Write some (constant) benchmark information on the dataset used during training.
    :param data_obj: The input data object
    :param nsamples: number of samples for training
    :param shape_sample: shape of sample
    :return: Writes data_info.json-file to disk
    """
    method = write_dataset_info.__name__

    js_file = os.path.join(os.getcwd(), "data_info.json")

    data_mem = data_obj.data_info["memory_datasets"]
    data_info = {"training data size": data_mem["train"], "validation data size": data_mem["val"],
                 "nsamples": nsamples, "shape_samples": shape_sample}

    if not os.path.isfile(js_file):
        # write data to JSON-file
        with open(js_file, "w") as jsf:
            json.dump(data_info, jsf)
    else:
        with open(js_file, "r") as jsf:
            data_old = json.load(jsf)

        if data_old != data_info:
            print("%{0}: WARNING: New dataset info is different from existing one (file: '{1}'). Check for correctness."
                  .format(method, js_file))
            print(data_info)
            print(data_old)













