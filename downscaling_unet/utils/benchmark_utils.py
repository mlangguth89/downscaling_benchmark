__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-24"
__update__ = "2022-01-24"

import os
from collections import OrderedDict
import numpy as np
import pandas as pd

class BenchmarkCSV(object):

    expected_cols = ["Experiment number", "Job ID", "#Nodes", "#GPUs", "#MPI tasks", "#CPUs",
                     "Loading data time", "Total runtime", "Total training time",
                     "Avg. training time per epoch", "Training time first epoch",
                     "Min. training time per epoch", "Max. training time per epoch",
                     "Avg. training time per iteration", "Final training loss", "Final validation loss",
                     "Saving model time"]

    def __init__(self, csvfile):
        self.csv_file, self.mode, self.data = BenchmarkCSV.check_csvfile(csvfile)

        if self.mode == "a":
            self.exp_number = self.get_exp_number()
        elif self.mode == "w":
            self.exp_number = 1

    def get_exp_number(self):
        
        all_exps = sorted(self.data["Experiment number"].values)

        return all_exps[-1] + 1

    def populate_csv_from_dict(self, benchmark_dict: dict):
        
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

        method = BenchmarkCSV.check_csvfile.__name__

        # sanity
        assert csvfile.endswith(".csv"), "%{0}: Parsed file must be a csv-file".format(method)

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
    def check_collist(column_list, ignore_case: bool = False):

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








