__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-24"
__update__ = "2022-01-24"

import os
import pandas as pd

class BenchmarkCSV(object):

    def __init__(self, csvfile):
        self.csv_file, self.mode, self.data = BenchmarkCSV.check_csvfile(csvfile)

        if self.mode == "a":
            self.exp_number = self.get_exp_number()
        elif self.mode == "w":
            self.exp_number = 1

    def get_exp_number(self):
        all_exps = self.data["Experiment number"].values

        return all_exps[-1]

    @staticmethod
    def check_csvfile(csvfile: str):

        method = BenchmarkCSV.check_csvfile.__name__

        expected_cols = ["Experiment Number", "Job ID", "# Nodes", "# CPUs", "#MPI tasks", "#CPUs",
                         "Loading data time [s]", "Total runtime [s]", "Total training time [s]",
                         "Average training time per epoch [s]", "Training time first epoch [s]",
                         "Min. training time per epoch [s]", "Max. training time per epoch [s]",
                         "Average training time per iteration", "Final training loss", "Final validation loss" ,
                         "Saving model time [s]"]

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
            for expected_col in expected_cols:
                if expected_col in columns:
                    pass
                else:
                    raise ValueError("%{0}: Expected column '{1}' not found in data from csv-file '{2}'."
                                     .format(method, expected_col, csvfile) + "The following columns are required: {0}"
                                     .format(", ".join(expected_cols)))

            mode = "a"
        else:
            mode = "w"
            data = None

        return csvfile, mode, data






