# ********** Info **********
# @Creation: 2022-03-29
# @Author: Bing Gong
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: preproces_downscaling_data for precipitation RADKLIM data (from one image to sequence)
# ********** Info **********

import tensorflow as tf
import os
import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

class PreciProcess(object):

    def __init__(self, src_dir: str = None, out_dir=None, month: int = 1, year: int = 2011, seq_len: int = 24,
                 shift: int = 6, threshold: float = 0.1):

        self.month = month
        self.year = year
        self.src_dir = src_dir
        self.out_dir = out_dir
        self.seq_len = seq_len
        self.shift = shift
        self.threshold = threshold

        avail_years = list(range(2011, 2019))
        if year not in avail_years:
            raise ValueError("The year is only available from", avail_years)

        if seq_len % self.shift != 0:
            raise ("The seq_len must be deivided by shift")

        self.gen_full_src_dir()

    def gen_full_src_dir(self):
        """
        To generate the full path of input and output directory
        """

        self.input_dir = os.path.join(self.src_dir, str(self.year), '{}-{:02}/'.format(self.year, self.month))
        self.output_dir = os.path.join(self.out_dir, str(self.year))
        self.output_nc_fl = os.path.join(self.output_dir, '{}-{:02}.nc'.format(self.year, self.month))

        if not os.path.isdir(self.input_dir):
            raise NotADirectoryError("The input directory {} does not exist".format(self.input_dir))

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok = True)
            print("%{0}: Created output-directory for remapped data '{1}'".format(self.input_dir, self.output_dir))
        else:
            print(self.output_dir, "has been created and exist")

    def preprocess_to_sequences(self):

        # get the data from one month netcdf files
        dt = xr.open_mfdataset(os.path.join(self.input_dir, "*.nc"))
        print("All the netcdf files in {} is opened".format(self.input_dir))
        da = dt["yw_hourly"]
        x = dt["x"]
        y = dt["y"]
        self.timestamps = dt["time"].values
        self.data_arr = np.squeeze(da.values)

        # preprocess the data to seqences with certain shift
        dataset = tf.data.Dataset.from_tensor_slices(self.data_arr).window(self.seq_len, shift = self.shift,
                                                                           drop_remainder = True)
        dataset = dataset.flat_map(lambda window: window.batch(self.seq_len))
        remainder = (self.seq_len - self.shift)
        self.ts = (self.timestamps)[:-remainder][::self.shift]

        # Get the sequeneces of data and calculate the mean values, and save to list
        exp_list = []
        counts = 0
        for next_element in dataset.take(20000):
            exp = np.nanmean(next_element.numpy())
            exp_list.append(exp)
            counts = counts + 1

        print("{} sequences are generated for year {}, month {}".format(counts, self.year, self.month))

        # pos_min_2_max = np.argsort(exp_list) #sort the mean values and get the position of the order

        sel_examples_pos = [x > 0.1 for x in
                            exp_list]  # get the position that the mean of sequence is large than 0.1 mm h-1
        # The position must be saved for processing the coarsed data (use the same position from the list)

        pos = 0
        sequences = []
        sequences_ts = []
        for next_element in dataset.take(2000):
            sequence = next_element.numpy()
            if sel_examples_pos[pos]:
                # save squences to file
                sequences.append(sequence)
                sequences_ts.append(self.ts[pos])
            else:
                pass
            pos = pos + 1

        print("{} sequences larger than threshold {} are saved out of the total sequences {} for  year {}, month {}".format(sequences.shape[0], self.threshold, counts, self.year,self.month))
        # sanity check, compare the example from sequences and from netcdf files
        seq_ck = sequences[2][0]
        ts_ck = sequences_ts[2]
        da_ck = da.sel(time = ts_ck)
        print("da_ck", da_ck)
        print("ts_ck", ts_ck)
        if np.nanmax(seq_ck - da_ck) == 0:
            print("The data preprcessing for precipition data is succesfully!")
            print("The sequences shape is ", np.array(sequences).shape)

        else:
            raise (
                "The data is not processed properly ! The generated sequences does not match the values from original netcdf files!!")

        # save sequences to netcef file
        ds = xr.Dataset(
            {'yw_hourly': (['init_time', 'seq_len', 'y', 'x'], sequences), },
            coords = {"init_time": sequences_ts, 'x': x, 'y': y, "seq_len": np.arange(1, self.seq_len + 1)}, )
        ds.attrs["unit"] = "mm h-1"
        ds.attrs["grid_mapping"] = "polar_stereographic"
        ds.attrs["License"] = "DWD Licenses"
        ds.attrs["Authors"] = "Bing Gong (Jülich Supercomputing Center), Yan Ji(Jülich Supercomputing Center)"
        ds.attrs["Source"] = "DWD C-Band Weather Radar Network, Original RADOLAN Data"
        ds.attrs["Created_Date"] = "2022-03-29"
        ds.to_netcdf(self.output_nc_fl)
        print("The file {} is saved!".format(self.output_nc_fl))

        #plot the mean of sqeuences
        plt_name = os.path.join(self.output_dir, "means_seq_year_{}_month_{}.img")
        plt.ylim(0, 0.8)
        plt.plot(np.sort(exp_list))
        plt.savefig(plt_name)
        print("The mean value of sequence ")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_parent_dir", "-src_dir", dest="src_dir", type=str,
                        default="/p/largedata/slmet/slmet111/met_data/dwd/radklim-yw/netcdf/orig_grid/yw_hourly",
                        help="Top-level directory under which radklim data are stored with subdirectories " +
                             "<year>/<month>.")
    parser.add_argument("--out_parent_dir", "-out_dir", dest="out_dir", type=str, required=True,
                        help="Top-level directory under which remapped data will be stored.")
    parser.add_argument("--years", "-y", dest="years", type=int, nargs="+", default=[2016, 2017, 2018, 2019, 2020],
                        help="Years of data to be preprocessed.")
    parser.add_argument("--months", "-m", dest="months", type=int, nargs="+", default=range(3, 10),
                        help="Months of data to be preprocessed.")
    parser.add_argument("--prep_threshold", "-threshold", dest="p_threshold", type=float,  default=0.1,
                        help="The precipitation threshold to filter the data.")

    parser.add_argument("--sequence_length", "-seq_len", dest="seq_len", type=int,  default=24,
                        help="The sequence length per sample")
    parser.add_argument("--shift", "-s", dest="shift", type=int,  default=6,
                        help="The shift for preprocessing the image to sequences, the sequence_length should be divided by this the number of shift")

    args = parser.parse_args()
    dir_in = args.src_dir
    dir_out = args.out_dir
    years = args.years
    months = args.months
    seq_len = args.seq_len
    threshold = args.threshold
    shift = args.shift

    PreciProcessObj =PreciProcess(src_dir=dir_in, out_dir=dir_out, month=args.months, year=years, seq_len=seq_len, shift=shift, threshold=threshold)
    PreciProcessObj.preprocess_to_sequences()


