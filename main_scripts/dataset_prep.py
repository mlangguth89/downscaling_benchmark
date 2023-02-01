# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT


__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-13"


import xarray as xr
import torch
import numpy as np
import pathlib
import math
import torchvision
import os
import json
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')



class PrecipDatasetInter(torch.utils.data.IterableDataset):
    """
    This is the class used for generate dataset generator for precipitation downscaling
    """

    def __init__(self, file_path: str = None, batch_size: int = 4, patch_size: int = 16,
                 vars_in: list = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in",
                                  "u700_in","v700_in","yw_hourly_in"],
                 vars_out: list = ["yw_hourly_tar"], sf: int = 10,
                 seed: int = 1234, k: float = 0.01, mode: str = "train", stat_path: str = None):
        """
        file_path : the path to the directory of .nc files
        vars_in   : the list contains the input variable names
        var_out   : the list contains the output variable name
        batch_size: the number of samples per iteration
        patch_size: the patch size for low-resolution image,
                    the corresponding high-resolution patch size should be muliply by scale factor (sf)
        sf        : the scaling factor from low-resolution to high-resolution
        seed      : specify a seed so that we can generate the same random index for shuffle function
        stat_dir  : the path to the directory of training .nc files
        """

        super(PrecipDatasetInter).__init__()

        self.file_path = file_path
        self.patch_size = patch_size
        self.sf = sf  # scaling factor
        self.vars_in = vars_in
        self.var_out = vars_out
        self.batch_size = batch_size
        self.seed = seed
        self.k = k 
        self.mode = mode
        self.stat_path = stat_path
        self.vars_in_patches_list = []
        self.vars_out_patches_list = []
        self.times_patches_list = []
        
        prcpids = ['yw','cp','lsp']
        self.prcp_indexes = []
        i = 0

        while i < len(vars_in):
            for j in range(len(prcpids)):
                if prcpids[j] in vars_in[i]:
                    self.prcp_indexes.append(i)
            i += 1
        print('self.prcp_indexes: {}'.format(self.prcp_indexes))

        # Search for files
        p = pathlib.Path(self.file_path)
        #assert(p.is_dir())
        #self.files = glob.glob(os.path.join(file_path, 'preproc_ifs_radklim*.nc'))
        #for path in p.rglob('preproc_ifs_radklim*.nc'):
        #    print("pathname",path.name)
        files = sorted(p.rglob('preproc_ifs_radklim_*.nc'))
        if len(files) < 1:
            raise RuntimeError('No files found.')
        print("Going to open the following files:", files)
       
        self.vars_in_patches_list, self.vars_out_patches_list, self.times_patches_list  = self.process_netcdf(files)
        print('self.times_patches_list: {}'.format(self.times_patches_list))

        if self.mode == "train":
            self.vars_in_patches_mean = self.vars_in_patches_list.mean(dim=(0,2,3))
            self.vars_in_patches_std = self.vars_in_patches_list.std(dim=(0,2,3))
            self.vars_out_patches_mean = self.vars_out_patches_list.mean()
            self.vars_out_patches_std = self.vars_out_patches_list.std()
        else:
            stat_file = os.path.join(stat_path, "statistics.json")
            with open(stat_file,'r') as f:
                stat_data = json.load(f)
            self.vars_in_patches_mean = []
            self.vars_in_patches_std = []
            for i in range(len(self.vars_in)):
                self.vars_in_patches_mean.append(stat_data[self.vars_in[i]+'_mean'])
                self.vars_in_patches_std.append(stat_data[self.vars_in[i]+'_std'])
            self.vars_out_patches_mean = stat_data[self.var_out[0]+'_mean']
            self.vars_out_patches_std = stat_data[self.var_out[0]+'_std']

        print("The total number of samples after filtering NaN values:", len(self.vars_in_patches_list))
        
        self.n_samples = len(self.vars_in_patches_list)
        #print("var_out size",self.vars_out_patches_list)

        self.idx_perm = self.shuffle()
        self.save_stats()

    def process_netcdf(self, filenames: int = None):
        """
        process netcdf files: filter the Nan Values, split to patches
        """
        print("Loading data from the file:", filenames)
        dt = xr.open_mfdataset(filenames)
        # get input variables, and select the regions
        inputs = dt[self.vars_in].isel(lon = slice(2, 114)).sel(lat = slice(47.5, 60))
        output = dt[self.var_out].isel(lon_tar = slice(16, 113 * 10 + 6)).sel(lat_tar = slice(47.41, 60))

        n_lat = inputs["lat"].values.shape[0]
        n_lon = inputs["lon"].values.shape[0]

        assert inputs.dims["time"] == output.dims["time"]
        assert inputs.dims["lat"] * self.sf == output.dims["lat_tar"]

        n_patches_x = int(np.floor(n_lon) / self.patch_size)
        n_patches_y = int(np.floor(n_lat) / self.patch_size)
        num_patches_img = n_patches_x * n_patches_y

        inputs_nparray = inputs.to_array(dim = "variables").squeeze().values
        outputs_nparray = output.to_array(dim = "variables").squeeze().values

        # log-transform -> log(x+k)-log(k)
        inputs_nparray[self.prcp_indexes] = np.log(inputs_nparray[self.prcp_indexes]+self.k)-np.log(self.k)
        outputs_nparray = np.log(outputs_nparray+self.k)-np.log(self.k)
        print('inputs_nparray shape: {}'.format(inputs_nparray.shape))
        print('inputs_nparray[self.prcp_indexes] shape: {}'.format(inputs_nparray[self.prcp_indexes].shape))

        da_in = torch.from_numpy(inputs_nparray)
        da_out = torch.from_numpy(outputs_nparray)
        del inputs_nparray, outputs_nparray
        gc.collect()
        times = inputs["time"].values  # get the timestamps
        times = np.transpose(np.stack([dt["time"].dt.year,dt["time"].dt.month,dt["time"].dt.day,dt["time"].dt.hour]))        
        
        print("Original input shape:", da_in.shape)

        # split into small patches, the return dim are [vars, samples,n_patch_x, n_patch_y, patch_size, patch_size]
        vars_in_patches = da_in.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        vars_in_patches_shape = list(vars_in_patches.shape)

        #sanity check to make sure the number of patches is as we expected
        assert n_patches_x * n_patches_y == int(vars_in_patches_shape[2] * vars_in_patches_shape[3])
        
        vars_in_patches = torch.reshape(vars_in_patches, [vars_in_patches_shape[0],
                                                          vars_in_patches_shape[1] * vars_in_patches_shape[2] *
                                                          vars_in_patches_shape[3],
                                                          vars_in_patches_shape[4], vars_in_patches_shape[5]])

        vars_in_patches = torch.transpose(vars_in_patches, 0, 1)
        print("Input shape:", vars_in_patches.shape)

        ## Replicate times for patches in the same image
        times_patches = torch.from_numpy(np.array([ x for x in times for _ in range(num_patches_img)]))
        
        ## sanity check 
        assert len(times_patches) ==  vars_in_patches_shape[1] * vars_in_patches_shape[2] * vars_in_patches_shape[3]

        vars_out_patches = da_out.unfold(1, self.patch_size * self.sf,
                                         self.patch_size * self.sf).unfold(2,
                                                                       self.patch_size * self.sf,
                                                                       self.patch_size * self.sf)
        vars_out_patches_shape = list(vars_out_patches.shape)
        vars_out_patches = torch.reshape(vars_out_patches,
                                         [vars_out_patches_shape[0] * vars_out_patches_shape[1] *
                                          vars_out_patches_shape[2],
                                          vars_out_patches_shape[3], vars_out_patches_shape[4]])

        print("Output reshape", vars_out_patches.shape)

        no_nan_idx = []

        # get the indx if there any nan in the sample
        [no_nan_idx.append(i) for i in range(vars_out_patches.shape[0]) if not torch.isnan(vars_out_patches[i]).any()]

        print("There are No. {} patches out of {} without Nan Values ".format(len(no_nan_idx), len(vars_out_patches)))

        # change the index from List to LongTensor type
        no_nan_idx = torch.LongTensor(no_nan_idx)

        # Only get the patch that without NaN values
        vars_out_pathes_no_nan = torch.index_select(vars_out_patches, 0, no_nan_idx)
        vars_in_patches_no_nan = torch.index_select(vars_in_patches, 0, no_nan_idx)
        times_no_nan = torch.index_select(times_patches, 0, no_nan_idx)
        assert len(vars_out_pathes_no_nan) == len(vars_in_patches_no_nan)

        return vars_in_patches_no_nan, vars_out_pathes_no_nan, times_no_nan

    def shuffle(self):
        """
        shuffle the index 
        """
        print("Shuffling the index ....")
        multiformer_np_rng = np.random.default_rng(self.seed)
        idx_perm = multiformer_np_rng.permutation(self.n_samples)

        # restrict to multiples of batch size
        idx = int(math.floor(self.n_samples/self.batch_size)) * self.batch_size

        idx_perm = idx_perm[:idx]
        print("idx_perm",idx_perm)
        return idx_perm

    def save_stats(self):
        output_file = os.path.join(self.file_path, "statistics.json")
        stats = {}
        for i in range(len(self.vars_in)):
            key = self.vars_in[i]+'_mean'
            stats.update({key:float(self.vars_in_patches_mean[i])})
            key = self.vars_in[i]+'_std'
            stats.update({key:float(self.vars_in_patches_std[i])}) 
            
        key = self.var_out[0]+'_mean'
        stats.update({key:float(self.vars_out_patches_mean)})
        key = self.var_out[0]+'_std'
        stats.update({key:float(self.vars_out_patches_std)})

        #save to output directory
        with open(output_file,'w') as f:
            json.dump(stats, f)
        print("The statistic has been stored to the json file: ", output_file)

    def __iter__(self):

        iter_start, iter_end = 0, int(len(self.idx_perm)/self.batch_size)  # todo
        self.idx = 0

        transform_x = torchvision.transforms.Normalize(self.vars_in_patches_mean, self.vars_in_patches_std)

        for bidx in range(iter_start, iter_end):

            #initialise x, y for each batch
            # x  stores the low resolution images, y for high resolution,
            # t is the corresponding timestamps, cidx is the index
            x = torch.zeros(self.batch_size, len(self.vars_in), self.patch_size, self.patch_size)
            y = torch.zeros(self.batch_size, self.patch_size * self.sf, self.patch_size * self.sf )
            t = torch.zeros(self.batch_size, 4, dtype = torch.int)
            cidx = torch.zeros(self.batch_size, 1, dtype = torch.int) #store the index

            for jj in range(self.batch_size):

                cid = self.idx_perm[self.idx]

                x[jj] = transform_x(self.vars_in_patches_list[cid])
                y[jj] = (self.vars_out_patches_list[cid] - self.vars_out_patches_mean) / self.vars_out_patches_std
                t[jj] = self.times_patches_list[cid]
                cidx[jj] = torch.tensor(cid, dtype=torch.int)

                self.idx += 1
            yield  {'L': x, 'H': y, "idx": cidx, "T":t}

def run():
    data_loader = PrecipDatasetInter(file_path="/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train")
    print("created data_loader")
    for batch_idx, train_data in enumerate(data_loader):
        inputs = train_data["L"]
        target = train_data["H"]
        idx = train_data["idx"]
        times = train_data["T"]
        print("inputs", inputs.size())
        print("target", target.size())
        print("idx", idx)
        print("batch_idx", batch_idx)
        print("timestamps,", times)

if __name__ == "__main__":
    run()











