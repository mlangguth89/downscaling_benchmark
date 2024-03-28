# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Class for calculating scores.
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2022-Xx-XX"
__update__ = "2024-03-25"

from typing import List
import numpy as np
import xarray as xr
from skimage.util import view_as_blocks

class Scores:
    """
    Class to calculate scores and skill scores.
    """

    known_geodims = {"lon_dims": ["longitude", "lon", "rlon"],
                     "lat_dims": ["latitude", "lat", "rlat"]} 

    def __init__(self, data_fcst: xr.DataArray, data_ref: xr.DataArray, dims: List[str]):
        """
        :param data_fcst: forecast data to evaluate
        :param data_ref: reference or ground truth data
        """
        self.metrics_dict = {"mse": self.calc_mse, "rmse": self.calc_rmse, "bias": self.calc_bias,
                             "grad_amplitude": self.calc_spatial_variability, "psnr": self.calc_psnr, 
                             "acc": self.calc_acc, "mae": self.calc_mae, "l1": self.calc_l1, "l2": self.calc_l2,
                             "ets": self.calc_ets, "fbi": self.calc_fbi, "pss": self.calc_pss, 
                             "me_std": self.calc_mestd}
        self.data_fcst = data_fcst
        self.data_dims = list(self.data_fcst.dims)
        self.data_ref = data_ref
        self.avg_dims = dims
        self.knwon_geodims = {"lat_dims": ["rlat", "lat", "latitude"], "lon_dims": ["rlon", "lon", "longitude"]}

    def __call__(self, score_name, **kwargs):
        try:
            score_func = self.metrics_dict[score_name]
        except:
            raise ValueError(f"{score_name} is not an implemented score." +
                             "Choose one of the following: {0}".format(", ".join(self.metrics_dict.keys())))

        return score_func(**kwargs)

    @property
    def data_fcst(self):
        return self._data_fcst

    @data_fcst.setter
    def data_fcst(self, da_fcst):
        if not isinstance(da_fcst, xr.DataArray):
            raise ValueError("data_fcst must be a xarray DataArray.")

        self._data_fcst = da_fcst

    @property
    def data_ref(self):
        return self._data_ref

    @data_ref.setter
    def data_ref(self, da_ref):
        if not isinstance(da_ref, xr.DataArray):
            raise ValueError("data_fcst must be a xarray DataArray.")

        if not list(da_ref.dims) == self.data_dims:
            raise ValueError("Dimensions of data_fcst and data_ref must match, but got:" +
                             "[{0}] vs. [{1}]".format(", ".join(list(da_ref.dims)),
                                                      ", ".join(self.data_dims)))

        self._data_ref = da_ref

    @property
    def avg_dims(self):
        return self._avg_dims

    @avg_dims.setter
    def avg_dims(self, dims):
        if dims is None:
            self.avg_dims = self.data_dims
            # print("Scores will be averaged across all data dimensions.")
        else:
            dim_stat = [avg_dim in self.data_dims for avg_dim in dims]
            if not all(dim_stat):
                ind_bad = [i for i, x in enumerate(dim_stat) if not x]
                raise ValueError("The following dimensions for score-averaging are not " +
                                 "part of the data: {0}".format(", ".join(np.array(dims)[ind_bad])))

            self._avg_dims = dims

    def get_2x2_event_counts(self, thresh):
        """
        Get counts of 2x2 contingency tables
        :param thres: threshold to define events
        :return: (a, b, c, d)-tuple of 2x2 contingency table
        """
        a = ((self.data_fcst >= thresh) & (self.data_ref >= thresh)).sum(dim=self.avg_dims)
        b = ((self.data_fcst >= thresh) & (self.data_ref < thresh)).sum(dim=self.avg_dims)
        c = ((self.data_fcst < thresh) & (self.data_ref >= thresh)).sum(dim=self.avg_dims)
        d = ((self.data_fcst < thresh) & (self.data_ref < thresh)).sum(dim=self.avg_dims)

        return a, b, c, d

    def calc_ets(self, thresh=0.1):
        """
        Calculates Equitable Threat Score (ETS) on data.
        :param thres: threshold to define events
        :return: ets-values
        """
        a, b, c, d = self.get_2x2_event_counts(thresh)
        n = a + b + c + d
        ar = (a + b)*(a + c)/n      # random reference forecast
        
        denom = (a + b + c - ar)

        ets = (a - ar)/denom
        ets = ets.where(denom > 0, np.nan)

        return ets
    
    def calc_fbi(self, thresh=0.1):
        """
        Calculates Frequency bias (FBI) on data.
        :param thres: threshold to define events
        :return: fbi-values
        """
        a, b, c, _ = self.get_2x2_event_counts(thresh)

        denom = a+c
        fbi = (a + b)/denom

        fbi = fbi.where(denom > 0, np.nan)

        return fbi
    
    def calc_pss(self, thresh=0.1):
        """
        Calculates Peirce Skill Score (PSS) on data.
        :param thres: threshold to define events
        :return: pss-values
        """
        a, b, c, d = self.get_2x2_event_counts(thresh)      

        denom = (a + c)*(b + d)
        pss = (a*d - b*c)/denom

        pss = pss.where(denom > 0, np.nan)

        return pss   

    def calc_l1(self, **kwargs):
        """
        Calculate the L1 error norm of forecast data w.r.t. reference data.
        L1 will be divided by the number of samples along the average dimensions.
        Similar to MAE, but provides just a number divided by number of samples along average dimensions.
        :return: L1-error 
        """
        if kwargs:
            print("Passed keyword arguments to calc_l1 are without effect.")   

        l1 = np.sum(np.abs(self.data_fcst - self.data_ref))

        len_dims = np.array([self.data_fcst.sizes[dim] for dim in self.avg_dims])
        l1 /= np.prod(len_dims)

        return l1
    
    def calc_l2(self, **kwargs):
        """
        Calculate the L2 error norm of forecast data w.r.t. reference data.
        Similar to RMSE, but provides just a number divided by number of samples along average dimensions.
        :return: L2-error 
        """
        if kwargs:
            print("Passed keyword arguments to calc_l2 are without effect.")   

        l2 = np.sum(np.square(self.data_fcst - self.data_ref))

        len_dims = np.array([self.data_fcst.sizes[dim] for dim in self.avg_dims])
        l2 /= np.prod(len_dims)

        return l2
    
    def calc_mae(self, **kwargs):
        """
        Calculate mean absolute error (MAE) of forecast data w.r.t. reference data
        :return: MAE averaged over provided dimensions
        """
        if kwargs:
            print("Passed keyword arguments to calc_mae are without effect.")   

        mae = np.abs(self.data_fcst - self.data_ref).mean(dim=self.avg_dims)

        return mae

    def calc_mse(self, **kwargs):
        """
        Calculate mse of forecast data w.r.t. reference data
        :return: averaged mse for each batch example, [batch,fore_hours]
        """
        if kwargs:
            print("Passed keyword arguments to calc_mse are without effect.")

        mse = np.square(self.data_fcst - self.data_ref).mean(dim=self.avg_dims)

        return mse

    def calc_rmse(self, **kwargs):

        rmse = np.sqrt(self.calc_mse(**kwargs))

        return rmse
    
    def calc_acc(self, clim_mean: xr.DataArray, spatial_dims: List = ["lat", "lon"]):
        """
        Calculate anomaly correlation coefficient (ACC).
        :param clim_mean: climatological mean of the data
        :param spatial_dims: names of spatial dimensions over which ACC are calculated. 
                             Note: No averaging is possible over these dimensions.
        :return acc: Averaged ACC (except over spatial_dims)
        """

        fcst_ano, obs_ano = self.data_fcst - clim_mean, self.data_ref - clim_mean

        acc = (fcst_ano*obs_ano).sum(spatial_dims)/np.sqrt(fcst_ano.sum(spatial_dims)*obs_ano.sum(spatial_dims))

        mean_dims = [x for x in self.avg_dims if x not in spatial_dims]
        if len(mean_dims) > 0:
            acc = acc.mean(mean_dims)

        return acc

    def calc_bias(self, **kwargs):

        if kwargs:
            print("Passed keyword arguments to calc_bias are without effect.")

        bias = (self.data_fcst - self.data_ref).mean(dim=self.avg_dims)

        return bias

    def calc_psnr(self, **kwargs):
        """
        Calculate PSNR of forecast data w.r.t. reference data
        :param kwargs: known keyword argument 'pixel_max' for maximum value of data
        :return: averaged PSNR
        """
        pixel_max = kwargs.get("pixel_max", 1.)

        mse = self.calc_mse()
        if np.count_nonzero(mse) == 0:
            psnr = mse
            psnr[...] = 100.
        else:
            psnr = 20. * np.log10(pixel_max / np.sqrt(mse))

        return psnr
    
    def calc_mestd(self, downscaling_fac: int = 4, **kwargs):
        """
        Calculate the mean error of standard deviation of forecast data w.r.t. reference data
        Note that the MESTD is always averaged over the spatial dimensions (domanin average is returned).
        Further averaging can be applied by passing the keyword argument 'non_spatial_avg_dims'.
        :param downscaling_fac: downscaling factor (defines patching)
        :return: mean error of standard deviation
        """
        xy_dims = [self.check_for_coords(self.data_dims, "lon"), self.check_for_coords(self.data_dims, "lat")]
        dims_no_xy = [dim for dim in self.data_dims if dim not in xy_dims]
    
        # ensure that spatial dimensions are the first two dimensions for later patching
        data_fcst = self.data_fcst.transpose(*xy_dims, *dims_no_xy,).copy()
        data_ref = self.data_ref.transpose(*xy_dims, *dims_no_xy).copy()

        data_sh = data_fcst.shape    

        # Patching forecast data, 
        # Note:
        # view_as_blocks results into an array with shape: (np, nq, 1, nd, nd, ...) 
        # where np and nq correspond to the number of patches in spatial directions and nd to the patch size (=downscaling_fac)
        data_fcst_patches = np.squeeze(view_as_blocks(data_fcst.values, block_shape=(downscaling_fac, downscaling_fac, *data_sh[2::])))
        # Patching ground truth data
        data_ref_patches = np.squeeze(view_as_blocks(data_ref.values, block_shape=(downscaling_fac, downscaling_fac, *data_sh[2::])))
        
        # Calculate standard deviation for each patch
        std_fcst_patches = np.std(data_fcst_patches, axis=(2, 3))
        std_ref_patches = np.std(data_ref_patches, axis=(2, 3))
        
        # Calculate mean error of the standard deviation over all patches
        mean_error_std = np.mean(np.abs(std_fcst_patches - std_ref_patches), axis=(0, 1))

        # convert back to DataArray
        mean_error_std = xr.DataArray(mean_error_std, coords={dim: data_fcst[dim] for dim in dims_no_xy}, dims=dims_no_xy)

        # apply further averaging if requested
        avg_dims = kwargs.get("non_spatial_avg_dims", None)
        if avg_dims is not None:
            mean_error_std = mean_error_std.mean(dim=avg_dims)
        
        return mean_error_std


    def calc_spatial_variability(self, **kwargs):
        """
        Calculates the ratio between the spatial variability of differental operator with order 1 (or 2) forecast and
        reference data using the calc_geo_spatial-method.
        :param kwargs: 'order' to control the order of spatial differential operator
                       'non_spatial_avg_dims' to add averaging in addition to spatial averaging performed with calc_geo_spatial
        :return: the ratio between spatial variabilty in the forecast and reference data field
        """
        order = kwargs.get("order", 1)
        avg_dims = kwargs.get("non_spatial_avg_dims", None)

        fcst_grad = self.calc_geo_spatial_diff(self.data_fcst, order=order)
        ref_grd = self.calc_geo_spatial_diff(self.data_ref, order=order)

        ratio_spat_variability = (fcst_grad / ref_grd)
        if avg_dims is not None:
            ratio_spat_variability = ratio_spat_variability.mean(dim=avg_dims)

        return ratio_spat_variability
    
    def calc_iqd(self, xnodes=None, nh=0, lfilter_zero=True):
        """
        Calculates squared integrated distance between simulation and observational data.
        Method: Retrieves the empirical CDF, calculates CDF(xnodes) for both data sets and
                then uses the squared differences at xnodes for trapezodial integration.
                Note, that xnodes should be selected in a way, that CDF(xvalues) increases
                more or less continuously by ~0.01 - 0.05 for increasing xnodes-elements
                to ensure accurate integration.

        :param data_simu : 1D-array of simulation data
        :param data_obs  : 1D-array of (corresponding) observational data
        :param xnodes    : x-values used as nodes for integration (optional, automatically set if not given
                           according to stochastic properties of precipitation data)
        :param nh        : accumulation period (affects setting of xnodes)
        :param lfilter_zero: Flag to filter out zero values from CDF calculation
        :return: integrated quadrated distance between CDF of data_simu and data_obs
        """

        data_simu = self.data_fcst.values.flatten()
        data_obs = self.data_ref.values.flatten() 

        # Scarlet: because data_simu and data_obs are flattened anyway this is not needed
        #if np.ndim(data_simu) != 1 or np.ndim(data_obs) != 1:
        #    raise ValueError("Input data arrays must be 1D-arrays.")

        if xnodes is None:
            if nh == 1:
                xnodes = [0., 0.005, 0.01, 0.015, 0.025, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6,
                          0.8, 1., 1.25, 1.5, 1.8, 2.4, 3., 3.75, 4.5, 5.25, 6., 7., 9., 12., 20., 30., 50.]
            elif 1 < nh <= 6:
                ### obtained manually based on observational data between May and July 2017
                ### except for the first step and the highest node-values,
                ### CDF is increased by 0.03 - 0.05 with every step ensuring accurate integration
                xnodes = [0.00, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.15, 1.5, 1.9, 2.4,
                         3., 4., 5., 6., 7.5, 10., 15., 25., 40., 60.]
            else:
                xnodes = [0.00, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1., 1.3, 1.7, 2.1, 2.5,
                          3., 3.5, 4., 4.75, 5.5, 6.3, 7.1, 8., 9., 10., 12.5, 15., 20., 25., 35., 50., 70., 100.]

        data_simu_filt = data_simu[~np.isnan(data_simu)]
        data_obs_filt = data_obs[~np.isnan(data_obs)]
        if lfilter_zero:
            data_simu_filt = np.sort(data_simu_filt[data_simu_filt > 0.])
            data_obs_filt  = np.sort(data_obs_filt[data_obs_filt > 0.])
        else:
            data_simu_filt = np.sort(data_simu_filt)
            data_obs_filt  = np.sort(data_obs_filt)

        nd_points_simu = np.shape(data_simu_filt)[0]
        nd_points_obs  = np.shape(data_obs_filt)[0]

        prob_simu = 1. * np.arange(nd_points_simu)/ (nd_points_simu - 1)
        prob_obs  = 1. * np.arange(nd_points_obs)/ (nd_points_obs -1)

        cdf_simu = self.get_cdf_of_x(data_simu_filt,prob_simu)
        cdf_obs  = self.get_cdf_of_x(data_obs_filt,prob_obs)

        yvals_simu = cdf_simu(xnodes)
        yvals_obs  = cdf_obs(xnodes)

        if yvals_obs[-1] < 0.999:
            print("CDF of last xnodes {0:5.2f} for observation data is smaller than 99.9%." +
                  "Consider setting xnodes manually!")

        if yvals_simu[-1] < 0.999:
            print("CDF of last xnodes {0:5.2f} for simulation data is smaller than 99.9%." +
                  "Consider setting xnodes manually!")

        # finally, perform trapezodial integration
        return np.trapz(np.square(yvals_obs - yvals_simu), xnodes)

    def calc_seeps(self, seeps_weights: xr.DataArray, t1: xr.DataArray, t3: xr.DataArray, spatial_dims: List):
        """
        Calculates stable equitable error in probabiliyt space (SEEPS), see Rodwell et al., 2011
        :param seeps_weights: SEEPS-parameter matrix to weight contingency table elements
        :param t1: threshold for light precipitation events
        :param t3: threshold for strong precipitation events
        :param spatial_dims: list/name of spatial dimensions of the data
        :return seeps skill score (i.e. 1-SEEPS)
        """

        def seeps(data_ref, data_fcst, thr_light, thr_heavy, seeps_weights):
            ob_ind = (data_ref > thr_light).astype(int) + (data_ref >= thr_heavy).astype(int)
            fc_ind = (data_fcst > thr_light).astype(int) + (data_fcst >= thr_heavy).astype(int)
            indices = fc_ind * 3 + ob_ind  # index of each data point in their local 3x3 matrices
            seeps_val = seeps_weights[indices, np.arange(len(indices))]  # pick the right weight for each data point
            
            return 1.-seeps_val
        
        if self.data_fcst.ndim == 3:
            assert len(spatial_dims) == 2, f"Provide two spatial dimensions for three-dimensional data."
            data_fcst, data_ref = self.data_fcst.stack({"xy": spatial_dims}), self.data_ref.stack({"xy": spatial_dims})
            seeps_weights = seeps_weights.stack({"xy": spatial_dims})
            t3 = t3.stack({"xy": spatial_dims})
            lstack = True
        elif self.data_fcst.ndim == 2:
            data_fcst, data_ref = self.data_fcst, self.data_ref
            lstack = False
        else:
            raise ValueError(f"Data must be a two-or-three-dimensional array.")

        # check dimensioning of data
        assert data_fcst.ndim <= 2, f"Data must be one- or two-dimensional, but has {data_fcst.ndim} dimensions. Check if stacking with spatial_dims may help." 

        if data_fcst.ndim == 1:
            seeps_values_all = seeps(data_ref, data_fcst, t1.values, t3, seeps_weights)
        else:
            data_fcst, data_ref = data_fcst.transpose(..., "xy"), data_ref.transpose(..., "xy")
            seeps_values_all = xr.full_like(data_fcst, np.nan)
            seeps_values_all.name = "seeps"
            for it in range(data_ref.shape[0]):
                data_fcst_now, data_ref_now = data_fcst[it, ...], data_ref[it, ...]
                # in case of missing data, skip computation
                if np.all(np.isnan(data_fcst_now)) or np.all(np.isnan(data_ref_now)):
                    continue

                seeps_values_all[it,...] = seeps(data_ref_now, data_fcst_now, t1.values, t3, seeps_weights.values)

        if lstack:
            seeps_values_all = seeps_values_all.unstack()

        seeps_values = seeps_values_all.mean(dim=self.avg_dims)

        return seeps_values

    def calc_geo_spatial_diff(self, scalar_field: xr.DataArray, order: int = 1, r_e: float = 6371.e3, dom_avg: bool = True):
        """
        Calculates the amplitude of the gradient (order=1) or the Laplacian (order=2) of a scalar field given on a regular,
        geographical grid (i.e. dlambda = const. and dphi=const.)
        :param scalar_field: scalar field as data array with latitude and longitude as coordinates
        :param order: order of spatial differential operator
        :param r_e: radius of the sphere
        :return: the amplitude of the gradient/laplacian at each grid point or over the whole domain (see avg_dom)
        """
        method = Scores.calc_geo_spatial_diff.__name__
        # sanity checks
        assert isinstance(scalar_field, xr.DataArray), f"Scalar_field of {method} must be a xarray DataArray."
        assert order in [1, 2], f"Order for {method} must be either 1 or 2."

        dims = list(scalar_field.dims)

        lat_name, lon_name = self.check_for_coords(dims, "lat"), self.check_for_coords(dims, "lon")

        lat, lon = np.deg2rad(scalar_field[lat_name]), np.deg2rad(scalar_field[lon_name])
        dphi, dlambda = lat[1].values - lat[0].values, lon[1].values - lon[0].values

        if order == 1:
            dvar_dlambda = 1. / (r_e * np.cos(lat) * dlambda) * scalar_field.differentiate(lon_name)
            dvar_dphi = 1. / (r_e * dphi) * scalar_field.differentiate(lat_name)
            dvar_dlambda = dvar_dlambda.transpose(*scalar_field.dims)  # ensure that dimension ordering is not changed

            var_diff_amplitude = np.sqrt(dvar_dlambda ** 2 + dvar_dphi ** 2)
            if dom_avg: var_diff_amplitude = var_diff_amplitude.mean(dim=[lat_name, lon_name])
        else:
            raise ValueError(f"Second-order differentation is not implemenetd in {method} yet.")

        return var_diff_amplitude


    def check_for_coords(self, coord_names_data, dim_query: str, return_index: bool = False):
        """
        Check if one of the known geographical coordinates is part of the passed list.
        :param coord_names_data: list of coordinate names
        :param dim_query: dimension to be checked for (either 'lat' or 'lon')
        :param return_index: flag to return the index of the found coordinate instead of name
        :return: index of the first found coordinate and the name of the coordinate
        """
        assert dim_query in ["lat", "lon"], "dim_query must be either 'lat' or 'lon'."

        dim_key = dim_query + "_dims"
        known_geodims = self.known_geodims[dim_key]

        stat = False
        for i, coord in enumerate(known_geodims):
            if coord in coord_names_data:
                stat = True
                break

        if stat:
            return_val = i if return_index else known_geodims[i]
            return return_val
        else:
            raise ValueError("Could not find one of the following coordinates in the passed dictionary: {0}"
                                .format(",".join(known_geodims)))
    @staticmethod
    def get_cdf_of_x(sample_in, prob_in):
        """
        Wrappper for interpolating CDF-value for given data
        :param sample_in : input values to derive discrete CDF
        :param prob_in   : corresponding CDF
        :return: lambda function converting arbitrary input values to corresponding CDF value
        """
        return lambda xin: np.interp(xin, sample_in, prob_in)

