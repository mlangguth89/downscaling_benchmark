'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data

from torch.utils.data import DataLoader
from .dataset_prep import PrecipDatasetInter

def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset


# def create_dataset_precp(file_path = "/Users/gongbing/PycharmProjects/downscaling_maelstrom/data"):
#     from data.dataset_prep import PrecipDatasetInter
#     data_loader = PrecipDatasetInter(file_path = file_path)
#     return data_loader




def create_loader_prep(file_path: str = None, batch_size: int = 4, patch_size: int = 16,
                 vars_in: list = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in","yw_hourly_in"],
                 #vars_in: list = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in","u700_in","v700_in","yw_hourly_in"],
                 var_out: list = ["yw_hourly_tar"], sf: int = 10,
                 seed: int = 1234, k: float = 0.01, mode: str = "train", stat_path: str = None):

    """
    file_path : the path to the directory of .nc files
    vars_in   : the list contains the input variable namsaes
    var_out   : the list contains the output variable name
    batch_size: the number of samples per iteration
    patch_size: the patch size for low-resolution image,
                the corresponding high-resolution patch size should be muliply by scale factor (sf)
    sf        : the scaling factor from low-resolution to high-resolution
    seed      : specify a seed so that we can generate the same random index for shuffle function
    """

    dataset = PrecipDatasetInter(file_path, batch_size, patch_size, vars_in, var_out, sf, seed, k, mode, stat_path)

    return DataLoader(dataset) # , batch_size


