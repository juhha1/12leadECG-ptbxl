import argparse, os, ast, json
import pandas as pd
from argparse import Namespace
from load_data import get_dataset
from models.Xception2D import xception as xception2d

import torch

def get_opt():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #############
    # For training
    parser.add_argument('--num_workers', type = int, default = 8, dest = 'num_workers',
                        help = 'Number of workers for dataloader multi-processing (default: 8)')
    parser.add_argument('--batch_size', type = int, default = 8, dest = 'batch_size',
                        help = 'Batch size (default: 8)')
    parser.add_argument('--device', type = str, default ='cuda:0', dest = 'device',
                        help = 'Torch Device (default: cuda:0)')
    parser.add_argument('--val_index', type = int, default = 1, dest = 'val_index',
                        help = 'Validation index # between 1-10 (default 1)')
    parser.add_argument('--epoch', type = int, default = 30, dest = 'EPOCH',
                        help = 'EPOCH (default: 30)')
    #############
    # For model
    # model configuration json file
    parser.add_argument('--model_config_name', type = str, default = None, dest = 'model_config_name',
                        help = 'Filename for model config json file')
    # for base configuration
    parser.add_argument('--save_dir', type = str, default = None, dest = 'save_dir',
                        help = 'Save directory')
    ############
    # For data
    parser.add_argument('--target_label', type = str, default = 'diagnostic', dest = 'target_label',
                        help = 'Target class (diagnostic, form, rhythm, diagnostic_class, diagnostic_subclass, all), (default: diagnostic)')
    # for all signal, stft, cwt
    parser.add_argument('--data_type', type = str, default = 'signal', dest = 'data_type',
                        help = 'Data type (signal, stft, cwt), (default: signal)')
    parser.add_argument('--fs', type = int, default = 100, dest = 'fs',
                        help = 'sampling frequency (100, 500), (default: 500)')
    parser.add_argument('--upsampling_factor', type = int, default = 1, dest = 'upsampling_factor',
                        help = 'resampling (upsampling) factor for signal (default: 1)')
    parser.add_argument('--sig_scaling', type = str, default = None, dest = 'sig_scaling',
                        help = 'signal scaling (minmax, standard, both, None)')
    # for stft
    parser.add_argument('--nperseg', type = int, default = 128, dest = 'nperseg',
                        help = 'nperseg for STFT (default: 128)')
    parser.add_argument('--t_len', type = int, default = 500, dest = 't_len',
                        help = 'time-length for STFT (default: 500)')
    parser.add_argument('--stft_scaling', type = str, default = None, dest = 'stft_scaling',
                        help = 'signal scaling (min_max, standard, both, None)')
    # for cwt
    parser.add_argument('--width_min', type = float, default = 0.1, dest = 'width_min',
                        help = 'minimum width (default: 0.1)')
    parser.add_argument('--width_max', type = float, default = 30.1, dest = 'width_max',
                        help = 'maximum width (default: 30.1)')
    parser.add_argument('--width_inc', type = float, default = 0.1, dest = 'width_inc',
                        help = 'width increment unit (default: 0.1)')
    parser.add_argument('--wavelet', type = str, default = 'ricker', dest = 'wavelet_type',
                        help = 'wavelet type (ricker, morlet2), (default: ricker)')
    parser.add_argument('--cwt_scaling', type = str, default = None, dest = 'cwt_scaling',
                        help = 'signal scaling (min_max, standard, both, None)')
    parser.add_argument('--cwt_resize', type = bool, default = False, dest = 'cwt_resize',
                        help = 'cwt resize or not? (default: False)')
    parser.add_argument('--cwt_width', type = int, default = 1000, dest = 'cwt_width',
                        help = 'cwt resize width (default: 1000)')
    parser.add_argument('--cwt_height', type = int, default = 300, dest = 'cwt_height',
                        help = 'cwt resize height (default: 300)')
    config, _ = parser.parse_known_args()
    return config


def main():
    # loading reference files
    df_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input/physionet.org/files/ptb-xl/1.0.1')

    df = pd.read_csv(os.path.join(df_dir, 'ptbxl_database.csv'))
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    df['label'] = df['scp_codes'].apply(lambda x: set(x.keys()))
    df_scp = pd.read_csv(os.path.join(df_dir, 'scp_statements.csv'))
    df_scp.index = df_scp['Unnamed: 0'].values
    df_scp = df_scp.iloc[:,1:]
    df_scp['diagnostic'] = [idx_code if val == 1 else None for idx_code, val in zip(df_scp.index, df_scp['diagnostic'].values)]
    df_scp['all'] = df_scp.index
    df_val = df[df['strat_fold'] == 1]
    df_train = df[df['strat_fold'] != 1]
    
    # loading configurations
    opt = get_opt()
    
    # Get dataset and dataloader
    num_workers = opt.num_workers
    batch_size = opt.batch_size
    gpu_id = opt.gpu_id
    valid_index = opt.val_index
    
    df_val = df[df['strat_fold'] == valid_index]
    df_train = df[df['strat_fold'] != valid_index]
    
    dataset_train = get_dataset(opt, df_train, df_scp, df_dir)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, num_workers = num_workers, batch_size = batch_size, pin_memory = True, shuffle = True)
    
    dataset_test = get_dataset(opt, df_val, df_scp, df_dir)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, num_workers = num_workers, batch_size = batch_size, pin_memory = True, shuffle = False)

    # Get model
    d = dataset_train[0]
    input_channel = d[0].shape[0]
    num_classes = d[1].shape[0]
    del d

    print(len(dataset_train))
    print(dataset_train[0][0].shape)
    

if __name__ == '__main__':
    main()