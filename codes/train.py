 # For data
from load_data import SignalDataset, CWTDataset, STFTDataset, get_dataset, standard_normalization, minmax_normalization, both_normalization
from load_data import get_dataset
from train_options import get_opt
# For model
from models.Xception2D import xception as xception2d
from models.Xception1D import xception as xception1d

import torch
import torch.nn as nn
import random, os, ast, json, tqdm, visdom
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

seed = 99
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

def get_model(model_config, input_channel, num_classes):
    model_name = model_config['model_name']
    exec('list_kernel_size=' + model_config['list_kernel_size'], globals())
    exec('list_strides=' + model_config['list_strides'], globals())
    act_fn = model_config['act_fn']
    bn = model_config['bn']
    if model_name == 'xception1d':
        model = xception1d(input_channel = input_channel, num_classes = num_classes, act_fn = act_fn, bn = bn, list_kernel_size = list_kernel_size, list_strides = list_strides)
    elif model_name == 'xception2d':
        model = xception2d(input_channel = input_channel, num_classes = num_classes, act_fn = act_fn, bn = bn, list_kernel_size = list_kernel_size, list_strides = list_strides)
    return model


def main():
    # loading reference files
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    df_dir = os.path.join(base_dir, 'input/physionet.org/files/ptb-xl/1.0.1')

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
    valid_index = opt.val_index
    
    df_val = df[df['strat_fold'] == valid_index]
    df_train = df[df['strat_fold'] != valid_index]
    
    dataset_train = get_dataset(opt, df_train, df_scp, df_dir)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, num_workers = num_workers, batch_size = batch_size, pin_memory = True, shuffle = True)
    
    dataset_val = get_dataset(opt, df_val, df_scp, df_dir)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, num_workers = num_workers, batch_size = batch_size, pin_memory = True, shuffle = False)
        
    # Get model
    d = dataset_train[0]
    input_channel = d[0].shape[0]
    num_classes = d[1].shape[0]
    del d
    model_config_name = opt.model_config_name
    model_config = json.load(open(os.path.join(base_dir, model_config_name), 'r'))
    model = get_model(model_config, input_channel, num_classes)
           
    # Train
    device = train_opt.device
    EPOCH = train_opt.EPOCH
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    list_train_loss, list_train_auc = [], []
    list_val_loss, list_val_auc = [], []
    list_class_auc = []
    
    # Make a temp file
    os.makedirs('./tmps', exist_ok = True)
    import tempfile
    temp_dir = tempfile.TemporaryDirectory(suffix=None, prefix=None, dir='./tmps')
    temp_dir = temp_dir.name
    os.makedirs(temp_dir)
    
    json.dump(vars(opt), open(os.path.join(temp_dir, 'options.json'), 'w'))

    for epoch in range(EPOCH):
        # train
        list_real, list_pred, temp_loss = [], [], []
        pbar = tqdm.tqdm(total = len(dataloader_train), position = 0, desc = f'Train Epoch: {epoch + 1}/{EPOCH}')
        model.train()
        for data, label in dataloader_train:
            data = data.to(device)
            label = label.to(device)
            
            model.zero_grad()
            pred = model(data)
            
            loss = criterion(torch.sigmoid(pred), label)
            loss.backward()
            optimizer.step()
            
            list_real.append(label.detach().cpu().numpy())
            list_pred.append(pred.detach().cpu().numpy())
            temp_loss.append(loss.item())
            pbar.update(1)
            pbar.set_postfix({'Loss': np.mean(temp_loss)})
        list_real = np.concatenate(list_real, axis = 0)
        list_pred = np.concatenate(list_pred, axis = 0)
        list_train_loss.append(np.mean(temp_loss))
        list_train_auc.append(roc_auc_score(list_real, list_pred))
        pbar.set_postfix({'Loss': list_train_loss[-1], 'TRAIN_AUC': list_train_auc[-1]})
        pbar.close()
        
        # test
        list_real, list_pred, temp_loss = [], [], []
        pbar = tqdm.tqdm(total = len(dataloader_val), position = 0, desc = f'Val Epoch: {epoch + 1}/{EPOCH}')
        model.eval()
        with torch.no_grad():
            for data, label in dataloader_val:
                data = data.to(device)
                label = label.to(device)
                
                pred = model(data)
                loss = criterion(torch.sigmoid(pred), label)
                list_real.append(label.detach().cpu().numpy())
                list_pred.append(pred.detach().cpu().numpy())
                temp_loss.append(loss.item())
                pbar.update(1)
                pbar.set_postfix({'Loss': np.mean(temp_loss)})
        list_real = np.concatenate(list_real, axis = 0)
        list_pred = np.concatenate(list_pred, axis = 0)
        list_val_loss.append(np.mean(temp_loss))
        list_val_auc.append(roc_auc_score(list_real, list_pred))
        pbar.set_postfix({'Loss': list_val_loss[-1], 'VAL_AUC': list_val_auc[-1]})
        pbar.close()
        # save result for best model
        list_aucs = []
        classes = dataset_train.mlb.classes_
        for r,p in zip(list_real.transpose(), list_pred.transpose()):
            fpr, tpr, _ = roc_curve(r, p)
            auc_score = auc(fpr, tpr)
            list_aucs.append(auc_score)
        list_class_auc.append(list_aucs)
        result_df1 = pd.DataFrame(list_class_auc)
        result_df1.columns = classes
        result_df2 = pd.DataFrame({'list_train_loss': list_train_loss, 'list_val_loss': list_val_loss, 'list_train_auc': list_train_auc, 'list_val_auc': list_val_auc})
        result_df = pd.concat([result_df2, result_df1], axis = 1)
        result_df.to_csv(os.path.join(temp_dir, 'result_df.csv'))
        if list_val_loss[-1] == np.min(list_val_loss):
            # save weight
            torch.save(model.state_dict(), os.path.join(temp_dir, f'model.pth'))
            torch.save(model.features.state_dict(), os.path.join(temp_dir, f'features.pth'))
            print(f"Best model saved at Loss {list_val_loss[-1]:2f} in {os.path.join(temp_dir, f'model.pth')}")
    
if __name__ == '__main__':
    main()