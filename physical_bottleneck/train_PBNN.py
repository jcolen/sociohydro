import numpy as np
import os
import torch
from tqdm.auto import tqdm
from argparse import ArgumentParser

from data_processing import *
from pbnn import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--county', nargs='+', default=['cook_IL'])
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--modeltype', type=str, default='PBNN')
    parser.add_argument('--pretrain_model', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--stationary', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--savedir', type=str, default='.')
    parser.add_argument('--housing_method', type=str, default='constant')
    args = parser.parse_args()

    datasets = []
    for county in args.county:
        if args.stationary:
            datasets.append(StationaryDataset(county, housing_method=args.housing_method))
        else:
            datasets.append(CensusDataset(county, housing_method=args.housing_method))
    dataset = torch.utils.data.ConcatDataset(datasets)
    
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(args.modeltype)().to(device)
    
    if args.load_model:
        info = torch.load(args.load_model, map_location='cpu')
        model.load_state_dict(info['state_dict'])
    
    if args.pretrain:
        if args.pretrain_model:
            info = torch.load(args.pretrain_model, map_location='cpu')
            modeltype = os.path.basename(args.pretrain_model)
            modeltype = modeltype.split('.')[0]
            print(f'Pretraining to a {modeltype}')
            pretrain_model = eval(modeltype)().to(device)
            pretrain_model.load_state_dict(info['state_dict'])
        else:
            pretrain_model = None
        
        pretrain(model, dataset, args.pretrain_epochs, args.batch_size, device, args.savedir, pretrain_model)
    
    train(model, dataset, args.n_epochs, args.batch_size, device, args.savedir)

