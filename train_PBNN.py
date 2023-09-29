import numpy as np
import os
import torch
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser

from data_processing import *
from pbnn import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--county', nargs='+', default=['cook_IL'])
    parser.add_argument('--n_epochs', type=int, default=250)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--modeltype', type=str, default='PBNN')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--savedir', type=str, default='.')
    parser.add_argument('--housing_method', type=str, default='constant')
    args = parser.parse_args()

    with open(f'{args.savedir}/{args.modeltype}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    datasets = []
    for county in args.county:
        datasets.append(CensusDataset(county, housing_method=args.housing_method))
    dataset = torch.utils.data.ConcatDataset(datasets)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(args.modeltype)().to(device)
    
    if args.load_model:
        info = torch.load(args.load_model, map_location='cpu')
        model.load_state_dict(info['state_dict'])
    
    with torch.autograd.set_detect_anomaly(True):
        train(model, dataset, args.n_epochs, args.batch_size, device, args.savedir)

