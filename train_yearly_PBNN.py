import numpy as np
import os
import glob
import torch
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser

from data_processing import *
from yearly_pbnn import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=250)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--modeltype', type=str, default='SourcedOnlyPBNN')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--savename', type=str, default='./yearly/SourcedOnlyPBNN')
    parser.add_argument('--housing_method', type=str, default='constant')
    args = parser.parse_args()
    
    counties = [os.path.basename(c)[:-9] for c in glob.glob('yearly/processed/*_mesh.xml')]
    N = len(counties)
    Nc = len(counties) * 3 // 5
    train_counties = list(np.random.choice(counties, Nc))
    val_counties = [c for c in counties if not c in train_counties]
    print('Training on ', train_counties, flush=True)
    print('Validating on ', val_counties, flush=True)
    args.__dict__['train_county'] = train_counties
    args.__dict__['val_county'] = val_counties

    with open(f'{args.savename}_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train_datasets = []
    for county in args.train_county:
        if county == 'California_San Bernardino':
            print('Skipping San Bernardino in Training dataset')
            continue
        train_datasets.append(YearlyDataset(county, housing_method=args.housing_method))
        train_datasets[-1].training()
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        
    val_datasets = []
    for county in args.val_county:
        if county == 'California_San Bernardino':
            print('Skipping San Bernardino in Validation dataset')
            continue
        val_datasets.append(YearlyDataset(county, housing_method=args.housing_method))
        val_datasets[-1].validate()
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(args.modeltype)().to(device)
    
    if args.load_model:
        info = torch.load(args.load_model, map_location='cpu')
        model.load_state_dict(info['state_dict'])
    
    with torch.autograd.set_detect_anomaly(True):
        train(model, train_dataset, val_dataset, 
              args.n_epochs, args.batch_size, device, args.savename)
        
        for dataset in val_dataset.datasets:
            compute_saliency(model, dataset, device, args.savename)
