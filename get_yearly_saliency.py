import numpy as np
import h5py
import torch
import json
from tqdm.auto import tqdm

from data_processing import *
from yearly_pbnn import *
from sys import argv

if __name__ == '__main__':
    directory = argv[1]
    print(f'Processing directory {directory}')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SourcedOnlyPBNN().to(device)
    info = torch.load(f'{directory}/SourcedOnlyPBNN.ckpt')
    model.load_state_dict(info['state_dict'])
    
    with open(f'{directory}/{model.__class__.__name__}_args.txt', 'r') as f:
        args = json.load(f)
        
    for county in args['val_county']:
        
        if county == 'California_San Bernardino':
            print('Skipping San Bernardino!')
            continue
        
        dataset = YearlyDataset(county, housing_method=args['housing_method'])
        dataset.validate()
        
        compute_saliency(model, dataset, device,
                         savename=f'{directory}/{model.__class__.__name__}')
        
        
