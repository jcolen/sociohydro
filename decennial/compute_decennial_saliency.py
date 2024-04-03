import numpy as np
import os
import glob
import torch
import json
from pprint import pprint
from argparse import ArgumentParser

from census_dataset import *
from census_pbnn import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_id', type=str, default='revision_v3')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CensusPBNN().to(device)

    savename = f'models/{model.__class__.__name__}_{args.model_id}'
    info = torch.load(f'{savename}.ckpt')
    model.load_state_dict(info['state_dict'])

    with open(f'{savename}_args.txt', 'r') as f:
        params = json.load(f)
    pprint(params)
    
    with torch.autograd.set_detect_anomaly(True):
        for county in params['val_county']:
            dataset = CensusDataset(county)
            compute_saliency(model, dataset, device, savename)
