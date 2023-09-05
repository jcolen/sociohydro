import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import h5py
from tqdm.auto import tqdm
import time

from convnext_models import *
from data_processing import *

from argparse import ArgumentParser

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['conv', 'fcn'], default='conv')
    parser.add_argument('--n_steps', type=int, default=10000)
    args = parser.parse_args()

    dataset = SociohydrodynamicsDataset(test_bbox=[20, 53, 20, 30])
    device = torch.device('cuda:0')

    if args.model == 'conv':
        model_kwargs = dict(n_input=2, n_hidden=128)
        model = Conv_ODEFunc(**model_kwargs).to(device)
    else:
        model_kwargs = dict(n_input=1200, n_hidden=2000)
        model = FCN_ODEFunc(**model_kwargs).to(device)

    print(f'Training a {model.__class__.__name__}', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loader = iter(dataset)

    end = time.time()
    ode_kwargs = dict(method='rk4', options=dict(step_size=0.1))
    ode_kwargs = dict(atol=1e-5, rtol=1e-5)

    best_loss = 1e10
    for step in range(args.n_steps):
        optimizer.zero_grad()
        batch = dataset.test_batch
        t = batch['t'].to(device)
        wb = torch.stack([batch['w'], batch['b']], dim=-3).to(device)
        wb_pred = odeint(model, wb[0:1], t, **ode_kwargs).squeeze()
        loss = F.l1_loss(wb, wb_pred)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f'Iter {step:04d} | Test Loss: {loss.item():.6g}\tTime: {time.time() - end}', flush=True)
            if loss.item() < best_loss:
                torch.save(dict(
                    hparams=model_kwargs,
                    state_dict=model.state_dict(),
                    loss=loss.item()),
                f'data/{model.__class__.__name__}.ckpt')

                best_loss = loss.item()

        end = time.time()
