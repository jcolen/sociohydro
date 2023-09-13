import numpy as np
import os
import torch
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser

from data_processing import *
from pbnn import *

def pretrain(model, dataset, n_epochs, batch_size, device, savedir='pipeline',
             pretrain_model=None,):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    loss_history = []

    if pretrain_model is None:
        gammas_init = torch.zeros(2, dtype=torch.float, device=device)
    else:
        gammas_init = pretrain_model.gammas.detach()

    print(f'Pretraining with target Gamma = {gammas_init.cpu().numpy()}')

    step = 0
    batch_size = 8

    with tqdm(total=n_epochs*len(dataset)) as ebar:
        for epoch in range(n_epochs):
            d_ad.set_working_tape(d_ad.Tape())
            for i in range(len(dataset)):
                batch = dataset[i]
                batch['wb0'] = batch['wb0'].to(device)
                batch['mask'] = torch.BoolTensor(batch['mask']).to(device)

                with d_ad.stop_annotating(), torch.no_grad():
                    if pretrain_model is None:
                        batch['Dij'] = batch['Dij'].to(device)
                        batch['gammas'] = gammas_init
                    else:
                        Dij, _, gammas = pretrain_model.forward(batch['wb0'][None])
                        batch['Dij'] = Dij
                        batch['gammas'] = gammas

                loss, Dij, gammas = model.pretrain(batch)
                loss_history.append(loss.item())
                step += 1
                ebar.update()

                if step % batch_size == 0:
                    ebar.set_postfix(
                        loss=np.mean(loss_history[-batch_size:]),
                        gammas=gammas)

                    opt.step()
                    d_ad.set_working_tape(d_ad.Tape())
                    opt.zero_grad()


            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'loss_history': loss_history,
                },
                f'{savedir}/{model.__class__.__name__}_pretrain.ckpt')

            sch.step()

def train(model, dataset, n_epochs, batch_size, device, savedir='pipeline'):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    train_loss = []
    val_loss = []
    step = 0
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            for ds in dataset.datasets:
                ds.training()
                
            d_ad.set_working_tape(d_ad.Tape())

            with tqdm(total=len(dataset), leave=False) as ebar:
                for i in range(len(dataset)):
                    batch = dataset[i]
                    batch['wb0'] = batch['wb0'].to(device)

                    Dij, Dij_mesh, constants, J, dJdD = model.training_step(batch)
                    train_loss.append(J)
                    step += 1
                    ebar.update()

                    if step % batch_size == 0:
                        pbar.set_postfix(
                            loss=np.mean(train_loss[-batch_size:]),
                            gammas=constants.detach().cpu().numpy())

                        opt.step()
                        d_ad.set_working_tape(d_ad.Tape())
                        opt.zero_grad()

            for ds in dataset.datasets:
                ds.validate()
            val_loss.append(0)

            with tqdm(total=len(dataset), leave=False) as ebar:
                with torch.no_grad():
                    for i in range(len(dataset)):
                        d_ad.set_working_tape(d_ad.Tape())

                        batch = dataset[i]
                        batch['wb0'] = batch['wb0'].to(device)

                        Dij, Dij_mesh, constants, J = model.validation_step(batch)
                        val_loss[epoch] += J
                        ebar.update()


            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                },
                f'{savedir}/{model.__class__.__name__}.ckpt')

            sch.step()
            pbar.update()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--county', nargs='+', default=['cook_IL'])
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--housing_method', choices=['constant', 'varying'])
    parser.add_argument('--savedir', type=str, default='.')
    args = parser.parse_args()

    with open(f'{args.savedir}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    datasets = []
    for county in args.county:
        datasets.append(CensusDataset(county, housing_method=args.housing_method))
    dataset = torch.utils.data.ConcatDataset(datasets)

    '''
    Step 1: PBNN with only diagonal diffusion coefficients
    '''
    model1 = DiagonalOnlyPBNN().to(device)

    # Pretrain model
    pretrain(model1, dataset, args.pretrain_epochs, args.batch_size, device, args.savedir,
             pretrain_model=None)

    # Train diagonal PBNN
    train(model1, dataset, args.n_epochs, args.batch_size, device, args.savedir)

    '''
    Step 2: PBNN with symmetric off-diagonal diffusion coefficients
    '''
    model2 = SymmetricCrossDiffusionPBNN().to(device)
    model2.load_state_dict(model1.state_dict())

    train(model2, dataset, args.n_epochs, args.batch_size, device, args.savedir)

    '''
    Step 3: Full PBNN with off-diagonal coefficients
    '''
    model3 = PBNN().to(device)

    # Pretrain to match reciprocal PBNN
    pretrain(model3, dataset, args.pretrain_epochs, args.batch_size, device, args.savedir,
             pretrain_model=model2)

    train(model3, dataset, args.n_epochs, args.batch_size, device, args.savedir)




