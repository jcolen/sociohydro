import numpy as np
import os
import torch
from tqdm.auto import tqdm
from argparse import ArgumentParser

from data_processing import *
from pbnn import *

def pretrain(model, dataset, n_epochs, batch_size, device, pretrain_model=None, savedir='dynamic'):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    loss_history = []

    if pretrain_model is None:
        gammas_init = torch.zeros(2, dtype=torch.float, device=device)
    else:
        gammas_init = pretrain_model.gammas.detach()
      
    print(f'Pretraining with target Gamma = {gammas_init.cpu().numpy()}')

    n_epochs = 100
    step = 0
    batch_size = 8

    with tqdm(total=n_epochs) as pbar, \
         tqdm(total=n_epochs*len(dataset)) as ebar:
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
                    pbar.set_postfix(
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
            pbar.update()
            
def train(model, dataset, n_epochs, batch_size, device, savedir='dynamic'):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    train_loss = []
    val_loss = []
    step = 0
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            dataset.training()
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

            dataset.validate()
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
    parser.add_argument('--county', type=str, default='cook_IL')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--modeltype', type=str, default='PBNN')
    parser.add_argument('--pretrain_model', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--stationary', action='store_true')
    parser.add_argument('--no_pretrain', action='store_true')
    args = parser.parse_args()
    
    if args.dynamic or not args.stationary:
        dataset = CensusDataset(args.county)
        savedir = 'dynamic'
    elif args.stationary:
        dataset = StationaryDataset(args.county)
        savedir = 'stationary'
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(args.modeltype)().to(device)
    
    if args.load_model:
        info = torch.load(args.load_model, map_location='cpu')
        model.load_state_dict(info['state_dict'])
    
    if not args.no_pretrain:
        if args.pretrain_model:
            info = torch.load(args.pretrain_model, map_location='cpu')
            modeltype = os.path.basename(args.pretrain_model)
            modeltype = modeltype.split('.')[0]
            print(f'Pretraining to a {modeltype}')
            pretrain_model = eval(modeltype)().to(device)
            pretrain_model.load_state_dict(info['state_dict'])
        else:
            pretrain_model = None
        
        pretrain(model, dataset, args.n_epochs, args.batch_size, device, pretrain_model, savedir)
    
    train(model, dataset, args.n_epochs, args.batch_size, device, savedir)

