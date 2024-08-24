import os
import torch
import yaml
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pysindy as ps
import seaborn as sns

from pprint import pprint
from sklearn.linear_model import LinearRegression
from argparse import ArgumentParser

from fipy_nn import SociohydroParameterNetwork
from fipy_dataset import FipyDataset
from fvm_utils import plot_mesh


def loss_plot(info, model_dir=None, printout=False):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=150)
    ax.plot(info['train_loss'], label='Train')
    ax.plot(info['val_loss'], label='Validation')
    ax.legend(framealpha=0.)
    ax.set(xlabel='Epoch', ylabel='Loss', yscale='log')
    ax.tick_params(which='both', direction='in')

    if model_dir:
        fig.savefig(f'{model_dir}/loss_curve.png', bbox_inches='tight')


def plot_sample_prediction(mesh, inputs, targets, outputs, feature_terms, growth, vmax=0.03, model_dir=None, printout=False):
    
    fig, ax = plt.subplots(2, 5,
                           sharey=True,
                           sharex=True,
                           constrained_layout=True,
                           dpi=144,
                           figsize=(10, 3))

    plot_mesh(inputs[0], mesh, ax[0, 0],
              cmap=plt.cm.Blues, vmin=0, vmax=1)
    plot_mesh(inputs[1], mesh, ax[1, 0],
              cmap=plt.cm.Reds, vmin=0, vmax=1)
    ax[0, 0].set(ylabel="White", title=r"$\phi_i$")
    ax[1, 0].set(ylabel="Black")

    vmax = np.max([np.abs(targets[0]).max(),
                   np.abs(targets[1]).max()])
    plot_mesh(targets[0], mesh, ax[0, 1],
              cmap=plt.cm.PiYG, vmin=-vmax, vmax=vmax)
    plot_mesh(targets[1], mesh, ax[1, 1],
              cmap=plt.cm.BrBG, vmin=-vmax, vmax=vmax)
    ax[0, 1].set(title=r"$\partial_t \phi_i$")


    plot_mesh(outputs[0], mesh, ax[0, 2],
              cmap=plt.cm.PiYG, vmin=-vmax, vmax=vmax)
    plot_mesh(outputs[1], mesh, ax[1, 2],
              cmap=plt.cm.BrBG, vmin=-vmax, vmax=vmax)
    ax[0, 2].set(title=r"Coefs + NN")

    plot_mesh(feature_terms[0], mesh, ax[0, 3],
              cmap=plt.cm.PiYG, vmin=-vmax, vmax=vmax)
    plot_mesh(feature_terms[1], mesh, ax[1, 3],
              cmap=plt.cm.BrBG, vmin=-vmax, vmax=vmax)
    ax[0, 3].set(title=r"Coefs only")

    plot_mesh(growth[0], mesh, ax[0, 4],
              cmap=plt.cm.PiYG, vmin=-vmax, vmax=vmax)
    plot_mesh(growth[1], mesh, ax[1, 4],
              cmap=plt.cm.BrBG, vmin=-vmax, vmax=vmax)
    ax[0, 4].set(title=r"NN only")

    for a in ax.ravel():
        a.set_aspect(1)
        a.set(xticks=[], yticks=[])

    if model_dir:
        fig.savefig(f'{model_dir}/sample_prediction.png', bbox_inches='tight')


def scatter_plot_single(ax, x, y, title='', s=2, alpha=0.8, lim=0.05):
    
    lim = np.max([np.abs(x).max(),
                  np.abs(y).max()]) * 1.2
    # Scatterplot
    ax.scatter(x[0], y[0], color='steelblue', s=s, alpha=alpha)
    ax.scatter(x[1], y[1], color='firebrick', s=s, alpha=alpha)

    # Linear fits
    xx = np.linspace(-lim, lim, 2)
    fitter = LinearRegression(fit_intercept=False)

    fitter.fit(x[0,:,None], y[0,:,None])
    yy = fitter.predict(xx[:,None])
    ax.plot(xx, yy, color='steelblue',
            label=f'Slope = {fitter.coef_[0,0]:.2g}')

    fitter.fit(x[1,:,None], y[1,:,None])
    yy = fitter.predict(xx[:,None])
    ax.plot(xx, yy, color='firebrick',
            label=f'Slope = {fitter.coef_[0,0]:.2g}')

    # Formatting and legend
    ticks = [-lim, 0, lim]
    ax.set(
        xlim=[-lim, lim],
        xticks=ticks,
        xlabel=r'$\partial_t \phi$ true',
        ylim=[-lim, lim], yticks=ticks,
        title=title, aspect=1
    )
    ax.plot(ticks, ticks, color='gray',
            alpha=0.5, linestyle='--', zorder=-10)
    ax.legend(loc='lower right', handlelength=1, fontsize=8)


def scatter_plot_predictions(targets, outputs, feature_terms, growth, model_dir=None, printout=False):
    fig, ax = plt.subplots(1, 3, dpi=144,
                           sharex=True,
                           sharey=True,
                           constrained_layout=True)

    scatter_plot_single(ax[0], targets, outputs, 'Coefs + NN')
    ax[0].set_ylabel('Model')

    scatter_plot_single(ax[1], targets, feature_terms, 'Coefs only')
    scatter_plot_single(ax[2], targets, growth, 'NN only')

    if model_dir:
        fig.savefig(f'{model_dir}/scatter_predictions.png',
                    bbox_inches='tight')


def plot_growth_model(model, device, N=10, vmax=0.02, model_dir=None, printout=False):
    phi = np.linspace(0, 1, N)
    phiWB = np.stack(np.meshgrid(phi, phi))
    phiWB = phiWB.reshape([2, -1])

    with torch.no_grad():
        x = torch.tensor(phiWB, dtype=torch.float, device=device)
        nn_out = model.local_network(x).cpu().numpy()

    sindy = ps.SINDy(
        optimizer=ps.STLSQ(threshold=1e-2, normalize_columns=True),
        feature_library=ps.PolynomialLibrary(degree=2),
        feature_names=['ϕW', 'ϕB'],
    )
    sindy.fit(x=phiWB.T, x_dot=nn_out.T)
    if printout:
        sindy.print(lhs=['gW', 'gB'])
    sindy_out = sindy.predict(phiWB.T).T

    phiWB = phiWB.reshape([2, N, N])
    nn_out = nn_out.reshape([2, N, N])
    sindy_out = sindy_out.reshape([2, N, N])

    fig, ax = plt.subplots(2, 3, dpi=144,
                           figsize=(6, 3),
                           sharex=True, sharey=True,
                           constrained_layout=True)
    
    c = ax[0,0].pcolormesh(phiWB[0], phiWB[1], nn_out[0], 
                           cmap='coolwarm',
                           vmin=-vmax, vmax=vmax)
    fig.colorbar(c, ax=ax[0,0], ticks=[-vmax, 0, vmax])
    
    c = ax[1,0].pcolormesh(phiWB[0], phiWB[1], nn_out[1], 
                           cmap='coolwarm_r',
                           vmin=-vmax, vmax=vmax)
    fig.colorbar(c, ax=ax[1,0], ticks=[-vmax, 0, vmax])
    ax[0,0].set_title('NN Growth', fontsize=8)

    c = ax[0,1].pcolormesh(phiWB[0], phiWB[1], sindy_out[0], cmap='coolwarm', vmin=-vmax, vmax=vmax)
    fig.colorbar(c, ax=ax[0,1], ticks=[-vmax, 0, vmax])
    
    c = ax[1,1].pcolormesh(phiWB[0], phiWB[1], sindy_out[1], cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
    fig.colorbar(c, ax=ax[1,1], ticks=[-vmax, 0, vmax])
    
    ax[0,1].set_title('SINDy Growth', fontsize=8)

    c = ax[0,2].pcolormesh(phiWB[0], phiWB[1],
                           np.abs(nn_out[0]-sindy_out[0]),
                           cmap='Reds', vmin=0, vmax=vmax)
    fig.colorbar(c, ax=ax[0,2], ticks=[0, vmax])
    
    c = ax[1,2].pcolormesh(phiWB[0], phiWB[1],
                           np.abs(nn_out[1]-sindy_out[1]),
                           cmap='Blues', vmin=0, vmax=vmax)
    fig.colorbar(c, ax=ax[1,2], ticks=[0, vmax])
    
    ax[0,2].set_title('Abs. Error', fontsize=8)

    for a in ax.ravel():
        a.set(
            xlabel=r'$\phi_W$', xlim=[0,1], xticks=[0,1],
            ylabel=r'$\phi_B$', ylim=[0,1], yticks=[0,1],
            aspect=1,
        )

    eqns = sindy.equations()
    title  = f'$g_W = {eqns[0]}$\n'
    title += f'$g_B = {eqns[1]}$'
    fig.suptitle(title, fontsize=8)

    if model_dir:
        fig.savefig(f'{model_dir}/growth_model.png', bbox_inches='tight')


def compare_coefficients(model, config, model_dir=None, printout=False):
    # Load exact parameters
    paramfile = f"./data/{config['county']}_small/{config['county']}_small_NYCinferredParams_params.json"
    with open(paramfile) as f:
        params = json.load(f)

    term_names = [
        r'$T_i$',
        r'$\Gamma_i$',
        r'$\nu_{iii}$',
        r'$\nu_{iij}$',
        r'$\nu_{ijj}$',
        r'$k_{ii}$',
        r'$k_{ij}$'
    ]
    
    coefW = [
        params['tempW'],
        params['gammaW'],
        params['nuWWW'],
        params['nuWWB'],
        params['nuWBB'],
        params['kWW'],
        params['kWB']
    ]

    coefB = [
        params['tempB'],
        params['gammaB'],
        params['nuBBB'],
        params['nuBWB'],
        params['nuBWW'],
        params['kBB'],
        params['kBW']
    ]
    

    # Put everything into a dataframe
    
    W_df = pd.DataFrame({
        'term':  term_names,
        'coef': coefW,
    })
    W_df['target'] = 'White'

    B_df = pd.DataFrame({
        'term':  term_names,
        'coef': coefB,
    }) 
    B_df['target'] = 'Black'

    true_df = pd.concat([W_df, B_df])

    # Collect NN parameters
    with torch.no_grad():
        coefs = model.get_coefs().detach().cpu().numpy()

    # Put everything into a dataframe
    W_df = pd.DataFrame({
        'term':  term_names,
        'coef': -coefs[0, [0, 6, 3, 4, 5, 1, 2]],
    })
    W_df.loc[0, 'coef'] *= -1
    W_df['target'] = 'White'

    B_df = pd.DataFrame({
        'term':  term_names,
        'coef': -coefs[1, [0, 6, 3, 4, 5, 1, 2]],
    })
    B_df.loc[0, 'coef'] *= -1
    B_df['target'] = 'Black'

    ml_df = pd.concat([W_df, B_df])

    #Put it all together and plot
    total_df = pd.concat([true_df, ml_df])

    fig, ax = plt.subplots(1, 1, dpi=144, figsize=(6, 3))
    sns.stripplot(
        data=true_df,
        x='term',
        y='coef',
        hue='target',
        palette=['steelblue', 'firebrick'],
        dodge=True,
        s=10, marker='o',
        ax=ax,
    )
    sns.move_legend(ax, loc='center left', bbox_to_anchor=[1.1, 0.75], 
                    framealpha=0., title='Exact', fontsize=8)

    ax2 = ax.twinx()
    sns.stripplot(
        data=ml_df,
        x='term',
        y='coef',
        hue='target',
        palette=['steelblue', 'firebrick'],
        dodge=True,
        s=10, marker='s',
        ax=ax2,
    )
    sns.move_legend(ax2, loc='center left', bbox_to_anchor=[1.1, 0.25], 
                    framealpha=0., title='Inferred', fontsize=8)


    ax.axhline(0, color='grey', linestyle='--', zorder=-1)
    ax.set(ylim=[-35, 35], yticks=[-30, 0, 30], ylabel='Coefficient', xlabel='Term')
    ylim = np.ceil(np.max(np.abs(ml_df.coef)))
    ax2.set(ylim=[-ylim, ylim], yticks=[-ylim, 0, ylim], ylabel=None);

    # Collect and save information
    joint_df = pd.concat([ml_df.assign(source='Inferred'), true_df.assign(source='Exact')])
    joint_df = joint_df.pivot(columns='source', index=['target', 'term'], values='coef')

    if printout:
        print(joint_df)

    if model_dir:
        fig.savefig(f'{model_dir}/compare_coefficients.png', bbox_inches='tight')
        joint_df.to_csv(f'{model_dir}/compare_coefficients.csv')

    return joint_df


def scatter_plot_coefficients(joint_df, model_dir=None, printout=False):
    xmin, xmax = np.floor(np.min(joint_df.Exact)), np.ceil(np.max(joint_df.Exact))

    fig, ax = plt.subplots(1, 1, dpi=150, figsize=(3, 3))

    for target, c in zip(['White', 'Black'], ['steelblue', 'firebrick']):
        x, y = joint_df.loc[target, 'Exact'], joint_df.loc[target, 'Inferred']
        ax.scatter(x, y, c=c)

        fitter = LinearRegression().fit(x.values[:, None], y.values[:, None])
        m, b = fitter.coef_.squeeze(), fitter.intercept_.squeeze()
        xx = np.array([xmin, xmax])[:, None]
        yy = fitter.predict(xx)
        ax.plot(xx, yy, c=c, label=f'{target}, y = {m:.02f} x + {b:.02f}')
        
    ax.legend(loc='lower right', framealpha=0, fontsize=8, handlelength=1)
    ax.set(xlabel='True coefficients', ylabel='Inferred coefficients')
    ax.tick_params(which='both', direction='in')

    if model_dir:
        fig.savefig(f'{model_dir}/scatter_coefficients.png', bbox_inches='tight')


def load_model(model_dir, printout=True):
    with open(f"{model_dir}/config.yml", "r") as f:
        config = yaml.safe_load(f)

    info = torch.load(f"{model_dir}/model.ckpt", map_location="cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SociohydroParameterNetwork(grid=config["grid"],
                                       grouped=config.get("grouped", False)).to(device)
    model.load_state_dict(info["state_dict"])
    model.eval()
    if printout:
        pprint(config)
        model.print()
    
    return config, info, device, model


def get_growth(model_dir, N=10, save=False):
    config, info, device, model = load_model(model_dir, printout=False)
    phi = np.linspace(0, 1, N)
    phiWB = np.stack(np.meshgrid(phi, phi))
    phiWB = phiWB.reshape([2, -1])

    with torch.no_grad():
        x = torch.tensor(phiWB, dtype=torch.float, device=device)
        nn_out = model.local_network(x).cpu().numpy()

    sindy = ps.SINDy(
        optimizer=ps.STLSQ(threshold=1e-2, normalize_columns=True),
        feature_library=ps.PolynomialLibrary(degree=2),
        feature_names=['ϕW', 'ϕB'],
    )
    sindy.fit(x=phiWB.T, x_dot=nn_out.T)


    if save:
        df = pd.DataFrame({"features":sindy.get_feature_names(),
                           "coeffsW":sindy.coefficients()[0],
                           "coeffsB":sindy.coefficients()[1]})
        df.to_csv(os.path.join(model_dir, "growth_terms.csv"), index=False)

    return sindy


def analysis_pipeline(model_dir,
                      county="Georgia_Fulton",
                      printout=True):

    # load model
    config, info, device, model = load_model(model_dir, printout=printout)
    
    # Generate sample prediction
    dataset = FipyDataset(path=f"./data/{county}_small/fipy_output",
                          grid=config['grid'],
                          remove_extra=False)
    sample = dataset[100]

    with torch.no_grad():
        outputs, (feature_terms, growth)= model(
            sample['inputs'].to(device), sample['features'].to(device), batched=False)

        inputs = sample['inputs'].cpu().numpy()
        targets = sample['targets'].cpu().numpy()
        outputs = outputs.cpu().numpy()
        feature_terms = feature_terms.cpu().numpy()
        growth = growth.cpu().numpy()

    mesh = sample['W0_mesh'].mesh

    # save growth terms
    df = get_growth(model_dir, save=True)

    # Run all plots
    loss_plot(info, model_dir=model_dir)
    
    plot_sample_prediction(mesh, inputs,
                           targets, outputs,
                           feature_terms, growth,
                           model_dir=model_dir,
                           printout=printout)
    
    scatter_plot_predictions(targets, outputs,
                             feature_terms, growth,
                             model_dir=model_dir,
                             printout=printout)
    
    plot_growth_model(model, device,
                      model_dir=model_dir,
                      printout=printout)
    
    joint_df = compare_coefficients(model, config,
                                    model_dir=model_dir,
                                    printout=printout)
    
    scatter_plot_coefficients(joint_df,
                              model_dir=model_dir,
                              printout=printout)

    return joint_df

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='models/Georgia_Fulton_noGrid_SociohydroParameterNetwork_210824_1556')
    parser.add_argument('--printout', action='store_true')

    analysis_pipeline(args.model_dir, printout=args.printout)