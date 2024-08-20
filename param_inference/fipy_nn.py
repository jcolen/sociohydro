import torch
import torch.nn as nn
import torch.nn.functional as F

import string

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class ParameterNetwork(nn.Module):
    """ Network using learnable linear coefficients and a local NN to predict dynamics """
    def __init__(self, 
                 num_classes=2,
                 num_coefs=7,
                 num_hidden=64,
                 num_layers=2,
                 grid=False,
                 labels=None,
                 features=None):
        """ Initialize the neural network """
        super().__init__()

        self.num_classes = 2
        self.num_coefs = 7
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.grid = grid

        # What do we call the things that we predict
        if labels is None:
            labels = [string.ascii_lowercase[i] for i in range(num_classes)]
        self.labels = labels

        # What do we call the features for each coefficient
        if features is None:
            features = []
            for label in labels:
                features.append([f'{label}_{i}' for i in range(num_coefs)])
        self.features = features

        # Initialize coefficients
        self.coefs = nn.Parameter(
            torch.zeros([self.num_classes, self.num_coefs], dtype=torch.float), 
            requires_grad=True
        )

        # Initialize local neural network
        conv_block = nn.Conv2d if grid else nn.Conv1d
        layers = []
        for i in range(num_layers):
            layers.append(conv_block(
                num_classes if i == 0 else num_hidden,
                num_hidden,
                kernel_size=1
            ))
            layers.append(Sin())

        layers.append(conv_block(num_hidden, num_classes, kernel_size=1))
        self.local_network = nn.Sequential(*layers)
    
    def get_coefs(self):
        """ Subclasses may use activations to apply e.g. positivity constraints"""
        return self.coefs
    
    def print(self, indent=0):
        """ Print the equations represented by the model """
        coefs = self.get_coefs()

        for i in range(self.num_classes):
            eqn_string = f'dt {self.labels[i]} = '
            for j in range(self.num_coefs):
                eqn_string += f'{coefs[i,j]:.2g} {self.features[i][j]} + '
            eqn_string += f'NN({self.labels})'
            for j in range(indent):
                eqn_string = '\t' + eqn_string
            print(eqn_string)
    
    def forward(self, inputs, features, batched=True):
        """ Apply the linear coefficients to features and the neural network to inputs 
            inputs - variables of size [B, Nc, ...]
            features - variables of size [B, Nc, Nf, ...]
            batched - set to False if there is no leading batch dimension
        """

        if not batched:
            inputs = inputs[None]
            features = features[None]

        # Multiply features by coefs
        coefs = self.get_coefs()[None] # [1, Nc, Nf], added leading batch dimension
        feature_terms = torch.einsum('bij...,bij->bij...', features, coefs)
        feature_terms = feature_terms.sum(dim=2) #[B, Nc, ...]

        # Get growth terms from neural network
        growth = self.local_network(inputs)

        output = feature_terms + growth

        if batched:
            return output
        else:
            return output[0]

class SociohydroParameterNetwork(ParameterNetwork):
    """ Extension of parameter network using grouped sociohydrodynamic terms """
    def __init__(self, 
                 num_hidden=64, 
                 num_layers=2, 
                 grid=False):
        """ Using the proposed equations, we always have 7 coefficients per class """
        num_classes = 2
        num_coefs = 7

        labels = ['ϕW', 'ϕB']
        features = [
            ['T_W', 'k_WW', 'k_WB', 'ν_WWW', 'ν_WWB', 'ν_WBB', 'Γ_W'],
            ['T_B', 'k_BB', 'k_BW', 'ν_BBB', 'ν_BWB', 'ν_BWW', 'Γ_B'],
        ]

        super().__init__(
            num_classes=num_classes, 
            num_coefs=num_coefs, 
            num_hidden=num_hidden, 
            num_layers=num_layers, 
            grid=grid,
            labels=labels,
            features=features
        )
    
    def get_coefs(self):
        """ Use activations to apply inequality constraints on selected coefficients """
        coefs = torch.empty_like(self.coefs)
        coefs[:, 0] = self.coefs[:, 0].exp()   # Diffusion must be POSITIVE
        coefs[:, 1:6] = self.coefs[:, 1:6]
        coefs[:, 6] = -self.coefs[:, 6].exp() # Gamma must be NEGATIVE (for stability)
        return coefs