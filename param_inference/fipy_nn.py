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
                 grouped=False,
                 grid=False,
                 labels=None,
                 features=None):
        """ Initialize the neural network """
        super().__init__()

        self.num_classes = 2
        self.num_coefs = 7
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.grouped = grouped
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
        kwargs = {'kernel_size': 1, 'groups': num_classes if grouped else 1}
        layers = []
        for i in range(num_layers):
            layers.append(conv_block(
                num_classes if i == 0 else num_hidden,
                num_hidden,
                **kwargs
            ))
            layers.append(Sin())

        layers.append(conv_block(num_hidden, num_classes, **kwargs))
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
            return output, (feature_terms, growth)
        else:
            return output[0], (feature_terms[0], growth[0])

class SociohydroParameterNetwork(ParameterNetwork):
    """ Extension of parameter network using grouped sociohydrodynamic terms """
    def __init__(self, 
                 num_hidden=64, 
                 num_layers=2,
                 grouped=False,
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
            grouped=grouped,
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

class LocalCoefficientsParameterNetwork(ParameterNetwork):
    """ Extension of parameter network to allow coefficients that are local functions of phi """
    def __init__(self, 
                 num_coefs=7,
                 num_hidden=64,
                 num_layers=2,
                 grouped=False,
                 grid=False,
                 features=None):
        """ You may want to hardcode features depending on what this expects to get
            from fvm_utils.calc_gradients
            I've hardcoded num_classes to allow for gradient computation
        """
        num_classes = 2
        labels = ['ϕW', 'ϕB']

        super().__init__(
            num_classes=num_classes, 
            num_coefs=num_coefs, 
            num_hidden=num_hidden, 
            num_layers=num_layers,
            grouped=grouped,
            grid=grid,
            labels=labels,
            features=features
        )

        # Replace the last layer to allow multiple local functions of inputs
        conv_block = nn.Conv2d if grid else nn.Conv1d
        kwargs = {'kernel_size': 1, 'groups': num_classes if grouped else 1}
        last_layer = conv_block(num_hidden, num_classes * (num_coefs + 1), **kwargs)

        local_layers = list(self.local_network.children())[:-1]
        self.local_network = nn.Sequential(*local_layers, last_layer)

    def forward(self, inputs, features, batched=False):
        """ Apply the local coefficients to features and the neural network to inputs
            inputs - variables of size [B, Nc, ...]
            features - variables of size [B, Nc, Nf, ...]
            batched - set to False if there is no leading batch dimension
        """
    
        if not batched:
            inputs = inputs[None]
            features = features[None]

        # Get shape
        b = inputs.shape[0]
        c = inputs.shape[1]
        f = features.shape[2]
        hwl = inputs.shape[2:]

        # Ensure inputs require grad so we can take a gradient
        ϕW = inputs[:, 0].clone().detach().requires_grad_(True)
        ϕB = inputs[:, 1].clone().detach().requires_grad_(True)
        inp = torch.stack([ϕW, ϕB], dim=1)

        # Compute terms and coefs as local features
        local_features = self.local_network(inp) #[b, c*(f+1), ...]
        local_features = local_features.reshape([b, c, f+1, *hwl])

        coefs = local_features[:, :, :-1] #[b, c, f, ...]
        growth = local_features[:, :, -1] #[b, c, ...]

        # Compute second derivative of coefs w.r.t. phi
        def grad_func(output, input, **kwargs):
            grad_outputs = torch.ones_like(input)
            kwargs = dict(grad_outputs=grad_outputs, **kwargs)

            grad = []
            for cc in range(output.shape[1]):
                gcc = []
                for ff in range(output.shape[2]):
                    gcc.append(torch.autograd.grad(output[:, cc, ff], input, **kwargs)[0])
                grad.append(torch.stack(gcc, dim=1))
            
            grad = torch.stack(grad, dim=1)
            return grad

        c_W = grad_func(coefs, ϕW, create_graph=True)
        c_B = grad_func(coefs, ϕB, create_graph=True)
        c_WW = grad_func(c_W, ϕW, retain_graph=True)
        c_WB = grad_func(c_W, ϕB, retain_graph=True)
        c_BW = grad_func(c_B, ϕW, retain_graph=True)
        c_BB = grad_func(c_B, ϕB, retain_graph=True)

        # Aggregate feature terms
        feature_terms = coefs * features
        feature_terms = feature_terms.sum(dim=2) #[b, c, ...]

        output = feature_terms * growth

        if batched:
            return output, (feature_terms, growth)
        else:
            return output[0], (feature_terms[0], growth[0])