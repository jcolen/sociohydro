import torch
import torch.nn as nn

'''
Building blocks
'''
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding='same', padding_mode='replicate', groups=in_channels)
        self.conv2 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(4*out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sin(x)
        x = self.conv3(x)
        return x

'''
Neural ODE forecasters
'''
class Conv_ODEFunc(nn.Module):
    def __init__(self, n_input=2, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvNextBlock(2, n_hidden),
            ConvNextBlock(n_hidden, n_hidden),
            nn.Conv2d(n_hidden, 2, kernel_size=1)
        )

    def forward(self, t, y):
        return self.net(y)

class FCN_ODEFunc(nn.Module):
    def __init__(self, n_input=1380, n_hidden=2048,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            Sin(),
            nn.Linear(n_hidden, n_hidden),
            Sin(),
            nn.Linear(n_hidden, n_input)
        )

    def forward(self, t, y):
        b, c, h, w = y.shape
        y = y.reshape([b, -1])
        dt = self.net(y)
        return dt.reshape([b, c, h, w])

'''
ConvNext forecasters
'''

class ConvNN(nn.Module):
    def __init__(self, n_input=2, n_hidden=128, n_steps=4):
        '''
        n_steps = number of steps per time step		
        '''
        super().__init__()

        self.dt_net = Conv_ODEFunc(n_input, n_hidden)
        self.n_steps = n_steps

    def forward(self, t, y):
        yp = torch.zeros([t.shape[0], *y.shape], dtype=y.dtype, device=y.device)
        for i in range(len(t)):
            yp[i] = y
            for j in range(self.n_steps):
                y = y + self.dt_net(t[i], y)

        yp[-1] = y
        return yp

class FCN(nn.Module):
    def __init__(self, n_input=1380, n_hidden=4096, n_steps=4):
        super().__init__()
        self.dt_net = FCN_ODEFunc(n_input, n_hidden)
        self.n_steps = n_steps

    def forward(self, t, y):
        yp = torch.zeros([t.shape[0], *y.shape], dtype=y.dtype, device=y.device)
        for i in range(len(t)):
            yp[i] = y
            for j in range(self.n_steps):
                y = y + self.dt_net(t[i], y)

        yp[-1] = y
        return yp

class ConvLSTM(nn.Module):
    def __init__(self, n_input=2, n_hidden=128, n_steps=4):
        super().__init__()
        self.conv = nn.Sequential(
            ConvNextBlock(2, n_hidden, kernel_size=7),
            ConvNextBlock(n_hidden, n_hidden, kernel_size=7),
            ConvNextBlock(n_hidden, n_hidden, kernel_size=7),
        )
        self.lstm = nn.LSTM(input_size=n_hidden,
                           proj_size=2,
                           hidden_size=n_hidden,
                           num_layers=2,
                           batch_first=True)
        self.n_hidden = n_hidden
        self.n_steps = n_steps

    def forward(self, t, y):
        Nt = t.shape[0]
        b, c, h, w = y.shape
        yp = torch.zeros([Nt, b, c, h, w], dtype=y.dtype, device=y.device)

        hn = torch.zeros([2, b*h*w, 2], dtype=y.dtype, device=y.device)
        cn = torch.zeros([2, b*h*w, self.n_hidden], dtype=y.dtype, device=y.device)

        for i in range(len(t)):
            yp[i] = y

            for i in range(self.n_steps):
                #Update local state using neighbor information
                dy = self.conv(y)

                #Update local state using LSTM
                dy = dy.permute(0, 2, 3, 1).reshape([b*h*w, 1, self.n_hidden])
                dy, (hn, cn) = self.lstm(dy, (hn, cn))
                dy = dy.reshape([b, h, w, c]).permute(0, 3, 1, 2)

                #Update with residual rule
                y = y + dy

        yp[-1] = y
        return yp