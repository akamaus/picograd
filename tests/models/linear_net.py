import torch


class LinearNet(torch.nn.Module):
    def __init__(self, inp_chans, out_chans):
        self.meta_parameters = {'name': 'LinearNet',
                                'args': {'inp_chans': inp_chans, 'out_chans': out_chans}
                               }
        super().__init__()
        self.layer = torch.nn.Linear(inp_chans, out_chans)

    def forward(self, x):
        return self.layer(x)
