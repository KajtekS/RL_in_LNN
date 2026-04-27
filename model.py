import torch.nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class Model(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, units: int):
        '''
        :param input_size: input size of given tensor
        :param output_size: output of our model
        :param units: units used in model

        Creates using NCPS LNN.
        '''

        super().__init__()
        self.wiring = AutoNCP(units, output_size)
        self.lnn = CfC(input_size, self.wiring, batch_first=True)

    #Note: we use hx to give memory for our model
    def forward(self, x, hx=None):
        out, hx = self.lnn(x, hx)

        return out, hx