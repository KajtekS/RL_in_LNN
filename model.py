from ncps.torch import CfC
from ncps.wirings import AutoNCP

class Model:
    def __init__(self, input_size : int, output_size : int, units : int):
        wiring = AutoNCP(units, output_size)
        self.lnn = CfC(input_size, wiring, batch_first=True)