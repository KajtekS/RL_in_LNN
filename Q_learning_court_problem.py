from collections import deque, namedtuple
from random import random
import torch.optim as optim
from sympy.printing.pytorch import torch

from model import Model

#Consts
LR = 1e-3

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    with torch.no_grad():
        return policy_net()

class ReplayMemory():
    def __init__(self, max_capacity: int):
        self.memory = deque([], max_capacity)
    def __len__(self):
        return len(self.memory)
    def push(self, *args):
        self.memory.append(*args)
    def sample(self, batch_size):
        return random.sample(list(self.memory), batch_size)

policy_net = Model(4, 2)
target_net = Model(4, 2)

optimizer = optim.AdamW(policy_net.lnn.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(1000)

