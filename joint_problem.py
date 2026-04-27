import torch
import torch.nn.functional as F
import gymnasium as gym
import lightning as L
from torch.utils.data import DataLoader

from JointSolver import JointSolver
from MLP import MLP
from model import Model


#Conclusion is model weights less and gives similar results as MLP

if __name__ == '__main__':
    lnn_model = Model(6, 3, 64).lnn
    lnn_model = torch.compile(lnn_model)
    solver = JointSolver(lnn_model)

    train_loader = DataLoader(range(500), batch_size=1, num_workers=7)
    trainer = L.Trainer(max_epochs=1, log_every_n_steps=10, enable_progress_bar=True)
    trainer.fit(solver, train_loader)

    mlp = MLP(6, 3)
    lnn_model = torch.compile(mlp)
    solver = JointSolver(mlp)

    train_loader = DataLoader(range(500), batch_size=1, num_workers=7)
    trainer = L.Trainer(max_epochs=1, log_every_n_steps=10, enable_progress_bar=True)
    trainer.fit(solver, train_loader)