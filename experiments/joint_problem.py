import torch
import torch.nn.functional as F
import gymnasium as gym
import lightning as L
from torch.utils.data import DataLoader

from src.solvers.JointSolver import JointSolver
from src.models.acrobot_MLP import MLP
from src.models.net_model_CfC import CfCNet

#Conclusion is model weights less and gives similar results as MLP
#Try to change in calc_loss using mean/sum at end.

if __name__ == '__main__':
    lnn_model = CfCNet(6, 3, 64).lnn
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