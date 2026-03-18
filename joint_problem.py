import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import gymnasium as gym
import matplotlib.pyplot as plt
import lightning as L

from model import Model


class JointSolver(L.LightningModule):
    def __init__(self, lnn_model):
        super().__init__()
        self.lnn = lnn_model
        self.env = gym.make("Acrobot-v1")

    def training_step(self, batch, batch_idx):
        state, _ = self.env.reset()
        log_probs = []
        rewards = []
        done = False
        hx = None

        while not done:
            state_tensor = torch.FloatTensor(state).view(1, 1, -1).to(self.device)

            output, hx = self.lnn(state_tensor, hx)

            probs = F.softmax(output.view(-1), dim=0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

        loss = self.calc_loss(log_probs, rewards)

        self.log("train_loss", loss, prog_bar=True)
        self.log("episode_reward", sum(rewards), prog_bar=True)

        return loss

    def calc_loss(self, log_probs, rewards):
        gamma = 0.99

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, device=self.device)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for lp, G in zip(log_probs, returns):
            policy_loss.append(-lp * G)

        return torch.stack(policy_loss).sum()

    def configure_optimizers(self):
        return torch.optim.Adam(self.lnn.parameters(), lr=1e-3)

if __name__ == '__main__':
    lnn_model = Model(6, 3, 64).lnn
    solver = JointSolver(lnn_model)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(range(1000), batch_size=1, num_workers=7)
    trainer = L.Trainer(max_epochs=1, log_every_n_steps=10, enable_progress_bar=True)
    trainer.fit(solver, train_loader)