import torch
import torch.nn.functional as F
import gymnasium as gym
import lightning as L
from torch.utils.data import DataLoader

from MLP import MLP
from model import Model

input_size = 4
output_size = 2
units = 16

class CartPole(L.LightningModule):
    def __init__(self, lnn):
        super().__init__()
        self.lnn = lnn
        self.env = gym.make("CartPole-v1")

    def training_step(self, batch, batch_idx):
        state, info = self.env.reset()
        log_probs = []
        rewards = []
        hx = None

        done = False

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
        return torch.optim.Adam(self.lnn.parameters(), lr=1e-4)

if __name__ == '__main__':
    lnn_model = Model(4, 2, 32).lnn
    solver = CartPole(lnn_model)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(range(500), batch_size=1, num_workers=7)
    trainer = L.Trainer(max_epochs=1, log_every_n_steps=10, enable_progress_bar=True)
    trainer.fit(solver, train_loader)

    mlp_model = MLP(4, 2)
    solver = CartPole(mlp_model)

    train_loader = DataLoader(range(500), batch_size=1, num_workers=7)
    trainer = L.Trainer(max_epochs=1, log_every_n_steps=10, enable_progress_bar=True)
    trainer.fit(solver, train_loader)