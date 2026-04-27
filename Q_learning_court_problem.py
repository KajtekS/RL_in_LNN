from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import gymnasium as gym
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from model import Model

INPUT_DIM = 6
ACTION_DIM = 3
LR = 1e-4

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class RLDataset(Dataset):
    def __init__(self, memory, sample_size=1000):
        self.memory = memory
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    def __getitem__(self, i):
        return self.memory.sample(1)[0]


class ReplayMemory():
    def __init__(self, max_capacity: int, sequence_length=16):
        self.memory = deque([], maxlen=max_capacity)
        self.seq_len = sequence_length

    def push(self, transition):
        self.memory.append(transition)

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        if len(self.memory) < self.seq_len:
            dummy = [self.memory[0]] * self.seq_len if len(self.memory) > 0 else []
            return [dummy] * batch_size

        samples = []
        while len(samples) < batch_size:
            for _ in range(batch_size):
                start_idx = random.randint(0, len(self.memory) - self.seq_len)
                seq = [self.memory[i] for i in range(start_idx, start_idx + self.seq_len)]

            if not any(t.done for t in seq[:-1]):
                samples.append(seq)

        return samples


class Q_learning_trainer(L.LightningModule):
    def __init__(self, policy, target, memory, gamma=0.99, tau=0.005):
        super().__init__()
        self.policy = policy
        self.target = target
        self.gamma = gamma
        self.tau = tau
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = memory

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def training_step(self, batch, batch_idx):
        states_np = np.array([[t.state for t in seq] for seq in batch], dtype=np.float32)
        next_states_np = np.array([[t.next_state for t in seq] for seq in batch], dtype=np.float32)
        actions_np = np.array([seq[-1].action for seq in batch], dtype=np.int64)
        rewards_np = np.array([seq[-1].reward for seq in batch], dtype=np.float32)
        dones_np = np.array([seq[-1].done for seq in batch], dtype=np.float32)

        b_states = torch.from_numpy(states_np).to(self.device)
        b_next_states = torch.from_numpy(next_states_np).to(self.device)
        b_actions = torch.from_numpy(actions_np).to(self.device)
        b_rewards = torch.from_numpy(rewards_np).to(self.device)
        b_dones = torch.from_numpy(dones_np).to(self.device)

        q_values, _ = self.policy(b_states)

        last_step_q = q_values[:, -1, :]  # [64, 3]
        current_q = last_step_q.gather(1, b_actions.unsqueeze(-1))

        with torch.no_grad():
            next_q_values, _ = self.target(b_next_states)
            max_next_q = next_q_values[:, -1, :].max(1)[0]

            expected_q = b_rewards + (self.gamma * max_next_q * (1-b_dones))

        loss = self.loss_fn(current_q, expected_q.unsqueeze(-1))

        current_batch_size = len(batch)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=current_batch_size
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        target_dict = self.target.state_dict()
        policy_dict = self.policy.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * self.tau + target_dict[key] * (1 - self.tau)
        self.target.load_state_dict(target_dict)

        self.build_memory()

    def configure_optimizers(self):
        return optim.Adam(self.policy.parameters(), lr=LR)

    def build_memory(self):
        new_states = []
        env = gym.make("Acrobot-v1")
        state, _ = env.reset()
        hx = None
        with torch.no_grad():
            while len(new_states) < 50:
                state_tensor = torch.as_tensor(state, device=self.device).float().unsqueeze(0)

                output, hx = self.policy(state_tensor, hx)

                if hx is not None:
                    hx = hx.detach()

                probs = F.softmax(output.view(-1), dim=0)
                action = torch.argmax(probs).item()

                if np.random.random() < 0.1:
                    action = env.action_space.sample()

                next_state, reward, terminated, truncated, _ = env.step(action)

                if terminated:
                    reward += 3.0

                new_states.append(Transition(state, action, next_state, reward, terminated or truncated))
                state = next_state

                if terminated or truncated:
                    hx = None
                    state, _ = env.reset()

            self.memory.memory.extend(new_states)


def build_memory(rep_memo: ReplayMemory) -> None:
    env = gym.make("Acrobot-v1")
    state, _ = env.reset()
    capacity = rep_memo.memory.maxlen

    print(f"Inicjalizacja bufora: 0/{capacity}", end="\r")
    while len(rep_memo) < capacity:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        if terminated:
            reward += 3.0

        rep_memo.push(Transition(state, action, next_state, reward, terminated or truncated))
        state = next_state

        if terminated or truncated:
            state, _ = env.reset()

        if len(rep_memo) % 100 == 0:
            print(f"Inicjalizacja bufora: {len(rep_memo)}/{capacity}", end="\r")
    print("\nBufor gotowy.")
    env.close()


if __name__ == '__main__':
    policy_net = Model(INPUT_DIM, ACTION_DIM, 128)
    target_net = Model(INPUT_DIM, ACTION_DIM, 128)
    target_net.load_state_dict(policy_net.state_dict())

    memory = ReplayMemory(max_capacity=50000, sequence_length=32)
    build_memory(memory)

    model_wrapper = Q_learning_trainer(policy_net, target_net, memory, tau=0.001)

    dataset = RLDataset(memory, sample_size=32)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda x: x)

    trainer = L.Trainer(
        max_epochs=400,
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=5
    )

    trainer.fit(model_wrapper, train_loader)