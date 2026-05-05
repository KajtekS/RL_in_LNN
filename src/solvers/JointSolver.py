import lightning as L
import gymnasium as gym
import torch
import torch.nn.functional as F


#Method created in order to train models on AcroRobot

class JointSolver(L.LightningModule):
    def __init__(self, lnn_model):
        super().__init__()
        self.lnn = lnn_model
        self.env = gym.make("Acrobot-v1")

    #Step of train is one game. We perfrom steps until 500-th or reaching line by acrorobot
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

    #There are two possible functions of counting loss.
    # -You can use sum or mean, but sum gives better results

    def calc_loss(self, log_probs, rewards):
        '''
        :param log_probs: probibalities of making move
        :param rewards: reward from move
        :return: policy loss
        '''
        #Discount factor
        gamma = 0.99

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        #To tensor
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        log_probs = torch.stack(log_probs)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        #Counts sum of policy loss
        policy_loss = -(log_probs * returns).mean()

        return policy_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.lnn.parameters(), lr=1e-3)
