import torch
import torch.nn.functional as F
import gymnasium as gym
import lightning as L
from model import Model

from joint_problem import JointSolver

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def visualize_game_joint(checkpoint_path, model_config=(6, 3, 64)):
    input_size, output_size, hidden_size = model_config
    lnn_inner = Model(input_size, output_size, hidden_size).lnn

    try:
        solver = JointSolver.load_from_checkpoint(checkpoint_path, lnn_model=lnn_inner)
        print(f"Pomyślnie załadowano: {checkpoint_path}")
    except Exception as e:
        print(f"Błąd ładowania checkpointu: {e}")
        return

    env = gym.make("Acrobot-v1", render_mode="human")
    solver.eval()

    device = solver.device
    solver.to(device)

    state, _ = env.reset()
    done = False
    hx = None
    total_reward = 0

    print("Rozpoczynanie wizualizacji...")

    with torch.no_grad():
        while not done:
            state_tensor = torch.FloatTensor(state).view(1, 1, -1).to(device)

            output, hx = solver.lnn(state_tensor, hx)

            probs = F.softmax(output.view(-1), dim=0)
            action = torch.argmax(probs).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            env.render()

            done = terminated or truncated

    print(f"Rozgrywka zakończona. Suma nagród: {total_reward}")
    env.close()


def visualize_game_pool_court(checkpoint_path, model_config=(4, 2, 32)):
    input_size, output_size, hidden_size = model_config
    lnn_inner = Model(input_size, output_size, hidden_size).lnn

    try:
        solver = JointSolver.load_from_checkpoint(checkpoint_path, lnn_model=lnn_inner)
        print(f"Pomyślnie załadowano: {checkpoint_path}")
    except Exception as e:
        print(f"Błąd ładowania checkpointu: {e}")
        return

    env = gym.make("CartPole-v1", render_mode="human")
    solver.eval()

    device = solver.device
    solver.to(device)

    state, _ = env.reset()
    done = False
    hx = None
    total_reward = 0

    print("Rozpoczynanie wizualizacji...")

    with torch.no_grad():
        while not done:
            state_tensor = torch.FloatTensor(state).view(1, 1, -1).to(device)

            output, hx = solver.lnn(state_tensor, hx)

            probs = F.softmax(output.view(-1), dim=0)
            action = torch.argmax(probs).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            env.render()

            done = terminated or truncated

    print(f"Rozgrywka zakończona. Suma nagród: {total_reward}")
    env.close()


if __name__ == '__main__':
    PATH = "./lightning_logs/version_19/checkpoints/epoch=0-step=300.ckpt"

    visualize_game_pool_court(PATH)