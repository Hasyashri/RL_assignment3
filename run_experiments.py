"""
run_experiments.py
------------------
Runs multiple DQN training configurations for Pong, compares results,
and generates combined performance plots.
"""

from model import DQNCNN
from replay_buffer import ReplayBuffer
from agent import DQNAgent
from train import train_dqn
import gymnasium as gym
import ale_py
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import glob
import os

# Global constants
image_shape = (84, 80)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_seed(42)

def make_env():
    """Creates Pong environment safely."""
    try:
        env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
    except Exception:
        env = gym.make('Pong-v4', render_mode='rgb_array')
    return env



def plot_combined_results(results_dir='results'):
    """Generates one comparative plot showing all experiments."""
    plt.figure(figsize=(12,6))
    for file in glob.glob(f"{results_dir}/*_rewards.npy"):
        data = np.load(file)
        label = os.path.basename(file).replace('_rewards.npy', '')
        plt.plot(np.convolve(data, np.ones(5)/5, mode='valid'), label=label)
    plt.title("DQN Pong – Comparison of Configurations")
    plt.xlabel("Episode")
    plt.ylabel("5-Episode Moving Avg Reward")
    plt.legend()
    plt.savefig(f"{results_dir}/comparison_plot.png")
    plt.close()

def run_all():
    """Runs all configurations and saves results."""
    configs = [
        {'batch': 8,  'target': 10},
        {'batch': 16, 'target': 10},
        {'batch': 8,  'target': 3},
        {'batch': 16, 'target': 3}
    ]

    for cfg in configs:
        env = make_env()
        n_actions = env.action_space.n

        policy_net = DQNCNN(4, n_actions).to(DEVICE)
        target_net = DQNCNN(4, n_actions).to(DEVICE)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
        buffer = ReplayBuffer(100_000)
        agent = DQNAgent(n_actions, DEVICE)

        label = f"batch{cfg['batch']}_target{cfg['target']}_"
        train_dqn(
            env, policy_net, target_net, optimizer, buffer, agent,
            image_shape=image_shape, num_episodes=500,
            batch_size=cfg['batch'], target_update=cfg['target'],
            save_dir='results', model_prefix=label
        )
        env.close()

    plot_combined_results('results')

if __name__ == "__main__":
    run_all()
