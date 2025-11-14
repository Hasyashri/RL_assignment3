"""
train.py
---------
Handles DQN training loop, evaluation, and logging.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import DQNCNN
from replay_buffer import ReplayBuffer
from agent import DQNAgent
from assignment3_utils import process_frame, transform_reward
from typing import List, Tuple
import os

# ---------------- Helper: Evaluate agent ----------------
def evaluate_agent(env, model, image_shape, episodes=5):
    """
    Run the trained model without exploration (epsilon=0).
    Return average reward.
    """
    model.eval()
    scores = []

    for ep in range(episodes):
        obs, _ = env.reset()
        frame = np.squeeze(process_frame(obs, image_shape))  # remove extra dim
        state = np.stack([frame]*4, axis=0)                  # stack 4 frames
        done = False
        total_reward = 0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_t)
            action = q_values.argmax(1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = np.squeeze(process_frame(obs, image_shape))
            state = np.append(state[1:], [frame], axis=0)
            total_reward += reward

        scores.append(total_reward)

    avg = np.mean(scores)
    print(f"✅ Evaluation avg reward over {episodes} episodes: {avg:.2f}")
    return avg

# ---------------- DQN Training ----------------
def train_dqn(
    env,
    policy_net: DQNCNN,
    target_net: DQNCNN,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    agent: DQNAgent,
    image_shape: Tuple[int, int],
    num_episodes: int = 500,
    batch_size: int = 8,
    target_update: int = 10,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.995,
    save_dir: str = 'results',
    model_prefix: str = ''
) -> Tuple[List[float], List[float]]:
    """
    Train a DQN agent on Pong environment.
    """
    device = next(policy_net.parameters()).device
    epsilon = eps_start
    rewards_per_episode = []
    avg_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        frame = np.squeeze(process_frame(obs, image_shape))
        state = np.stack([frame]*4, axis=0)  # shape: (4, H, W)
        done = False
        total_reward = 0

        while not done:
            # ---------------- Select Action ----------------
            action = agent.select_action(state, policy_net, epsilon)

            # ---------------- Take Step ----------------
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = np.squeeze(process_frame(obs, image_shape))
            next_state = np.append(state[1:], [frame], axis=0)

            # ---------------- Store Experience ----------------
            buffer.push(state, action, transform_reward(reward), next_state, done)
            state = next_state
            total_reward += reward

            # ---------------- Learn ----------------
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states_v = torch.tensor(states, dtype=torch.float32).to(device)
                next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
                actions_v = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards_v = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                dones_v = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                # Q(s,a)
                q_vals = policy_net(states_v).gather(1, actions_v)

                # Q_target(s',a')
                next_q_vals = target_net(next_states_v).max(1)[0].unsqueeze(1).detach()

                q_targets = rewards_v + gamma * next_q_vals * (1 - dones_v)

                # Compute loss and backprop
                loss = torch.nn.MSELoss()(q_vals, q_targets)
                assert not torch.isnan(loss), 'Loss became NaN!'
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ---------------- Decay epsilon ----------------
            epsilon = max(epsilon * eps_decay, eps_end)

        # ---------------- Update target network ----------------
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # ---------------- Logging ----------------
        rewards_per_episode.append(total_reward)
        avg_5 = np.mean(rewards_per_episode[-5:])
        avg_rewards.append(avg_5)

        if episode % 10 == 0:
            print(f"Ep {episode:03d} | Reward: {total_reward:.1f} | Avg5: {avg_5:.2f} | Eps: {epsilon:.2f}")

    # ---------------- Evaluation ----------------
    evaluate_agent(env, policy_net, image_shape)

    # ---------------- Save results ----------------
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(policy_net.state_dict(), f'{save_dir}/{model_prefix}dqn_pong.pth')
    np.save(f'{save_dir}/{model_prefix}_rewards.npy', rewards_per_episode)

    # ---------------- Plot ----------------
    plt.figure(figsize=(12,6))
    plt.plot(rewards_per_episode, label='Episode Reward')
    plt.plot(avg_rewards, label='5-ep Avg')
    plt.title(f"DQN Pong: {model_prefix}")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f'{save_dir}/{model_prefix}_reward_plot.png')
    plt.close()

    return rewards_per_episode, avg_rewards

